"""
validate_replication.py — Vérifie la fidélité de la réplication ET les invariants
                          des extensions copule.

1) Reconstruit le facteur TSMOM (toutes classes + par classe) ;
2) le compare au facteur TSMOM OFFICIEL d'AQR (aqr_tsmom_factors.csv) ;
3) vérifie l'identité comptable de la décomposition Lo-MacKinlay (Table 5B) ;
4) vérifie l'identité Futures = Spot + Roll (Table 6) ;
5) (NOUVEAU) vérifie les invariants mathématiques des extensions A/B/C lus depuis
   outputs/tables (familles valides, λ ∈ [0,1], cohérence de la sélection AIC).
   Ces invariants sont rapides : ils valident la cohérence des sorties produites.
   Pour une RE-CALCUL complet des extensions, lancer regenerate_all.py.

Usage :
    TSMOM_DATA=data/data.xlsx TSMOM_DATA_DIR=data python validate_replication.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

from src.data_loader import load_raw
from src.returns import build_daily_excess_returns, daily_to_monthly_returns
from src.volatility import ewma_ex_ante_vol, vol_for_signal
from src.strategy import tsmom_instrument_returns, diversified_tsmom, tsmom_by_asset_class
from src.crosssectional import decomposition_by_asset_class
from src.rollyield import spot_roll_monthly
from src.config import LOOKBACK_MONTHS, TABLES_DIR

S, E = "1985-01-31", "2009-12-31"
prices = load_raw()
daily = build_daily_excess_returns(prices)
monthly = daily_to_monthly_returns(daily)
mvol = vol_for_signal(ewma_ex_ante_vol(daily), monthly.index)
inst = tsmom_instrument_returns(monthly, mvol, k=LOOKBACK_MONTHS)
mine_all = diversified_tsmom(inst)
mine_ac = tsmom_by_asset_class(inst)

m = (mine_all.index >= S) & (mine_all.index <= E)
tp = mine_all.loc[m].dropna()
print("="*64)
print(f"1) TSMOM diversifié 1985-2009 : Sharpe={tp.mean()/tp.std()*np.sqrt(12):.2f}  "
      f"vol={tp.std()*np.sqrt(12):.1%}  (papier : SR>1, vol~12%)")

aqr_path = Path("data/external/aqr_tsmom_factors.csv")
if not aqr_path.exists(): aqr_path = Path("aqr_tsmom_factors.csv")
if aqr_path.exists():
    aqr = pd.read_csv(aqr_path); aqr.columns = ["date","ALL","CM","EQ","FI","FX"]
    aqr["date"] = pd.to_datetime(aqr["date"]); aqr = aqr.set_index("date")
    aqr.index = aqr.index + pd.offsets.MonthEnd(0)
    def corr(a, b):
        df = pd.concat([a.rename("m"), b.rename("a")], axis=1).dropna()
        df = df.loc[(df.index>=S)&(df.index<=E)]
        return df["m"].corr(df["a"])
    print("\n2) Corrélation avec le facteur TSMOM officiel AQR :")
    print(f"   ALL={corr(mine_all,aqr['ALL']):.3f}  CM={corr(mine_ac['Commodity'],aqr['CM']):.3f}  "
          f"EQ={corr(mine_ac['Equity'],aqr['EQ']):.3f}  "
          f"FI={corr(mine_ac['Bond'],aqr['FI']):.3f}  FX={corr(mine_ac['Currency'],aqr['FX']):.3f}")
else:
    print("\n2) aqr_tsmom_factors.csv introuvable — comparaison AQR sautée.")

dec = decomposition_by_asset_class(monthly.loc[m])
err = (dec["XS_Total(sum)"] - dec["XS_Total(emp)"]).abs().max()
print(f"\n3) Décomposition Lo-MacKinlay 12->1 : identité Auto+Cross+Mean=empirique, "
      f"écart max={err*1e4:.2f} bp (doit être ~0)")

comp = spot_roll_monthly(prices)
idn = (comp["total"] - comp["spot"] - comp["roll"]).abs().max().max()
print(f"4) Décomposition Futures=Spot+Roll : écart max={idn:.1e} (doit être 0)")

# =====================================================================
# 5) Invariants des extensions copule (A/B/C) — lus depuis outputs/tables
#    (prints volontairement ASCII-only pour les consoles Windows cp1252)
# =====================================================================
ALLOWED = {"Gaussian", "Student-t", "Clayton", "Gumbel", "Frank"}
TOL = 1e-6

def _col(df, *subs):
    """1re colonne contenant l'une des sous-chaines (insensible a la casse)."""
    for c in df.columns:
        cl = str(c).lower()
        if any(s.lower() in cl for s in subs):
            return c
    return None

def _in01(series):
    v = pd.to_numeric(series, errors="coerce")
    return bool(((v >= -TOL) & (v <= 1 + TOL)).all())

print("\n5) Invariants des extensions copule (A/B/C) :")
checks = []   # (label, ok, detail)

# --- Extension A : selection de copule plein-echantillon ---
pA = TABLES_DIR / "ext_A_copula_selection.csv"
if pA.exists():
    a = pd.read_csv(pA)
    fam_col = _col(a, "family")
    fam_ok = set(a[fam_col]).issubset(ALLOWED) if fam_col else False
    rng_ok = _in01(a["lambda_L"]) and _in01(a["lambda_U"])

    def _struct_ok(row):
        f = row[fam_col]; lL = float(row["lambda_L"]); lU = float(row["lambda_U"])
        if f in ("Gaussian", "Frank"): return abs(lL) < 1e-6 and abs(lU) < 1e-6
        if f == "Clayton":             return abs(lU) < 1e-6           # queue basse seule
        if f == "Gumbel":              return abs(lL) < 1e-6           # queue haute seule
        if f == "Student-t":           return abs(lL - lU) < 1e-6      # symetrique
        return False
    struct_ok = bool(a.apply(_struct_ok, axis=1).all())

    checks += [
        ("Ext A : familles dans les 5 familles autorisees", fam_ok, ""),
        ("Ext A : lambda_L, lambda_U dans [0,1]",            rng_ok, ""),
        ("Ext A : lambda coherents avec la structure de la famille", struct_ok, ""),
    ]
else:
    print("   ext_A_copula_selection.csv absent — Ext A sautee.")

# --- Extension B : rolling copula ---
pB = TABLES_DIR / "ext_B_rolling_copula.csv"
if pB.exists():
    b = pd.read_csv(pB)
    fam_col = _col(b, "best_family")
    fam_ok = set(b[fam_col]).issubset(ALLOWED) if fam_col else False
    rng_ok = _in01(b["lambda_L"]) and _in01(b["lambda_U"])
    aicb = pd.to_numeric(b["AIC_best"],  errors="coerce")
    aicg = pd.to_numeric(b["AIC_gauss"], errors="coerce")
    # best = argmin AIC sur un ensemble qui CONTIENT la gaussienne -> AIC_best <= AIC_gauss
    gap = float((aicb - aicg).max())
    sel_ok = bool(gap <= 1e-6)
    checks += [
        ("Ext B : familles dans les 5 familles autorisees", fam_ok, ""),
        ("Ext B : lambda dans [0,1] sur toutes les fenetres", rng_ok, ""),
        ("Ext B : AIC_best <= AIC_gauss (selection coherente)", sel_ok,
         f"max(AIC_best-AIC_gauss)={gap:.2e}"),
    ]
else:
    print("   ext_B_rolling_copula.csv absent — Ext B sautee.")

# --- Extension C : dependance de queue par classe ---
pC = TABLES_DIR / "ext_C_copula_by_class.csv"
if pC.exists():
    c = pd.read_csv(pC)
    fam_col  = _col(c, "best copula")
    daic_col = _col(c, "aic vs", "delta", "\u0394aic")
    lLp_col  = _col(c, "_l (param", "l_l (param")
    lUp_col  = _col(c, "_u (param", "l_u (param")
    fam_ok  = set(c[fam_col]).issubset(ALLOWED) if fam_col else False
    daic_ok = bool((pd.to_numeric(c[daic_col], errors="coerce") >= -1e-6).all()) \
              if daic_col else False
    rng_ok  = True
    for col in (lLp_col, lUp_col):
        if col is not None:
            rng_ok &= _in01(c[col])
    checks += [
        ("Ext C : familles dans les 5 familles autorisees", fam_ok, ""),
        ("Ext C : DeltaAIC vs Gaussian >= 0 (best >= gaussienne)", daic_ok, ""),
        ("Ext C : lambda_L, lambda_U parametriques dans [0,1]", rng_ok, ""),
    ]
else:
    print("   ext_C_copula_by_class.csv absent — Ext C sautee.")

if checks:
    for label, ok, detail in checks:
        tag = " OK " if ok else "FAIL"
        print(f"   [{tag}] {label}" + (f"  ({detail})" if detail else ""))
    n_ok = sum(1 for _, ok, _ in checks if ok)
    print(f"   -> {n_ok}/{len(checks)} invariants valides.")
    if n_ok < len(checks):
        print("   (!) Un invariant a echoue : relancer regenerate_all.py et verifier.")
else:
    print("   (aucune sortie d'extension trouvee — lancer regenerate_all.py d'abord)")

print("="*64)
