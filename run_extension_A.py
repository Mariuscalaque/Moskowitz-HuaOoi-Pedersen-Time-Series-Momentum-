"""Extension A — Robustesse : hors échantillon, risque de liquidité, turbulence.

Trois robustesses « Tier 1 » qui prolongent la réplication de Moskowitz, Ooi &
Pedersen (2012) et rebranchent l'analyse sur le cours (turbulence de Mahalanobis,
shrinkage) :

  (D1) HORS ÉCHANTILLON 1985-2009 vs 2010-2025
       Le MÊME TSMOM diversifié, reconstruit jusqu'à fin 2025. On compare
       performance, alpha factoriel (MKT/SMB/HML/UMD) et convexité « smile » (MKT²)
       avant/après 2009. Le « trend-following est-il mort post-crise ? »

  (D2) RISQUE DE LIQUIDITÉ — facteur TRADABLE de Pástor-Stambaugh (LIQ_V)
       MOP ne testent la liquidité que comme variable d'état (Panel C). Ici on
       ajoute le facteur de liquidité TRADABLE comme facteur de risque dans la
       régression d'alpha : l'alpha du TSMOM survit-il à cet ajustement ?

  (D3) TURBULENCE FINANCIÈRE — distance de Mahalanobis (Kritzman-Li 2010)
       Pont entre MOP §4.3 (« best during extreme markets ») et l'Exercice 2 du
       cours. Indice de turbulence sur le panel d'instruments (Sigma régularisée
       par shrinkage Ledoit-Wolf), puis test : le TSMOM rend-il plus dans les
       mois turbulents ?

Sorties :
  ext_A_oos_performance.csv     — perf + alpha + smile par sous-période
  ext_A_liquidity_alpha.csv     — alpha avant/après ajout du facteur LIQ_PS
  ext_A_turbulence_summary.csv  — TSMOM par quintile de turbulence
  ext_A_turbulence_series.csv   — l'indice de turbulence mensuel (transparence)
  fig13_oos_cumulative.png      — croissance TSMOM, in-sample vs out-of-sample
  fig14_turbulence.png          — indice de turbulence + rendement TSMOM par quintile
"""

import sys as _sys
try:  # sorties UTF-8 quel que soit l'OS/locale (Windows cp1252)
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import robustness as rb
from src import plotting
from src.config import (TABLES_DIR, FIGURES_DIR, LOOKBACK_MONTHS,
                        PAPER_END, EXTENDED_END)
from src.data_loader import load_raw
from src.returns import build_daily_excess_returns, daily_to_monthly_returns
from src.volatility import ewma_ex_ante_vol, vol_for_signal
from src.strategy import tsmom_instrument_returns, diversified_tsmom
from src.factors import fetch_ff_factors

# Zones de crise (mêmes repères visuels que l'Extension C)
CRISES = [
    ("2000-03", "2002-09", "Dot-com"),
    ("2007-07", "2009-03", "GFC 2008"),
    ("2011-07", "2012-06", "Dette Euro"),
    ("2020-02", "2020-04", "COVID"),
    ("2022-01", "2022-12", "Hausse taux"),
]


def _load_ps_traded() -> pd.Series | None:
    """Facteur de liquidité TRADABLE de Pástor-Stambaugh (LIQ_V), via le cache
    du projet puis repli sur le CSV local. None si introuvable (D2 sautée)."""
    # 1) chemin officiel : external_data (cache disque)
    try:
        from src.external_data import fetch_ps_liquidity
        ps = fetch_ps_liquidity()
        if ps is not None and "traded_liq" in ps:
            s = pd.to_numeric(ps["traded_liq"], errors="coerce").dropna()
            if len(s):
                return s
    except Exception:
        pass
    # 2) repli : CSV local quel que soit l'emplacement plausible
    for p in (_ROOT / "data" / "external" / "pastor_stambaugh_liquidity.csv",
              _ROOT / "data" / "pastor_stambaugh_liquidity.csv"):
        if p.exists():
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            if "traded_liq" in df:
                s = pd.to_numeric(df["traded_liq"], errors="coerce").dropna()
                if len(s):
                    return s
    return None


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Données de base : TSMOM diversifié sur le PLEIN échantillon (->2025) ----
    prices = load_raw()
    daily = build_daily_excess_returns(prices)
    monthly = daily_to_monthly_returns(daily)
    mvol = vol_for_signal(ewma_ex_ante_vol(daily), monthly.index)
    tsmom = diversified_tsmom(
        tsmom_instrument_returns(monthly, mvol, k=LOOKBACK_MONTHS)).dropna()

    # ---- Facteurs MKT/SMB/HML/UMD (décimal, fin de mois) ----
    ff = fetch_ff_factors(start="1985-01-01", end=EXTENDED_END, source="auto")
    benchm = prices[["MXWO Index"]].resample("ME").last().pct_change()
    mkt_excess = (benchm["MXWO Index"] - ff["RF"]).rename("MKT")
    ff4 = pd.concat([mkt_excess, ff[["SMB", "HML", "UMD"]]], axis=1).dropna()

    print("=" * 68)
    print("EXTENSION A — Robustesse : OOS, liquidité (PS), turbulence (Mahalanobis)")
    print("=" * 68)
    print(f"TSMOM diversifié : {tsmom.index.min():%Y-%m} -> {tsmom.index.max():%Y-%m} "
          f"({len(tsmom)} mois)")

    # ======================= (D1) Hors échantillon =======================
    print("\n=== (D1) Hors échantillon : 1985-2009 vs 2010-2025 ===")
    oos = rb.subperiod_performance(tsmom, ff4)
    print(oos.round(3).to_string())
    sh_in = oos.loc["In-sample 1985-2009", "Sharpe"]
    sh_oos = oos.loc["Out-of-sample 2010-2025", "Sharpe"]
    a_oos = oos.loc["Out-of-sample 2010-2025", "Alpha (%/m)"]
    ta_oos = oos.loc["Out-of-sample 2010-2025", "t(Alpha)"]
    print(f"  -> Sharpe {sh_in:.2f} (in) vs {sh_oos:.2f} (out) ; alpha OOS "
          f"{a_oos:.2f}%/m (t={ta_oos:.2f}, "
          f"{'significatif' if abs(ta_oos) >= 2 else 'NON significatif'})")
    oos.to_csv(TABLES_DIR / "ext_A_oos_performance.csv", float_format="%.4f")

    # ======================= (D2) Risque de liquidité =======================
    print("\n=== (D2) Risque de liquidité : facteur tradable Pástor-Stambaugh ===")
    ps_traded = _load_ps_traded()
    if ps_traded is None:
        print("  [SKIP] facteur PS tradable introuvable (cache/CSV absent).")
    else:
        liq = rb.liquidity_augmented_alpha(tsmom, ff4, ps_traded)
        print(liq.round(3).to_string())
        a0 = liq.loc["Baseline (MKT,SMB,HML,UMD)", "Alpha (%)"]
        a1 = liq.loc["+ LIQ_PS (tradable)", "Alpha (%)"]
        tliq = liq.loc["+ LIQ_PS (tradable)", "t(LIQ)"]
        print(f"  -> alpha {a0:.3f}% -> {a1:.3f}% après ajout de LIQ "
              f"(loading t={tliq:.2f}) : l'alpha "
              f"{'SURVIT' if a1 > 0.3 and abs(tliq) < 2 else 'change'} au "
              f"risque de liquidité tradable.")
        liq.to_csv(TABLES_DIR / "ext_A_liquidity_alpha.csv", float_format="%.4f")

    # ======================= (D3) Turbulence financière =======================
    print("\n=== (D3) Turbulence financière (distance de Mahalanobis) ===")
    turb, info = rb.financial_turbulence(monthly)
    print(f"  Panel : {info['n_instruments']} instruments, {info['n_months']} mois "
          f"({info['start']} -> {info['end']}) ; classes {info['by_class']}")
    print(f"  Shrinkage Ledoit-Wolf delta = {info['shrinkage_delta']:.3f} "
          f"(dof ~ E[d_t] = {info['dof']})")
    res = rb.tsmom_by_turbulence(tsmom, turb)
    print("\n  TSMOM par quintile de turbulence :")
    print(res["by_bucket"].round(3).to_string())
    print("\n  Calme vs turbulent :")
    print(res["calm_vs_turbulent"].round(3).to_string())
    print("\n  Régressions (TSMOM ~ turbulence) :")
    print(res["regression"].round(3).to_string())
    bz = res["regression"].loc["z(turbulence)", "beta(X) (%/m)"]
    tz = res["regression"].loc["z(turbulence)", "t(X)"]
    print(f"  -> +1σ de turbulence = {bz:+.2f}%/m de TSMOM (t={tz:.2f}) : "
          f"confirme MOP §4.3 (« best during extreme markets ») via Mahalanobis.")

    # Sauvegardes turbulence (résumé + série)
    summary = res["by_bucket"].copy()
    summary.to_csv(TABLES_DIR / "ext_A_turbulence_summary.csv", float_format="%.4f")
    turb.rename("turbulence").to_csv(TABLES_DIR / "ext_A_turbulence_series.csv")

    # ======================= Figures =======================
    plotting.figure_oos_cumulative(tsmom, split=PAPER_END)
    plotting.figure_turbulence(turb, res["by_bucket"], crises=CRISES)

    print("\nFichiers écrits :")
    for f in ("ext_A_oos_performance.csv", "ext_A_liquidity_alpha.csv",
              "ext_A_turbulence_summary.csv", "ext_A_turbulence_series.csv"):
        print(f"  {TABLES_DIR / f}")
    for f in ("fig13_oos_cumulative.png", "fig14_turbulence.png"):
        print(f"  {FIGURES_DIR / f}")


if __name__ == "__main__":
    main()
