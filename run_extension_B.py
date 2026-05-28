"""Extension B — Le « smile » de TSMOM comme DÉPENDANCE DE QUEUE (copules).

OBJET (lien réplication ↔ cours) : la Table 3C du papier teste le profil de
STRADDLE de TSMOM par une régression TSMOM ~ MKT + MKT². Dans notre réplication,
le terme MKT² ressort NON significatif (t ≈ 1.1). Ce test paramétrique est
symétrique, global, et dilue l'effet d'option. Or la prédiction économique
(TSMOM gagne dans les grands mouvements des DEUX côtés du marché) est une
affirmation de DÉPENDANCE DE QUEUE. On la teste donc avec l'outillage copules
du cours, là où le polynôme échoue.

Méthode = approche CANONIQUE / CML (méthode ③ du cours) :
  (1) marges filtrées AR(1)-GARCH(1,1)-t -> résidus iid (invariants de Meucci) ;
  (2) PIT empirique (rangs, Deheuvels) -> pseudo-observations (invariance) ;
  (3) sélection de copules (AIC/BIC) + dépendance de queue λ_L, λ_U + GOF CvM ;
  (4) MESURES DE DÉPENDANCE du cours (Spearman/Kendall/Blomqvist/SW/Hoeffding)
      + impossibilité d'Embrechts (1999) ;
  (5) encadrement de FRÉCHET (copule empirique vs C⁻, C⁺, produit) ;
  (+) copule de PANIQUE de Meucci (mélange convexe de 2 gaussiennes).
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
import matplotlib.pyplot as plt

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import copula_extension as ce
from src.factors import fetch_ff_factors
import os as _os
from src.config import FIGURES_DIR as _FIG_DEF, TABLES_DIR as _TAB_DEF
FIGURES_DIR = Path(_os.environ.get("TSMOM_FIG_DIR", str(_FIG_DEF)))
TABLES_DIR = Path(_os.environ.get("TSMOM_TAB_DIR", str(_TAB_DEF)))

_TSMOM_PATH = Path(_os.environ.get("TSMOM_SERIES_CSV", "outputs/tables/diversified_tsmom_series.csv"))
_DATA_PATH = Path("data/data_monthly_returns.csv")
_SMILE_PATH = Path("outputs/tables/table3_panelC_smile.csv")

# ---- 1. TSMOM diversifié (issu de la réplication corrigée) ----
tsmom = (pd.read_csv(_TSMOM_PATH, parse_dates=["date"])
         .set_index("date")["TSMOM"].dropna())

# ---- 2. Marché en excès = MXWO - RF (même définition que le test du smile) ----
mx = (pd.read_csv(_DATA_PATH, parse_dates=["date"])
      .set_index("date")["MXWO Index"])
_ff = fetch_ff_factors(start="1900-01-01", end="2100-12-31", source="auto")
rf = _ff["RF"]
mkt = (mx - rf.reindex(mx.index)).dropna().rename("MKT")

# ---- (1) Filtrage AR(1)-GARCH(1,1)-t des marges (invariants de Meucci) ----
print("=== (1) Filtrage des marges AR(1)-GARCH(1,1)-t (invariants iid) ===")
z_ts, info_ts = ce.filter_marginal(tsmom)
z_mk, info_mk = ce.filter_marginal(mkt)
print("TSMOM :", {k: round(v, 3) for k, v in info_ts.items()})
print("MKT   :", {k: round(v, 3) for k, v in info_mk.items()})

# ---- (2) PIT empirique (méthode canonique) -> pseudo-observations ----
u, idx = ce.to_pseudo_obs(z_ts.rename("TS"), z_mk.rename("MK"))
print(f"\nÉchantillon joint : N = {len(u)} mois ({idx.min():%Y-%m} -> {idx.max():%Y-%m})")

# ---- (3) Sélection des copules (AIC/BIC) + dépendance de queue ----
print("\n=== (3) Selection des copules (AIC/BIC) et dependance de queue ===")
fit = ce.fit_all_copulas(u)

# (+) Copule de PANIQUE de Meucci ajoutee comme famille candidate
panic = ce.fit_panic_copula(u)
fit.loc["Panic (Meucci)"] = {
    "logL": panic["logL"], "AIC": panic["AIC"], "BIC": panic["BIC"],
    "lambda_L": panic["lambda_L_emp_sim"], "lambda_U": panic["lambda_U_emp_sim"],
    "param": panic["param"], "N": panic["N"],
}
fit_show = fit.copy()
for c in ["logL", "AIC", "BIC", "lambda_L", "lambda_U"]:
    fit_show[c] = fit_show[c].astype(float).round(3)
print(fit_show.to_string())
best_aic = fit.drop(index="Panic (Meucci)")["AIC"].astype(float).idxmin()
best_bic = fit.drop(index="Panic (Meucci)")["BIC"].astype(float).idxmin()
print(f"\nMeilleure copule parametrique  AIC: {best_aic}   |   BIC: {best_bic}")
print(f"Copule de panique (Meucci) : rho_calm={panic['rho_calm']:.3f}  "
      f"rho_panic={panic['rho_panic']:.3f}  p={panic['p_panic']:.3f}  AIC={panic['AIC']:.2f}")

# ---- Dependance de queue NON-PARAMETRIQUE ----
print("\n=== Dependance de queue empirique (model-free) ===")
emp10 = ce.empirical_tail_dependence(u, q=0.10)
emp15 = ce.empirical_tail_dependence(u, q=0.15)
for e in (emp10, emp15):
    print({k: round(v, 3) for k, v in e.items()})

# ---- (4) MESURES DE DEPENDANCE DU COURS (pp. 31-34) + Embrechts ----
print("\n=== (4) Mesures de dependance du cours (sur la copule) ===")
dm = ce.dependence_measures(u)
_zt = z_ts.reindex(idx); _zm = z_mk.reindex(idx)
pearson_levels = float(np.corrcoef((_zt - _zt.mean()) / _zt.std(),
                                   (_zm - _zm.mean()) / _zm.std())[0, 1])
print(f"  Pearson (niveaux, residus)   : {pearson_levels:+.3f}   "
      f"[NON invariant, NON-(5) : pas une vraie mesure de dependance]")
print(f"  Spearman rho_S (rang)        : {dm['Spearman_rho']:+.3f}")
print(f"  Kendall  tau   (rang)        : {dm['Kendall_tau']:+.3f}")
print(f"  Blomqvist beta = 4.C(1/2)-1  : {dm['Blomqvist_beta']:+.3f}")
print(f"  Schweizer-Wolff sigma        : {dm['Schweizer_Wolff_sigma']:.3f}")
print(f"  Hoeffding Phi                : {dm['Hoeffding_Phi']:.3f}")
print("  Rappel Embrechts (1999) : aucune mesure ne verifie A LA FOIS l'invariance")
print("  par transfo monotone (4) et independance<=>delta=0 (5). Les mesures de RANG")
print("  verifient (1)-(4) ; la dependance de QUEUE (lambda) capte le (5) LOCAL.")

# ---- Test d'adequation CvM : Gaussienne (H0) et meilleure famille ----
print("\n=== Test d'adequation Cramer-von Mises (bootstrap parametrique) ===")
def parse_param(row):
    d = {}
    for kv in row["param"].split(", "):
        k, v = kv.split("=")
        if k in ("rho", "df", "theta"):
            d[k] = float(v)
    return d
gof_rows = []
for fam in ["Gaussian", best_aic]:
    par = parse_param(fit.loc[fam])
    g = ce.cvm_gof(u, fam, par, n_boot=250)
    gof_rows.append(g)
    print(f"  {fam:10s}  Sn={g['Sn']:.4f}  p-value={g['p_value']:.3f}")
gauss_rejected = gof_rows[0]["p_value"] < 0.05
best_not_rejected = gof_rows[1]["p_value"] > 0.05

# ---- Correlations d'exceedance (asymetrie krach vs boom) ----
exc = ce.exceedance_correlation(z_ts.reindex(idx), z_mk.reindex(idx))
print("\n=== Correlations d'exceedance (residus standardises) ===")
print(exc.round(3).to_string(index=False))

# ---- (5) Encadrement de FRECHET ----
fdiag = ce.frechet_diagonal(u)

# ---- VERDICT : le smile que la regression rate, la copule le voit-elle ? ----
smile_beta = smile_t = np.nan
if _SMILE_PATH.exists():
    sm = pd.read_csv(_SMILE_PATH, index_col=0)
    row = sm.iloc[0]
    smile_beta = float(row.get("MKT_sq", np.nan))
    smile_t = float(row.get("t(MKT_sq)", np.nan))
exc_tail = exc[exc["threshold"] >= 0.8]
excess_neg = float((exc_tail["corr_neg"] - exc_tail["corr_neg_gauss"]).mean())
excess_pos = float((exc_tail["corr_pos"] - exc_tail["corr_pos_gauss"]).mean())
print("\n" + "=" * 64)
print("VERDICT — Le straddle : regression (Table 3C) vs copules (Ext. B)")
print("=" * 64)
print(f"  Test parametrique du papier : beta(MKT2)={smile_beta:.3f}  t={smile_t:.2f}  "
      f"-> {'NON significatif' if abs(smile_t) < 2 else 'significatif'}")
print(f"  lambda_L empirique (q=10%)  : {emp10['lambda_L_emp']:.3f}  (co-krach)")
print(f"  lambda_U empirique (q=10%)  : {emp10['lambda_U_emp']:.3f}  (co-boom)")
print(f"  Exces de queue vs gaussien  : krachs {excess_neg:+.3f} | booms {excess_pos:+.3f}")
print(f"  Meilleure copule (AIC)      : {best_aic}  (lambda={fit.loc[best_aic,'lambda_L']:.3f})")
print(f"  GOF : gaussienne rejetee ? {gauss_rejected} | {best_aic} non rejetee ? {best_not_rejected}")
if not gauss_rejected:
    print("  -> Lecture HONNETE : au niveau diversifie l'AIC prefere la dependance")
    print("     de queue mais le GOF ne REJETTE PAS la gaussienne (N limite, peu de")
    print("     co-extremes). Evidence SUGGESTIVE du straddle, non concluante ici ;")
    print("     l'Extension D montre qu'elle est NETTE pour les actions (Delta AIC>>0).")

# ======================= SORTIES ========================
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

res = fit.copy()
res.insert(0, "Family", res.index)
res.to_csv(TABLES_DIR / "ext_B_copula_selection.csv", index=False)

summary = pd.DataFrame({
    "Best copula (AIC)": [best_aic],
    "Best copula (BIC)": [best_bic],
    "lambda_L (best)": [float(fit.loc[best_aic, "lambda_L"])],
    "lambda_U (best)": [float(fit.loc[best_aic, "lambda_U"])],
    "lambda_L emp (q=.10)": [emp10["lambda_L_emp"]],
    "lambda_U emp (q=.10)": [emp10["lambda_U_emp"]],
    "asymmetry emp (L-U, q=.10)": [emp10["asymmetry_L_minus_U"]],
    "Spearman rho": [dm["Spearman_rho"]],
    "Kendall tau": [dm["Kendall_tau"]],
    "Blomqvist beta": [dm["Blomqvist_beta"]],
    "Schweizer-Wolff sigma": [dm["Schweizer_Wolff_sigma"]],
    "Hoeffding Phi": [dm["Hoeffding_Phi"]],
    "Pearson (levels)": [pearson_levels],
    "Panic rho_calm": [panic["rho_calm"]],
    "Panic rho_panic": [panic["rho_panic"]],
    "Panic p": [panic["p_panic"]],
    "Smile beta(MKT^2)": [smile_beta],
    "Smile t(MKT^2)": [smile_t],
    "Tail excess vs gauss (neg)": [excess_neg],
    "Gaussian rejected? (p<.05)": [gauss_rejected],
    "Best not rejected? (p>.05)": [best_not_rejected],
    "N": [len(u)],
}).T.rename(columns={0: "value"})
summary.to_csv(TABLES_DIR / "ext_B_summary.csv")
print("\n=== RESUME ===")
print(summary.to_string())

# Figure 8 — 3 panneaux : (a) exceedance, (b) pseudo-obs, (c) Frechet
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.7))
ax1.plot(exc["threshold"], exc["corr_neg"], "o-", color="#c0392b",
         label="Krachs joints (empirique)")
ax1.plot(exc["threshold"], exc["corr_pos"], "s-", color="#2980b9",
         label="Booms joints (empirique)")
ax1.plot(exc["threshold"], exc["corr_neg_gauss"], "--", color="#c0392b", alpha=0.55,
         label="Krachs - benchmark gaussien")
ax1.plot(exc["threshold"], exc["corr_pos_gauss"], "--", color="#2980b9", alpha=0.55,
         label="Booms - benchmark gaussien")
ax1.axhline(0, color="grey", lw=0.7)
ax1.set_xlabel("Seuil |z| (ecarts-types)")
ax1.set_ylabel("Correlation d'exceedance TSMOM-Marche")
ax1.set_title(f"(a) Le straddle dans les queues\n(smile parametrique : t={smile_t:.1f}, NS)")
ax1.legend(fontsize=7)

ax2.scatter(u[:, 0], u[:, 1], s=10, alpha=0.5, color="#34495e")
ax2.set_xlabel("u = F(residu TSMOM)")
ax2.set_ylabel("v = F(residu Marche)")
ax2.set_title("(b) Pseudo-observations (copule empirique)")

ax3.plot(fdiag["t"], fdiag["C_plus"],  color="#c0392b", lw=1.3, label="C+ comonotone (max)")
ax3.plot(fdiag["t"], fdiag["C_indep"], color="#27ae60", lw=1.3, ls="--", label="Pi independance (t2)")
ax3.plot(fdiag["t"], fdiag["C_minus"], color="#2980b9", lw=1.3, label="C- contre-monotone")
ax3.plot(fdiag["t"], fdiag["C_emp"], "o", color="#000", ms=2.5, label="C empirique (Deheuvels)")
ax3.fill_between(fdiag["t"], fdiag["C_minus"], fdiag["C_plus"], color="grey", alpha=0.08)
ax3.set_xlabel("t")
ax3.set_ylabel("C(t, t)")
ax3.set_title("(c) Encadrement de Frechet (diagonale)\nC- <= C_emp <= C+ ; ecart a l'independance")
ax3.legend(fontsize=7)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig8_copula_tail_dependence.png", dpi=130)
print("\nFichiers ecrits :")
print(f"  {TABLES_DIR / 'ext_B_copula_selection.csv'}")
print(f"  {TABLES_DIR / 'ext_B_summary.csv'}")
print(f"  {FIGURES_DIR / 'fig8_copula_tail_dependence.png'}")