"""Exécute l'Extension A sur les données réelles du projet.

CORRECTIONS (revue méthodologique) :
  - Chargement du RF via le loader robuste du projet (src.factors.fetch_ff_factors)
    au lieu d'un open() en dur sur un chemin/nom de fichier fragile
    ("FF_Research_Data_Factors.csv" sans tiret -> FileNotFoundError). Plus aucune
    dépendance au répertoire de lancement ; téléchargement automatique en repli.
  - Figure d'exceedance : ajout du BENCHMARK GAUSSIEN (Boyer-Gibson-Loretan). Seul
    l'écart empirique − gaussien est interprétable comme dépendance de queue.
"""
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
from src.config import FIGURES_DIR, TABLES_DIR

# Chemin vers les données : local en priorité, fallback container
from pathlib import Path as PathLib
_TSMOM_PATH = PathLib("outputs/tables/diversified_tsmom_series.csv")
_DATA_PATH = PathLib("data/data_monthly_returns.csv")

# ---- 1. Charger TSMOM diversifié (déjà produit par ta réplication) ----
tsmom = (pd.read_csv(_TSMOM_PATH, parse_dates=["date"])
         .set_index("date")["TSMOM"].dropna())

# ---- 2. Marché en excès = MXWO - RF, MÊME définition que ton test du smile ----
mx = (pd.read_csv(_DATA_PATH, parse_dates=["date"])
      .set_index("date")["MXWO Index"])
# RF mensuel (décimal, fin de mois) via le loader robuste du projet :
# cherche le CSV FF où qu'il soit, sinon télécharge — plus de chemin en dur.
_ff = fetch_ff_factors(start="1900-01-01", end="2100-12-31", source="auto")
rf = _ff["RF"]                                   # DÉJÀ en décimal (pas de /100)
mkt = (mx - rf.reindex(mx.index)).dropna().rename("MKT")

# ---- Filtrage AR(1)-GARCH(1,1)-t des deux marges ----
print("=== (1) Filtrage des marges AR(1)-GARCH(1,1)-t ===")
z_ts, info_ts = ce.filter_marginal(tsmom)
z_mk, info_mk = ce.filter_marginal(mkt)
print("TSMOM :", {k: round(v, 3) for k, v in info_ts.items()})
print("MKT   :", {k: round(v, 3) for k, v in info_mk.items()})

# ---- (2) PIT -> pseudo-observations ----
u, idx = ce.to_pseudo_obs(z_ts.rename("TS"), z_mk.rename("MK"))
print(f"\nÉchantillon joint : N = {len(u)} mois ({idx.min():%Y-%m} → {idx.max():%Y-%m})")

# ---- (3) Ajustement des copules ----
print("\n=== (3) Sélection des copules (AIC/BIC) et dépendance de queue ===")
fit = ce.fit_all_copulas(u)
fit_show = fit.copy()
for c in ["logL", "AIC", "BIC", "lambda_L", "lambda_U"]:
    fit_show[c] = fit_show[c].astype(float).round(3)
print(fit_show.to_string())
best_aic = fit["AIC"].astype(float).idxmin()
best_bic = fit["BIC"].astype(float).idxmin()
print(f"\nMeilleure copule  AIC: {best_aic}   |   BIC: {best_bic}")

# ---- Dépendance de queue NON-PARAMÉTRIQUE ----
print("\n=== Dépendance de queue empirique (model-free) ===")
emp10 = ce.empirical_tail_dependence(u, q=0.10)
emp15 = ce.empirical_tail_dependence(u, q=0.15)
for e in (emp10, emp15):
    print({k: round(v, 3) for k, v in e.items()})

# ---- Test d'adéquation CvM sur la gaussienne (H0) et la meilleure famille ----
print("\n=== Test d'adéquation Cramér-von Mises (bootstrap paramétrique) ===")
def parse_param(row):
    d = {}
    for kv in row["param"].split(", "):
        k, v = kv.split("=")
        d[{"rho": "rho", "df": "df", "theta": "theta"}[k]] = float(v)
    return d
gof_rows = []
for fam in ["Gaussian", best_aic]:
    par = parse_param(fit.loc[fam])
    g = ce.cvm_gof(u, fam, par, n_boot=250)
    gof_rows.append(g)
    print(f"  {fam:10s}  Sn={g['Sn']:.4f}  p-value={g['p_value']:.3f}")

# ---- Corrélations d'exceedance (asymétrie krach vs boom) ----
exc = ce.exceedance_correlation(z_ts.reindex(idx), z_mk.reindex(idx))
print("\n=== Corrélations d'exceedance (résidus standardisés) ===")
print(exc.round(3).to_string(index=False))

# ======================= SORTIES ========================
# Mêmes dossiers que les extensions B et C : outputs/tables et outputs/figures
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Tableau principal au format des autres tables du projet
res = fit.copy()
res.insert(0, "Family", res.index)
res.to_csv(TABLES_DIR / "ext_A_copula_selection.csv", index=False)

# Résumé synthétique
summary = pd.DataFrame({
    "Best copula (AIC)": [best_aic],
    "Best copula (BIC)": [best_bic],
    "lambda_L (best)": [float(fit.loc[best_aic, "lambda_L"])],
    "lambda_U (best)": [float(fit.loc[best_aic, "lambda_U"])],
    "lambda_L emp (q=.10)": [emp10["lambda_L_emp"]],
    "lambda_U emp (q=.10)": [emp10["lambda_U_emp"]],
    "asymmetry emp (L-U, q=.10)": [emp10["asymmetry_L_minus_U"]],
    "Gaussian rejected? (p<.05)": [gof_rows[0]["p_value"] < 0.05],
    "Best not rejected? (p>.05)": [gof_rows[1]["p_value"] > 0.05],
    "N": [len(u)],
}).T.rename(columns={0: "value"})
summary.to_csv(TABLES_DIR / "ext_A_summary.csv")
print("\n=== RÉSUMÉ ===")
print(summary.to_string())

# Figure : (a) corrélations d'exceedance + benchmark gaussien  (b) nuage pseudo-obs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
ax1.plot(exc["threshold"], exc["corr_neg"], "o-", color="#c0392b",
         label="Krachs joints (empirique)")
ax1.plot(exc["threshold"], exc["corr_pos"], "s-", color="#2980b9",
         label="Booms joints (empirique)")
# Benchmark gaussien (Boyer-Gibson-Loretan) : ce que produit une normale de même ρ
ax1.plot(exc["threshold"], exc["corr_neg_gauss"], "--", color="#c0392b", alpha=0.55,
         label="Krachs — benchmark gaussien")
ax1.plot(exc["threshold"], exc["corr_pos_gauss"], "--", color="#2980b9", alpha=0.55,
         label="Booms — benchmark gaussien")
ax1.axhline(0, color="grey", lw=0.7)
ax1.set_xlabel("Seuil |z| (écarts-types)")
ax1.set_ylabel("Corrélation d'exceedance TSMOM–Marché")
ax1.set_title("(a) Le straddle dans les queues\n(empirique vs normale de même ρ)")
ax1.legend(fontsize=7)
ax2.scatter(u[:, 0], u[:, 1], s=10, alpha=0.5, color="#34495e")
ax2.set_xlabel("u = F(résidu TSMOM)")
ax2.set_ylabel("v = F(résidu Marché)")
ax2.set_title("(b) Pseudo-observations (copule empirique)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig8_copula_tail_dependence.png", dpi=130)
print("\nFichiers écrits :")
print(f"  {TABLES_DIR / 'ext_A_copula_selection.csv'}")
print(f"  {TABLES_DIR / 'ext_A_summary.csv'}")
print(f"  {FIGURES_DIR / 'fig8_copula_tail_dependence.png'}")