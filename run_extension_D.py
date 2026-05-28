"""Extension D — Dépendance de queue TSMOM–Marché par classe d'actifs.

Idée : les Extensions B et B ont analysé la dépendance de queue de TSMOM
GLOBAL (tous actifs confondus) avec le marché actions. Cette extension décompose
cette dépendance PAR CLASSE D'ACTIFS pour répondre à la question :
  - Quelles classes d'actifs contribuent le plus au profil « straddle » ?
  - Est-ce que l'asymétrie krach/boom est universelle ou concentrée ?
  - La sélection de famille (Student-t vs Gumbel vs Clayton) est-elle homogène ?

Procédure (identique à Extension B, répétée par classe) :
  (1) Filtrage AR(1)-GARCH(1,1)-t de chaque TSMOM de classe et du marché.
  (2) PIT non-paramétrique → pseudo-observations uniformes.
  (3) Sélection copule par AIC, rapport λ_L / λ_U.
  (4) Corrélations d'exceedance (Longin-Solnik) sur résidus standardisés.
  (5) Dépendance de queue empirique à q=10 %.

CORRECTIONS (revue méthodologique) :
  - RF HARMONISÉ avec A et B : on utilise désormais le RF Fama-French (loader
    robuste du projet) au lieu du taux USD 1M/12 de data.xlsx, pour que la jambe
    « marché » soit rigoureusement identique d'une extension à l'autre.
  - ASYMÉTRIE reportée sur les λ EMPIRIQUES (λ_L_emp − λ_U_emp). La Student-t est
    symétrique par construction (λ_L = λ_U) : reporter l'asymétrie paramétrique
    affichait 0 pour les classes où elle gagne (Actions, Changes) et MASQUAIT
    l'asymétrie réelle que révèlent les mesures non-paramétriques.
  - Figure d'exceedance : ajout du benchmark gaussien (Boyer-Gibson-Loretan).

Sorties :
  ext_D_copula_by_class.csv          — tableau récapitulatif par classe
  fig11_exceedance_by_class.png      — corrélations d'exceedance (4 panneaux)
  fig12_copula_summary_by_class.png  — résumé λ_L, λ_U, asymétrie, AIC gain
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
from src.config import (
    LOOKBACK_MONTHS, PAPER_START, PAPER_END,
    asset_class_of,
)
import os as _os
from src.config import FIGURES_DIR as _FIG_DEF, TABLES_DIR as _TAB_DEF
FIGURES_DIR = Path(_os.environ.get("TSMOM_FIG_DIR", str(_FIG_DEF)))
TABLES_DIR = Path(_os.environ.get("TSMOM_TAB_DIR", str(_TAB_DEF)))
from src.factors import fetch_ff_factors
from src.data_loader import load_raw
from src.returns import build_daily_excess_returns, daily_to_monthly_returns
from src.volatility import ewma_ex_ante_vol, vol_for_signal
from src.strategy import tsmom_instrument_returns, tsmom_by_asset_class

# ------------------------------------------------------------------ #
#  Palette et labels                                                   #
# ------------------------------------------------------------------ #
CLASS_META = {
    "Equity":    {"label": "Actions",            "color": "#922b21"},
    "Bond":      {"label": "Obligations",        "color": "#1a5276"},
    "Commodity": {"label": "Matières premières", "color": "#d35400"},
    "Currency":  {"label": "Changes",            "color": "#1e8449"},
}
ORDERED = ["Equity", "Bond", "Commodity", "Currency"]

FAMILY_COLORS = {
    "Gaussian":  "#95a5a6",
    "Student-t": "#f39c12",
    "Clayton":   "#c0392b",
    "Gumbel":    "#27ae60",
    "Frank":     "#8e44ad",
}

# ------------------------------------------------------------------ #
#  (1) Chargement des données de base                                  #
# ------------------------------------------------------------------ #
print("=== Extension D — Dépendance de queue TSMOM–Marché par classe d'actifs ===\n")
print("(1) Chargement des données brutes...")

prices = load_raw()
daily  = build_daily_excess_returns(prices)
monthly = daily_to_monthly_returns(daily)
dvol   = ewma_ex_ante_vol(daily)
mvol   = vol_for_signal(dvol, monthly.index)

# ------------------------------------------------------------------ #
#  (2) Stratégie TSMOM par classe sur la période-papier               #
# ------------------------------------------------------------------ #
print("(2) Calcul du TSMOM par classe d'actifs...")

inst     = tsmom_instrument_returns(monthly, mvol, k=LOOKBACK_MONTHS)
tsmom_ac = tsmom_by_asset_class(inst)          # Equity, Bond, Commodity, Currency

m_mask   = (tsmom_ac.index >= PAPER_START) & (tsmom_ac.index <= PAPER_END)
tsmom_ac = tsmom_ac.loc[m_mask].dropna(how="all")

# ------------------------------------------------------------------ #
#  (3) Rendement de marché en excès (MXWO - RF mensuel)               #
#      RF HARMONISÉ avec A/B : Fama-French (et non USD 1M/12).         #
# ------------------------------------------------------------------ #
print("(3) Construction du rendement marché (RF Fama-French, comme A/B)...")

mxwo_monthly = (prices["MXWO Index"]
                .resample("ME").last()
                .pct_change())
_ff        = fetch_ff_factors(start="1900-01-01", end="2100-12-31", source="auto")
rf_monthly = _ff["RF"]                                  # décimal, fin de mois
mkt        = (mxwo_monthly - rf_monthly.reindex(mxwo_monthly.index)).dropna().rename("MKT")

# ------------------------------------------------------------------ #
#  (4) Analyse copule par classe                                       #
# ------------------------------------------------------------------ #
print("\n(4) Analyse copule par classe (AR(1)-GARCH(1,1)-t, PIT, AIC)...")

results = {}   # {ac_key: {best_family, lambda_L, lambda_U, lambda_L_emp,
               #            lambda_U_emp, exc_df, aic_gauss, aic_best, fit_df}}

for ac in ORDERED:
    if ac not in tsmom_ac.columns:
        print(f"  [{ac}] — SKIPPED (données insuffisantes)")
        continue

    lbl = CLASS_META[ac]["label"]
    print(f"\n  [{lbl}]")

    ts_s = tsmom_ac[ac].dropna()

    # Alignement TSMOM-classe et marché
    common = ts_s.index.intersection(mkt.index)
    ts_s   = ts_s.reindex(common)
    mk_s   = mkt.reindex(common)

    # Filtrage AR(1)-GARCH(1,1)-t
    z_ts, info_ts = ce.filter_marginal(ts_s)
    z_mk, info_mk = ce.filter_marginal(mk_s)
    print(f"    TSMOM-{lbl} : persist={info_ts.get('alpha+beta (persist.)', 0):.3f}")
    print(f"    MKT         : persist={info_mk.get('alpha+beta (persist.)', 0):.3f}")

    # PIT → pseudo-observations
    u, idx = ce.to_pseudo_obs(z_ts.rename("TS"), z_mk.rename("MK"))
    print(f"    N={len(u)}  ({idx.min():%Y-%m} → {idx.max():%Y-%m})")

    # Ajustement des copules
    fit = ce.fit_all_copulas(u)
    best_aic = fit["AIC"].astype(float).idxmin()
    lam_L = float(fit.loc[best_aic, "lambda_L"])
    lam_U = float(fit.loc[best_aic, "lambda_U"])
    aic_gauss = float(fit.loc["Gaussian", "AIC"])
    aic_best  = float(fit.loc[best_aic,   "AIC"])
    print(f"    Meilleure copule (AIC): {best_aic}  λ_L={lam_L:.3f}  λ_U={lam_U:.3f}")
    print(f"    ΔAIC vs Gaussian: {aic_gauss - aic_best:.2f}")

    # Dépendance de queue empirique (model-free, q=10 %)
    emp = ce.empirical_tail_dependence(u, q=0.10)

    # Mesures de dépendance du cours (rang) par classe
    dm = ce.dependence_measures(u)

    # Corrélations d'exceedance sur résidus standardisés (+ benchmark gaussien)
    exc = ce.exceedance_correlation(z_ts.reindex(idx), z_mk.reindex(idx))

    results[ac] = {
        "label":       lbl,
        "best_family": best_aic,
        "lambda_L":    lam_L,
        "lambda_U":    lam_U,
        "lambda_L_emp": emp["lambda_L_emp"],
        "lambda_U_emp": emp["lambda_U_emp"],
        "spearman":    dm["Spearman_rho"],
        "kendall":     dm["Kendall_tau"],
        "blomqvist":   dm["Blomqvist_beta"],
        "exc_df":      exc,
        "aic_gauss":   aic_gauss,
        "aic_best":    aic_best,
        "fit_df":      fit,
    }

# ------------------------------------------------------------------ #
#  (5) Sauvegarde du tableau récapitulatif                             #
# ------------------------------------------------------------------ #
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

summary_rows = []
for ac, r in results.items():
    summary_rows.append({
        "Asset class":   ac,
        "Label":         r["label"],
        "Best copula":   r["best_family"],
        "λ_L (param)":  round(r["lambda_L"],     3),
        "λ_U (param)":  round(r["lambda_U"],     3),
        "λ_L (emp 10%)": round(r["lambda_L_emp"], 3),
        "λ_U (emp 10%)": round(r["lambda_U_emp"], 3),
        # Asymétrie sur les λ EMPIRIQUES (la param. Student-t est symétrique -> 0)
        "Asym. emp (λ_L−λ_U)": round(r["lambda_L_emp"] - r["lambda_U_emp"], 3),
        # Mesures de dépendance de RANG du cours (invariantes, vraies mesures)
        "Spearman ρ": round(r["spearman"], 3),
        "Kendall τ": round(r["kendall"], 3),
        "Blomqvist β": round(r["blomqvist"], 3),
        "ΔAIC vs Gaussian":    round(r["aic_gauss"] - r["aic_best"], 2),
    })
summary_df = pd.DataFrame(summary_rows)
out_csv = TABLES_DIR / "ext_D_copula_by_class.csv"
summary_df.to_csv(out_csv, index=False)
print(f"\nTableau sauvegardé : {out_csv}")
print(summary_df.to_string(index=False))

# ------------------------------------------------------------------ #
#  (6) Figure 11 — Corrélations d'exceedance par classe (4 panneaux)  #
#      avec benchmark gaussien (Boyer-Gibson-Loretan)                  #
# ------------------------------------------------------------------ #
print("\n(5) Génération de fig11_exceedance_by_class.png ...")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    "Extension D — Dépendance de queue TSMOM–Marché par classe d'actifs\n"
    "(corrélations d'exceedance empiriques vs normale de même ρ — résidus AR-GARCH-t)",
    fontsize=11,
)

panel_labels = list("abcd")
for idx_ac, (ac, ax, pl) in enumerate(
        zip(ORDERED, axes.flat, panel_labels)):
    if ac not in results:
        ax.axis("off")
        continue

    r   = results[ac]
    exc = r["exc_df"]
    col = CLASS_META[ac]["color"]
    lbl = r["label"]

    ax.plot(exc["threshold"], exc["corr_neg"],
            "o-",  color=col,       lw=1.8,
            label="Krachs joints (empirique)")
    ax.plot(exc["threshold"], exc["corr_pos"],
            "s--", color=col,       lw=1.6, alpha=0.7,
            label="Booms joints (empirique)")
    # Benchmarks gaussiens (sans dépendance de queue) — référence à dépasser
    ax.plot(exc["threshold"], exc["corr_neg_gauss"],
            ":", color="#7f8c8d", lw=1.4, label="Krachs — benchmark gaussien")
    ax.plot(exc["threshold"], exc["corr_pos_gauss"],
            ":", color="#bdc3c7", lw=1.4, label="Booms — benchmark gaussien")
    ax.axhline(0, color="grey", lw=0.8)

    ax.set_title(
        f"({pl}) {lbl}\n"
        f"Copule : {r['best_family']} | "
        f"λ_L={r['lambda_L']:.3f}  λ_U={r['lambda_U']:.3f}",
        fontsize=9,
    )
    ax.set_xlabel("Seuil |z| (écarts-types)", fontsize=8)
    ax.set_ylabel("Corrélation d'exceedance", fontsize=8)
    ax.set_ylim(-0.15, 1.05)
    ax.legend(fontsize=6.5)
    ax.tick_params(labelsize=8)

plt.tight_layout()
fig11_path = FIGURES_DIR / "fig11_exceedance_by_class.png"
plt.savefig(fig11_path, dpi=130)
plt.close()
print(f"  Sauvegardé : {fig11_path}")

# ------------------------------------------------------------------ #
#  (7) Figure 12 — Résumé comparatif λ_L, λ_U, asymétrie, AIC gain   #
# ------------------------------------------------------------------ #
print("(6) Génération de fig12_copula_summary_by_class.png ...")

labels    = [CLASS_META[ac]["label"]   for ac in ORDERED if ac in results]
colors    = [CLASS_META[ac]["color"]   for ac in ORDERED if ac in results]
x         = np.arange(len(labels))
width     = 0.35

lam_L_p   = [results[ac]["lambda_L"]     for ac in ORDERED if ac in results]
lam_U_p   = [results[ac]["lambda_U"]     for ac in ORDERED if ac in results]
lam_L_e   = [results[ac]["lambda_L_emp"] for ac in ORDERED if ac in results]
lam_U_e   = [results[ac]["lambda_U_emp"] for ac in ORDERED if ac in results]
# Asymétrie sur les λ EMPIRIQUES (cf. correction : la param. Student-t est symétrique)
asymmetry = [results[ac]["lambda_L_emp"] - results[ac]["lambda_U_emp"]
             for ac in ORDERED if ac in results]
delta_aic = [results[ac]["aic_gauss"] - results[ac]["aic_best"]
             for ac in ORDERED if ac in results]
best_fams  = [results[ac]["best_family"] for ac in ORDERED if ac in results]
bar_colors = [FAMILY_COLORS.get(f, "#333") for f in best_fams]

fig2, axes2 = plt.subplots(2, 2, figsize=(13, 9))
fig2.suptitle(
    "Extension D — Résumé comparatif : λ_L, λ_U, asymétrie, famille",
    fontsize=11,
)

# --- (a) λ_L ---
ax = axes2[0, 0]
b1 = ax.bar(x - width / 2, lam_L_p, width,
            color=colors,            label="λ_L param. (copule)")
b2 = ax.bar(x + width / 2, lam_L_e, width,
            color=colors, alpha=0.45, hatch="///",
            label="λ_L emp. (q=10%)")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("λ_L")
ax.set_title("(a) Dépendance queue basse λ_L  (co-krachs)")
ax.legend(fontsize=8)
for bar, val in zip(b1, lam_L_p):
    if val > 0.005:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

# --- (b) λ_U ---
ax = axes2[0, 1]
b3 = ax.bar(x - width / 2, lam_U_p, width,
            color=colors,            label="λ_U param. (copule)")
b4 = ax.bar(x + width / 2, lam_U_e, width,
            color=colors, alpha=0.45, hatch="///",
            label="λ_U emp. (q=10%)")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("λ_U")
ax.set_title("(b) Dépendance queue haute λ_U  (co-booms)")
ax.legend(fontsize=8)
for bar, val in zip(b3, lam_U_p):
    if val > 0.005:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

# --- (c) Asymétrie λ_L − λ_U (EMPIRIQUE) ---
ax = axes2[1, 0]
bar_asym = ax.bar(labels, asymmetry, color=colors, edgecolor="white")
ax.axhline(0, color="black", lw=0.9)
ax.set_ylabel("λ_L − λ_U (empirique)")
ax.set_title("(c) Asymétrie λ_L − λ_U (empirique, q=10%)\n(positif = krachs > booms)")
for bar, val in zip(bar_asym, asymmetry):
    va   = "bottom" if val >= 0 else "top"
    yoff = 0.003 if val >= 0 else -0.003
    ax.text(bar.get_x() + bar.get_width() / 2,
            val + yoff, f"{val:+.3f}",
            ha="center", va=va, fontsize=8, fontweight="bold")

# --- (d) ΔAIC vs Gaussienne ---
ax = axes2[1, 1]
bar_aic = ax.bar(labels, delta_aic, color=bar_colors, edgecolor="white")
ax.axhline(0, color="black", lw=0.9)
ax.set_ylabel("ΔAIC  (> 0 = meilleure fit)")
ax.set_title("(d) Gain AIC vs Gaussienne\n(couleur = famille sélectionnée)")

# Étiquettes famille sur chaque barre
for bar, fam, val in zip(bar_aic, best_fams, delta_aic):
    ypos = max(val + 0.5, 1.5)
    ax.text(bar.get_x() + bar.get_width() / 2, ypos,
            fam, ha="center", va="bottom", fontsize=7, rotation=0)

# Légende familles
from matplotlib.patches import Patch
legend_elems = [Patch(facecolor=FAMILY_COLORS[f], label=f)
                for f in sorted(set(best_fams))]
ax.legend(handles=legend_elems, fontsize=8, loc="upper right")

plt.tight_layout()
fig12_path = FIGURES_DIR / "fig12_copula_summary_by_class.png"
plt.savefig(fig12_path, dpi=130)
plt.close()
print(f"  Sauvegardé : {fig12_path}")

print("\n=== Extension D terminée. ===")
print(f"  → {out_csv}")
print(f"  → {fig11_path}")
print(f"  → {fig12_path}")