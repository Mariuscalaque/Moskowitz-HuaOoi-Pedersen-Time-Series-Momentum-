"""Extension B — Stabilité temporelle de la dépendance TSMOM–Marché (Rolling Copula).

Idée : l'Extension A a estimé la copule SUR TOUT L'ÉCHANTILLON d'un coup.
Mais la structure de dépendance TSMOM–Marché est-elle stable dans le temps ?
On fait tourner la copule sur des fenêtres roulantes pour montrer que :
  - λ_L (queue basse) bondit en période de crise (2008, 2020, …)
  - La meilleure famille CHANGE de régime : Gaussienne en période calme,
    Clayton/Student-t en crise
  - Implication : TSMOM n'est PAS un hedge stable du marché — son profil
    de dépendance est lui-même conditionnel au régime.

CORRECTIONS (revue méthodologique) :
  - RF chargé via le loader robuste du projet (src.factors.fetch_ff_factors),
    plus de open() en dur -> identique à l'Extension A, zéro problème de chemin.
  - Fenêtre portée à 60 mois (la dépendance de queue est ininterprétable sur 36 obs).
  - REFILTER : si True, les marges sont refiltrées AR-GARCH-t DANS chaque fenêtre
    -> aucune fuite d'information future. Si False (défaut), filtrage marginal
    GLOBAL (standard type Patton 2006) : la copule seule roule, le GARCH global
    utilise l'information plein-échantillon -> lecture purement DESCRIPTIVE a
    posteriori, assumée et plus stable. Mets REFILTER=True pour une version sans
    look-ahead (plus lente, GARCH estimé sur 60 points).

Sorties :
  ext_B_rolling_copula.csv        — série roulante complète
  ext_B_regime_statistics.csv     — stats par régime (Gaussian vs tail-dep.)
  fig9_rolling_copula.png         — figure principale 4 panneaux
  fig10_copula_regime_freq.png    — distribution λ_L + fréquence des familles
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Résolution du chemin projet
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import copula_extension as ce
from src.config import FIGURES_DIR, TABLES_DIR
from src.factors import fetch_ff_factors

# Chemins vers les données
from pathlib import Path as PathLib
_TSMOM_CSV = PathLib("outputs/tables/diversified_tsmom_series.csv")
_DATA_CSV  = PathLib("data/data_monthly_returns.csv")

WINDOW   = 60      # fenêtre roulante (mois) — >= 60 recommandé pour la queue
REFILTER = False   # True = refiltrage AR-GARCH-t par fenêtre (zéro look-ahead, + lent)

# =========================================================
# (1) Charger les données — strictement identique à Ext A
# =========================================================
tsmom = (pd.read_csv(_TSMOM_CSV, parse_dates=["date"])
         .set_index("date")["TSMOM"].dropna())

mx = (pd.read_csv(_DATA_CSV, parse_dates=["date"])
      .set_index("date")["MXWO Index"])

# RF mensuel (décimal, fin de mois) via le loader robuste du projet
_ff = fetch_ff_factors(start="1900-01-01", end="2100-12-31", source="auto")
rf  = _ff["RF"]                                   # DÉJÀ en décimal (pas de /100)
mkt = (mx - rf.reindex(mx.index)).dropna().rename("MKT")

# =========================================================
# (2) Filtrage des marges AR(1)-GARCH(1,1)-t
#     - REFILTER=False : filtrage GLOBAL une fois (résidus passés au rolling)
#     - REFILTER=True  : on passe les séries BRUTES, refiltrage dans la fenêtre
# =========================================================
print("=== (1) Filtrage des marges AR(1)-GARCH(1,1)-t ===")
z_ts, info_ts = ce.filter_marginal(tsmom)
z_mk, info_mk = ce.filter_marginal(mkt)
print("TSMOM :", {k: round(v, 3) for k, v in info_ts.items()})
print("MKT   :", {k: round(v, 3) for k, v in info_mk.items()})

if REFILTER:
    s1, s2 = tsmom.rename("TS"), mkt.rename("MK")   # brutes -> refiltrées en fenêtre
    print("\nMode : REFILTER=True (refiltrage par fenêtre, sans look-ahead)")
else:
    s1, s2 = z_ts.rename("TS"), z_mk.rename("MK")    # résidus globaux
    print("\nMode : REFILTER=False (filtrage marginal global, descriptif)")

# =========================================================
# (3) Estimation rolling de la copule
# =========================================================
print(f"\n=== (2) Rolling copula (fenêtre = {WINDOW} mois) — patience ===")
rolling = ce.rolling_copula(s1, s2, window=WINDOW, refilter=REFILTER)
print(f"\n  {len(rolling)} fenêtres  ({rolling.index.min():%Y-%m} → {rolling.index.max():%Y-%m})")

print("\nDistribution des meilleures copules (AIC) :")
print(rolling["best_family"].value_counts().to_string())

print(f"\nλ_L moyen  = {rolling['lambda_L'].mean():.3f}")
print(f"λ_L 90e p. = {rolling['lambda_L'].quantile(0.90):.3f}")
print(f"λ_L max    = {rolling['lambda_L'].max():.3f}")

# =========================================================
# (4) Statistiques par régime
# =========================================================
rolling["regime"] = rolling["best_family"].apply(
    lambda f: "Gaussian" if f == "Gaussian" else "Tail-dependent"
)
regime_stats = (rolling.groupby("regime")[["lambda_L", "lambda_U", "AIC_best", "AIC_gauss"]]
                .describe().round(4))
print("\n=== (3) Statistiques λ_L par régime ===")
print(regime_stats.to_string())

# =========================================================
# (5) Figures
# =========================================================

# Zones de crise (pour le shading visuel)
CRISES = [
    ("2000-03", "2002-09", "Dot-com"),
    ("2007-07", "2009-03", "GFC 2008"),
    ("2011-07", "2012-06", "Dette Euro"),
    ("2020-02", "2020-04", "COVID"),
    ("2022-01", "2022-12", "Hausse taux"),
]

FAMILY_COLORS = {
    "Gaussian":  "#95a5a6",
    "Student-t": "#f39c12",
    "Clayton":   "#c0392b",
    "Gumbel":    "#27ae60",
    "Frank":     "#8e44ad",
}


def shade_crises(ax, alpha=0.13):
    """Ombre les périodes de crise sur un axe matplotlib."""
    for start, end, _ in CRISES:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        # clamp aux limites de la série roulante
        s = max(s, rolling.index.min())
        e = min(e, rolling.index.max())
        if s < e:
            ax.axvspan(s, e, alpha=alpha, color="#e74c3c", zorder=0)


# ------------------------------------------------------------------
# Figure 9 — quatre panneaux principaux
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
_mode = "refiltrage par fenêtre, sans look-ahead" if REFILTER else "filtrage global, descriptif"
fig.suptitle(
    f"Extension B — Stabilité temporelle de la dépendance TSMOM–Marché\n"
    f"(Fenêtre roulante = {WINDOW} mois, AR(1)-GARCH(1,1)-t — {_mode})",
    fontsize=12,
)

# (a) λ_L dans le temps
ax = axes[0, 0]
shade_crises(ax)
ax.fill_between(rolling.index, rolling["lambda_L"], alpha=0.35, color="#c0392b")
ax.plot(rolling.index, rolling["lambda_L"], color="#c0392b", lw=1.6)
mean_l = rolling["lambda_L"].mean()
ax.axhline(mean_l, color="black", lw=1.1, ls="--",
           label=f"Moyenne = {mean_l:.3f}")
ax.set_title("(a) Dépendance de queue basse λ_L  (risque de co-krach)")
ax.set_ylabel("λ_L")
ax.set_ylim(bottom=0)
ax.legend(fontsize=8)
ax.tick_params(axis="x", rotation=25)

# (b) λ_U dans le temps
ax = axes[0, 1]
shade_crises(ax)
ax.fill_between(rolling.index, rolling["lambda_U"], alpha=0.35, color="#2980b9")
ax.plot(rolling.index, rolling["lambda_U"], color="#2980b9", lw=1.6)
mean_u = rolling["lambda_U"].mean()
ax.axhline(mean_u, color="black", lw=1.1, ls="--",
           label=f"Moyenne = {mean_u:.3f}")
ax.set_title("(b) Dépendance de queue haute λ_U  (co-booms)")
ax.set_ylabel("λ_U")
ax.set_ylim(bottom=0)
ax.legend(fontsize=8)
ax.tick_params(axis="x", rotation=25)

# (c) Meilleure famille de copule sur la timeline
ax = axes[1, 0]
y_pos = {"Gaussian": 0, "Student-t": 1, "Clayton": 2, "Gumbel": 3, "Frank": 4}
shade_crises(ax, alpha=0.10)
for fam, grp in rolling.groupby("best_family"):
    ax.scatter(
        grp.index,
        [y_pos.get(fam, 0)] * len(grp),
        color=FAMILY_COLORS.get(fam, "#333"),
        marker="|", s=120, linewidths=1.8,
        label=f"{fam} ({len(grp)})"
    )
ax.set_yticks(list(y_pos.values()))
ax.set_yticklabels(list(y_pos.keys()))
ax.set_title("(c) Meilleure copule par période (sélection AIC)")
ax.legend(fontsize=7, loc="upper right", ncol=2)
ax.tick_params(axis="x", rotation=25)

# (d) λ_L vs volatilité roulante de TSMOM (double axe)
ax  = axes[1, 1]
ax2 = ax.twinx()
shade_crises(ax, alpha=0.10)
l1, = ax.plot(rolling.index, rolling["lambda_L"],
              color="#c0392b", lw=1.8, label="λ_L (axe G)")
rolling_vol = tsmom.reindex(rolling.index).rolling(12, min_periods=6).std()
l2, = ax2.plot(rolling.index, rolling_vol,
               color="#2c3e50", lw=1.2, ls="--", alpha=0.75,
               label="Vol TSMOM 12m (axe D)")
ax.set_title("(d) λ_L et volatilité roulante de TSMOM")
ax.set_ylabel("λ_L",                 color="#c0392b")
ax2.set_ylabel("Vol. roulante (σ)",  color="#2c3e50")
ax.tick_params(axis="y", labelcolor="#c0392b")
ax2.tick_params(axis="y", labelcolor="#2c3e50")
ax.tick_params(axis="x", rotation=25)
ax.legend([l1, l2], [l.get_label() for l in [l1, l2]], fontsize=8)

fig.text(0.5, 0.01,
         "Zones rouges = périodes de crise (Dot-com, GFC 2008, Dette Euro, COVID, Hausse taux)",
         ha="center", fontsize=8, color="#c0392b")
plt.tight_layout(rect=[0, 0.03, 1, 1])
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / "fig9_rolling_copula.png", dpi=130)
print(f"\nFigure sauvegardée : {FIGURES_DIR / 'fig9_rolling_copula.png'}")

# ------------------------------------------------------------------
# Figure 10 — distribution λ_L + fréquence des familles
# ------------------------------------------------------------------
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
fig2.suptitle("Extension B — Résumé des régimes de dépendance", fontsize=11)

# Histogramme de λ_L avec annotations
q25 = rolling["lambda_L"].quantile(0.25)
q75 = rolling["lambda_L"].quantile(0.75)
q90 = rolling["lambda_L"].quantile(0.90)
ax1.hist(rolling["lambda_L"], bins=28, color="#c0392b", alpha=0.65, edgecolor="white")
ax1.axvline(mean_l, color="black",   ls="--", lw=1.6,
            label=f"Moyenne = {mean_l:.3f}")
ax1.axvline(q75,   color="#e67e22", ls=":",  lw=1.4,
            label=f"75e  perc. = {q75:.3f}")
ax1.axvline(q90,   color="#8e44ad", ls=":",  lw=1.4,
            label=f"90e  perc. = {q90:.3f}")
ax1.set_xlabel("λ_L  (dépendance de queue basse)")
ax1.set_ylabel("Nombre de fenêtres")
ax1.set_title("(e) Distribution de λ_L sur toute la période")
ax1.legend(fontsize=8)

# Graphique en barres horizontales — % du temps par famille
vc     = rolling["best_family"].value_counts()
colors = [FAMILY_COLORS.get(f, "#333") for f in vc.index]
bars   = ax2.barh(vc.index, vc.values / len(rolling) * 100,
                  color=colors, edgecolor="white", height=0.6)
ax2.set_xlabel("% du temps où la famille est sélectionnée")
ax2.set_title(f"(f) Fréquence de sélection par famille\n"
              f"(fenêtre {WINDOW} m | {len(rolling)} périodes)")
ax2.set_xlim(right=ax2.get_xlim()[1] * 1.18)
for bar, cnt in zip(bars, vc.values):
    pct = cnt / len(rolling) * 100
    ax2.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
             f"{pct:.1f}%  (n={cnt})", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig10_copula_regime_freq.png", dpi=130)
print(f"Figure sauvegardée : {FIGURES_DIR / 'fig10_copula_regime_freq.png'}")

# =========================================================
# (6) Exports CSV
# =========================================================
TABLES_DIR.mkdir(parents=True, exist_ok=True)
rolling.reset_index().to_csv(TABLES_DIR / "ext_B_rolling_copula.csv", index=False)
# describe() multi-index aplati pour un CSV lisible
regime_stats_flat = regime_stats.copy()
regime_stats_flat.columns = [f"{a}_{b}" for a, b in regime_stats_flat.columns]
regime_stats_flat.to_csv(TABLES_DIR / "ext_B_regime_statistics.csv")

# =========================================================
# (7) Résumé synthétique
# =========================================================
print("\n" + "=" * 55)
print("RÉSUMÉ EXTENSION B — Rolling Copula TSMOM–Marché")
print("=" * 55)
print(f"Fenêtre roulante       : {WINDOW} mois")
print(f"Mode de filtrage       : {'refiltrage par fenêtre' if REFILTER else 'global (descriptif)'}")
print(f"Nombre de fenêtres     : {len(rolling)}")
print(f"Période couverte       : {rolling.index.min():%Y-%m} → {rolling.index.max():%Y-%m}")
print(f"λ_L moyen (plein éch.) : {mean_l:.3f}")
print(f"λ_L au 90e percentile  : {q90:.3f}")
print(f"λ_L maximum observé    : {rolling['lambda_L'].max():.3f}")
print(f"Famille dominante      : {rolling['best_family'].value_counts().idxmax()}")
pct_td = (rolling["best_family"] != "Gaussian").mean() * 100
print(f"% périodes tail-dep.   : {pct_td:.1f}%")
print("\nFichiers produits :")
print("  ext_B_rolling_copula.csv    — série roulante complète")
print("  ext_B_regime_statistics.csv — stats par régime (aplaties)")
print("  fig9_rolling_copula.png     — figure 4 panneaux")
print("  fig10_copula_regime_freq.png — distribution + fréquences")
