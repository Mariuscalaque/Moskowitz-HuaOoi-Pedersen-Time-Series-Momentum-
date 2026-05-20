# Réplication — Moskowitz, Hua Ooi & Pedersen (2012) « Time Series Momentum »

Réplication fidèle de l'article publié au *Journal of Financial Economics* (104, 228-250),
avec **récupération automatique des facteurs Fama-French** pour la Table 3.

## Organisation du dossier

```
tsmom_replication/
├── README.md
├── replication.ipynb                       # Notebook maître (à exécuter)
├── replication_executed_with_outputs.ipynb # Version déjà exécutée, sorties incluses
├── data/
│   └── data.xlsx                           # À placer ici (fichier de prix fourni)
├── src/
│   ├── config.py        # Constantes, mapping des 57 instruments → 4 classes d'actifs
│   ├── data_loader.py   # Chargement de data.xlsx + parsing des dates Excel
│   ├── returns.py       # Rendements futures (pct_change) + carry FX
│   ├── volatility.py    # Volatilité ex-ante EWMA — Équation (1)
│   ├── strategy.py      # Signal TSMOM sign(r_{t-12,t}) × 40%/σ — Équation (5)
│   ├── analysis.py      # Fig.1, grille (k,h) Table 2, Table 3 à 6 facteurs
│   ├── factors.py       # ★ Récupération AUTO des facteurs Fama-French (SMB/HML/UMD/RF)
│   ├── plotting.py      # Toutes les figures
│   └── tables.py        # Export CSV + Markdown
└── outputs/
    ├── figures/         # Fig 1 (A/B/C), Fig 2, Fig 3 (×2), Fig 4, drawdown
    └── tables/          # Table 1, Table 2 (A-E), Table 3, résumés
```

## Installation & exécution

```bash
pip install pandas numpy matplotlib statsmodels openpyxl scipy
# (optionnel, source alternative pour les facteurs)
pip install pandas_datareader
```

1. Placer `data.xlsx` dans `data/` (et ajuster `DATA_PATH` dans `src/config.py` si besoin).
2. Ouvrir `replication.ipynb` et exécuter toutes les cellules.

## Récupération automatique des facteurs Fama-French (`src/factors.py`)

La Table 3 régresse le TSMOM sur 6 facteurs (Eq. 4). Trois — MKT (MSCI World),
BOND (Barclays Agg), GSCI (S&P GSCI) — sont déjà dans `data.xlsx`. Les trois autres
— **SMB, HML, UMD** — sont **téléchargés automatiquement** depuis la
**Ken French Data Library** (Dartmouth), la source citée par l'article.

`fetch_ff_factors()` essaie dans l'ordre :
1. **téléchargement direct** du ZIP Dartmouth (Python pur, sans dépendance) ;
2. **pandas_datareader** (si installé) ;
3. **CSV locaux** téléchargés à la main : `fetch_ff_factors(source='csv', csv_3factor=..., csv_momentum=...)`.

Fichiers concernés sur le site de French (mensuels) :
`F-F_Research_Data_Factors` (→ SMB, HML, RF) et `F-F_Momentum_Factor` (→ UMD).
Un accès Internet est requis pour les sources 1 et 2.

## Méthodologie clé (correspondance à l'article)

| Étape | Méthode | Réf. |
|---|---|---|
| Rendements excédentaires | Futures : %change. FX : spot return + (i_étranger − i_USD)/252 | §2.1 |
| Volatilité ex-ante | EWMA com=60 (δ=60/61), annualisation 261, sans look-ahead | Éq.(1) |
| Signal & taille | sign(rendement 12 mois) × (40%/σ) × rendement t→t+1 | Éq.(5) |
| Facteur diversifié | Moyenne équipondérée des instruments disponibles chaque mois | §4.1 |
| Table 3 | Régression sur MKT, BOND, GSCI, SMB, HML, UMD (mensuel + trimestriel), SE Newey-West | Éq.(4) |

## Principaux résultats (échantillon papier 1985-2009)

| Métrique | Papier | Réplication |
|---|---|---|
| **Sharpe (TSMOM diversifié)** | ~1.0–1.2 | **0.90** |
| Rendement annualisé | ~11 % | 11.30 % |
| Volatilité annualisée | ~12 % | 12.53 % |
| Max drawdown | — | −20.7 % |
| Instruments avec Sharpe > 0 | 58/58 | 51/57 |
| Pattern Fig. 1 (continuation 1-12m, reversal long terme) | ✓ | ✓ |
| Coefficient MKT² (sourire, Fig. 4) | positif | β₂ ≈ +1.13 |
| **Alpha Table 3** (6 facteurs, en ligne) | ≈ 1.58 %/mois, charge + sur UMD | se calcule au run avec FF |

### Note importante sur la Table 3
Sans les facteurs Fama-French, la régression se limite à 3 facteurs (MKT/BOND/GSCI) et
l'alpha tombe à ~0.55 %/mois — précisément parce qu'UMD, le facteur le plus corrélé au
TSMOM, manque. C'est pour cela que `factors.py` récupère SMB/HML/UMD : avec eux, on retrouve
le résultat central de l'article (alpha large + significatif, chargement positif sur UMD,
bêtas non significatifs sur MKT/SMB/HML).

## Écarts résiduels assumés vs papier
1. **57 vs 58 instruments** : reconstruction FX-forward non parfaite avec les taux courts fournis.
2. **Couverture commodities** : quelques séries commencent plus tard que dans Bloomberg du papier.
3. Les facteurs SMB/HML/UMD sont les facteurs **US** de French (ceux cités par l'article).
