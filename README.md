# Réplication — Moskowitz, Hua Ooi & Pedersen (2012) "Time Series Momentum"

Réplication fidèle de l'article publié au *Journal of Financial Economics* (104, 228-250).

## Organisation du dossier

```
tsmom_replication/
├── README.md
├── replication.ipynb                       # Notebook maître (à exécuter)
├── replication_executed_with_outputs.ipynb # Version déjà exécutée avec sorties
├── data/
│   └── data.xlsx                           # À placer ici (fichier de prix fourni)
├── src/
│   ├── config.py        # Constantes, mapping des 57 instruments → 4 classes d'actifs
│   ├── data_loader.py   # Chargement de data.xlsx + parsing des dates Excel
│   ├── returns.py       # Rendements futures (pct_change) + carry FX
│   ├── volatility.py    # Volatilité ex-ante EWMA — Équation (1) du papier
│   ├── strategy.py      # Signal TSMOM sign(r_{t-12,t}) × 40%/σ — Équation (5)
│   ├── analysis.py      # Régressions pooled (Fig.1), grille (k,h) (Table 2), facteurs (Table 3)
│   ├── plotting.py      # Toutes les figures
│   └── tables.py        # Export CSV + Markdown de toutes les tables
└── outputs/
    ├── figures/         # 8 PNG : Fig 1 (A/B/C), Fig 2, Fig 3 (×2), Fig 4, drawdown bonus
    └── tables/          # 22 fichiers CSV + .md : Table 1, Table 2 (A-E), Table 3, etc.
```

## Comment lancer

1. Placer `data.xlsx` dans `data/`
2. Ouvrir `replication.ipynb` dans Jupyter et exécuter toutes les cellules

Tout est modulaire : on peut aussi importer les fonctions directement depuis `src/`.

## Méthodologie clé (correspondance au papier)

| Étape | Méthode | Équation papier |
|---|---|---|
| Rendements excédentaires | Futures: %change. FX forward: spot return + (i_étranger - i_USD)/252 | §2.1 |
| Volatilité ex-ante | EWMA avec com=60, facteur annualisation = 261 | Éq.(1) |
| Signal & taille | sign(rendement 12 mois) × (40%/σ) × rendement t→t+1 | Éq.(5) |
| Facteur diversifié | Moyenne équipondérée des instruments disponibles chaque mois | §3.2 |

## Principaux résultats obtenus (échantillon 1985-2009 du papier)

| Métrique | Papier | Notre réplication |
|---|---|---|
| Sharpe (TSMOM diversifié) | ~1.0–1.2 | **0.90** |
| Rendement annualisé | ~11 % | 11.30 % |
| Volatilité annualisée | ~12 % | 12.53 % |
| Instruments avec Sharpe > 0 | 58/58 | 51/57 |
| Pattern Figure 1 (continuation 1-12m, reversal long terme) | ✓ | ✓ |
| Coefficient MKT² (sourire) | positif | β² = +1.13 (Fig 4) |

L'amélioration par rapport à la tentative précédente (Sharpe 0.78) provient de :
- Calcul correct du carry FX (signe inversé pour cotations USDXXX)
- Volatilité EWMA strictement sans look-ahead (lag d'un jour)
- Repondération mensuelle correcte du portefeuille diversifié (pas annuelle)
- Univers d'instruments aligné sur le papier (57 vs 58)

## Écarts résiduels vs papier (assumés et documentés)

1. **Table 3** : le papier régresse sur 4 facteurs Fama-French + UMD. On utilise MKT + GSCI + BOND (UMD/SMB/HML indisponibles dans nos données). L'alpha est donc plus faible mécaniquement.
2. **57 vs 58 instruments** : on n'a pas pu reconstruire parfaitement toutes les paires FX forward avec les taux courts disponibles.
3. **Couverture commodities** : quelques séries commencent plus tard que dans Bloomberg du papier.
