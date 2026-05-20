# Réplication — *Time Series Momentum* (Moskowitz, Hua Ooi & Pedersen, 2012)

Réplication complète de l'article : **6 tables** et **7 figures**.
Le projet produit systématiquement tout ce qui est réalisable avec `data.xlsx`,
et complète les analyses nécessitant des données externes (téléchargées
automatiquement et mises en cache).

## Installation

```bash
pip install -r requirements.txt          # pandas, numpy, statsmodels, matplotlib, openpyxl
pip install cot-reports                   # (optionnel) facilite le téléchargement CFTC
```

Place `data.xlsx` dans `data/` (déjà attendu là par `config.py`).

## Lancer la réplication complète

```bash
python run_replication.py                 # 1985-2009 (échantillon papier)
python run_replication.py --no-external   # sans aucun téléchargement
python run_replication.py --end 2025-12-31
```

Sorties : `outputs/tables/*.csv` (+ `.md`) et `outputs/figures/*.png`.

## Notebook

`replication_complete.ipynb` reproduit l'article pas-à-pas (un bloc par table/
figure, avec interprétation comparée à l'article) en s'appuyant sur `src/`.
Il est livré **pré-exécuté** sur l'échantillon papier ; relance-le **avec une
connexion internet** pour remplir les cellules 🌐 (Fama-French, VME, CFTC…).

## Structure

```
src/
  config.py          paramètres, mapping des instruments par classe
  data_loader.py     lecture/nettoyage de data.xlsx
  returns.py         rendements excédentaires journaliers/mensuels (futures + FX)
  volatility.py      volatilité ex-ante EWMA (Eq. 1)
  strategy.py        signal TSMOM, scaling 40% vol, portefeuille diversifié (Eq. 5)
  crosssectional.py  XSMOM + décomposition Lo-MacKinlay        (Table 5)
  rollyield.py       décomposition rendement = spot + roll      (Table 6)
  factors.py         facteurs Fama-French (SMB/HML/UMD/RF)      (Table 3A)
  external_data.py   AQR / Pástor-Stambaugh / Baker-Wurgler / CFTC + CACHE
  analysis.py        régressions Fig.1, Tables 2/3/4/5/6, event study, VAR
  tables.py          mise en forme et sauvegarde des tables
  plotting.py        Figures 1 à 7
  pipeline.py        orchestrateur (produit tout)
run_replication.py   point d'entrée
```

## Couverture article → sortie

| Élément | Fichier de sortie | Données |
|---|---|---|
| Table 1 — stats descriptives | `table1_summary_stats` | data.xlsx |
| Table 2 — alphas (k,h) | `table2_panelA_all` | data.xlsx |
| Table 3A — facteurs Fama-French | `table3_panelA_ff` | FF (auto, `factors.py`) |
| Table 3B — facteurs VME | `table3_panelB_vme` | **AQR** (externe) |
| Table 3C — extrêmes marché/vol/liq/sent | `table3_panelC_extremes` | VIX+TED (data.xlsx) ; PS+BW (externes) |
| Table 4 — corrélations intra/inter-classes | `table4_panelA/B_*` | data.xlsx |
| Table 5 — TSMOM vs XSMOM (A/B/C) | `table5_panelA/B/C_*` | A,B : data.xlsx ; C : indices HF (externe) |
| Table 6 — spot / roll / positions | `table6_predictors` | spot/roll (M1/M2 data.xlsx) ; positions **CFTC** |
| Figure 1 — prédictibilité par lag | `fig1_panelA/B` | data.xlsx |
| Figure 2 — Sharpe par instrument | `fig2_sharpe_by_instrument` | data.xlsx |
| Figure 3 — cumulé TSMOM vs passif | `fig3_cumulative` | data.xlsx |
| Figure 4 — smile | `fig4_smile` | data.xlsx |
| Figure 5 — positions spéculateurs | `fig5_net_speculator` | **CFTC** |
| Figure 6 — event study (A rendements / B positions) | `fig6_event_study` | A : data.xlsx ; B : CFTC |
| Figure 7 — réponse impulsionnelle | `fig7_impulse_response` | univarié (data.xlsx) ; bivarié si CFTC |

## Données externes

Tout passe par `src/external_data.py`, qui télécharge **et met en cache** dans
`data/external/` (fichier brut + CSV nettoyé + `_manifest.json` traçant URL,
date, période). Aux exécutions suivantes, le cache est relu (fonctionne ensuite
hors-ligne).

- **AQR** (facteurs VME, + série TSMOM officielle de validation) : Excel direct.
- **Pástor-Stambaugh** (liquidité) et **Baker-Wurgler** (sentiment) : fichiers
  dont le nom porte un millésime — si l'URL par défaut renvoie 404, passe
  `url="..."` à jour ou `local_file="..."`.
- **CFTC** (positions spéculateurs, Legacy futures-only depuis 1986) : via la
  librairie `cot-reports` (recommandé) ou téléchargement direct des ZIP annuels.
  Couvre uniquement le sous-univers **listé aux US** (pas DAX/CAC/TOPIX/Gilt/
  Bund/LME/NOK/SEK) ; voir `DEFAULT_COT_MARKETS` dans `external_data.py`.

Non automatisé : indices hedge funds Credit Suisse (Table 5C) — inscription
requise depuis la fusion CSAM→UBS (2024). Utiliser un substitut gratuit (SG
Trend / Barclay CTA / HFR) puis le passer en cible à `table5_what_tsmom_explains`.

## Notes méthodologiques

- **Table 5B** : décomposition de Lo-MacKinlay (1990) ; identité comptable
  vérifiée (Auto + Cross + Mean = profit empirique, à la précision machine).
- **Table 6** : décomposition spot/roll *approximée* à partir des contrats de
  1re et 2e échéance (M1/M2) ; le papier utilise des séries spot/roll dédiées.
- Toutes les régressions de performance utilisent des erreurs-types HAC
  (Newey-West) ; les régressions de panel (Fig.1, Table 6) des SE clustées par date.
