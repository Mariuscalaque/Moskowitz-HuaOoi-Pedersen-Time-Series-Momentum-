# Time Series Momentum — Réplication & Extensions

Réplication complète, puis extension, de :

> **Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012).**
> *Time Series Momentum.* **Journal of Financial Economics**, 104(2), 228-250.

Le projet reproduit l'**intégralité** de l'article (6 tables, 7 figures) à partir
de données de futures, puis l'étend à l'échantillon **1985-2025** et prépare une
analyse de la **structure de dépendance par copules**.

---

## 1. L'article en bref

Moskowitz, Ooi & Pedersen (MOP) documentent une anomalie qu'ils nomment
**time series momentum (TSMOM)** : pour presque tous les contrats à terme liquides
— indices actions, devises, matières premières, obligations souveraines — le
**rendement passé sur 12 mois d'un actif prédit positivement son propre rendement
futur**, indépendamment des autres actifs. C'est une prédictibilité *temporelle*
(le passé d'un actif prédit son futur), à distinguer du **momentum transversal
(XSMOM)** classique (Jegadeesh-Titman), qui compare les actifs entre eux.

Résultats principaux du papier :

1. **Prédictibilité universelle.** Continuation des rendements sur 1-12 mois, puis
   **réversion partielle** au-delà d'un an — compatible avec une sous-réaction
   initiale suivie d'une sur-réaction différée (théories de sentiment).
2. **Facteur diversifié performant.** Une stratégie TSMOM 12-mois (signal = signe
   du rendement passé 12 m, position calée à volatilité constante) diversifiée sur
   ~58 instruments offre un **Sharpe > 1** (≈ 2,5× le marché actions), avec peu
   d'exposition aux facteurs de risque standards.
3. **Performance dans les extrêmes.** Le TSMOM ressemble à un **straddle sur le
   marché** : il gagne le plus lors des grands mouvements (hausse *et* baisse),
   d'où une convexité positive vs le marché (le « smile »).
4. **Décomposition.** L'**auto-covariance** des rendements explique l'essentiel des
   profits TSMOM *et* XSMOM ; les effets lead-lag (cross-covariance) jouent en sens
   inverse. TSMOM **capture** le facteur momentum actions UMD et les rendements des
   hedge funds Managed Futures.
5. **Qui trade la tendance.** Les **spéculateurs** chevauchent le trend ~1 an puis
   se retirent avant le retournement ; les **hedgers** prennent l'autre côté. Les
   spéculateurs profitent du TSMOM aux dépens des hedgers (prime de liquidité).
   La décomposition prix spot / roll yield distingue diffusion d'information et
   pression de couverture.

---

## 2. Ce que fait ce dépôt

### 2.1 Réplication (1985-2009)

Le package `src/` reconstruit toute la chaîne, du prix brut aux tables/figures :

| Sortie | Contenu |
|---|---|
| **Table 1** | Statistiques descriptives par instrument + positions nettes spéculateurs |
| **Figure 1** | Prédictibilité par lag (régressions taille & signe, par classe) |
| **Table 2** | t-stats des alphas pour tous les couples (look-back $k$, holding $h$) |
| **Figure 2** | Sharpe du TSMOM 12-mois par instrument |
| **Table 3** | Alpha & chargements factoriels (Fama-French ; VME d'AQR ; smile/liquidité/sentiment) |
| **Figure 3** | Performance cumulée TSMOM vs passive long (vol-matched) |
| **Figure 4** | Le « smile » : rendement TSMOM vs marché |
| **Table 4** | Corrélations TSMOM intra- et inter-classes vs passive long |
| **Table 5** | TSMOM vs XSMOM : régressions + décomposition Lo-MacKinlay |
| **Table 6** | Prédicteurs : prix spot, roll yield, positions CFTC |
| **Figure 5** | Positions nettes moyennes des spéculateurs par instrument |
| **Figure 6** | Event study autour des signaux TSMOM (rendements + positions) |
| **Figure 7** | Réponse impulsionnelle (VAR bivarié) à un choc de rendement, **avec bandes bootstrap** |

### 2.2 Extension out-of-sample (1985-2025)

`src/extension_extended_sample.py` rejoue le **même** pipeline sur 480 mois et
le découpe en sous-périodes (1985-2009 « papier », 2010-2025 « OOS », full).
Il met en évidence une **forte décroissance hors-échantillon** du trend-following.

### 2.3 Étape suivante (en cours) : copules

Reformulation de deux résultats du papier en **structure de dépendance** :
la corrélation inter-classes (Table 4) et la convexité TSMOM-marché (smile) sont
des questions de **dépendance de queue** que la corrélation de Pearson ne capture
pas. Avec ~480 mois, l'estimation des queues devient statistiquement crédible.

---

## 3. Fidélité de la réplication

Validée contre le **facteur TSMOM officiel d'AQR** : corrélation **0,74** (toutes
classes) à **0,89** (actions) sur 1985-2009. Le Sharpe reconstruit (0,98) est en
deçà de celui d'AQR (1,41) car les données sont des **séries génériques Bloomberg**
(roll reconstruit heuristiquement) et non les séries total-return propriétaires
d'AQR, sur un univers partiel. **Le sens de tous les résultats est reproduit** ;
les écarts sont des écarts de *niveau*, documentés dans la synthèse du notebook.

| Résultat clé | Papier | Réplication |
|---|---|---|
| TSMOM Sharpe / vol | > 1 / ~12 % | 0,98 / 12,1 % |
| Table 3A — alpha (t) / loading UMD | 1,58 % (8,0) / 0,28 | 0,86 % (4,6) / 0,24 |
| Table 4 — corr. Commodities | 0,07 | 0,06 |
| Table 5A — β(TSMOM~XSMOM, ALL) / R² | 0,66 / 44 % | 0,56 / 54 % |
| Table 5B — auto-cov. actions (vol-scalé) | 0,74 % | 0,75 % |
| Identités comptables (5B ; Futures=Spot+Roll) | — | < 1,4 bp ; exacte |

Extension out-of-sample :

| Période | Sharpe | Alpha 4F (%/an, t) | corr. AQR (ALL) |
|---|---|---|---|
| 1985-2009 | 0,98 | 10,1 % (4,3) | 0,74 |
| 2010-2025 | 0,31 | 2,1 % (0,79) | 0,87 |

→ l'alpha devient **non significatif** après publication (crowding du trend-following).
La hausse de corrélation avec AQR confirme que la décroissance **n'est pas** un
artefact de reconstruction.

---

## 4. Méthodologie & conventions

- **Volatilité ex-ante** (Eq. 1) : EWMA des rendements quotidiens au carré,
  centre de masse 60 jours, annualisée ×261, **décalée d'un mois** (pas de
  look-ahead).
- **Signal TSMOM** : signe du rendement excédentaire des 12 derniers mois ;
  position = `40 % / σ_{t-1}` (Eq. 5), moyenne équipondérée des instruments.
- **Roll des futures** : rendement du contrat *réellement détenu* (front M1 →
  M2 au roll), pas un `pct_change` naïf sur la série générique. Garde-fou prix ≤ 0.
- **Obligations** cotées `100 − rendement` converties via la duration cible.
- **Devises** : `FX_CROSS_PAIRS = True` → 36 paires croisées (élimine le facteur
  USD commun, rapproche les corrélations FX du papier). Mettre `False` pour
  10 paires vs-USD (utile pour la Fig. 2, voir notebook).
- **Décomposition Lo-MacKinlay** sur rendements **scalés par la volatilité**.
- **Corrélations Table 4** calculées **paire par paire** (intersection propre).
- **Event study** : rendements **dé-moyennés par instrument** (convention du papier).
- **IRF (Fig. 7)** : VAR panel à coefficients communs, 24 retards, Cholesky
  rendement-en-premier ; **bandes 90 % par cluster bootstrap** (sur instruments).

---

## 5. Structure du dépôt

```
src/
├── config.py             # paramètres, mapping instruments → classes d'actifs
├── data_loader.py        # chargement data.xlsx
├── returns.py            # rendements excédentaires journaliers → mensuels
├── volatility.py         # volatilité ex-ante EWMA (Eq. 1)
├── strategy.py           # signal TSMOM, position, diversification (Eq. 5)
├── crosssectional.py     # XSMOM + décomposition Lo-MacKinlay (Eq. 6-7)
├── rollyield.py          # décomposition Futures = Spot + Roll
├── factors.py            # facteurs Fama-French / VME pour Table 3
├── external_data.py      # téléchargement + cache des données externes
├── analysis.py           # toutes les régressions, corrélations, event study, IRF
├── tables.py             # mise en forme des 6 tables
├── plotting.py           # les 7 figures
├── pipeline.py           # orchestration bout-en-bout
└── extension_extended_sample.py   # extension out-of-sample 1985-2025

regenerate.py             # régénère TOUS les outputs en une passe (source de vérité)
validate_replication.py   # contrôles rapides (Sharpe, corr AQR, identités)
replication_complete.ipynb # notebook narratif (mode AFFICHAGE, n'écrit rien)
data.xlsx                 # prix des futures 1985-2025
```

### Données externes (pour Table 3 et CFTC)

Facteurs Fama-French (`FF_*.csv`), VME d'AQR, liquidité Pástor-Stambaugh,
sentiment Baker-Wurgler, positions CFTC (`cftc_net_spec_positions.csv`),
facteur TSMOM officiel AQR (`aqr_tsmom_factors.csv`, pour la validation).

---

## 6. Utilisation

```bash
# Dépendances : numpy, pandas, statsmodels, matplotlib, openpyxl

# 1) Réplication (échantillon papier 1985-2009)
python regenerate.py

# 2) Échantillon étendu
python regenerate.py --end 2025-12-31

# 3) Sans téléchargements externes (cœur seulement)
python regenerate.py --no-external

# 4) Contrôles de fidélité rapides
python validate_replication.py
```

Le **notebook** `replication_complete.ipynb` est en **mode affichage** : il
recalcule et commente tout à l'écran mais **n'écrase jamais** `outputs/`
(les écritures sont redirigées vers un dossier temporaire). `regenerate.py`
reste l'unique source de vérité pour les fichiers officiels.

Extension en script :

```python
import src.extension_extended_sample as ext
res  = ext.build_tsmom_full()
perf = ext.performance_by_subperiod(res["tsmom"], res["tsmom_ac"])
dec  = ext.alpha_decay(res["tsmom"], res["prices"],
                       ff_csv_3="FF_Research_Data_Factors.csv",
                       ff_csv_mom="FF_Momentum_Factor.csv")
corr = ext.corr_with_aqr(res["tsmom"], res["tsmom_ac"], "aqr_tsmom_factors.csv")
```

---

## 7. Feuille de route

- [x] Réplication complète des 6 tables et 7 figures (1985-2009)
- [x] Validation contre le facteur AQR officiel (corr 0,74-0,89)
- [x] Correctifs : `min_count`, corrélations pairwise, décomposition vol-scalée,
      event study dé-moyenné, bandes bootstrap sur l'IRF
- [x] Extension out-of-sample 1985-2025 (décroissance d'alpha documentée)
- [ ] Coûts de transaction / turnover (performance nette)
- [ ] Copules : dépendance de queue inter-sleeves (re-lecture de la Table 4) et
      convexité TSMOM-marché (re-test non-paramétrique du smile) ; copules
      dynamiques / à changement de régime pour la dimension temporelle

---

*Réplication académique à but pédagogique. Les données restent la propriété de
leurs fournisseurs respectifs (Bloomberg, AQR, Ken French, CFTC, etc.).*
