"""
rollyield.py — Décomposition rendement futures = rendement spot + roll yield,
pour la Table 6 de Moskowitz, Hua Ooi & Pedersen (2012).

Idée : pour un contrat à terme, le rendement total de la position « front month
roulé » se décompose en une composante « prix spot » et une composante « roll »
(structure de la courbe). On approxime, à partir des prix de 1re (F1) et 2e (F2)
échéance disponibles dans data.xlsx (colonnes …1 Comdty / …2 Comdty) :

    roll_yield_t ≈ (F1_t - F2_t) / F2_t           (positif en backwardation)
    r_total_t    ≈ %variation de F1               (rendement futures)
    r_spot_t     ≈ r_total_t - roll_yield_t        (résidu = variation « prix »)

La Table 6 régresse le rendement futures mensuel sur :
  - « Full TSMOM »   : rendement total des 12 derniers mois,
  - « Spot price MOM »: variation spot des 12 derniers mois,
  - « Roll MOM »      : roll return des 12 derniers mois,
  - (+ positions spéculateurs CFTC, fournies à part).
"""

import pandas as pd
import numpy as np

# Paires (front, second) détectées automatiquement, mais on liste les
# correspondances « X1 -> X2 » par substitution du chiffre d'échéance.
def _second_contract_name(front: str) -> str | None:
    """'CL1 Comdty' -> 'CL2 Comdty', 'C 1 Comdty' -> 'C 2 Comdty'."""
    if "1 Comdty" in front:
        return front.replace("1 Comdty", "2 Comdty")
    return None


def available_roll_pairs(prices: pd.DataFrame) -> dict:
    """Renvoie {front_ticker: second_ticker} pour les contrats ayant un M2."""
    pairs = {}
    for c in prices.columns:
        c2 = _second_contract_name(c)
        if c2 and c2 in prices.columns:
            pairs[c] = c2
    return pairs


def spot_roll_monthly(prices: pd.DataFrame) -> dict:
    """
    Construit trois DataFrames mensuels (fin de mois) alignés :
      'total' : rendement futures total (%var F1, composé en mensuel),
      'roll'  : roll yield mensuel (moyenne du basis (F1-F2)/F2 sur le mois),
      'spot'  : rendement spot ≈ total - roll.
    Limité aux instruments disposant d'un contrat M2.
    """
    pairs = available_roll_pairs(prices)
    f1 = prices[list(pairs.keys())]
    f2 = prices[[pairs[c] for c in pairs]].copy()
    f2.columns = list(pairs.keys())   # aligne les noms sur le front

    # rendement total journalier puis mensuel
    daily_total = f1.pct_change()
    monthly_total = (1 + daily_total).resample("ME").prod() - 1

    # roll yield journalier = (F1 - F2)/F2 ; on agrège en mensuel par la moyenne
    daily_roll = (f1 - f2) / f2
    monthly_roll = daily_roll.resample("ME").mean()

    monthly_spot = monthly_total - monthly_roll
    return {"total": monthly_total, "roll": monthly_roll, "spot": monthly_spot}


def momentum_signals_12m(components: dict, k: int = 12) -> dict:
    """Pour chaque composante (total/spot/roll), renvoie le cumul des k mois."""
    out = {}
    for key, df in components.items():
        log = np.log1p(df.clip(lower=-0.999))
        out[key] = np.expm1(log.rolling(k).sum())
    return out
