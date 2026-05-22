"""
rollyield.py — Décomposition rendement futures = variation du prix « spot » +
roll return, pour la Table 6 de Moskowitz, Ooi & Pedersen (2012), Section 6.3.

CORRECTION : on implémente désormais la VRAIE décomposition du papier, et non
plus un proxy de pente de courbe. Le papier pose l'identité comptable

    Futures return_{t-12,t} = Price change_{t-12,t} + Roll return_{t-12,t}

où « Price change » est la variation du prix du contrat le plus proche de
l'échéance (nearest-to-expiration), et « Futures return » est le rendement
RÉELLEMENT investi (donc roulé). Le roll return est le résidu.

Mise en œuvre, à partir des prix M1 (front) et du rendement roulé déjà calculé
par `returns.py` :
    r_total = rendement futures roulé (investable)            -> components['total']
    r_spot  = variation du prix front non roulée (niveau M1)  -> components['spot']
    r_roll  = r_total - r_spot                                -> components['roll']

Le « spot » capte l'information (variation du niveau de prix) ; le « roll »
capte la pente de la courbe / pression de couverture. Sur les futures
financiers le roll est proche de zéro ; sur les commodities il peut être
substantiel (backwardation/contango).
"""

import pandas as pd
import numpy as np

from .returns import (futures_daily_excess_returns, safe_pct_change,
                      yield_quoted_bond_return)
from .config import (COMMODITY_FUTURES, BOND_FUTURES, EQUITY_FUTURES,
                     YIELD_QUOTED_BONDS)


def _front_price_change_daily(prices: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Variation journalière du NIVEAU de prix du contrat front (non roulée).
    Sert de proxy « spot price change » au sens de MOP §6.3. Les obligations
    cotées en rendement (AUS) utilisent la même conversion duration que le total,
    de sorte que le roll y soit nul (cohérent : pas de roll sur un financier)."""
    out = {}
    for c in cols:
        if c not in prices.columns:
            continue
        if c in YIELD_QUOTED_BONDS:
            out[c] = yield_quoted_bond_return(prices[c], YIELD_QUOTED_BONDS[c])
        else:
            out[c] = safe_pct_change(prices[c])
    return pd.DataFrame(out)


def spot_roll_monthly(prices: pd.DataFrame) -> dict:
    """
    Trois DataFrames mensuels (fin de mois) alignés et cohérents avec
    l'identité Futures = Spot + Roll :
      'total' : rendement futures ROULÉ (investable),
      'spot'  : variation du prix front (niveau M1, non roulée),
      'roll'  : roll return = total - spot.
    """
    cols = list(EQUITY_FUTURES) + list(BOND_FUTURES) + list(COMMODITY_FUTURES)
    cols = [c for c in cols if c in prices.columns]

    # rendement total = rendement roulé (réutilise la logique corrigée de returns.py)
    daily_total = futures_daily_excess_returns(prices)[cols]
    monthly_total = (1 + daily_total).resample("ME").prod() - 1

    # variation spot = niveau du prix front, non roulé
    daily_spot = _front_price_change_daily(prices, cols)
    monthly_spot = (1 + daily_spot).resample("ME").prod() - 1

    monthly_spot = monthly_spot.reindex(columns=monthly_total.columns)
    monthly_roll = monthly_total - monthly_spot
    return {"total": monthly_total, "roll": monthly_roll, "spot": monthly_spot}


def momentum_signals_12m(components: dict, k: int = 12) -> dict:
    """Pour chaque composante (total/spot/roll), rendement cumulé (somme) des
    k derniers mois — le signal de la Table 6 (« Full / Spot / Roll MOM »).
    On somme (cohérent avec l'identité additive total = spot + roll)."""
    out = {}
    for key, df in components.items():
        out[key] = df.rolling(k).sum()
    return out
