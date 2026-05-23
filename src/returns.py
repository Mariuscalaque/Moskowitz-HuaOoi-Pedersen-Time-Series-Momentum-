"""
Build daily and monthly excess returns for each instrument.

CORRECTIONS (réplication fidèle MOP 2012) :
  * Rendements futures ROULÉS : les séries génériques "front-month" de Bloomberg
    ("CL1 Comdty", ...) ne sont PAS ajustées du roll. Un simple pct_change compte
    le saut de prix au changement de contrat comme un faux rendement (cf. WTI le
    21/03/2007 : +5% purement dû au roll). Lorsqu'on dispose du 2e contrat (M2),
    on reconstruit le rendement du contrat réellement détenu :
        - jour normal : F1_t / F1_{t-1} - 1
        - jour de roll : F1_t / F2_{t-1} - 1   (on a roulé dans l'ex-M2)
  * Garde-fou prix <= 0 : le 20/04/2020 le WTI front-month cote -37.63, ce qui
    casse le rendement multiplicatif (pct_change = -306%). On neutralise tout
    prix non strictement positif (NaN) -> pas de rendement aberrant.

Pour les futures sans M2 disponible (actions, obligations, métaux LME) on
utilise le rendement « sûr » (pct_change protégé). Pour les devises, on calcule
le rendement excédentaire du forward FX : variation du spot + carry (i_FCY-i_USD).
"""

import pandas as pd
import numpy as np
from itertools import combinations

from .config import (
    EQUITY_FUTURES,
    BOND_FUTURES,
    COMMODITY_FUTURES,
    CURRENCY_FORWARDS,
    USD_RATE,
    YIELD_QUOTED_BONDS,
    FX_CROSS_PAIRS,
)

# Seuils de détection du roll (réglables). Un roll se manifeste par un écart
# anormal entre le rendement mesuré sur F1 et celui mesuré sur F2.
ROLL_GAP_THRESHOLD = 0.015     # |r_F1 - r_F2| au-delà duquel on suspecte un roll
ROLL_IMPROVE_MARGIN = 0.005    # le rendement « roulé » doit être nettement plus petit


def safe_pct_change(price: pd.Series) -> pd.Series:
    """% change protégé : tout prix <= 0 est mis à NaN (évite l'artefact WTI 2020)."""
    pr = price.astype(float).where(price > 0)
    return pr / pr.shift(1) - 1.0


def rolled_return_with_m2(f1: pd.Series, f2: pd.Series) -> pd.Series:
    """
    Rendement journalier roulé propre à partir des prix M1 (f1) et M2 (f2).

    Hors roll, F1 et F2 bougent quasi à l'identique ; au roll, F1 bascule sur
    l'ancien M2 et son % de variation incorpore le saut de base. On détecte ce
    cas et on remplace alors le rendement par celui du contrat réellement
    détenu (l'ex-M2) : F1_t / F2_{t-1} - 1.
    """
    f1 = f1.astype(float).where(f1 > 0)
    f2 = f2.astype(float).where(f2 > 0)
    r_f1 = f1 / f1.shift(1) - 1.0
    r_f2 = f2 / f2.shift(1) - 1.0
    r_roll = f1 / f2.shift(1) - 1.0          # rendement si l'on a roulé dans l'ex-M2
    gap = (r_f1 - r_f2).abs()
    is_roll = (gap > ROLL_GAP_THRESHOLD) & (r_roll.abs() < r_f1.abs() - ROLL_IMPROVE_MARGIN)
    return r_f1.where(~is_roll, r_roll)


def _second_contract_name(front: str):
    """'CL1 Comdty' -> 'CL2 Comdty' ; renvoie None si pas de schéma '1 Comdty'."""
    if front.endswith("1 Comdty"):
        return front[:-len("1 Comdty")] + "2 Comdty"
    return None


def yield_quoted_bond_return(quote: pd.Series, target_duration: float) -> pd.Series:
    """
    Rendement journalier d'un future obligataire coté en « 100 − rendement »
    (convention AUS). La cote q vérifie yield = 100 − q, donc Δyield = −Δq, et
    le rendement obligataire ≈ −D·Δyield = D·Δq/100 (D = duration-cible).
    Le signe est identique à celui d'un pct_change sur la cote.
    """
    q = quote.astype(float).where(quote > 0)
    return target_duration * q.diff() / 100.0


def futures_daily_excess_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Rendements journaliers excédentaires des futures actions / obligations /
    matières premières. Pour les commodities ayant un M2 dans `prices`, on
    utilise le rendement roulé ; sinon, le pct_change protégé. Les obligations
    cotées en rendement (AUS) sont converties via la duration.
    """
    out = {}
    # Actions et obligations
    for c in list(EQUITY_FUTURES) + list(BOND_FUTURES):
        if c not in prices.columns:
            continue
        if c in YIELD_QUOTED_BONDS:                      # AUS : cote 100 − rendement
            out[c] = yield_quoted_bond_return(prices[c], YIELD_QUOTED_BONDS[c])
        else:                                            # prix normal -> pct_change sûr
            out[c] = safe_pct_change(prices[c])
    # Matières premières : rendement roulé si M2 dispo, sinon pct_change sûr
    for c in COMMODITY_FUTURES:
        if c not in prices.columns:
            continue
        c2 = _second_contract_name(c)
        if c2 and c2 in prices.columns:
            out[c] = rolled_return_with_m2(prices[c], prices[c2])
        else:
            out[c] = safe_pct_change(prices[c])
    return pd.DataFrame(out)


def fx_daily_excess_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Rendement journalier excédentaire d'un forward FX (long FCY contre USD) :
        r_t ≈ Δlog(S^{USD/FCY})_t + (i_FCY - i_USD) * (1/252)
    Les taux 1M sont annualisés en % (style Bloomberg) -> convertis en taux jour.
    """
    out = {}
    usd_rate = prices[USD_RATE] / 100.0 / 252.0

    for spot_ticker, meta in CURRENCY_FORWARDS.items():
        if spot_ticker not in prices.columns:
            continue
        s = prices[spot_ticker].astype(float).copy()
        if meta["invert"]:           # 'USDXXX' = FCY par USD -> on veut USD par FCY
            s = 1.0 / s
        spot_ret = s.pct_change()
        rate_fcy_t = prices[meta["rate_fcy"]] / 100.0 / 252.0
        carry = (rate_fcy_t - usd_rate).shift(1)   # carry connu la veille
        out[spot_ticker] = spot_ret + carry

    return pd.DataFrame(out)


def _ccy_code(spot_ticker: str) -> str:
    """'AUDUSD Curncy' -> 'AUD' ; 'USDCAD Curncy' -> 'CAD' (on retire 'USD')."""
    pair = spot_ticker.split()[0]      # 'AUDUSD' / 'USDCAD'
    return pair.replace("USD", "")      # 'AUD' / 'CAD'


def build_fx_cross_pairs(fx_vs_usd: pd.DataFrame) -> pd.DataFrame:
    """
    Rendements excédentaires journaliers des PAIRES CROISÉES, à partir des
    rendements vs-USD déjà calculés (chacun = long-devise-vs-USD en excès).

    Pour deux devises i, j (toutes deux exprimées vs USD), le rendement de la
    paire croisée « long i / short j » vaut r_i − r_j : la jambe USD (spot ET
    carry) s'annule. Cela élimine le facteur USD commun qui gonfle les
    corrélations FX. On splice d'abord le Deutsche Mark dans l'Euro, puis on
    forme les C(9,2)=36 paires.
    """
    fx = fx_vs_usd.copy()
    if "DEMUSD Curncy" in fx.columns and "EURUSD Curncy" in fx.columns:
        fx["EURUSD Curncy"] = fx["EURUSD Curncy"].fillna(fx["DEMUSD Curncy"])
        fx = fx.drop(columns=["DEMUSD Curncy"])
    cols = list(fx.columns)
    out = {}
    for a, b in combinations(cols, 2):
        name = f"{_ccy_code(a)}/{_ccy_code(b)} Cross"
        out[name] = fx[a] - fx[b]
    return pd.DataFrame(out)


def build_daily_excess_returns(prices: pd.DataFrame,
                               fx_cross: bool = FX_CROSS_PAIRS) -> pd.DataFrame:
    """Concatène futures (roulés) et devises en rendements journaliers excédentaires.

    `fx_cross=True` (défaut) construit les 36 paires croisées ; `False` garde les
    10 paires vs-USD (ancien comportement)."""
    fut = futures_daily_excess_returns(prices)
    fx_vs_usd = fx_daily_excess_returns(prices)
    fx = build_fx_cross_pairs(fx_vs_usd) if fx_cross else fx_vs_usd
    return pd.concat([fut, fx], axis=1)


def daily_to_monthly_returns(daily_ret: pd.DataFrame) -> pd.DataFrame:
    """Compose les rendements journaliers en mensuels (horodatés fin de mois)."""
    return (1.0 + daily_ret).resample("ME").prod() - 1.0
