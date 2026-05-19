"""
Build daily and monthly excess returns for each instrument.

For futures (equity, bond, commodity) the % change of the contract price is
already an *excess* return (margin-financed). For currencies, we compute
the standard FX forward excess return: %change in the spot leg plus the
foreign-minus-USD interest-rate carry over the period.
"""

import pandas as pd
import numpy as np

from .config import (
    EQUITY_FUTURES,
    BOND_FUTURES,
    COMMODITY_FUTURES,
    CURRENCY_FORWARDS,
    USD_RATE,
)


def _pct_change(price: pd.Series) -> pd.Series:
    """Simple % change (futures return)."""
    return price.pct_change()


def futures_daily_excess_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily excess returns for equity/bond/commodity futures (just %change)."""
    cols = list(EQUITY_FUTURES) + list(BOND_FUTURES) + list(COMMODITY_FUTURES)
    cols = [c for c in cols if c in prices.columns]
    return prices[cols].apply(_pct_change)


def fx_daily_excess_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    FX-forward daily excess returns.

    For a long-foreign-vs-USD position:
        r_t ≈ Δlog(S^FCY/USD)_t + (i_FCY - i_USD) * (1/252)

    The interest-rate columns are *annualized percent* rates (Bloomberg style),
    so we convert to a daily rate.
    """
    out = {}
    usd_rate = prices[USD_RATE] / 100.0 / 252.0  # daily decimal

    for spot_ticker, meta in CURRENCY_FORWARDS.items():
        if spot_ticker not in prices.columns:
            continue
        s = prices[spot_ticker].copy()
        # Quote convention: if invert=True, the spot is USD per 1 FCY's inverse,
        # i.e. "USDXXX" gives FCY per USD; we want USD per FCY, so invert.
        if meta["invert"]:
            s = 1.0 / s
        spot_ret = s.pct_change()

        rate_fcy_t = prices[meta["rate_fcy"]] / 100.0 / 252.0
        carry = (rate_fcy_t - usd_rate).shift(1)  # use yesterday's rate as carry

        out[spot_ticker] = spot_ret + carry

    return pd.DataFrame(out)


def build_daily_excess_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Combine futures and FX-forward daily excess returns into one frame."""
    fut = futures_daily_excess_returns(prices)
    fx = fx_daily_excess_returns(prices)
    return pd.concat([fut, fx], axis=1)


def daily_to_monthly_returns(daily_ret: pd.DataFrame) -> pd.DataFrame:
    """
    Compound daily returns into monthly returns (month-end stamps).
    """
    # (1 + r_daily).prod() - 1, per month-end
    return (1.0 + daily_ret).resample("ME").prod() - 1.0
