"""
TSMOM signal construction and portfolio formation, per Eq. (5):

    r^TSMOM,s_{t,t+1} = sign(r^s_{t-12,t}) · (TARGET_VOL / σ^s_t) · r^s_{t,t+1}

The diversified TSMOM factor is an equal-weighted average across all
*available* instruments at each month-end.
"""

import pandas as pd
import numpy as np

from .config import TARGET_VOL, LOOKBACK_MONTHS, asset_class_of


def past_k_month_returns(monthly_ret: pd.DataFrame, k: int = LOOKBACK_MONTHS) -> pd.DataFrame:
    """
    Cumulative return over the previous k months, ending at month t.
    The output at row t is the compound return from t-k+1 through t inclusive,
    i.e. the signal that is *known* at the end of month t and used to size
    the position for month t+1.
    """
    log_ret = np.log1p(monthly_ret)
    cum_log = log_ret.rolling(window=k).sum()
    return np.expm1(cum_log)


def tsmom_signal(monthly_ret: pd.DataFrame, k: int = LOOKBACK_MONTHS) -> pd.DataFrame:
    """+1 / -1 sign of the past-k-month cumulative return (NaN if insufficient data)."""
    past = past_k_month_returns(monthly_ret, k=k)
    return np.sign(past).where(past.notna())


def tsmom_instrument_returns(monthly_ret: pd.DataFrame,
                              monthly_vol: pd.DataFrame,
                              k: int = LOOKBACK_MONTHS,
                              target_vol: float = TARGET_VOL) -> pd.DataFrame:
    """
    Per-instrument TSMOM excess return at time t+1:

        sign(past_k_t) * (target_vol / σ_t) * r_{t+1}
    """
    sig = tsmom_signal(monthly_ret, k=k)                          # known at end of month t
    scale = (target_vol / monthly_vol).replace([np.inf, -np.inf], np.nan)
    # Position is *set* at end of t, applied to t+1 return → shift the signal/scale by 1
    pos = (sig * scale).shift(1)
    return pos * monthly_ret


def diversified_tsmom(instrument_tsmom: pd.DataFrame) -> pd.Series:
    """
    Equal-weighted diversified TSMOM portfolio across all available instruments
    at each month (i.e. take the cross-sectional mean over non-NaN columns).
    """
    return instrument_tsmom.mean(axis=1)


def tsmom_by_asset_class(instrument_tsmom: pd.DataFrame) -> pd.DataFrame:
    """Equal-weighted TSMOM within each asset class."""
    classes = {c: asset_class_of(c) for c in instrument_tsmom.columns}
    by_class = {}
    for ac in ("Commodity", "Equity", "Bond", "Currency"):
        cols = [c for c, k in classes.items() if k == ac]
        if cols:
            by_class[ac] = instrument_tsmom[cols].mean(axis=1)
    return pd.DataFrame(by_class)


def passive_long(monthly_ret: pd.DataFrame,
                 monthly_vol: pd.DataFrame,
                 target_vol: float = TARGET_VOL) -> pd.Series:
    """
    Diversified passive long: same volatility scaling, but always long.
    Useful as a benchmark in Fig. 3.
    """
    scale = (target_vol / monthly_vol).replace([np.inf, -np.inf], np.nan)
    inst_ret = scale.shift(1) * monthly_ret
    return inst_ret.mean(axis=1)
