"""
Ex-ante volatility estimation per Moskowitz, Hua Ooi, Pedersen (2012) Eq. (1):

    σ_t² = 261 · Σ_{i≥0} (1-δ) δ^i (r_{t-1-i} - r̄_t)²

where r̄_t is the analogous EWMA of past returns, δ is chosen so that the
centre of mass of the weights is 60 days (⇒ δ = 60/61).

The model is applied to daily returns; the result is an *annualized* variance.
We always use the t-1 value when sizing time-t positions to avoid look-ahead.
"""

import pandas as pd
import numpy as np

from .config import EWMA_COM_DAYS, ANNUALIZATION_DAYS


def ewma_ex_ante_vol(daily_returns: pd.DataFrame,
                     com: int = EWMA_COM_DAYS,
                     annualization: int = ANNUALIZATION_DAYS,
                     min_periods: int = 60) -> pd.DataFrame:
    """
    EWMA daily volatility scaled to annual, exactly as in Eq. (1).

    Parameters
    ----------
    daily_returns : DataFrame
        Daily excess returns, one column per instrument.
    com : int
        Centre-of-mass of the EWMA. com=60 ⇒ δ = 60/61.
    annualization : int
        Scalar 261 in the paper.
    min_periods : int
        Minimum observations before reporting a vol number.

    Returns
    -------
    DataFrame of annualized ex-ante volatilities (σ_t), same shape.
    """
    # ewm.var with com=N gives an EWMA second moment of returns about their
    # exponential mean — exactly the (1-δ) Σ δ^i (r_{t-i} - r̄_t)^2 in the paper.
    ewm_var = daily_returns.ewm(com=com, min_periods=min_periods, adjust=True).var()
    annual_var = annualization * ewm_var
    return np.sqrt(annual_var)


def vol_for_signal(daily_vol_annual: pd.DataFrame,
                   monthly_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Convert the daily ex-ante vol series to a monthly series stamped
    at month-end. We use the *last* daily vol available BEFORE the month-end
    (no look-ahead).
    """
    # Re-sample to month-end, taking the last observation in each month
    monthly_vol = daily_vol_annual.resample("ME").last()
    return monthly_vol.reindex(monthly_index)
