"""
Statistical analysis: pooled lag regressions (Fig. 1),
trading-strategy alpha grid (Table 2), and factor regressions (Table 3).
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

from .strategy import past_k_month_returns, tsmom_instrument_returns, diversified_tsmom
from .config import TARGET_VOL


# =======================================================================
# Fig. 1 — pooled lag regressions of vol-scaled returns on lagged signals
# =======================================================================

def pooled_lag_regression_size(monthly_ret: pd.DataFrame,
                                monthly_vol: pd.DataFrame,
                                max_lag: int = 60) -> pd.DataFrame:
    """
    Panel A of Fig. 1:
        r^s_t / σ^s_{t-1}   =   α  +  β_h · ( r^s_{t-h} / σ^s_{t-h-1} )   +   ε

    Pooled across all instruments and dates; clustered SE by month.

    Returns
    -------
    DataFrame indexed by lag h (1..max_lag) with columns beta, tstat.
    """
    scaled = monthly_ret / monthly_vol.shift(1)  # use σ_{t-1} for time-t return
    out = []
    for h in range(1, max_lag + 1):
        y = scaled.stack(future_stack=True).rename("y")
        x = scaled.shift(h).stack(future_stack=True).rename("x")
        df = pd.concat([y, x], axis=1).dropna()
        if len(df) < 100:
            out.append({"lag": h, "beta": np.nan, "tstat": np.nan, "nobs": len(df)})
            continue
        # cluster SE by time (level 0 of the original index = date)
        groups = df.index.get_level_values(0)
        X = sm.add_constant(df["x"])
        model = sm.OLS(df["y"], X).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )
        out.append({
            "lag": h,
            "beta": model.params["x"],
            "tstat": model.tvalues["x"],
            "nobs": int(model.nobs),
        })
    return pd.DataFrame(out).set_index("lag")


def pooled_lag_regression_sign(monthly_ret: pd.DataFrame,
                                monthly_vol: pd.DataFrame,
                                max_lag: int = 60) -> pd.DataFrame:
    """
    Panel B of Fig. 1:
        r^s_t / σ^s_{t-1}   =   α  +  β_h · sign( r^s_{t-h} )   +   ε
    """
    scaled = monthly_ret / monthly_vol.shift(1)
    signs = np.sign(monthly_ret).replace(0, np.nan)
    out = []
    for h in range(1, max_lag + 1):
        y = scaled.stack(future_stack=True).rename("y")
        x = signs.shift(h).stack(future_stack=True).rename("x")
        df = pd.concat([y, x], axis=1).dropna()
        if len(df) < 100:
            out.append({"lag": h, "beta": np.nan, "tstat": np.nan, "nobs": len(df)})
            continue
        groups = df.index.get_level_values(0)
        X = sm.add_constant(df["x"])
        model = sm.OLS(df["y"], X).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )
        out.append({
            "lag": h,
            "beta": model.params["x"],
            "tstat": model.tvalues["x"],
            "nobs": int(model.nobs),
        })
    return pd.DataFrame(out).set_index("lag")


# =======================================================================
# Table 2 — alphas for (k, h) trading strategies
# =======================================================================

def tsmom_strategy_returns_kh(monthly_ret: pd.DataFrame,
                               monthly_vol: pd.DataFrame,
                               k: int,
                               h: int,
                               target_vol: float = TARGET_VOL) -> pd.Series:
    """
    Build TSMOM portfolio with lookback k and holding period h.

    Following Jegadeesh-Titman, the time-t return is the average of the
    h currently-active sub-portfolios (one opened last month, one two
    months ago, etc.). No overlap, single time series of monthly returns.
    """
    past_k = past_k_month_returns(monthly_ret, k=k)
    sig = np.sign(past_k).where(past_k.notna())
    scale = (target_vol / monthly_vol).replace([np.inf, -np.inf], np.nan)
    pos_t = sig * scale  # set at end of month t for month t+1

    # The sub-portfolio "opened at end of month t-j" earns r_{t+1} with position pos_{t-j}.
    # Equivalently: position used for r_{t+1} is the AVERAGE of pos_t, pos_{t-1}, ..., pos_{t-h+1}.
    avg_pos = pos_t.rolling(window=h, min_periods=1).mean()
    inst_ret = avg_pos.shift(1) * monthly_ret
    return inst_ret.mean(axis=1)


def factor_alpha_tstat(strategy_returns: pd.Series,
                        factors: pd.DataFrame) -> tuple:
    """
    Regress strategy returns on factors and return (alpha, tstat_alpha, R^2).
    Newey-West HAC SE (3 lags, monthly).
    """
    df = pd.concat([strategy_returns.rename("y"), factors], axis=1).dropna()
    if len(df) < 24:
        return (np.nan, np.nan, np.nan)
    X = sm.add_constant(df[factors.columns])
    model = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    return (model.params["const"], model.tvalues["const"], model.rsquared)


def table2_grid(monthly_ret: pd.DataFrame,
                 monthly_vol: pd.DataFrame,
                 factors: pd.DataFrame,
                 k_grid=(1, 3, 6, 9, 12, 24, 36, 48),
                 h_grid=(1, 3, 6, 9, 12, 24, 36, 48)) -> pd.DataFrame:
    """
    Build the (k, h) grid of t-stats on the alpha of each TSMOM strategy.
    Mirrors Panel A of Table 2 in the paper.
    """
    grid = pd.DataFrame(index=k_grid, columns=h_grid, dtype=float)
    grid.index.name = "Lookback (k)"
    grid.columns.name = "Holding (h)"
    for k in k_grid:
        for h in h_grid:
            r = tsmom_strategy_returns_kh(monthly_ret, monthly_vol, k=k, h=h)
            alpha, tstat, _ = factor_alpha_tstat(r, factors)
            grid.loc[k, h] = tstat
    return grid


# =======================================================================
# Table 3 — performance of diversified TSMOM
# =======================================================================

def factor_regression_summary(strategy_returns: pd.Series,
                               factors: pd.DataFrame,
                               freq: str = "monthly") -> dict:
    """
    Run regression of strategy on factors. Returns dict with coefficients,
    t-stats, alpha (monthly or quarterly), R², and Sharpe.
    """
    df = pd.concat([strategy_returns.rename("y"), factors], axis=1).dropna()
    X = sm.add_constant(df[factors.columns])
    model = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    out = {
        "freq": freq,
        "alpha": model.params["const"],
        "alpha_tstat": model.tvalues["const"],
        "betas": {c: model.params[c] for c in factors.columns},
        "betas_tstat": {c: model.tvalues[c] for c in factors.columns},
        "R2": model.rsquared,
        "N": int(model.nobs),
    }
    return out


# =======================================================================
# Misc — Sharpe ratios per instrument (Fig. 2)
# =======================================================================

def annualized_sharpe(returns: pd.Series, ppy: int = 12) -> float:
    r = returns.dropna()
    if r.std() == 0 or len(r) < 12:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(ppy))


def sharpe_by_instrument(instrument_tsmom: pd.DataFrame) -> pd.Series:
    return instrument_tsmom.apply(annualized_sharpe).sort_values(ascending=False)


# =======================================================================
# Table 3 (version complète, 6 facteurs) — format de l'article
# =======================================================================

def _to_quarterly(monthly_returns: pd.Series) -> pd.Series:
    """Rendements trimestriels non chevauchants (composition)."""
    return (1.0 + monthly_returns).resample("QE").prod() - 1.0


def _quarterly_factors(factors: pd.DataFrame) -> pd.DataFrame:
    """Composition trimestrielle non chevauchante des facteurs."""
    return (1.0 + factors).resample("QE").prod() - 1.0


def table3_full(diversified_tsmom_ret: pd.Series,
                factors: pd.DataFrame,
                hac_lags_monthly: int = 3,
                hac_lags_quarterly: int = 1) -> pd.DataFrame:
    """
    Reproduit la Table 3 du papier : régression du TSMOM diversifié (en excès)
    sur MKT, BOND, GSCI, SMB, HML, UMD — en mensuel (ligne 1) et trimestriel
    non chevauchant (ligne 2). SE Newey-West (HAC).

    Renvoie un DataFrame : lignes = {Monthly, Quarterly},
    colonnes = [Alpha, t(Alpha), MKT, t(MKT), ..., UMD, t(UMD), R2, N].
    L'alpha est exprimé en % (×100) pour comparer aux 1.58%/mois du papier.
    """
    cols_order = list(factors.columns)
    results = {}

    # ---- Mensuel ----
    dfm = pd.concat([diversified_tsmom_ret.rename("y"), factors], axis=1).dropna()
    Xm = sm.add_constant(dfm[cols_order])
    mm = sm.OLS(dfm["y"], Xm).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags_monthly})

    # ---- Trimestriel non chevauchant ----
    yq = _to_quarterly(diversified_tsmom_ret)
    fq = _quarterly_factors(factors)
    dfq = pd.concat([yq.rename("y"), fq], axis=1).dropna()
    Xq = sm.add_constant(dfq[cols_order])
    mq = sm.OLS(dfq["y"], Xq).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags_quarterly})

    def _row(model):
        row = {"Alpha (%)": model.params["const"] * 100.0,
               "t(Alpha)": model.tvalues["const"]}
        for c in cols_order:
            row[c] = model.params[c]
            row[f"t({c})"] = model.tvalues[c]
        row["R2"] = model.rsquared
        row["N"] = int(model.nobs)
        return row

    results["Monthly"] = _row(mm)
    results["Quarterly"] = _row(mq)
    out = pd.DataFrame(results).T
    # ordre des colonnes
    ordered = ["Alpha (%)", "t(Alpha)"]
    for c in cols_order:
        ordered += [c, f"t({c})"]
    ordered += ["R2", "N"]
    return out[ordered]


def diversified_performance(diversified_tsmom_ret: pd.Series,
                            ppy: int = 12) -> dict:
    """Stats de performance brute du TSMOM diversifié (pour le résumé)."""
    r = diversified_tsmom_ret.dropna()
    mean_a = r.mean() * ppy
    vol_a = r.std() * np.sqrt(ppy)
    sharpe = mean_a / vol_a if vol_a > 0 else np.nan
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    # skewness / kurtosis
    return {
        "ann_mean": mean_a,
        "ann_vol": vol_a,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "skew": r.skew(),
        "kurtosis": r.kurtosis(),
        "N_months": len(r),
    }
