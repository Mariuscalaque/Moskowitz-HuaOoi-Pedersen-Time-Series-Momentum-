"""
Pretty-printing of the paper's Tables 1, 2, 3 — saved as CSV and Markdown.
"""

import pandas as pd
import numpy as np

from .config import (
    TABLES_DIR,
    EQUITY_FUTURES, BOND_FUTURES, COMMODITY_FUTURES, CURRENCY_FORWARDS,
    asset_class_of, pretty_name,
)
from .analysis import annualized_sharpe


def _save(df: pd.DataFrame, name: str, float_fmt: str = "%.3f"):
    df.to_csv(TABLES_DIR / f"{name}.csv", float_format=float_fmt)
    md = df.to_markdown(floatfmt=".3f")
    (TABLES_DIR / f"{name}.md").write_text(md)
    return TABLES_DIR / f"{name}.csv"


# -----------------------------------------------------------------------
# Table 1 — Summary statistics of futures contract excess returns
# -----------------------------------------------------------------------

def table1_summary_stats(monthly_ret: pd.DataFrame,
                          daily_ret: pd.DataFrame) -> pd.DataFrame:
    """
    Per-instrument: first valid date, annualized mean and vol.
    Order: Commodities, Equity, Bonds, Currencies.
    """
    rows = []
    instruments = (
        [(t, "Equity")    for t in EQUITY_FUTURES]
      + [(t, "Bond")      for t in BOND_FUTURES]
      + [(t, "Commodity") for t in COMMODITY_FUTURES]
      + [(t, "Currency")  for t in CURRENCY_FORWARDS]
    )
    for t, ac in instruments:
        if t not in monthly_ret.columns:
            continue
        s = monthly_ret[t].dropna()
        if len(s) == 0:
            continue
        rows.append({
            "Asset class": ac,
            "Instrument": pretty_name(t),
            "Ticker": t,
            "Start": s.index[0].strftime("%Y-%m"),
            "Ann. mean": s.mean() * 12,
            "Ann. vol":  s.std() * np.sqrt(12),
            "N months":  len(s),
        })
    df = pd.DataFrame(rows)
    df["AC_order"] = df["Asset class"].map(
        {"Commodity": 0, "Equity": 1, "Bond": 2, "Currency": 3}
    )
    df = df.sort_values(["AC_order", "Instrument"]).drop(columns="AC_order")
    _save(df.set_index(["Asset class", "Instrument"]),
          "table1_summary_stats", float_fmt="%.4f")
    return df


# -----------------------------------------------------------------------
# Table 2 — t-statistics of alphas for (k, h) strategies
# -----------------------------------------------------------------------

def table2_save(grid: pd.DataFrame, panel_name: str = "panelA_all"):
    out = grid.copy()
    _save(out, f"table2_{panel_name}", float_fmt="%.2f")
    return out


# -----------------------------------------------------------------------
# Table 3 — Performance of diversified TSMOM
# -----------------------------------------------------------------------

def table3_performance(tsmom_monthly: pd.Series,
                        factor_regs: dict) -> pd.DataFrame:
    """
    Assemble a Table 3-style summary from a dict of regressions.
    Each entry of `factor_regs` is a dict from `analysis.factor_regression_summary`.
    """
    rows = []
    for name, res in factor_regs.items():
        row = {
            "Regression": name,
            "Freq": res["freq"],
            "Alpha": res["alpha"],
            "α t-stat": res["alpha_tstat"],
            "R²": res["R2"],
            "N": res["N"],
        }
        for f, b in res["betas"].items():
            row[f"β({f})"] = b
            row[f"t({f})"] = res["betas_tstat"][f]
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Regression")
    _save(df, "table3_performance", float_fmt="%.3f")
    return df


# -----------------------------------------------------------------------
# Performance summary — global stats for the TSMOM portfolio
# -----------------------------------------------------------------------

def performance_summary(returns_dict: dict) -> pd.DataFrame:
    """
    returns_dict: {name: monthly excess return series}.
    Build mean/vol/Sharpe/skew/kurt/MaxDD/CAGR.
    """
    rows = []
    for name, r in returns_dict.items():
        r = r.dropna()
        if len(r) == 0:
            continue
        ann_ret = r.mean() * 12
        ann_vol = r.std() * np.sqrt(12)
        sharpe = annualized_sharpe(r)
        # Max drawdown
        eq = (1 + r).cumprod()
        peak = eq.cummax()
        dd = eq / peak - 1
        rows.append({
            "Series": name,
            "N months": len(r),
            "Ann. mean": ann_ret,
            "Ann. vol":  ann_vol,
            "Sharpe":    sharpe,
            "Skew":      r.skew(),
            "Excess kurt": r.kurtosis(),
            "Max DD":    dd.min(),
            "CAGR":      eq.iloc[-1] ** (12.0 / len(r)) - 1,
        })
    df = pd.DataFrame(rows).set_index("Series")
    _save(df, "performance_summary", float_fmt="%.4f")
    return df
