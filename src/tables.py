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
    # encoding="utf-8" explicite : sous Windows write_text() utilise sinon
    # l'encodage local (cp1252) qui ne sait pas encoder certains caractères.
    df.to_csv(TABLES_DIR / f"{name}.csv", float_format=float_fmt, encoding="utf-8")
    md = df.to_markdown(floatfmt=".3f")
    (TABLES_DIR / f"{name}.md").write_text(md, encoding="utf-8")
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
    # Pilote par les colonnes RÉELLEMENT présentes (compatible vs-USD ET paires
    # croisées) : on classe chaque instrument via asset_class_of.
    instruments = [(t, asset_class_of(t)) for t in monthly_ret.columns
                   if asset_class_of(t) in ("Equity", "Bond", "Commodity", "Currency")]
    for t, ac in instruments:
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


# -----------------------------------------------------------------------
# EXTENSIONS — Tables 3 (B/C), 4, 5, 6
# -----------------------------------------------------------------------

def table3_panel_save(panel_df: pd.DataFrame, name: str):
    """Sauvegarde un panel de Table 3 (sortie de analysis.table3_full)."""
    _save(panel_df, name, float_fmt="%.3f")
    return panel_df


def table4_save(within: pd.DataFrame, across: dict):
    """Table 4 : Panel A (intra-classe) + Panel B (matrices inter-classes)."""
    _save(within, "table4_panelA_within_class", float_fmt="%.3f")
    for key, mat in across.items():
        _save(mat, f"table4_panelB_across_{key.lower().replace(' ', '_')}",
              float_fmt="%.3f")
    return within


def table5_save(panelA: pd.DataFrame, panelB: pd.DataFrame,
                panelC: pd.DataFrame | None = None):
    """Table 5 : A (TSMOM~XSMOM), B (décomposition), C (ce que TSMOM explique)."""
    _save(panelA, "table5_panelA_tsmom_on_xsmom", float_fmt="%.3f")
    _save(panelB, "table5_panelB_decomposition", float_fmt="%.5f")
    if panelC is not None and len(panelC):
        _save(panelC, "table5_panelC_what_tsmom_explains", float_fmt="%.3f")
    return panelA


def table6_save(df: pd.DataFrame):
    """Table 6 : prédicteurs spot/roll/positions."""
    _save(df, "table6_predictors", float_fmt="%.4f")
    return df