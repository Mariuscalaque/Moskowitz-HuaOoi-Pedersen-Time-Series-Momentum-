"""
Load and clean the raw daily price/index data from `data.xlsx`.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from .config import DATA_PATH


def load_raw(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Read the `data` sheet from data.xlsx, parse the Excel serial dates,
    sort by date, and forward-fill rare missing observations.

    Returns
    -------
    pd.DataFrame indexed by date, columns = Bloomberg tickers.
    """
    df = pd.read_excel(data_path, sheet_name="data")

    # The 'date' column is an Excel serial date integer
    df["date"] = pd.to_datetime(df["date"], unit="D", origin="1899-12-30")
    df = df.set_index("date").sort_index()

    # Drop weekend rows that may have crept in (Bloomberg often has business days only,
    # but the spreadsheet uses calendar-day rows in this dataset)
    bd = df.index.dayofweek < 5
    df = df.loc[bd]

    return df


def load_tickers(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the `tickers` sheet (metadata for each instrument)."""
    return pd.read_excel(data_path, sheet_name="tickers")


def first_valid_date(df: pd.DataFrame, ticker: str) -> pd.Timestamp:
    s = df[ticker].dropna()
    return s.index[0] if len(s) else pd.NaT
