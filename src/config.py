"""
Configuration: constants, asset mappings, and parameters
following Moskowitz, Hua Ooi, Pedersen (2012) "Time Series Momentum".
"""

from pathlib import Path

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path("/mnt/project/data.xlsx")
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

# ---------- Sample periods ----------
# Paper sample (Moskowitz et al. 2012): Jan 1985 - Dec 2009
PAPER_START = "1985-01-01"
PAPER_END = "2009-12-31"
# Extended out-of-sample: through Dec 2025
EXTENDED_END = "2025-12-31"

# ---------- TSMOM parameters (from paper) ----------
LOOKBACK_MONTHS = 12      # k=12: 12-month past return defines the sign
HOLDING_MONTHS = 1        # h=1: 1-month holding period
TARGET_VOL = 0.40         # 40% target annual vol per position (paper §4.1)

# EWMA volatility: center of mass = 60 days ⇒ delta = 60/61
EWMA_COM_DAYS = 60
EWMA_DELTA = EWMA_COM_DAYS / (EWMA_COM_DAYS + 1)
ANNUALIZATION_DAYS = 261  # paper Eq. (1): scalar 261 for annualized variance

# ---------- Asset class mapping (Bloomberg ticker → asset class) ----------
# Futures we use to construct TSMOM, by asset class.
# We deliberately pick *front-month* (M1) commodity contracts and the
# main futures for equity/bond/currency, in line with the paper.

EQUITY_FUTURES = {
    "XP1 Index": "ASX SPI 200 (AUS)",
    "CF1 Index": "CAC 40 (FR)",
    "GX1 Index": "DAX (GER)",
    "ST1 Index": "FTSE/MIB (IT)",
    "TP1 Index": "TOPIX (JP)",
    "EO1 Index": "AEX (NL)",
    "IB1 Index": "IBEX 35 (ES)",
    "Z 1 Index": "FTSE 100 (UK)",
    "SP1 Index": "S&P 500 (US)",
}

BOND_FUTURES = {
    "YM1 Comdty": "3Y AUS",
    "XM1 Comdty": "10Y AUS",
    "DU1 Comdty": "2Y EURO (Schatz)",
    "OE1 Comdty": "5Y EURO (Bobl)",
    "RX1 Comdty": "10Y EURO (Bund)",
    "UB1 Comdty": "30Y EURO (Buxl)",
    "CN1 Comdty": "10Y CAN",
    "JB1 Comdty": "10Y JP",
    "G 1 Comdty": "10Y UK (Gilt)",
    "TU1 Comdty": "2Y US",
    "FV1 Comdty": "5Y US",
    "TY1 Comdty": "10Y US",
    "US1 Comdty": "30Y US",
}

# Commodity front-month (M1) futures
COMMODITY_FUTURES = {
    "LMAHDS03 Comdty": "Aluminum",
    "LMCADS03 Comdty": "Copper",
    "LMNIDS03 Comdty": "Nickel",
    "LMZSDS03 Comdty": "Zinc",
    "CO1 Comdty": "Brent Crude",
    "QS1 Comdty": "Gas Oil",
    "CT1 Comdty": "Cotton",
    "KC1 Comdty": "Coffee",
    "CC1 Comdty": "Cocoa",
    "SB1 Comdty": "Sugar",
    "LC1 Comdty": "Live Cattle",
    "LH1 Comdty": "Lean Hogs",
    "C 1 Comdty": "Corn",
    "S 1 Comdty": "Soybeans",
    "SM1 Comdty": "Soy Meal",
    "BO1 Comdty": "Soy Oil",
    "W 1 Comdty": "Wheat",
    "CL1 Comdty": "WTI Crude",
    "HO1 Comdty": "Heating Oil",
    "NG1 Comdty": "Natural Gas",
    "GC1 Comdty": "Gold",
    "SI1 Comdty": "Silver",
    "PL1 Comdty": "Platinum",
    # Unleaded Gasoline spliced with RBOB
    "HU1 Comdty": "Unleaded Gasoline",
    "XB1 Comdty": "RBOB Gasoline",
}

# Currency forwards: spot + interest-rate carry
# spot ticker → (foreign 1M-rate ticker, "is USDXXX" flag i.e. quoted as USD-per-FCY needs inversion)
# AUDUSD/EURUSD/NZDUSD/GBPUSD: FCY per USD = NO (i.e. they're USD per FCY: AUD=>USD price of 1 AUD)
#   Wait — AUDUSD is "USD per AUD" → going long AUD vs USD means betting AUDUSD goes UP.
# USDCAD/USDJPY/USDNOK/USDSEK/USDCHF: quoted as "FCY per USD" — long FCY means USDXXX goes DOWN
# DEMUSD: historical Deutsche Mark / USD — quoted as USD per DEM (long DEM means it goes UP)
CURRENCY_FORWARDS = {
    "AUDUSD Curncy":  {"name": "AUD/USD",  "rate_fcy": "AU0001M Index", "invert": False},
    "USDCAD Curncy":  {"name": "CAD/USD",  "rate_fcy": "CD0001M Index", "invert": True},
    "EURUSD Curncy":  {"name": "EUR/USD",  "rate_fcy": "EU0001M Index", "invert": False},
    "DEMUSD Curncy":  {"name": "DEM/USD",  "rate_fcy": "DM0001M Index", "invert": False},
    "USDJPY Curncy":  {"name": "JPY/USD",  "rate_fcy": "JY0001M Index", "invert": True},
    "NZDUSD Curncy":  {"name": "NZD/USD",  "rate_fcy": "NZ0001M Index", "invert": False},
    "USDNOK Curncy":  {"name": "NOK/USD",  "rate_fcy": "NIBOR1M Index", "invert": True},
    "USDSEK Curncy":  {"name": "SEK/USD",  "rate_fcy": "SK0001M Index", "invert": True},
    "USDCHF Curncy":  {"name": "CHF/USD",  "rate_fcy": "SF0001M Index", "invert": True},
    "GBPUSD Curncy":  {"name": "GBP/USD",  "rate_fcy": "BP0001M Index", "invert": False},
}

USD_RATE = "US0001M Index"  # USD 1M rate (used as the financing leg for FX)

# ---------- Benchmarks / risk factors ----------
BENCHMARKS = {
    "MXWO Index":     "MSCI World",       # MKT proxy
    "SPGSCI Index":   "S&P GSCI",         # commodity benchmark
    "LBUSTRUU Index": "Barclays Agg Bond",  # bond benchmark
}

# Pretty names for asset-class colors / labels
ASSET_CLASS_COLORS = {
    "Commodity":   "#8B4513",
    "Equity":      "#1f77b4",
    "Bond":        "#2ca02c",
    "Currency":    "#d62728",
}


def asset_class_of(ticker: str) -> str:
    if ticker in EQUITY_FUTURES:
        return "Equity"
    if ticker in BOND_FUTURES:
        return "Bond"
    if ticker in COMMODITY_FUTURES:
        return "Commodity"
    if ticker in CURRENCY_FORWARDS:
        return "Currency"
    return "Other"


def pretty_name(ticker: str) -> str:
    for d in (EQUITY_FUTURES, BOND_FUTURES, COMMODITY_FUTURES):
        if ticker in d:
            return d[ticker]
    if ticker in CURRENCY_FORWARDS:
        return CURRENCY_FORWARDS[ticker]["name"]
    return ticker


def all_instruments() -> list:
    """All instrument tickers used to build TSMOM (excluding Unleaded/RBOB splice)."""
    return (
        list(EQUITY_FUTURES.keys())
        + list(BOND_FUTURES.keys())
        + list(COMMODITY_FUTURES.keys())
        + list(CURRENCY_FORWARDS.keys())
    )
