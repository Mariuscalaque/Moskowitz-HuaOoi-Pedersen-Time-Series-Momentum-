"""
Configuration: constants, asset mappings, and parameters
following Moskowitz, Hua Ooi, Pedersen (2012) "Time Series Momentum".
"""

import os
from pathlib import Path

# ---------- Paths ----------
# PROJECT_ROOT = dossier qui contient src/, data/, outputs/ (résolu automatiquement,
# fonctionne quel que soit l'OS et l'emplacement du projet).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Chemin RELATIF au projet : data/data.xlsx
# (surchargeable via la variable d'environnement TSMOM_DATA si besoin)
DATA_PATH = Path(os.environ.get("TSMOM_DATA", PROJECT_ROOT / "data" / "data.xlsx"))
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Dossier des données (racine pour data.xlsx et les CSV externes mis en cache)
DATA_DIR = Path(os.environ.get("TSMOM_DATA_DIR", PROJECT_ROOT / "data"))
EXTERNAL_DIR = DATA_DIR / "external"

# Repli hors-ligne des facteurs Fama-French : si ces CSV (téléchargés à la main
# depuis le site de Ken French) existent, factors.py les lit AVANT toute tentative
# de téléchargement. Le téléchargement réussi est aussi mis en cache ici.
FF_3FACTOR_CSV = EXTERNAL_DIR / "F-F_Research_Data_Factors.csv"
FF_MOMENTUM_CSV = EXTERNAL_DIR / "F-F_Momentum_Factor.csv"

# Indices hedge funds Dow Jones / Credit Suisse (Table 5 Panel C). Données sous
# licence : à fournir manuellement (colonnes : date, ManagedFutures, GlobalMacro).
HEDGE_FUND_CSV = EXTERNAL_DIR / "djcs_hedge_fund_indices.csv"

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

# ---------- Devises : paires croisées vs vs-USD ----------
# Le papier utilise des « cross-currency pairs (from nine underlying currencies) ».
# Coter les 9 devises toutes contre l'USD injecte un facteur USD commun qui gonfle
# les corrélations FX (intra-classe 0.37/0.54 vs 0.10/0.04 dans le papier) et
# abaisse la corrélation avec le facteur AQR FX (0.56). Construire les C(9,2)=36
# paires croisées (la jambe USD s'annule) remonte la corr AQR FX à ~0.67 et
# rapproche les corrélations intra-classe de celles du papier.
#   True  -> 36 paires croisées (DÉFAUT, plus fidèle au papier ; corr AQR FX
#            0.55->0.67, Table 4 passive FX 0.54->0.05 ≈ 0.04 du papier)
#   False -> 10 paires vs-USD (ancien comportement ; corr AQR ALL 0.80 et
#            Sharpe 1.13 légèrement supérieurs car pas de sur-pondération FX)
# Compromis : en croisé, les 36 paires sur-pondèrent un peu la classe devises dans
# le facteur équipondéré par instrument (corr AQR ALL 0.80->0.76, Sharpe 1.13->1.03).
# NB : les positions CFTC ne couvrent que les devises vs-USD ; en mode croisé, la
# sleeve FX ne contribue pas aux analyses de positions (Fig 5/6B/7), ce qui est
# géré proprement (les autres classes restent couvertes).
FX_CROSS_PAIRS = True

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

# Futures obligataires cotés en « 100 − rendement » (convention australienne) :
# un pct_change sur la cote ne donne PAS un rendement obligataire. On convertit
# via r ≈ D · Δcote/100 (Δyield = −Δcote), avec la duration-cible de MOP App. A.2
# (2 ans pour le 3Y, 7 ans pour le 10Y). Le signe est identique au pct_change,
# donc le TSMOM (sign × 40%/σ × r) est inchangé ; seules les stats brutes
# (Table 1) sont corrigées.
YIELD_QUOTED_BONDS = {
    "YM1 Comdty": 2,   # 3Y AUS -> duration 2 ans
    "XM1 Comdty": 7,   # 10Y AUS -> duration 7 ans
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
    if ticker.endswith("Cross"):          # paire croisée FX, ex. "AUD/GBP Cross"
        return "Currency"
    return "Other"


def pretty_name(ticker: str) -> str:
    for d in (EQUITY_FUTURES, BOND_FUTURES, COMMODITY_FUTURES):
        if ticker in d:
            return d[ticker]
    if ticker in CURRENCY_FORWARDS:
        return CURRENCY_FORWARDS[ticker]["name"]
    if ticker.endswith("Cross"):          # "AUD/GBP Cross" -> "AUD/GBP"
        return ticker.replace(" Cross", "")
    return ticker


def all_instruments() -> list:
    """All instrument tickers used to build TSMOM (excluding Unleaded/RBOB splice)."""
    return (
        list(EQUITY_FUTURES.keys())
        + list(BOND_FUTURES.keys())
        + list(COMMODITY_FUTURES.keys())
        + list(CURRENCY_FORWARDS.keys())
    )