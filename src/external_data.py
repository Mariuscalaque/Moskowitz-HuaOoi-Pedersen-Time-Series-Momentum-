"""
external_data.py — Récupération AUTOMATIQUE des données externes manquantes
pour compléter la réplication de Moskowitz, Hua Ooi & Pedersen (2012),
« Time Series Momentum ».

Sources couvertes (faisables sans friction) :
  1. AQR Data Library
        - Facteurs « Value & Momentum Everywhere » (Asness-Moskowitz-Pedersen)
          -> Table 3, Panel B
        - Série officielle des rendements TSMOM du papier (validation/benchmark)
  2. Pástor-Stambaugh — facteur de liquidité agrégée  -> Table 3, Panel C
  3. Baker-Wurgler — indice de sentiment investisseur  -> Table 3, Panel C
  4. CFTC — Commitments of Traders (Legacy, futures-only), positions
     spéculateurs/hedgers  -> Table 6, Figures 5, 6B, 7

Rappels :
  - VIX et le spread TED (extrêmes de marché/volatilité du Panel C) sont DÉJÀ
    dans data.xlsx (colonnes `VIX Index` et `.TEDSP Index`) : pas besoin de les
    télécharger ici.
  - Les indices hedge funds Credit Suisse (Table 5, Panel C) ne sont PAS inclus
    car leur accès demande une inscription depuis la fusion CSAM->UBS (2024) ;
    voir la note en fin de fichier pour les substituts (SG Trend, Barclay CTA…).

------------------------------------------------------------------------------
SYSTÈME DE CACHE (la « trace » demandée)
------------------------------------------------------------------------------
À chaque téléchargement réussi, on écrit dans `data/external/` :
  - raw/<nom_fichier_original>   : le fichier brut tel que téléchargé (archive)
  - <nom>.csv                    : la version nettoyée, prête à l'emploi
  - _manifest.json               : journal {nom: {url, downloaded_at, rows,
                                                  start, end, raw_file, cache}}
Aux appels suivants, si le cache existe et `force_refresh=False`, on relit le
CSV local — donc le code fonctionne ENSUITE même sans connexion internet.

Chaque fonction `fetch_*` accepte :
  - force_refresh : bool   -> re-télécharge même si le cache existe
  - url           : str    -> surcharge l'URL par défaut (utile si AQR/Wharton
                              changent le nom du fichier annuel)
  - local_file    : str    -> ingère un fichier téléchargé à la main (repli
                              hors-ligne, comme `source='csv'` dans factors.py)
"""

from __future__ import annotations

import io
import json
import os
import re
import zipfile
import contextlib
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# --- Chemin du dossier data (réutilise la config du projet si présente) -------
try:                                   # exécution comme module du package src/
    from .config import DATA_PATH
except Exception:                      # exécution directe / hors package
    try:
        from config import DATA_PATH   # type: ignore
    except Exception:
        DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "data.xlsx"

DATA_DIR = Path(DATA_PATH).resolve().parent          # .../data
EXTERNAL_DIR = DATA_DIR / "external"                 # .../data/external
RAW_DIR = EXTERNAL_DIR / "raw"                        # .../data/external/raw
MANIFEST_PATH = EXTERNAL_DIR / "_manifest.json"

# --- URLs par défaut (surchargeables ; AQR/Wharton changent le millésime) -----
# AQR : page « Value and Momentum Everywhere: Factors, Monthly »
AQR_VME_URL = ("https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/"
               "Value-and-Momentum-Everywhere-Factors-Monthly.xlsx")
# AQR : page « Time Series Momentum: Factors, Monthly » (rendements du papier)
AQR_TSMOM_URL = ("https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/"
                 "Time-Series-Momentum-Factors-Monthly.xlsx")
# Pástor-Stambaugh : page perso de R. Stambaugh (le millésime de fin évolue)
PS_LIQ_URL = "https://finance.wharton.upenn.edu/~stambaug/liq_data_1962_2023.txt"
# Baker-Wurgler : page perso de J. Wurgler (NYU Stern) — fichier Excel mensuel
BW_SENT_URL = ("https://pages.stern.nyu.edu/~jwurgler/data/"
               "Investor_Sentiment_Data_20190327_POST.xlsx")
# CFTC : fichiers annuels « Legacy futures-only » (un .zip par année)
CFTC_LEGACY_FUT_URL = "https://www.cftc.gov/files/dea/history/deacot{year}.zip"


# ============================================================================
# Infrastructure de cache
# ============================================================================
def _ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _write_manifest_entry(name: str, entry: dict) -> None:
    _ensure_dirs()
    man = _read_manifest()
    man[name] = entry
    MANIFEST_PATH.write_text(json.dumps(man, indent=2, ensure_ascii=False),
                             encoding="utf-8")


def _cache_csv_path(name: str) -> Path:
    return EXTERNAL_DIR / f"{name}.csv"


def _save_cache(df: pd.DataFrame, name: str, source_url: str,
                raw_bytes: bytes | None = None,
                raw_filename: str | None = None) -> None:
    """Sauvegarde le DataFrame nettoyé (+ éventuellement le fichier brut) et
    journalise l'opération dans le manifeste."""
    _ensure_dirs()
    csv_path = _cache_csv_path(name)
    df.to_csv(csv_path)

    raw_rel = None
    if raw_bytes is not None and raw_filename:
        raw_path = RAW_DIR / raw_filename
        raw_path.write_bytes(raw_bytes)
        raw_rel = str(raw_path.relative_to(DATA_DIR))

    try:
        start = str(df.index.min())
        end = str(df.index.max())
    except Exception:
        start = end = None

    _write_manifest_entry(name, {
        "source_url": source_url,
        "downloaded_at": _now_iso(),
        "rows": int(len(df)),
        "columns": list(map(str, df.columns)),
        "start": start,
        "end": end,
        "cache_csv": str(csv_path.relative_to(DATA_DIR)),
        "raw_file": raw_rel,
    })


def _load_cache(name: str) -> pd.DataFrame | None:
    csv_path = _cache_csv_path(name)
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, index_col=0)
    # ré-essaie de parser l'index en dates (la majorité de nos séries)
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    return df


def _download_bytes(url: str, timeout: int = 60) -> bytes:
    """Téléchargement binaire générique avec un User-Agent (sinon 403 fréquent)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _cache_or_download(name: str, url: str, force_refresh: bool,
                       parser, raw_filename: str,
                       local_file: str | None = None) -> pd.DataFrame:
    """
    Logique commune :
      1) cache présent et pas de force_refresh  -> relit le CSV local
      2) local_file fourni                       -> ingère le fichier manuel
      3) sinon                                    -> télécharge `url`
    `parser(raw_bytes) -> DataFrame` transforme le brut en série propre.
    """
    if not force_refresh and local_file is None:
        cached = _load_cache(name)
        if cached is not None:
            return cached

    if local_file is not None:
        raw = Path(local_file).read_bytes()
        src = f"local:{local_file}"
    else:
        raw = _download_bytes(url)
        src = url

    df = parser(raw)
    _save_cache(df, name, source_url=src, raw_bytes=raw, raw_filename=raw_filename)
    return df


# ============================================================================
# 1. AQR — facteurs Value & Momentum Everywhere + série TSMOM du papier
# ============================================================================
def _parse_aqr_excel(raw: bytes, header_tokens=("date",)) -> pd.DataFrame:
    """
    Parse un fichier Excel AQR : ~10-20 lignes de préambule (disclaimers),
    puis une ligne d'en-tête (« Date » + noms de facteurs), puis le bloc mensuel.
    On détecte dynamiquement la ligne d'en-tête (= juste avant la 1re ligne dont
    la 1re cellule est une date), pour être robuste aux changements de mise en page.
    """
    bio = io.BytesIO(raw)
    # 1er onglet de données (souvent index 0), tout en brut
    raw_df = pd.read_excel(bio, sheet_name=0, header=None)

    header_row = None
    for r in range(min(40, len(raw_df))):
        c0 = raw_df.iat[r, 0]
        if isinstance(c0, str) and any(tok in c0.strip().lower()
                                       for tok in header_tokens):
            header_row = r
            break
    if header_row is None:
        # repli : 1re ligne dont la cellule 0 se parse en date -> en-tête = r-1
        for r in range(1, min(40, len(raw_df))):
            try:
                pd.to_datetime(raw_df.iat[r, 0])
                header_row = r - 1
                break
            except Exception:
                continue
    if header_row is None:
        raise ValueError("Impossible de localiser l'en-tête dans le fichier AQR.")

    cols = [str(x).strip() for x in raw_df.iloc[header_row].tolist()]
    data = raw_df.iloc[header_row + 1:].copy()
    data.columns = cols
    data = data.rename(columns={cols[0]: "date"})
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).set_index("date").sort_index()
    # colonnes en numérique
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna(axis=1, how="all")
    data.index = data.index + pd.offsets.MonthEnd(0)  # caler en fin de mois
    data.index.name = "date"
    return data


def fetch_aqr_vme(force_refresh: bool = False, url: str = AQR_VME_URL,
                  local_file: str | None = None) -> pd.DataFrame:
    """Facteurs Value & Momentum Everywhere (mensuel) -> Table 3 Panel B.

    Colonnes typiques : VAL^... / MOM^... par classe + facteurs agrégés
    (VAL Everywhere, MOM Everywhere). Voir l'onglet descriptif du fichier AQR.
    """
    return _cache_or_download(
        "aqr_vme_factors", url, force_refresh,
        parser=lambda raw: _parse_aqr_excel(raw, header_tokens=("date",)),
        raw_filename="aqr_vme_factors_monthly.xlsx", local_file=local_file,
    )


def fetch_aqr_tsmom(force_refresh: bool = False, url: str = AQR_TSMOM_URL,
                    local_file: str | None = None) -> pd.DataFrame:
    """Rendements TSMOM officiels du papier (mensuel) — pour VALIDER ta
    réplication et investiguer l'écart d'alpha de la Table 3."""
    return _cache_or_download(
        "aqr_tsmom_factors", url, force_refresh,
        parser=lambda raw: _parse_aqr_excel(raw, header_tokens=("date",)),
        raw_filename="aqr_tsmom_factors_monthly.xlsx", local_file=local_file,
    )


# ============================================================================
# 2. Pástor-Stambaugh — liquidité agrégée (-> Table 3 Panel C)
# ============================================================================
def _parse_ps_liquidity(raw: bytes) -> pd.DataFrame:
    """
    Le fichier texte de Stambaugh est délimité par des espaces. On garde les
    lignes commençant par un mois YYYYMM, puis on nomme les colonnes :
      agg_liq    = niveau de liquidité agrégée
      innov_liq  = innovations (scaled) de liquidité  <- celle utilisée en facteur d'état
      traded_liq = facteur de liquidité « tradable » (LIQ_V)
    Valeurs -99 -> NaN.
    """
    text = raw.decode("latin-1", errors="ignore")
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if re.fullmatch(r"\d{6}", parts[0]):                 # YYYYMM
            nums = []
            for p in parts[1:]:
                try:
                    nums.append(float(p))
                except ValueError:
                    nums.append(np.nan)
            rows.append([parts[0]] + nums)
    if not rows:
        raise ValueError("Aucune ligne mensuelle YYYYMM trouvée (format PS inattendu).")

    width = max(len(r) for r in rows)
    names = ["yyyymm", "agg_liq", "innov_liq", "traded_liq"][:width]
    # complète les noms si le fichier a plus de colonnes que prévu
    names += [f"col{i}" for i in range(len(names), width)]
    df = pd.DataFrame([r + [np.nan] * (width - len(r)) for r in rows], columns=names)
    df = df.replace(-99, np.nan)
    df.index = pd.to_datetime(df["yyyymm"], format="%Y%m") + pd.offsets.MonthEnd(0)
    df.index.name = "date"
    return df.drop(columns=["yyyymm"]).apply(pd.to_numeric, errors="coerce")


def fetch_ps_liquidity(force_refresh: bool = False, url: str = PS_LIQ_URL,
                       local_file: str | None = None) -> pd.DataFrame:
    """Série de liquidité Pástor-Stambaugh (mensuel) -> Table 3 Panel C.

    Note : le nom du fichier porte l'année de fin (…1962_2023.txt). Si l'URL par
    défaut renvoie une 404, passe l'URL à jour via `url=...` ou télécharge le
    fichier à la main et utilise `local_file=...`.
    """
    return _cache_or_download(
        "pastor_stambaugh_liquidity", url, force_refresh,
        parser=_parse_ps_liquidity,
        raw_filename="pastor_stambaugh_liquidity.txt", local_file=local_file,
    )


# ============================================================================
# 3. Baker-Wurgler — sentiment investisseur (-> Table 3 Panel C)
# ============================================================================
def _parse_bw_sentiment(raw: bytes) -> pd.DataFrame:
    """
    Le classeur Baker-Wurgler contient un onglet de données mensuelles avec une
    colonne de date (YYYYMM) et les indices SENT et SENT⊥ (orthogonalisé).
    On repère l'onglet « DATA » si présent, sinon le premier, et on détecte la
    colonne de dates dynamiquement.
    """
    bio = io.BytesIO(raw)
    xls = pd.ExcelFile(bio)
    sheet = next((s for s in xls.sheet_names if "data" in s.lower()),
                 xls.sheet_names[0])
    raw_df = pd.read_excel(xls, sheet_name=sheet, header=None)

    # ligne d'en-tête = première dont une cellule contient « yearmo »/« date »/« sent »
    header_row = None
    for r in range(min(20, len(raw_df))):
        joined = " ".join(str(x).lower() for x in raw_df.iloc[r].tolist())
        if any(tok in joined for tok in ("yearmo", "date", "sent")):
            header_row = r
            break
    if header_row is None:
        header_row = 0

    cols = [str(x).strip() for x in raw_df.iloc[header_row].tolist()]
    data = raw_df.iloc[header_row + 1:].copy()
    data.columns = cols

    # repère la colonne de date (YYYYMM) : la 1re colonne dont >80% des valeurs
    # ressemblent à 6 chiffres
    date_col = None
    for c in data.columns:
        ser = data[c].astype(str).str.strip()
        frac = ser.str.fullmatch(r"\d{6}(\.0)?").mean()
        if frac > 0.8:
            date_col = c
            break
    if date_col is None:
        date_col = data.columns[0]

    idx = (data[date_col].astype(str).str.replace(".0", "", regex=False)
           .str.strip())
    out = data.drop(columns=[date_col]).apply(pd.to_numeric, errors="coerce")
    out.index = pd.to_datetime(idx, format="%Y%m", errors="coerce") + pd.offsets.MonthEnd(0)
    out.index.name = "date"
    out = out.dropna(how="all").loc[out.index.notna()]
    return out


def fetch_bw_sentiment(force_refresh: bool = False, url: str = BW_SENT_URL,
                       local_file: str | None = None) -> pd.DataFrame:
    """Indice de sentiment Baker-Wurgler (mensuel) -> Table 3 Panel C.

    Note : le nom du fichier porte une date de mise à jour. Si l'URL par défaut
    échoue, récupère l'URL courante sur la page de J. Wurgler (NYU Stern) et
    passe-la via `url=...`, ou utilise `local_file=...`.
    """
    return _cache_or_download(
        "baker_wurgler_sentiment", url, force_refresh,
        parser=_parse_bw_sentiment,
        raw_filename="baker_wurgler_sentiment.xlsx", local_file=local_file,
    )


# ============================================================================
# 4. CFTC — Commitments of Traders (Legacy, futures-only)
# ============================================================================
# Correspondance ticker Bloomberg -> sous-chaîne du nom de marché CFTC.
# UNIQUEMENT le sous-univers listé aux US (la CFTC ne couvre pas DAX, CAC,
# TOPIX, Gilt, Bund, LME, NOK/SEK…). À compléter/ajuster selon tes besoins.
DEFAULT_COT_MARKETS = {
    # Equity (US)
    "SP1 Index": "S&P 500",
    # Bonds (US)
    "TU1 Comdty": "2-YEAR U.S. TREASURY",
    "FV1 Comdty": "5-YEAR U.S. TREASURY",
    "TY1 Comdty": "10-YEAR U.S. TREASURY",
    "US1 Comdty": "U.S. TREASURY BONDS",
    # Energy
    "CL1 Comdty": "CRUDE OIL, LIGHT SWEET",
    "HO1 Comdty": "HEATING OIL",
    "NG1 Comdty": "NATURAL GAS",
    "HU1 Comdty": "GASOLINE",
    "XB1 Comdty": "GASOLINE RBOB",
    # Metals (COMEX)
    "GC1 Comdty": "GOLD",
    "SI1 Comdty": "SILVER",
    "PL1 Comdty": "PLATINUM",
    # Agriculture
    "C 1 Comdty": "CORN",
    "S 1 Comdty": "SOYBEANS",
    "SM1 Comdty": "SOYBEAN MEAL",
    "BO1 Comdty": "SOYBEAN OIL",
    "W 1 Comdty": "WHEAT",
    "CT1 Comdty": "COTTON",
    "KC1 Comdty": "COFFEE",
    "CC1 Comdty": "COCOA",
    "SB1 Comdty": "SUGAR",
    "LC1 Comdty": "LIVE CATTLE",
    "LH1 Comdty": "LEAN HOGS",
    # FX
    "AUDUSD Curncy": "AUSTRALIAN DOLLAR",
    "USDCAD Curncy": "CANADIAN DOLLAR",
    "EURUSD Curncy": "EURO FX",
    "USDJPY Curncy": "JAPANESE YEN",
    "NZDUSD Curncy": "NEW ZEALAND DOLLAR",
    "USDCHF Curncy": "SWISS FRANC",
    "GBPUSD Curncy": "BRITISH POUND",
}

# motifs de colonnes du rapport Legacy (les libellés varient un peu selon la source)
_RE_NC_LONG = re.compile(r"noncommercial.*long", re.I)
_RE_NC_SHORT = re.compile(r"noncommercial.*short", re.I)
_RE_COM_LONG = re.compile(r"(?<!non)commercial.*long", re.I)
_RE_COM_SHORT = re.compile(r"(?<!non)commercial.*short", re.I)
_RE_OI = re.compile(r"open.?interest", re.I)
_RE_MARKET = re.compile(r"market.*exchange|market.?and", re.I)
_RE_DATE = re.compile(r"as.?of.?date|report.?date|date", re.I)


def _find_col(columns, regex):
    for c in columns:
        if regex.search(str(c)):
            return c
    return None


def _fetch_cot_raw_via_library(years) -> pd.DataFrame:
    """Source A : librairie `cot_reports` (pip install cot-reports).
    On silencie son affichage par année et on redirige les fichiers bruts
    (annual.txt) qu'elle dépose vers data/external/raw, pas la racine du projet."""
    import cot_reports as cot  # type: ignore
    _ensure_dirs()
    frames = []
    cwd0 = os.getcwd()
    try:
        os.chdir(RAW_DIR)  # les annual.txt déposés par la lib vont dans data/external/raw
        with contextlib.redirect_stdout(io.StringIO()):
            for y in years:
                frames.append(cot.cot_year(year=y, cot_report_type="legacy_fut"))
    finally:
        os.chdir(cwd0)
    return pd.concat(frames, ignore_index=True)


def _fetch_cot_raw_via_direct(years) -> pd.DataFrame:
    """Source B : téléchargement direct des .zip annuels de la CFTC."""
    frames = []
    for y in years:
        raw = _download_bytes(CFTC_LEGACY_FUT_URL.format(year=y))
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            inner = zf.namelist()[0]
            with zf.open(inner) as fh:
                frames.append(pd.read_csv(fh, low_memory=False))
    return pd.concat(frames, ignore_index=True)


def _find_date_col(columns):
    """Colonne de date la plus fiable d'un rapport CFTC.
    Préfère un format AAAA-MM-JJ explicite ('yyyy' dans le nom), sinon retombe
    sur la 1re colonne contenant 'date' (souvent 'As of Date in Form YYMMDD')."""
    cols = list(columns)
    for c in cols:
        cl = str(c).lower()
        if "date" in cl and ("yyyy" in cl or "%y-%m-%d" in cl):
            return c
    return _find_col(cols, _RE_DATE)


def _parse_cot_dates(s: pd.Series) -> pd.Series:
    """Parse des dates CFTC robuste : essaie le format standard, puis YYMMDD."""
    d = pd.to_datetime(s, errors="coerce")
    if d.notna().mean() < 0.5:  # échec massif -> tenter YYMMDD (ex. '860930')
        d = pd.to_datetime(s.astype(str).str.zfill(6), format="%y%m%d",
                           errors="coerce")
    return d


def _net_speculator_positions(raw: pd.DataFrame,
                              markets: dict) -> pd.DataFrame:
    """
    À partir du rapport Legacy brut, calcule pour chaque marché ciblé la
    position nette des spéculateurs en % de l'open interest, en mensuel :

        net_spec_pct = (NC_long - NC_short) / open_interest

    (NC = non-commercial = spéculateurs ; commercial = hedgers.)
    Renvoie un DataFrame indexé fin de mois, une colonne par ticker Bloomberg.
    """
    cols = raw.columns
    c_mkt = _find_col(cols, _RE_MARKET)
    c_date = _find_date_col(cols)
    c_ncl = _find_col(cols, _RE_NC_LONG)
    c_ncs = _find_col(cols, _RE_NC_SHORT)
    c_oi = _find_col(cols, _RE_OI)
    missing = [n for n, c in [("market", c_mkt), ("date", c_date),
                              ("NC long", c_ncl), ("NC short", c_ncs),
                              ("open interest", c_oi)] if c is None]
    if missing:
        raise ValueError(f"Colonnes CFTC introuvables : {missing}. "
                         f"Colonnes dispo : {list(cols)[:12]}…")

    work = raw[[c_mkt, c_date, c_ncl, c_ncs, c_oi]].copy()
    work.columns = ["market", "date", "nc_long", "nc_short", "oi"]
    work["date"] = _parse_cot_dates(work["date"])
    for col in ("nc_long", "nc_short", "oi"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["date", "market"])
    work["market_u"] = work["market"].astype(str).str.upper()

    out = {}
    for ticker, needle in markets.items():
        sel = work[work["market_u"].str.contains(re.escape(needle.upper()),
                                                 na=False)]
        if sel.empty:
            continue
        # s'il reste plusieurs marchés (ex. plusieurs bourses), garde le plus liquide
        if sel["market_u"].nunique() > 1:
            best = sel.groupby("market_u")["oi"].mean().idxmax()
            sel = sel[sel["market_u"] == best]
        net_pct = ((sel["nc_long"] - sel["nc_short"]) / sel["oi"])
        s = pd.Series(net_pct.values, index=sel["date"]).sort_index()
        # hebdo (mardi) -> mensuel : dernière valeur du mois
        out[ticker] = s.resample("ME").last()

    if not out:
        raise ValueError("Aucun marché ciblé n'a été trouvé dans le rapport CFTC. "
                         "Vérifie le dictionnaire `markets`.")
    return pd.DataFrame(out).sort_index()


def fetch_cftc_cot(years=range(1986, 2010),
                   markets: dict | None = None,
                   force_refresh: bool = False,
                   source: str = "auto") -> pd.DataFrame:
    """
    Positions nettes des spéculateurs (% open interest), mensuelles, par marché
    -> Table 6, Figures 5, 6B, 7.

    Parameters
    ----------
    years   : itérable d'années (Legacy futures-only dispo depuis 1986).
    markets : dict {ticker Bloomberg -> sous-chaîne nom marché CFTC}.
              Par défaut, le sous-univers US (`DEFAULT_COT_MARKETS`).
    source  : 'auto' (librairie puis direct), 'library', ou 'direct'.

    Le brut concaténé est aussi sauvegardé pour archive.
    """
    markets = markets or DEFAULT_COT_MARKETS
    name = "cftc_net_spec_positions"

    if not force_refresh:
        cached = _load_cache(name)
        # Rejette un cache DÉGÉNÉRÉ (ex. stub d'une ligne issu d'un mauvais
        # fichier) : sous 24 mois, ce n'est pas une vraie série -> on ignore
        # et on re-télécharge, plutôt que de propager des données trompeuses.
        if cached is not None and len(cached.dropna(how="all")) >= 24:
            return cached
        if cached is not None:
            print(f"[CFTC] cache ignoré : seulement {len(cached.dropna(how='all'))} "
                  f"ligne(s) valide(s) (< 24) -> re-téléchargement.")

    years = list(years)
    if source in ("auto", "library"):
        try:
            raw = _fetch_cot_raw_via_library(years)
        except Exception as e:
            if source == "library":
                raise
            raw = _fetch_cot_raw_via_direct(years)
    elif source == "direct":
        raw = _fetch_cot_raw_via_direct(years)
    else:
        raise ValueError(f"source inconnue : {source}")

    # archive du brut (compressé) + table dérivée
    _ensure_dirs()
    raw_csv = RAW_DIR / "cftc_legacy_fut_raw.csv.gz"
    raw.to_csv(raw_csv, index=False, compression="gzip")

    net = _net_speculator_positions(raw, markets)
    _save_cache(net, name,
                source_url=f"CFTC legacy_fut {years[0]}-{years[-1]} ({source})")
    # complète le manifeste avec l'archive du brut
    man = _read_manifest()
    if name in man:
        man[name]["raw_file"] = str(raw_csv.relative_to(DATA_DIR))
        man[name]["markets_found"] = list(net.columns)
        MANIFEST_PATH.write_text(json.dumps(man, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    return net


# ============================================================================
# Orchestrateur
# ============================================================================
def fetch_all(force_refresh: bool = False, cot_years=range(1986, 2010)) -> dict:
    """
    Tente de récupérer toutes les sources faisables. Ne s'arrête pas à la
    première erreur : renvoie un dict {nom: DataFrame} pour les succès et
    affiche un récapitulatif des échecs (URL périmée, lib manquante…).
    """
    jobs = {
        "aqr_vme": lambda: fetch_aqr_vme(force_refresh=force_refresh),
        "aqr_tsmom": lambda: fetch_aqr_tsmom(force_refresh=force_refresh),
        "pastor_stambaugh": lambda: fetch_ps_liquidity(force_refresh=force_refresh),
        "baker_wurgler": lambda: fetch_bw_sentiment(force_refresh=force_refresh),
        "cftc_cot": lambda: fetch_cftc_cot(years=cot_years,
                                           force_refresh=force_refresh),
    }
    results, errors = {}, {}
    for key, fn in jobs.items():
        try:
            df = fn()
            results[key] = df
            print(f"[OK]   {key:18s} -> {df.shape[0]:>4d} lignes, "
                  f"{df.shape[1]} col., {df.index.min()} → {df.index.max()}")
        except Exception as e:
            errors[key] = str(e)
            print(f"[SKIP] {key:18s} -> {str(e)[:90]}")

    if errors:
        print("\nÉchecs (le plus souvent : URL annuelle périmée ou lib absente) :")
        for k, v in errors.items():
            print(f"   - {k}: {v[:120]}")
        print("Pistes : passe une `url=...` à jour, fournis `local_file=...`,")
        print("ou installe la librairie CFTC :  pip install cot-reports")

    print(f"\nCache & trace dans : {EXTERNAL_DIR}")
    print(f"Manifeste         : {MANIFEST_PATH}")
    return results


# ----------------------------------------------------------------------------
# NOTE — Indices hedge funds (Table 5, Panel C), NON automatisés ici
# ----------------------------------------------------------------------------
# Les indices Dow Jones/Credit Suisse (dont « Managed Futures ») demandent une
# inscription gratuite depuis la fusion CSAM->UBS (mai 2024). Récupère-les sur
# lab.credit-suisse.com, OU utilise un substitut de trend-following gratuit
# (SG Trend / SG CTA Index, Barclay/BarclayHedge CTA Index, HFR). Une fois le
# CSV en main :  pd.read_csv(...) puis aligne en fin de mois sur ton TSMOM.

if __name__ == "__main__":
    print(f"Dossier data détecté : {DATA_DIR}")
    print("Lancement de fetch_all() — nécessite une connexion internet.\n")
    fetch_all()
