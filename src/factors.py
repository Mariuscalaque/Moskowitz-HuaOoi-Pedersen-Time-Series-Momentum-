"""
factors.py — Facteurs Fama-French (SMB, HML, UMD) + RF pour la Table 3, et
chargeur des indices hedge funds pour la Table 5 Panel C.

CORRECTION (robustesse / hors-ligne) : l'ordre de résolution est désormais
   1) CSV LOCAUX (cache disque dans data/external/) s'ils existent  -> hors-ligne ;
   2) téléchargement direct des ZIP Dartmouth (mis EN CACHE après succès) ;
   3) pandas_datareader (si installé).
Ainsi, une fois les facteurs récupérés une première fois (ou déposés à la main),
toute la réplication tourne SANS connexion internet.

Le papier régresse le TSMOM diversifié sur (Eq. 4) :
    MKT  = excès du MSCI World     (MXWO de data.xlsx, en excès du RF)
    BOND = Barclays Aggregate Bond (LBUSTRUU, en excès)
    GSCI = S&P GSCI                (SPGSCI, en excès)
    SMB, HML, UMD = Fama-French    (ICI)
"""

import io
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (FF_3FACTOR_CSV, FF_MOMENTUM_CSV, HEDGE_FUND_CSV,
                     EXTERNAL_DIR, DATA_DIR, DATA_PATH)

_FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
_FF3_ZIP = _FF_BASE + "F-F_Research_Data_Factors_CSV.zip"
_MOM_ZIP = _FF_BASE + "F-F_Momentum_Factor_CSV.zip"


# ----------------------------------------------------------------------
# Parsing bas niveau (format French : bloc mensuel YYYYMM)
# ----------------------------------------------------------------------
def _parse_ff_csv_monthly(text: str) -> pd.DataFrame:
    """
    Extrait le bloc MENSUEL d'un CSV Ken French.

    Robuste :
      - l'en-tête est repéré comme une vraie ligne CSV (un token de facteur
        présent COMME CHAMP séparé par virgule, avec ≥2 champs) — on n'est donc
        pas piégé par une ligne de prose contenant « Mom », « RF », etc. ;
      - on collecte TOUTES les lignes dont la 1re cellule est une date YYYYMM
        (6 chiffres), sans arrêt prématuré sur les lignes vides ;
      - on s'arrête au bloc annuel (1re cellule à 4 chiffres = YYYY).
    Valeurs manquantes -99.99 / -999 -> NaN. Données en %.
    """
    tokens = {"Mkt-RF", "SMB", "HML", "Mom", "WML", "RF"}
    lines = text.splitlines()

    header, start_i = None, 0
    for i, line in enumerate(lines):
        fields = [p.strip() for p in line.split(",")]
        if len(fields) >= 2 and any(f in tokens for f in fields):
            header = [f for f in fields if f != ""]
            start_i = i + 1
            break
    if header is None:
        return pd.DataFrame()

    rows = []
    for line in lines[start_i:]:
        fields = [p.strip() for p in line.split(",")]
        d = fields[0]
        if d.isdigit() and len(d) == 6:            # ligne mensuelle YYYYMM
            rows.append(fields[: len(header) + 1])
        elif d.isdigit() and len(d) == 4:          # bloc annuel atteint -> stop
            break
        # toute autre ligne (vide, 2e en-tête, prose) est ignorée, pas d'arrêt

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index(0)
    df.columns = header[: df.shape[1]]
    df = df.apply(pd.to_numeric, errors="coerce").replace([-99.99, -999, -99.0], np.nan)
    df.index = pd.to_datetime(df.index, format="%Y%m") + pd.offsets.MonthEnd(0)
    df.index.name = "date"
    return df


def _download_ff_zip(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        return zf.read(zf.namelist()[0]).decode("latin-1")


# ----------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------
def _candidate_dirs() -> list[Path]:
    """Emplacements plausibles des CSV FF, dédupliqués, dans l'ordre de priorité.
    Couvre les décalages de répertoire de travail (notebook lancé ailleurs)."""
    cwd = Path.cwd()
    cands = [
        Path(EXTERNAL_DIR),
        Path(DATA_DIR), Path(DATA_DIR) / "external",
        Path(DATA_PATH).resolve().parent, Path(DATA_PATH).resolve().parent / "external",
        cwd, cwd / "data" / "external", cwd / "external",
        cwd.parent / "data" / "external",
    ]
    seen, out = set(), []
    for d in cands:
        try:
            rd = d.resolve()
        except Exception:
            continue
        if rd not in seen and rd.is_dir():
            seen.add(rd); out.append(rd)
    return out


def _find_one(directory: Path, patterns: list[str]) -> Path | None:
    """1er CSV du dossier correspondant à l'un des motifs (insensible à la casse)."""
    files = {f.name.lower(): f for f in directory.glob("*.csv")}
    # correspondance exacte d'abord
    for pat in patterns:
        for low, f in files.items():
            if low == pat.lower():
                return f
    # puis correspondance par sous-chaîne (gère suffixes de date, variantes)
    keys = [p.lower().replace(".csv", "") for p in patterns]
    for low, f in files.items():
        if any(k in low for k in keys):
            return f
    return None


def find_ff_csvs() -> tuple[Path | None, Path | None]:
    """Localise (3-factor, momentum) en balayant les emplacements candidats.
    Renvoie (None, None) si introuvables."""
    p3 = pm = None
    for d in _candidate_dirs():
        if p3 is None:
            p3 = _find_one(d, ["F-F_Research_Data_Factors.csv",
                               "F-F_Research_Data_Factors"])
        if pm is None:
            pm = _find_one(d, ["F-F_Momentum_Factor.csv", "F-F_Momentum_Factor",
                               "Momentum"])
        if p3 and pm:
            break
    return p3, pm


def _from_cache() -> pd.DataFrame | None:
    """Lit les CSV FF locaux où qu'ils soient (mode hors-ligne robuste)."""
    p3, pm = find_ff_csvs()
    if p3 is None or pm is None:
        return None
    with open(p3, "r", encoding="latin-1") as f:
        ff3 = _parse_ff_csv_monthly(f.read())
    with open(pm, "r", encoding="latin-1") as f:
        mom = _parse_ff_csv_monthly(f.read())
    mom.columns = ["UMD"]
    return ff3.join(mom, how="inner")


def _from_direct(write_cache: bool = True) -> pd.DataFrame:
    """Télécharge les ZIP Dartmouth et met en cache le texte brut."""
    txt3 = _download_ff_zip(_FF3_ZIP)
    txtm = _download_ff_zip(_MOM_ZIP)
    if write_cache:
        Path(EXTERNAL_DIR).mkdir(parents=True, exist_ok=True)
        Path(FF_3FACTOR_CSV).write_text(txt3, encoding="latin-1")
        Path(FF_MOMENTUM_CSV).write_text(txtm, encoding="latin-1")
    ff3 = _parse_ff_csv_monthly(txt3)
    mom = _parse_ff_csv_monthly(txtm); mom.columns = ["UMD"]
    return ff3.join(mom, how="inner")


def _from_datareader(start, end) -> pd.DataFrame:
    import pandas_datareader.data as web
    ff3 = web.DataReader("F-F_Research_Data_Factors", "famafrench", start=start, end=end)[0]
    mom = web.DataReader("F-F_Momentum_Factor", "famafrench", start=start, end=end)[0]
    mom.columns = ["UMD"]
    out = ff3.join(mom, how="inner")
    out.index = out.index.to_timestamp(how="end").normalize() + pd.offsets.MonthEnd(0)
    out.index.name = "date"
    return out


def load_ff_from_csv(path_3factor: str, path_momentum: str) -> pd.DataFrame:
    with open(path_3factor, "r", encoding="latin-1") as f:
        ff3 = _parse_ff_csv_monthly(f.read())
    with open(path_momentum, "r", encoding="latin-1") as f:
        mom = _parse_ff_csv_monthly(f.read())
    mom.columns = ["UMD"]
    return ff3.join(mom, how="inner")


# ----------------------------------------------------------------------
# API publique
# ----------------------------------------------------------------------
def fetch_ff_factors(start="1985-01-01", end="2009-12-31",
                     source="auto", csv_3factor=None, csv_momentum=None) -> pd.DataFrame:
    """
    SMB, HML, UMD, RF mensuels, en DÉCIMAL, fin de mois, filtrés sur [start, end].

    source : "auto" (cache local -> direct+cache -> datareader),
             "cache", "direct", "datareader", ou "csv" (chemins explicites).
    """
    raw = None
    if source == "csv":
        if not (csv_3factor and csv_momentum):
            raise ValueError("source='csv' nécessite csv_3factor et csv_momentum.")
        raw = load_ff_from_csv(csv_3factor, csv_momentum)
    elif source == "cache":
        raw = _from_cache()
        if raw is None:
            raise FileNotFoundError(f"CSV FF absents de {EXTERNAL_DIR}.")
    elif source == "direct":
        raw = _from_direct()
    elif source == "datareader":
        raw = _from_datareader(start, end)
    else:  # auto
        raw = _from_cache()
        if raw is None:
            try:
                raw = _from_direct()
            except Exception as e:
                try:
                    raw = _from_datareader(start, end)
                except Exception as e2:
                    raise RuntimeError(
                        "Échec des facteurs FF (cache vide, download KO).\n"
                        f"  direct: {e}\n  datareader: {e2}\n"
                        f"Déposez F-F_Research_Data_Factors.csv et "
                        f"F-F_Momentum_Factor.csv dans {EXTERNAL_DIR} "
                        "(source='cache')."
                    )

    ff = raw[["SMB", "HML", "UMD", "RF"]].copy() / 100.0
    return ff.loc[(ff.index >= pd.Timestamp(start)) & (ff.index <= pd.Timestamp(end))]


def build_table3_factors(monthly_data: pd.DataFrame,
                         ff_factors: pd.DataFrame) -> pd.DataFrame:
    """Matrice Eq. (4) : MKT/BOND/GSCI en excès du RF + SMB/HML/UMD."""
    df = monthly_data.join(ff_factors, how="inner")
    rf = df["RF"]
    X = pd.DataFrame(index=df.index)
    X["MKT"] = df["MXWO Index"] - rf
    X["BOND"] = df["LBUSTRUU Index"] - rf
    X["GSCI"] = df["SPGSCI Index"] - rf
    X["SMB"], X["HML"], X["UMD"] = df["SMB"], df["HML"], df["UMD"]
    return X.dropna()


# ----------------------------------------------------------------------
# Indices hedge funds (Table 5 Panel C) — données sous licence
# ----------------------------------------------------------------------
def load_hedge_fund_indices(path=None) -> pd.DataFrame | None:
    """
    Charge les indices hedge funds Dow Jones/Credit Suisse pour la Table 5C
    (« Managed Futures » et « Global Macro »). Les données étant sous licence,
    elles ne sont pas embarquées : déposez un CSV `date, ManagedFutures,
    GlobalMacro` (rendements mensuels) dans data/external/. Renvoie None si
    absent (le pipeline saute alors proprement le Panel C hedge funds).

    Substituts publics acceptables : SG Trend Index, BarclayHedge CTA Index.
    """
    p = Path(path) if path else Path(HEDGE_FUND_CSV)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
    df = df.set_index("date").apply(pd.to_numeric, errors="coerce")
    if df.abs().median().median() > 1.0:   # données en % -> décimal
        df = df / 100.0
    return df


if __name__ == "__main__":
    ff = fetch_ff_factors(source="auto")
    print(ff.tail()); print(f"{len(ff)} mois, {ff.index.min():%Y-%m} → {ff.index.max():%Y-%m}")
