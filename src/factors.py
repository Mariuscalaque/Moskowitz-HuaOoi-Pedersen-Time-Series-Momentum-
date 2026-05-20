"""
factors.py — Récupération AUTOMATIQUE des facteurs Fama-French
(SMB, HML, UMD) + taux sans risque RF, depuis la Ken French Data Library
(Dartmouth), pour la Table 3 de Moskowitz, Hua Ooi & Pedersen (2012).

Le papier régresse le TSMOM diversifié sur (Eq. 4) :
    MKT  = excès de rendement du MSCI World   -> garde MXWO de data.xlsx
    BOND = Barclays Aggregate Bond Index      -> LBUSTRUU de data.xlsx
    GSCI = S&P GSCI                            -> SPGSCI de data.xlsx
    SMB, HML, UMD = facteurs Fama-French       -> récupérés ICI
    + RF (taux sans risque) pour mettre MKT/BOND/GSCI en excès

Trois sources possibles, essayées dans l'ordre par fetch_ff_factors(source="auto") :
  1) téléchargement direct du ZIP sur le serveur de Dartmouth (Python pur,
     aucune dépendance externe) ;
  2) pandas_datareader (si installé et compatible) ;
  3) fichiers CSV téléchargés à la main (load_ff_from_csv).

Aucune donnée n'est stockée dans ce module : elle est récupérée à chaque appel.
"""

import io
import zipfile
import urllib.request

import numpy as np
import pandas as pd

# URL racine officielle de la Ken French Data Library
_FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
_FF3_ZIP = _FF_BASE + "F-F_Research_Data_Factors_CSV.zip"
_MOM_ZIP = _FF_BASE + "F-F_Momentum_Factor_CSV.zip"


# ----------------------------------------------------------------------
# Parsing bas niveau d'un CSV French (format à blocs : en-tête + section
# mensuelle + section annuelle séparées par des lignes vides)
# ----------------------------------------------------------------------
def _parse_ff_csv_monthly(text: str) -> pd.DataFrame:
    """
    Extrait UNIQUEMENT le bloc mensuel d'un CSV French.
    Les lignes mensuelles ont une date à 6 chiffres (YYYYMM) ;
    le bloc annuel (YYYY, 4 chiffres) et les notes sont ignorés.
    Valeurs manquantes codées -99.99 / -999 -> NaN. Données en %.
    """
    rows = []
    header = None
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if header is None:
            # l'en-tête est la 1re ligne dont les colonnes ressemblent aux facteurs
            if any(tok in line for tok in ("Mkt-RF", "SMB", "HML", "Mom", "RF")):
                header = [p for p in parts if p != ""]
            continue
        # ligne de données mensuelle : 1re cellule = 6 chiffres
        if parts and parts[0].isdigit() and len(parts[0]) == 6:
            vals = parts[: len(header) + 1]
            rows.append(vals)
        elif rows:
            # on a déjà collecté le bloc mensuel et on tombe sur une ligne
            # non mensuelle (ligne vide ou début du bloc annuel) -> stop
            break

    df = pd.DataFrame(rows).set_index(0)
    df.columns = header[: df.shape[1]]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([-99.99, -999, -99.0], np.nan)
    # index YYYYMM -> timestamp fin de mois
    df.index = pd.to_datetime(df.index, format="%Y%m") + pd.offsets.MonthEnd(0)
    df.index.name = "date"
    return df


def _download_ff_zip(url: str, timeout: int = 30) -> str:
    """Télécharge un ZIP French et renvoie le contenu texte du CSV qu'il contient."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        name = zf.namelist()[0]  # un seul CSV par ZIP
        return zf.read(name).decode("latin-1")


# ----------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------
def _fetch_via_direct() -> pd.DataFrame:
    """Source 1 : téléchargement direct des ZIP Dartmouth (Python pur)."""
    ff3 = _parse_ff_csv_monthly(_download_ff_zip(_FF3_ZIP))   # Mkt-RF, SMB, HML, RF
    mom = _parse_ff_csv_monthly(_download_ff_zip(_MOM_ZIP))   # Mom
    mom.columns = ["UMD"]
    out = ff3.join(mom, how="inner")
    return out


def _fetch_via_datareader(start, end) -> pd.DataFrame:
    """Source 2 : pandas_datareader (si installé/compatible)."""
    import pandas_datareader.data as web
    ff3 = web.DataReader("F-F_Research_Data_Factors", "famafrench",
                         start=start, end=end)[0] / 100.0
    mom = web.DataReader("F-F_Momentum_Factor", "famafrench",
                         start=start, end=end)[0] / 100.0
    mom.columns = ["UMD"]
    out = ff3.join(mom, how="inner")
    out.index = out.index.to_timestamp(how="end").normalize() + pd.offsets.MonthEnd(0)
    out.index.name = "date"
    out = out * 100.0  # remis en % pour homogénéiser avec _fetch_via_direct
    return out


def load_ff_from_csv(path_3factor: str, path_momentum: str) -> pd.DataFrame:
    """
    Source 3 : lecture de CSV téléchargés à la main depuis le site de French
    (F-F_Research_Data_Factors.CSV et F-F_Momentum_Factor.CSV).
    """
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
                     source="auto",
                     csv_3factor=None, csv_momentum=None) -> pd.DataFrame:
    """
    Renvoie SMB, HML, UMD, RF en mensuel, en valeurs DÉCIMALES (pas en %),
    indexées en fin de mois, filtrées sur [start, end].

    source : "auto" (direct -> datareader), "direct", "datareader", ou "csv".
    """
    if source in ("auto", "direct"):
        try:
            raw = _fetch_via_direct()
        except Exception as e:
            if source == "direct":
                raise
            try:
                raw = _fetch_via_datareader(start, end)
            except Exception as e2:
                raise RuntimeError(
                    "Échec du téléchargement automatique des facteurs FF.\n"
                    f"  - direct : {e}\n  - datareader : {e2}\n"
                    "Téléchargez les CSV à la main et utilisez source='csv'."
                )
    elif source == "datareader":
        raw = _fetch_via_datareader(start, end)
    elif source == "csv":
        if not (csv_3factor and csv_momentum):
            raise ValueError("source='csv' nécessite csv_3factor et csv_momentum.")
        raw = load_ff_from_csv(csv_3factor, csv_momentum)
    else:
        raise ValueError(f"source inconnue : {source}")

    ff = raw[["SMB", "HML", "UMD", "RF"]].copy() / 100.0   # % -> décimal
    ff = ff.loc[(ff.index >= pd.Timestamp(start)) & (ff.index <= pd.Timestamp(end))]
    return ff


def build_table3_factors(monthly_data: pd.DataFrame,
                         ff_factors: pd.DataFrame) -> pd.DataFrame:
    """
    Matrice de facteurs de la Table 3 (Eq. 4), alignée en fin de mois :
      MKT  = MXWO     - RF
      BOND = LBUSTRUU - RF
      GSCI = SPGSCI   - RF
      SMB, HML, UMD : Fama-French (déjà des facteurs long-short, en excès)

    `monthly_data` doit contenir 'MXWO Index', 'LBUSTRUU Index', 'SPGSCI Index'.
    """
    df = monthly_data.join(ff_factors, how="inner")
    rf = df["RF"]
    X = pd.DataFrame(index=df.index)
    X["MKT"]  = df["MXWO Index"]     - rf
    X["BOND"] = df["LBUSTRUU Index"] - rf
    X["GSCI"] = df["SPGSCI Index"]   - rf
    X["SMB"]  = df["SMB"]
    X["HML"]  = df["HML"]
    X["UMD"]  = df["UMD"]
    return X.dropna()


if __name__ == "__main__":
    ff = fetch_ff_factors(start="1985-01-01", end="2009-12-31", source="auto")
    print(ff.head())
    print(f"\n{len(ff)} mois, de {ff.index.min():%Y-%m} à {ff.index.max():%Y-%m}")
