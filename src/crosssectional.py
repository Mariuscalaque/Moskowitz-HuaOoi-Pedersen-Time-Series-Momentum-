"""
crosssectional.py — Momentum cross-sectionnel (XSMOM) et décomposition de
Lo & MacKinlay (1990) telle qu'utilisée par Moskowitz, Ooi & Pedersen (2012),
Section 5.2, Eq. (6)-(7), pour la Table 5 Panel B.

CORRECTION : la décomposition respecte désormais la structure du papier —
le SIGNAL est le rendement cumulé des 12 derniers mois, la CIBLE est le
rendement du mois suivant (t+1). On définit la matrice de comoments

    Ω = E[(R_{t-12,t} - 12μ)(r_{t,t+1} - μ)']            (N×N)

avec μ_i = E(r_{i,t,t+1}) = E(R_{i,t-12,t})/12 (exact si R_{t-12,t} est la
SOMME des 12 rendements mensuels). Alors (Eq. 6 / 7) :

  XSMOM (poids w_i = (1/N)(R_{i,t-12,t} - R̄_t)) :
     E[π^XS] = (N-1)/N²·tr(Ω) − (1/N²)(1'Ω1 − tr(Ω)) + 12·s²_μ
             =     Auto       +         Cross          +   Mean

  TSMOM (poids w_i = (1/N)·R_{i,t-12,t}) :
     E[π^TS] = tr(Ω)/N + 12·(μ'μ)/N
             =   Auto   +   Mean-squared
"""

import numpy as np
import pandas as pd

from .strategy import past_k_month_returns
from .config import LOOKBACK_MONTHS, asset_class_of


# ============================================================================
# Construction XSMOM (inchangée : rang dé-moyené, dollar-neutre)
# ============================================================================
def xsmom_weights(monthly_ret: pd.DataFrame, k: int = LOOKBACK_MONTHS) -> pd.DataFrame:
    """Poids dollar-neutres ∝ (rang - rang moyen) du rendement passé k mois,
    normalisés pour une jambe longue de +1 (et courte de -1)."""
    past = past_k_month_returns(monthly_ret, k=k)
    ranks = past.rank(axis=1, method="average")
    demeaned = ranks.sub(ranks.mean(axis=1), axis=0)
    pos_sum = demeaned.clip(lower=0).sum(axis=1).replace(0, np.nan)
    w = demeaned.div(pos_sum, axis=0)
    return w.where(past.notna())


def xsmom_returns(monthly_ret: pd.DataFrame, k: int = LOOKBACK_MONTHS) -> pd.Series:
    """Rendement XSMOM : Σ_i w_{i,t} · r_{i,t+1}."""
    w = xsmom_weights(monthly_ret, k=k)
    return (w.shift(1) * monthly_ret).sum(axis=1, min_count=1)


def xsmom_by_asset_class(monthly_ret: pd.DataFrame,
                         k: int = LOOKBACK_MONTHS) -> pd.DataFrame:
    """XSMOM construit séparément dans chaque classe + ALL."""
    classes = {c: asset_class_of(c) for c in monthly_ret.columns}
    out = {}
    for ac in ("Commodity", "Equity", "Bond", "Currency"):
        cols = [c for c, a in classes.items() if a == ac]
        if len(cols) >= 2:
            out[ac] = xsmom_returns(monthly_ret[cols], k=k)
    out["ALL"] = xsmom_returns(monthly_ret, k=k)
    return pd.DataFrame(out)


# ============================================================================
# Décomposition Lo-MacKinlay 12→1 (Panel B), conforme à MOP Eq. (6)-(7)
# ============================================================================
def lo_mackinlay_decomposition(monthly_ret: pd.DataFrame,
                               lookback: int = LOOKBACK_MONTHS,
                               min_obs: int = 120) -> dict:
    """
    Décomposition des profits XSMOM et TSMOM 12→1 selon Eq. (6)-(7).

    `monthly_ret` : rendements mensuels (instruments en colonnes). On retient un
    PANNEAU ÉQUILIBRÉ : instruments ayant au moins `min_obs` observations, puis
    fenêtre commune à ces survivants (Ω bien définie). Renvoie les composantes
    XSMOM (Auto/Cross/Mean) et TSMOM (Auto/MeanSq) ainsi que les profits
    empiriques correspondants (pour vérifier l'identité).

    CORRECTION : maintenant que `daily_to_monthly_returns` renvoie NaN (et non 0)
    pour les mois sans données, un `dropna(axis=1, how='any')` brut éliminerait
    presque tous les instruments (départs échelonnés). On sélectionne donc
    d'abord les instruments à historique suffisant (≥ min_obs), puis leur fenêtre
    commune. Sans cela, les zéros fantômes diluaient Ω (~10× trop petit).
    """
    keep = monthly_ret.columns[monthly_ret.notna().sum() >= min_obs]
    R = monthly_ret[keep].dropna(axis=0, how="any")
    N = R.shape[1]
    if N < 2 or len(R) < lookback + 24:
        nan = float("nan")
        return {k: nan for k in
                ("XS_Auto", "XS_Cross", "XS_Mean", "XS_Total(sum)", "XS_Total(emp)",
                 "TS_Auto", "TS_MeanSq", "TS_Total(sum)", "TS_Total(emp)")} | {"N": N}

    # rendement cumulé (SOMME) des 12 derniers mois, connu en fin de mois t
    R12 = R.rolling(lookback).sum()
    # cible : rendement du mois t+1
    r1_next = R.shift(-1)

    # alignement signal_t -> cible_{t+1}
    df = pd.concat({"sig": R12, "tgt": r1_next}, axis=1).dropna()
    sig = df["sig"].values          # T×N  (R_{t-12,t})
    tgt = df["tgt"].values          # T×N  (r_{t,t+1})
    T = sig.shape[0]

    mu = tgt.mean(axis=0)           # μ_i = E(r_{t,t+1})
    mu_bar = mu.mean()

    # Ω = (1/T) Σ (R12_t - 12μ)(r1_{t+1} - μ)'
    A = sig - 12.0 * mu             # T×N
    B = tgt - mu                    # T×N
    Omega = (A.T @ B) / T           # N×N

    trO = np.trace(Omega)
    sumO = Omega.sum()
    s2_mu = np.mean((mu - mu_bar) ** 2)        # variance transversale des μ_i

    # ---- XSMOM ----
    xs_auto = (N - 1) / N**2 * trO
    xs_cross = -(1.0 / N**2) * (sumO - trO)
    xs_mean = 12.0 * s2_mu
    xs_sum = xs_auto + xs_cross + xs_mean

    # ---- TSMOM ----
    ts_auto = trO / N
    ts_meansq = 12.0 * (mu @ mu) / N
    ts_sum = ts_auto + ts_meansq

    # ---- profits empiriques (mêmes poids que dans les équations) ----
    sig_bar = sig.mean(axis=1, keepdims=True)
    w_xs = (sig - sig_bar) / N
    pi_xs = (w_xs * tgt).sum(axis=1)
    w_ts = sig / N
    pi_ts = (w_ts * tgt).sum(axis=1)

    return {
        "XS_Auto": xs_auto, "XS_Cross": xs_cross, "XS_Mean": xs_mean,
        "XS_Total(sum)": xs_sum, "XS_Total(emp)": float(pi_xs.mean()),
        "TS_Auto": ts_auto, "TS_MeanSq": ts_meansq,
        "TS_Total(sum)": ts_sum, "TS_Total(emp)": float(pi_ts.mean()),
        "N": N,
    }


def decomposition_by_asset_class(monthly_ret: pd.DataFrame,
                                 lookback: int = LOOKBACK_MONTHS) -> pd.DataFrame:
    """Applique la décomposition 12→1 à chaque classe puis à l'ensemble."""
    classes = {c: asset_class_of(c) for c in monthly_ret.columns}
    rows = {}
    for ac in ("Commodity", "Equity", "Bond", "Currency"):
        cols = [c for c, a in classes.items() if a == ac]
        if len(cols) >= 2:
            rows[ac] = lo_mackinlay_decomposition(monthly_ret[cols], lookback=lookback)
    rows["ALL"] = lo_mackinlay_decomposition(monthly_ret, lookback=lookback)
    return pd.DataFrame(rows).T