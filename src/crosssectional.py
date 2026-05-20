"""
crosssectional.py — Cross-sectional momentum (XSMOM) et décomposition
des profits, pour la Table 5 de Moskowitz, Hua Ooi & Pedersen (2012).

XSMOM : à chaque fin de mois t, on classe les instruments par leur rendement
cumulé des 12 derniers mois et on prend des poids dollar-neutres proportionnels
au rang dé-moyenné (construction de type Asness-Moskowitz-Pedersen). Le
rendement t+1 est la somme pondérée des rendements des instruments.

Décomposition (Panel B) : on utilise le cadre de Lo & MacKinlay (1990) repris
par MOP en Section 4.2. Le profit espéré d'une stratégie de momentum relatif
se décompose en :
   - Auto   : contribution des AUTO-covariances (un actif prédit lui-même),
   - Cross  : contribution des covariances CROISÉES / effets lead-lag,
   - Mean   : contribution de la dispersion des moyennes inconditionnelles.
On vérifie l'identité comptable : Auto + Cross + Mean ≈ profit empirique moyen.
"""

import numpy as np
import pandas as pd

from .strategy import past_k_month_returns
from .config import LOOKBACK_MONTHS, asset_class_of


# ============================================================================
# Construction XSMOM
# ============================================================================
def xsmom_weights(monthly_ret: pd.DataFrame, k: int = LOOKBACK_MONTHS) -> pd.DataFrame:
    """
    Poids dollar-neutres de la stratégie de momentum cross-sectionnel.

    À chaque date, w_i ∝ (rang_i - rang_moyen), normalisés pour que la somme
    des poids longs vaille +1 (et donc des shorts -1). Les poids sont calés à
    la fin du mois t (ils s'appliquent au rendement t+1 via un shift en aval).
    """
    past = past_k_month_returns(monthly_ret, k=k)
    # rang transversal (méthode 'average'), normalisé entre ~0 et 1
    ranks = past.rank(axis=1, method="average")
    demeaned = ranks.sub(ranks.mean(axis=1), axis=0)
    # normalise : somme des poids positifs = 1
    pos_sum = demeaned.clip(lower=0).sum(axis=1).replace(0, np.nan)
    w = demeaned.div(pos_sum, axis=0)
    return w.where(past.notna())


def xsmom_returns(monthly_ret: pd.DataFrame, k: int = LOOKBACK_MONTHS) -> pd.Series:
    """Rendement XSMOM (toutes classes) : Σ_i w_{i,t} · r_{i,t+1}."""
    w = xsmom_weights(monthly_ret, k=k)
    return (w.shift(1) * monthly_ret).sum(axis=1, min_count=1)


def xsmom_by_asset_class(monthly_ret: pd.DataFrame,
                         k: int = LOOKBACK_MONTHS) -> pd.DataFrame:
    """XSMOM construit séparément à l'intérieur de chaque classe d'actifs."""
    classes = {c: asset_class_of(c) for c in monthly_ret.columns}
    out = {}
    for ac in ("Commodity", "Equity", "Bond", "Currency"):
        cols = [c for c, a in classes.items() if a == ac]
        if len(cols) >= 2:
            out[ac] = xsmom_returns(monthly_ret[cols], k=k)
    out["ALL"] = xsmom_returns(monthly_ret, k=k)
    return pd.DataFrame(out)


# ============================================================================
# Décomposition de Lo-MacKinlay (Panel B)
# ============================================================================
def lo_mackinlay_decomposition(monthly_ret: pd.DataFrame,
                               k: int = 1) -> dict:
    """
    Décomposition du profit espéré de la stratégie de momentum relatif à poids
    w_{i,t} = (1/N)(r_{i,t-k} - r̄_{t-k}) appliqués au rendement r_{i,t}.

    E[π] = Auto + Cross + Mean, avec (Lo & MacKinlay, 1990) :
        Auto  = (N-1)/N² · Σ_i  Cov(r_{i,t}, r_{i,t-k})           (own-serial)
        Cross = -(1/N²) · Σ_{i≠j} Cov(r_{i,t}, r_{j,t-k})         (lead-lag)
        Mean  = (N-1)/N² · Σ_i (μ_i - μ̄)²  = (N-1)/N · Var_x(μ)   (dispersion)

    On calcule sur le panneau des instruments présents en continu pour avoir une
    matrice de covariance bien définie. Renvoie les composantes + le profit
    empirique moyen (pour vérifier l'identité).
    """
    R = monthly_ret.dropna(axis=1, how="any").dropna(axis=0, how="any")
    if R.shape[1] < 2 or len(R) < k + 12:
        return {"Auto": np.nan, "Cross": np.nan, "Mean": np.nan,
                "Total (sum)": np.nan, "Total (empirical)": np.nan, "N": R.shape[1]}

    N = R.shape[1]
    Rt = R.iloc[k:].values            # r_{i,t}
    Rtk = R.iloc[:-k].values          # r_{i,t-k}
    T = Rt.shape[0]

    mu = R.mean().values
    mu_bar = mu.mean()

    # matrice de covariance "lag-k" : Gamma[i,j] = Cov(r_{i,t}, r_{j,t-k})
    Rt_c = Rt - Rt.mean(0)
    Rtk_c = Rtk - Rtk.mean(0)
    Gamma = (Rt_c.T @ Rtk_c) / (T - 1)        # N×N

    own = np.trace(Gamma)
    cross = Gamma.sum() - own

    auto_term = (N - 1) / N**2 * own
    cross_term = -(1.0 / N**2) * cross
    mean_term = (N - 1) / N * np.var(mu, ddof=0)

    # profit empirique moyen de la stratégie LM (poids 1/N · (r_{t-k} - r̄))
    w = (Rtk - Rtk.mean(1, keepdims=True)) / N
    pi = (w * Rt).sum(1)

    return {
        "Auto": auto_term,
        "Cross": cross_term,
        "Mean": mean_term,
        "Total (sum)": auto_term + cross_term + mean_term,
        "Total (empirical)": float(pi.mean()),
        "N": N,
    }


def decomposition_by_asset_class(monthly_ret: pd.DataFrame,
                                 k: int = 1) -> pd.DataFrame:
    """Applique la décomposition Lo-MacKinlay à chaque classe puis à l'ensemble."""
    classes = {c: asset_class_of(c) for c in monthly_ret.columns}
    rows = {}
    for ac in ("Commodity", "Equity", "Bond", "Currency"):
        cols = [c for c, a in classes.items() if a == ac]
        if len(cols) >= 2:
            rows[ac] = lo_mackinlay_decomposition(monthly_ret[cols], k=k)
    rows["ALL"] = lo_mackinlay_decomposition(monthly_ret, k=k)
    return pd.DataFrame(rows).T
