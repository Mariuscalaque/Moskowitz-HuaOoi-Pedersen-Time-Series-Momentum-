"""
robustness.py — Extensions de robustesse (« Tier 1 ») au-delà de la réplication
de Moskowitz, Ooi & Pedersen (2012, ci-après MOP) et des extensions copule A/B/C.

Trois analyses, toutes branchées sur l'infrastructure EXISTANTE du projet
(load_raw -> returns -> volatility -> strategy) et ne dépendant QUE de
numpy / pandas / statsmodels (pas de `arch`, contrairement aux extensions
copule) :

  (D1) HORS ÉCHANTILLON — 1985-2009 vs 2010-2025.
       MOP s'arrêtent en décembre 2009. On reconstruit le MÊME TSMOM diversifié
       jusqu'à fin 2025 et on compare, avant/après : performance brute,
       alpha factoriel (MKT/SMB/HML/UMD) et convexité « smile » (MKT²). Test
       direct de la robustesse out-of-sample (« le trend-following est-il mort
       depuis la crise ? »). -> subperiod_performance().

  (D2) RISQUE DE LIQUIDITÉ — facteur TRADABLE de Pástor-Stambaugh (LIQ_V).
       MOP ne testent la liquidité que comme VARIABLE D'ÉTAT (TED, VIX,
       innovation PS) dans le Panel C, et concluent à l'absence de lien. Ici on
       va plus loin : on ajoute le facteur de liquidité TRADABLE (rendement du
       portefeuille 10-1, qui porte une prime de risque) comme FACTEUR DE RISQUE
       dans la régression d'alpha. Question : l'alpha du TSMOM survit-il à un
       ajustement au risque de liquidité ? Test non fait par le papier.
       -> liquidity_augmented_alpha().

  (D3) TURBULENCE FINANCIÈRE — distance de Mahalanobis (Kritzman & Li, 2010).
       Pont direct entre le résultat phare de MOP §4.3 (« TSMOM performe le mieux
       dans les marchés extrêmes ») et l'Exercice 2 du cours (« stress émergent »,
       distance de Mahalanobis). On construit l'indice de turbulence
           d_t = (r_t - mu)' Sigma^{-1} (r_t - mu)
       sur le panel d'instruments, puis on teste si le TSMOM rend davantage dans
       les mois turbulents (quantile haut) que dans les mois calmes. Sigma est
       régularisée par shrinkage de Ledoit-Wolf vers l'identité (clin d'oeil au
       volet « estimateur de réduction / ill-posed problem » du cours).
       -> financial_turbulence(), tsmom_by_turbulence().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .analysis import table3_full, table3_smile_quarterly
from .config import asset_class_of


# ============================================================================
# Utilitaires communs
# ============================================================================
def _perf_stats(r: pd.Series, ppy: int = 12) -> dict:
    """Statistiques de performance brute d'une série de rendements mensuels."""
    r = r.dropna()
    if len(r) < 6:
        return {"N months": len(r), "Ann. mean": np.nan, "Ann. vol": np.nan,
                "Sharpe": np.nan, "Max DD": np.nan, "Skew": np.nan,
                "Exc. kurt": np.nan}
    ann_mean = r.mean() * ppy
    ann_vol = r.std() * np.sqrt(ppy)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    return {
        "N months": len(r),
        "Ann. mean": ann_mean,
        "Ann. vol": ann_vol,
        "Sharpe": ann_mean / ann_vol if ann_vol > 0 else np.nan,
        "Max DD": dd,
        "Skew": r.skew(),
        "Exc. kurt": r.kurtosis(),
    }


def _slice(s, start, end):
    return s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]


# ============================================================================
# (D1) Hors échantillon : 1985-2009 vs 2010-2025
# ============================================================================
def subperiod_performance(tsmom: pd.Series,
                          ff4: pd.DataFrame,
                          periods: dict | None = None) -> pd.DataFrame:
    """
    Compare le TSMOM diversifié sur plusieurs sous-périodes.

    Parameters
    ----------
    tsmom : série mensuelle du TSMOM diversifié (en excès), idéalement 1986->2025.
    ff4   : DataFrame mensuel des facteurs [MKT, SMB, HML, UMD] (décimal). MKT est
            le marché EN EXCÈS (MSCI World - RF) ; il sert aussi au « smile ».
    periods : dict {label: (start, end)}. Défaut : in-sample MOP, out-of-sample,
              et plein échantillon.

    Returns
    -------
    DataFrame indexé par sous-période, colonnes :
      N months, Ann. mean, Ann. vol, Sharpe, Max DD,
      Alpha (%/m), t(Alpha), beta(MKT^2), t(MKT^2)
    (Alpha = intercept de TSMOM ~ MKT+SMB+HML+UMD ; MKT^2 = coeff de convexité
     du « smile » trimestriel TSMOM ~ MKT + MKT².)
    """
    if periods is None:
        periods = {
            "In-sample 1985-2009": ("1985-01-01", "2009-12-31"),
            "Out-of-sample 2010-2025": ("2010-01-01", "2025-12-31"),
            "Full 1985-2025": ("1985-01-01", "2025-12-31"),
        }
    mkt_excess = ff4["MKT"]
    rows = {}
    for label, (a, b) in periods.items():
        r = _slice(tsmom, a, b)
        stats = _perf_stats(r)

        # Alpha factoriel (mensuel) sur MKT/SMB/HML/UMD — réutilise table3_full
        alpha = t_alpha = np.nan
        Xsub = ff4.loc[(ff4.index >= pd.Timestamp(a)) & (ff4.index <= pd.Timestamp(b))]
        try:
            if len(r.dropna()) >= 24 and len(Xsub) >= 24:
                t3 = table3_full(r, Xsub)
                alpha = float(t3.loc["Monthly", "Alpha (%)"])
                t_alpha = float(t3.loc["Monthly", "t(Alpha)"])
        except Exception:
            pass

        # Convexité « smile » (trimestriel) TSMOM ~ MKT + MKT²
        smile_b = smile_t = np.nan
        try:
            if len(r.dropna()) >= 24:
                sm_df = table3_smile_quarterly(r, _slice(mkt_excess, a, b))
                smile_b = float(sm_df.loc["Quarterly", "MKT_sq"])
                smile_t = float(sm_df.loc["Quarterly", "t(MKT_sq)"])
        except Exception:
            pass

        rows[label] = {**stats, "Alpha (%/m)": alpha, "t(Alpha)": t_alpha,
                       "beta(MKT^2)": smile_b, "t(MKT^2)": smile_t}

    cols = ["N months", "Ann. mean", "Ann. vol", "Sharpe", "Max DD",
            "Alpha (%/m)", "t(Alpha)", "beta(MKT^2)", "t(MKT^2)"]
    return pd.DataFrame(rows).T[cols]


# ============================================================================
# (D2) Risque de liquidité : facteur tradable Pástor-Stambaugh
# ============================================================================
def _hac_alpha(y: pd.Series, X: pd.DataFrame, lags: int = 3) -> dict:
    """OLS HAC (Newey-West) de y sur X ; renvoie alpha (%), t(alpha), betas, R², N."""
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    Xc = sm.add_constant(df[list(X.columns)])
    m = sm.OLS(df["y"], Xc).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {
        "alpha_pct": m.params["const"] * 100.0,
        "t_alpha": m.tvalues["const"],
        "betas": {c: m.params[c] for c in X.columns},
        "t_betas": {c: m.tvalues[c] for c in X.columns},
        "R2": m.rsquared,
        "N": int(m.nobs),
    }


def liquidity_augmented_alpha(tsmom: pd.Series,
                              ff4: pd.DataFrame,
                              ps_traded: pd.Series,
                              lags: int = 3) -> pd.DataFrame:
    """
    L'alpha du TSMOM survit-il à l'ajout du facteur de liquidité TRADABLE de
    Pástor-Stambaugh (LIQ_V) ?

    Trois régressions, TOUTES sur le MÊME échantillon commun (là où LIQ existe),
    pour que la comparaison d'alpha soit honnête :
      1. Baseline  : TSMOM ~ MKT + SMB + HML + UMD
      2. + LIQ_PS  : TSMOM ~ MKT + SMB + HML + UMD + LIQ
      3. Univarié  : TSMOM ~ LIQ

    Returns
    -------
    DataFrame (3 lignes) : Alpha (%), t(Alpha), beta(LIQ), t(LIQ), R2, N.
    """
    liq = ps_traded.rename("LIQ").reindex(ff4.index)
    base_cols = ["MKT", "SMB", "HML", "UMD"]

    # Échantillon commun = lignes où TSMOM, les 4 facteurs ET LIQ sont présents.
    common = pd.concat([tsmom.rename("y"), ff4[base_cols], liq], axis=1).dropna().index
    y = tsmom.reindex(common)
    Xbase = ff4.loc[common, base_cols]
    Xaug = pd.concat([Xbase, liq.reindex(common)], axis=1)

    r_base = _hac_alpha(y, Xbase, lags)
    r_aug = _hac_alpha(y, Xaug, lags)
    r_uni = _hac_alpha(y, liq.reindex(common).to_frame(), lags)

    def _row(res, with_liq):
        return {
            "Alpha (%)": res["alpha_pct"],
            "t(Alpha)": res["t_alpha"],
            "beta(LIQ)": res["betas"].get("LIQ", np.nan) if with_liq else np.nan,
            "t(LIQ)": res["t_betas"].get("LIQ", np.nan) if with_liq else np.nan,
            "R2": res["R2"],
            "N": res["N"],
        }

    out = pd.DataFrame({
        "Baseline (MKT,SMB,HML,UMD)": _row(r_base, False),
        "+ LIQ_PS (tradable)": _row(r_aug, True),
        "Univariate LIQ_PS": _row(r_uni, True),
    }).T
    return out[["Alpha (%)", "t(Alpha)", "beta(LIQ)", "t(LIQ)", "R2", "N"]]


# ============================================================================
# (D3) Turbulence financière : distance de Mahalanobis (Kritzman-Li 2010)
# ============================================================================
def _ledoit_wolf_identity(X: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Shrinkage linéaire de Ledoit-Wolf (2004) de la covariance échantillon vers
    la cible m·I (m = variance moyenne). Renvoie (Sigma_shrunk, intensité delta).

    X : matrice (n×p) DÉJÀ centrée. Sigma_shrunk est définie positive par
    construction -> inversion sûre pour la distance de Mahalanobis.
    """
    n, p = X.shape
    S = (X.T @ X) / n                          # covariance MLE (centrée)
    m = np.trace(S) / p                        # cible = m·I
    d2 = np.sum((S - m * np.eye(p)) ** 2) / p  # ||S - m I||_F^2 / p

    sq = (X ** 2).sum(axis=1)                  # x_k . x_k
    quad = np.einsum("ij,jk,ik->i", X, S, X)   # x_k' S x_k
    frobS2 = np.sum(S ** 2)
    per_k = sq ** 2 - 2.0 * quad + frobS2      # ||x_k x_k' - S||_F^2
    b2 = (per_k.mean() / p) / n
    b2 = min(b2, d2)                            # 0 <= b2 <= d2
    delta = b2 / d2 if d2 > 0 else 0.0
    Sigma = delta * m * np.eye(p) + (1.0 - delta) * S
    return Sigma, float(delta)


def financial_turbulence(monthly_ret: pd.DataFrame,
                         start: str = "1992-01-01",
                         coverage: float = 0.98,
                         shrink: bool = True) -> tuple[pd.Series, dict]:
    """
    Indice de turbulence financière de Kritzman-Li (2010) :
        d_t = (r_t - mu)' Sigma^{-1} (r_t - mu)
    sur le panel d'instruments (toutes classes confondues).

    Choix de robustesse :
      - on se restreint aux instruments présents à >= `coverage` sur [start, fin]
        (panel ÉQUILIBRÉ : sinon mu/Sigma seraient incohérents dans le temps) ;
      - mu et Sigma sont estimés sur tout l'échantillon retenu — c'est un indice
        DESCRIPTIF ex-post (comme dans Kritzman-Li, et comme le filtrage global
        assumé de l'Extension B), pas un signal investissable ;
      - Sigma est régularisée par shrinkage de Ledoit-Wolf si `shrink=True`.

    Returns
    -------
    (turbulence, info) :
       turbulence : Series mensuelle d_t (distance de Mahalanobis AU CARRÉ).
       info       : dict (nb instruments, période, intensité de shrinkage, dof).
    """
    sub = monthly_ret[monthly_ret.index >= pd.Timestamp(start)]
    cov = sub.notna().mean()
    keep = list(cov[cov >= coverage].index)
    panel = sub[keep].dropna()
    if panel.shape[0] < 60 or panel.shape[1] < 5:
        raise ValueError(f"Panel turbulence trop petit : {panel.shape}. "
                         f"Abaisse `coverage` ou avance `start`.")

    mu = panel.mean().to_numpy()
    Xc = panel.to_numpy() - mu
    if shrink:
        Sigma, delta = _ledoit_wolf_identity(Xc)
    else:
        Sigma, delta = np.cov(Xc, rowvar=False), 0.0
    Sinv = np.linalg.pinv(Sigma)

    d = np.einsum("ij,jk,ik->i", Xc, Sinv, Xc)     # Mahalanobis au carré
    turbulence = pd.Series(d, index=panel.index, name="turbulence")

    by_class = pd.Series([asset_class_of(c) for c in keep]).value_counts().to_dict()
    info = {
        "n_instruments": panel.shape[1],
        "n_months": panel.shape[0],
        "start": str(panel.index.min().date()),
        "end": str(panel.index.max().date()),
        "shrinkage_delta": delta,
        "dof": panel.shape[1],            # ~ E[d_t] sous normalité multivariée
        "by_class": by_class,
    }
    return turbulence, info


def tsmom_by_turbulence(tsmom: pd.Series,
                        turbulence: pd.Series,
                        n_buckets: int = 5,
                        top_q: float = 0.20,
                        lags: int = 3) -> dict:
    """
    Le TSMOM rend-il davantage quand le marché est turbulent ?
    (Test du résultat MOP §4.3 « best during extreme markets », via Mahalanobis.)

    Renvoie un dict avec :
      - 'by_bucket'  : DataFrame (perf du TSMOM par quantile de turbulence Q1..Qn)
      - 'calm_vs_turbulent' : DataFrame (calme vs top `top_q` de turbulence)
      - 'regression' : DataFrame (TSMOM ~ z(turbulence) ; et ~ dummy turbulent)
    """
    df = pd.concat([tsmom.rename("tsmom"), turbulence.rename("turb")], axis=1).dropna()

    # --- Perf par quantile de turbulence ---
    buckets = pd.qcut(df["turb"], n_buckets, labels=[f"Q{i+1}" for i in range(n_buckets)])
    rows = {}
    for q, sub in df.groupby(buckets, observed=True):
        s = _perf_stats(sub["tsmom"])
        rows[q] = {"N months": s["N months"], "Ann. mean": s["Ann. mean"],
                   "Sharpe": s["Sharpe"], "Avg turbulence": sub["turb"].mean()}
    by_bucket = pd.DataFrame(rows).T

    # --- Calme vs turbulent (top quantile) ---
    thr = df["turb"].quantile(1.0 - top_q)
    turb_mask = df["turb"] >= thr
    cvt = pd.DataFrame({
        "Calm (bottom %d%%)" % int((1 - top_q) * 100): _perf_stats(df.loc[~turb_mask, "tsmom"]),
        "Turbulent (top %d%%)" % int(top_q * 100): _perf_stats(df.loc[turb_mask, "tsmom"]),
    }).T[["N months", "Ann. mean", "Ann. vol", "Sharpe"]]

    # --- Régressions : TSMOM ~ z(turbulence) ; TSMOM ~ 1{turbulent} ---
    z = (df["turb"] - df["turb"].mean()) / df["turb"].std()
    reg_rows = {}
    for name, X in [("z(turbulence)", z.to_frame("X")),
                    ("1{turbulent top %d%%}" % int(top_q * 100),
                     turb_mask.astype(float).to_frame("X"))]:
        Xc = sm.add_constant(X)
        m = sm.OLS(df["tsmom"], Xc).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        reg_rows[name] = {"Intercept (%/m)": m.params["const"] * 100.0,
                          "t(Intercept)": m.tvalues["const"],
                          "beta(X) (%/m)": m.params["X"] * 100.0,
                          "t(X)": m.tvalues["X"], "R2": m.rsquared, "N": int(m.nobs)}
    regression = pd.DataFrame(reg_rows).T[["Intercept (%/m)", "t(Intercept)",
                                           "beta(X) (%/m)", "t(X)", "R2", "N"]]

    return {"by_bucket": by_bucket, "calm_vs_turbulent": cvt, "regression": regression}
