"""
Statistical analysis: pooled lag regressions (Fig. 1),
trading-strategy alpha grid (Table 2), and factor regressions (Table 3).
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

from .strategy import past_k_month_returns, tsmom_instrument_returns, diversified_tsmom
from .config import TARGET_VOL


# =======================================================================
# Fig. 1 — pooled lag regressions of vol-scaled returns on lagged signals
# =======================================================================

def pooled_lag_regression_size(monthly_ret: pd.DataFrame,
                                monthly_vol: pd.DataFrame,
                                max_lag: int = 60) -> pd.DataFrame:
    """
    Panel A of Fig. 1:
        r^s_t / σ^s_{t-1}   =   α  +  β_h · ( r^s_{t-h} / σ^s_{t-h-1} )   +   ε

    Pooled across all instruments and dates; clustered SE by month.

    Returns
    -------
    DataFrame indexed by lag h (1..max_lag) with columns beta, tstat.
    """
    scaled = monthly_ret / monthly_vol.shift(1)  # use σ_{t-1} for time-t return
    out = []
    for h in range(1, max_lag + 1):
        y = scaled.stack(future_stack=True).rename("y")
        x = scaled.shift(h).stack(future_stack=True).rename("x")
        df = pd.concat([y, x], axis=1).dropna()
        if len(df) < 100:
            out.append({"lag": h, "beta": np.nan, "tstat": np.nan, "nobs": len(df)})
            continue
        # cluster SE by time (level 0 of the original index = date)
        groups = df.index.get_level_values(0)
        X = sm.add_constant(df["x"])
        model = sm.OLS(df["y"], X).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )
        out.append({
            "lag": h,
            "beta": model.params["x"],
            "tstat": model.tvalues["x"],
            "nobs": int(model.nobs),
        })
    return pd.DataFrame(out).set_index("lag")


def pooled_lag_regression_sign(monthly_ret: pd.DataFrame,
                                monthly_vol: pd.DataFrame,
                                max_lag: int = 60) -> pd.DataFrame:
    """
    Panel B of Fig. 1:
        r^s_t / σ^s_{t-1}   =   α  +  β_h · sign( r^s_{t-h} )   +   ε
    """
    scaled = monthly_ret / monthly_vol.shift(1)
    signs = np.sign(monthly_ret).replace(0, np.nan)
    out = []
    for h in range(1, max_lag + 1):
        y = scaled.stack(future_stack=True).rename("y")
        x = signs.shift(h).stack(future_stack=True).rename("x")
        df = pd.concat([y, x], axis=1).dropna()
        if len(df) < 100:
            out.append({"lag": h, "beta": np.nan, "tstat": np.nan, "nobs": len(df)})
            continue
        groups = df.index.get_level_values(0)
        X = sm.add_constant(df["x"])
        model = sm.OLS(df["y"], X).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )
        out.append({
            "lag": h,
            "beta": model.params["x"],
            "tstat": model.tvalues["x"],
            "nobs": int(model.nobs),
        })
    return pd.DataFrame(out).set_index("lag")


# =======================================================================
# Table 2 — alphas for (k, h) trading strategies
# =======================================================================

def tsmom_strategy_returns_kh(monthly_ret: pd.DataFrame,
                               monthly_vol: pd.DataFrame,
                               k: int,
                               h: int,
                               target_vol: float = TARGET_VOL) -> pd.Series:
    """
    Build TSMOM portfolio with lookback k and holding period h.

    Following Jegadeesh-Titman, the time-t return is the average of the
    h currently-active sub-portfolios (one opened last month, one two
    months ago, etc.). No overlap, single time series of monthly returns.
    """
    past_k = past_k_month_returns(monthly_ret, k=k)
    sig = np.sign(past_k).where(past_k.notna())
    scale = (target_vol / monthly_vol).replace([np.inf, -np.inf], np.nan)
    pos_t = sig * scale  # set at end of month t for month t+1

    # The sub-portfolio "opened at end of month t-j" earns r_{t+1} with position pos_{t-j}.
    # Equivalently: position used for r_{t+1} is the AVERAGE of pos_t, pos_{t-1}, ..., pos_{t-h+1}.
    avg_pos = pos_t.rolling(window=h, min_periods=1).mean()
    inst_ret = avg_pos.shift(1) * monthly_ret
    return inst_ret.mean(axis=1)


def factor_alpha_tstat(strategy_returns: pd.Series,
                        factors: pd.DataFrame) -> tuple:
    """
    Regress strategy returns on factors and return (alpha, tstat_alpha, R^2).
    Newey-West HAC SE (3 lags, monthly).
    """
    df = pd.concat([strategy_returns.rename("y"), factors], axis=1).dropna()
    if len(df) < 24:
        return (np.nan, np.nan, np.nan)
    X = sm.add_constant(df[factors.columns])
    model = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    return (model.params["const"], model.tvalues["const"], model.rsquared)


def table2_grid(monthly_ret: pd.DataFrame,
                 monthly_vol: pd.DataFrame,
                 factors: pd.DataFrame,
                 k_grid=(1, 3, 6, 9, 12, 24, 36, 48),
                 h_grid=(1, 3, 6, 9, 12, 24, 36, 48)) -> pd.DataFrame:
    """
    Build the (k, h) grid of t-stats on the alpha of each TSMOM strategy.
    Mirrors Panel A of Table 2 in the paper.
    """
    grid = pd.DataFrame(index=k_grid, columns=h_grid, dtype=float)
    grid.index.name = "Lookback (k)"
    grid.columns.name = "Holding (h)"
    for k in k_grid:
        for h in h_grid:
            r = tsmom_strategy_returns_kh(monthly_ret, monthly_vol, k=k, h=h)
            alpha, tstat, _ = factor_alpha_tstat(r, factors)
            grid.loc[k, h] = tstat
    return grid


# =======================================================================
# Table 3 — performance of diversified TSMOM
# =======================================================================

def factor_regression_summary(strategy_returns: pd.Series,
                               factors: pd.DataFrame,
                               freq: str = "monthly") -> dict:
    """
    Run regression of strategy on factors. Returns dict with coefficients,
    t-stats, alpha (monthly or quarterly), R², and Sharpe.
    """
    df = pd.concat([strategy_returns.rename("y"), factors], axis=1).dropna()
    X = sm.add_constant(df[factors.columns])
    model = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    out = {
        "freq": freq,
        "alpha": model.params["const"],
        "alpha_tstat": model.tvalues["const"],
        "betas": {c: model.params[c] for c in factors.columns},
        "betas_tstat": {c: model.tvalues[c] for c in factors.columns},
        "R2": model.rsquared,
        "N": int(model.nobs),
    }
    return out


# =======================================================================
# Misc — Sharpe ratios per instrument (Fig. 2)
# =======================================================================

def annualized_sharpe(returns: pd.Series, ppy: int = 12) -> float:
    r = returns.dropna()
    if r.std() == 0 or len(r) < 12:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(ppy))


def sharpe_by_instrument(instrument_tsmom: pd.DataFrame) -> pd.Series:
    return instrument_tsmom.apply(annualized_sharpe).sort_values(ascending=False)


# =======================================================================
# Table 3 (version complète, 6 facteurs) — format de l'article
# =======================================================================

def _to_quarterly(monthly_returns: pd.Series) -> pd.Series:
    """Rendements trimestriels non chevauchants (composition)."""
    return (1.0 + monthly_returns).resample("QE").prod() - 1.0


def _quarterly_factors(factors: pd.DataFrame) -> pd.DataFrame:
    """Composition trimestrielle non chevauchante des facteurs."""
    return (1.0 + factors).resample("QE").prod() - 1.0


def table3_full(diversified_tsmom_ret: pd.Series,
                factors: pd.DataFrame,
                hac_lags_monthly: int = 3,
                hac_lags_quarterly: int = 1) -> pd.DataFrame:
    """
    Reproduit la Table 3 du papier : régression du TSMOM diversifié (en excès)
    sur MKT, BOND, GSCI, SMB, HML, UMD — en mensuel (ligne 1) et trimestriel
    non chevauchant (ligne 2). SE Newey-West (HAC).

    Renvoie un DataFrame : lignes = {Monthly, Quarterly},
    colonnes = [Alpha, t(Alpha), MKT, t(MKT), ..., UMD, t(UMD), R2, N].
    L'alpha est exprimé en % (×100) pour comparer aux 1.58%/mois du papier.
    """
    cols_order = list(factors.columns)
    results = {}

    # ---- Mensuel ----
    dfm = pd.concat([diversified_tsmom_ret.rename("y"), factors], axis=1).dropna()
    Xm = sm.add_constant(dfm[cols_order])
    mm = sm.OLS(dfm["y"], Xm).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags_monthly})

    # ---- Trimestriel non chevauchant ----
    yq = _to_quarterly(diversified_tsmom_ret)
    fq = _quarterly_factors(factors)
    dfq = pd.concat([yq.rename("y"), fq], axis=1).dropna()
    Xq = sm.add_constant(dfq[cols_order])
    mq = sm.OLS(dfq["y"], Xq).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags_quarterly})

    def _row(model):
        row = {"Alpha (%)": model.params["const"] * 100.0,
               "t(Alpha)": model.tvalues["const"]}
        for c in cols_order:
            row[c] = model.params[c]
            row[f"t({c})"] = model.tvalues[c]
        row["R2"] = model.rsquared
        row["N"] = int(model.nobs)
        return row

    results["Monthly"] = _row(mm)
    results["Quarterly"] = _row(mq)
    out = pd.DataFrame(results).T
    # ordre des colonnes
    ordered = ["Alpha (%)", "t(Alpha)"]
    for c in cols_order:
        ordered += [c, f"t({c})"]
    ordered += ["R2", "N"]
    return out[ordered]


def diversified_performance(diversified_tsmom_ret: pd.Series,
                            ppy: int = 12) -> dict:
    """Stats de performance brute du TSMOM diversifié (pour le résumé)."""
    r = diversified_tsmom_ret.dropna()
    mean_a = r.mean() * ppy
    vol_a = r.std() * np.sqrt(ppy)
    sharpe = mean_a / vol_a if vol_a > 0 else np.nan
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    # skewness / kurtosis
    return {
        "ann_mean": mean_a,
        "ann_vol": vol_a,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "skew": r.skew(),
        "kurtosis": r.kurtosis(),
        "N_months": len(r),
    }


# ============================================================================
# EXTENSIONS — Tables 3B/3C, 4, 5, 6 ; event study (Fig 6) ; VAR/IRF (Fig 7)
# ============================================================================
from .strategy import tsmom_by_asset_class, passive_long  # noqa: E402
from .config import asset_class_of  # noqa: E402


# ---- Table 3, Panels B & C : il suffit de fournir la bonne matrice de facteurs
# et de réutiliser table3_full(...). On ajoute ici les CONSTRUCTEURS de facteurs.

def build_vme_factor_matrix(vme: pd.DataFrame,
                            mkt_excess: pd.Series | None = None) -> pd.DataFrame:
    """
    Table 3 Panel B : facteurs « Value & Momentum Everywhere » (AQR).

    Le papier régresse le TSMOM sur les facteurs DIVERSIFIÉS toutes classes :
    VAL (value everywhere) et MOM (momentum everywhere) — soit, dans le fichier
    AQR, les deux PREMIÈRES colonnes de facteurs.

    Robuste au cas où le CSV a perdu ses en-têtes (colonnes 'nan', 'Unnamed…') :
    on identifie VAL/MOM par nom si possible, sinon par POSITION (col 0 = VAL,
    col 1 = MOM, convention du fichier AQR). On renomme en VAL_EVR / MOM_EVR.
    """
    cols = list(vme.columns)
    val_col = mom_col = None
    # 1) par nom exact si disponible
    for c in cols:
        cl = str(c).strip().upper()
        if cl == "VAL" and val_col is None:
            val_col = c
        elif cl == "MOM" and mom_col is None:
            mom_col = c
    # 2) sinon par position (en-têtes perdus -> colonnes anonymes)
    if val_col is None or mom_col is None:
        anonymous = all(str(c).lower().startswith(("nan", "unnamed")) for c in cols)
        if anonymous and len(cols) >= 2:
            val_col, mom_col = cols[0], cols[1]
    if val_col is None or mom_col is None:
        raise ValueError(
            "Facteurs VME : impossible d'identifier VAL/MOM everywhere. "
            "Vérifiez le chargement du fichier AQR (en-têtes 'VAL','MOM')."
        )
    X = pd.DataFrame({
        "VAL_EVR": pd.to_numeric(vme[val_col], errors="coerce"),
        "MOM_EVR": pd.to_numeric(vme[mom_col], errors="coerce"),
    })
    if mkt_excess is not None:
        X = X.join(mkt_excess.rename("MKT"), how="inner")
        X = X[["MKT", "VAL_EVR", "MOM_EVR"]]
    return X.dropna()


def build_extremes_factor_matrix(mkt_excess: pd.Series,
                                 vix: pd.Series | None = None,
                                 ted: pd.Series | None = None,
                                 ps_liquidity: pd.Series | None = None,
                                 bw_sentiment: pd.Series | None = None) -> pd.DataFrame:
    """Table 3 Panel C : marché + extrêmes de volatilité / liquidité / sentiment.
    MKT, MKT² (convexité), ΔVIX, niveau TED, innovation de liquidité (PS),
    sentiment (BW). Les composantes absentes sont simplement omises."""
    X = pd.DataFrame({"MKT": mkt_excess})
    X["MKT_sq"] = mkt_excess ** 2
    if vix is not None:
        X["dVIX"] = vix.reindex(mkt_excess.index).pct_change()
    if ted is not None:
        X["TED"] = ted.reindex(mkt_excess.index)
    if ps_liquidity is not None:
        X["LIQ"] = ps_liquidity.reindex(mkt_excess.index)
    if bw_sentiment is not None:
        X["SENT"] = bw_sentiment.reindex(mkt_excess.index)
    return X.dropna()


# ---- Table 3 Panel C : régressions SÉPARÉES, en TRIMESTRIEL (fidélité au papier) ----

def table3_smile_quarterly(diversified_tsmom_ret: pd.Series,
                           mkt_excess: pd.Series,
                           hac_lags: int = 1) -> pd.DataFrame:
    """
    Régression « straddle / smile » du papier (Table 3 Panel C, 1re ligne) :
        TSMOM_q ~ MKT + MKT²   sur rendements TRIMESTRIELS non chevauchants.
    C'est la spécification EXACTE du papier (coeff. MKT² = 1.99, t = 3.88).
    Régression DÉDIÉE et séparée — le papier ne mélange pas les blocs extrêmes.
    """
    yq = _to_quarterly(diversified_tsmom_ret)
    mq = _to_quarterly(mkt_excess)
    df = pd.concat([yq.rename("y"), mq.rename("MKT")], axis=1).dropna()
    df["MKT_sq"] = df["MKT"] ** 2
    m = sm.OLS(df["y"], sm.add_constant(df[["MKT", "MKT_sq"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return pd.DataFrame({"Quarterly": {
        "Alpha (%)": m.params["const"] * 100.0, "t(Alpha)": m.tvalues["const"],
        "MKT": m.params["MKT"], "t(MKT)": m.tvalues["MKT"],
        "MKT_sq": m.params["MKT_sq"], "t(MKT_sq)": m.tvalues["MKT_sq"],
        "R2": m.rsquared, "N": int(m.nobs),
    }}).T[["Alpha (%)", "t(Alpha)", "MKT", "t(MKT)", "MKT_sq", "t(MKT_sq)", "R2", "N"]]


def table3_extremes_blocks(diversified_tsmom_ret: pd.Series,
                           vix_level: pd.Series | None = None,
                           ted: pd.Series | None = None,
                           liq: pd.Series | None = None,
                           sent_level: pd.Series | None = None,
                           sent_change: pd.Series | None = None,
                           hac_lags: int = 1) -> pd.DataFrame:
    """
    Table 3 Panel C — liquidité / volatilité / sentiment, chacun dans une
    régression SÉPARÉE en trimestriel (comme le papier), pas un seul modèle
    fourre-tout. Chaque bloc : TSMOM_q ~ X (standardisé) + X·1{top 20%}
    (+ X·1{bottom 20%} pour le sentiment). Le papier utilise le NIVEAU du VIX
    (et non ΔVIX). Tous ces effets sont attendus NON significatifs : c'est le
    message du papier (TSMOM n'est pas expliqué par la liquidité/le sentiment).
    """
    yq = _to_quarterly(diversified_tsmom_ret)
    rows = {}

    def _block(name, raw, add_bottom=False):
        if raw is None:
            return
        xq = raw.resample("QE").mean()  # niveau/innovation trimestriel moyen
        df = pd.concat([yq.rename("y"), xq.rename("X")], axis=1).dropna()
        if len(df) < 12 or df["X"].std() == 0:
            return
        z = (df["X"] - df["X"].mean()) / df["X"].std()
        top = (df["X"] >= df["X"].quantile(0.8)).astype(float)
        cols = {"X": z, "X_top20": z * top}
        if add_bottom:
            cols["X_bot20"] = z * (df["X"] <= df["X"].quantile(0.2)).astype(float)
        XX = pd.DataFrame(cols, index=df.index)
        m = sm.OLS(df["y"], sm.add_constant(XX)).fit(
            cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        r = {"Alpha (%)": m.params["const"] * 100.0, "t(Alpha)": m.tvalues["const"],
             "beta(X)": m.params["X"], "t(X)": m.tvalues["X"],
             "beta(top20%)": m.params["X_top20"], "t(top20%)": m.tvalues["X_top20"],
             "beta(bot20%)": (m.params["X_bot20"] if add_bottom else np.nan),
             "t(bot20%)": (m.tvalues["X_bot20"] if add_bottom else np.nan),
             "R2": m.rsquared, "N": int(m.nobs)}
        rows[name] = r

    _block("VIX (level)", vix_level)
    _block("TED spread", ted)
    _block("Liquidity (PS)", liq)
    _block("Sentiment (BW)", sent_level, add_bottom=True)
    _block("Chg Sentiment", sent_change, add_bottom=True)

    if not rows:
        return pd.DataFrame()
    cols = ["Alpha (%)", "t(Alpha)", "beta(X)", "t(X)", "beta(top20%)",
            "t(top20%)", "beta(bot20%)", "t(bot20%)", "R2", "N"]
    return pd.DataFrame(rows).T[cols]


# ---- Table 4 : corrélations intra- et inter-classes -------------------------

def avg_pairwise_corr(df: pd.DataFrame, min_periods: int = 24) -> float:
    """Corrélation moyenne par paires (hors diagonale) des colonnes.

    CORRECTION : on calcule la corrélation PAR PAIRE sur l'intersection propre à
    chaque paire (`DataFrame.corr(min_periods=...)` gère le pairwise-complete),
    puis on moyenne le triangle supérieur. Auparavant l'appelant passait un
    `df.dropna()` (lignes complètes), ce qui restreignait l'échantillon à la
    fenêtre où TOUS les instruments existent — pour les actions, cela tombait sur
    2004-2009 (66 mois, en pleine crise) et gonflait la corrélation à 0.67 contre
    0.37 dans le papier. En pairwise on retrouve ~0.05/0.51/0.25/0.16
    (Commodity/Equity/Bond/Currency), bien plus proches du papier.
    """
    if df.shape[1] < 2:
        return np.nan
    c = df.corr(min_periods=min_periods).values
    iu = np.triu_indices(c.shape[0], k=1)
    vals = c[iu]
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan


def table4_within_class(inst_tsmom: pd.DataFrame,
                        inst_passive: pd.DataFrame) -> pd.DataFrame:
    """Panel A : corrélation moyenne par paires, par classe, pour les stratégies
    TSMOM et pour les positions passives longues."""
    classes = {c: asset_class_of(c) for c in inst_tsmom.columns}
    rows = {}
    for ac in ("Commodity", "Equity", "Bond", "Currency"):
        cols = [c for c, a in classes.items() if a == ac]
        if len(cols) >= 2:
            # NE PAS faire .dropna() ici : on laisse avg_pairwise_corr calculer
            # la corrélation par paire sur l'intersection propre à chaque paire.
            rows[ac] = {
                "TSMOM strategies": avg_pairwise_corr(inst_tsmom[cols]),
                "Passive long": avg_pairwise_corr(inst_passive[cols]),
            }
    return pd.DataFrame(rows).T


def table4_across_class(tsmom_by_ac: pd.DataFrame,
                        passive_by_ac: pd.DataFrame) -> dict:
    """Panel B : matrices de corrélation inter-classes (stratégies équipondérées
    par classe), pour TSMOM et pour passive long. Corrélation pairwise (chaque
    paire de classes sur son intersection) plutôt que sur la fenêtre commune aux
    quatre classes, pour ne pas tronquer inutilement l'échantillon."""
    return {
        "TSMOM": tsmom_by_ac.corr(min_periods=24),
        "Passive long": passive_by_ac.corr(min_periods=24),
    }


# ---- Table 5 Panel A : régression TSMOM(classe) sur XSMOM(classe) -----------

def _reg_tsmom_on_xsmom(y: pd.Series, x: pd.Series) -> dict | None:
    """Régression HAC (3 lags) de TSMOM (y) sur XSMOM (x). None si trop court."""
    df = pd.concat([y.rename("y"), x.rename("XSMOM")], axis=1).dropna()
    if len(df) < 24:
        return None
    m = sm.OLS(df["y"], sm.add_constant(df[["XSMOM"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3})
    return {
        "Alpha (%)": m.params["const"] * 100,
        "t(Alpha)": m.tvalues["const"],
        "beta(XSMOM)": m.params["XSMOM"],
        "t(XSMOM)": m.tvalues["XSMOM"],
        "R2": m.rsquared, "N": int(m.nobs),
    }


def table5_tsmom_on_xsmom(tsmom_by_ac: pd.DataFrame,
                          xsmom_by_ac: pd.DataFrame,
                          tsmom_all: pd.Series | None = None,
                          xsmom_all: pd.Series | None = None) -> pd.DataFrame:
    """
    Table 5 Panel A : régression HAC de TSMOM sur XSMOM.

    La ligne ALL (TSMOM diversifié ~ XSMOM diversifié) est la régression-phare
    du papier (β≈0.66, t≈15.17, R²=44%) et figure EN PREMIER. Pour l'obtenir,
    fournir `tsmom_all` (TSMOM diversifié toutes classes) et `xsmom_all` (XSMOM
    diversifié) ; à défaut, on essaie une éventuelle colonne 'ALL' commune.
    Suivent les régressions par classe d'actifs.
    """
    rows = {}

    # --- ligne ALL (en premier, comme dans le papier) ---
    if tsmom_all is None and "ALL" in tsmom_by_ac.columns:
        tsmom_all = tsmom_by_ac["ALL"]
    if xsmom_all is None and "ALL" in xsmom_by_ac.columns:
        xsmom_all = xsmom_by_ac["ALL"]
    if tsmom_all is not None and xsmom_all is not None:
        r = _reg_tsmom_on_xsmom(tsmom_all, xsmom_all)
        if r is not None:
            rows["ALL"] = r

    # --- lignes par classe d'actifs ---
    for ac in ("Commodity", "Equity", "Bond", "Currency"):
        if ac in tsmom_by_ac.columns and ac in xsmom_by_ac.columns:
            r = _reg_tsmom_on_xsmom(tsmom_by_ac[ac], xsmom_by_ac[ac])
            if r is not None:
                rows[ac] = r

    return pd.DataFrame(rows).T


# ---- Table 5 Panel C : « quels facteurs TSMOM explique-t-il ? » -------------

def table5_what_tsmom_explains(targets: dict, tsmom: pd.Series) -> pd.DataFrame:
    """Régresse chaque série-cible (facteurs externes, indices hedge funds…)
    sur le facteur TSMOM (+ const), en HAC. Renvoie alpha, beta(TSMOM), R²."""
    rows = {}
    for name, series in targets.items():
        df = pd.concat([series.rename("y"), tsmom.rename("TSMOM")], axis=1).dropna()
        if len(df) < 24:
            continue
        X = sm.add_constant(df[["TSMOM"]])
        m = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        rows[name] = {
            "Alpha (%)": m.params["const"] * 100,
            "t(Alpha)": m.tvalues["const"],
            "beta(TSMOM)": m.params["TSMOM"],
            "t(TSMOM)": m.tvalues["TSMOM"],
            "R2": m.rsquared, "N": int(m.nobs),
        }
    return pd.DataFrame(rows).T


# ---- Table 6 : prédicteurs (spot / roll / positions spéculateurs) -----------

def _pooled_predictive(y: pd.DataFrame, regressors: dict) -> dict:
    """Régression pooled de y_{t+1} (rendement futures) sur les régresseurs
    fournis (chacun un DataFrame instrument×date, connu en t). SE clustées par
    date. Renvoie coefficients et t-stats."""
    parts = {"y": y.shift(-1).stack(future_stack=True)}
    for nm, x in regressors.items():
        parts[nm] = x.stack(future_stack=True)
    df = pd.DataFrame(parts).dropna()
    if len(df) < 100:
        return {}
    groups = df.index.get_level_values(0)
    X = sm.add_constant(df[list(regressors.keys())])
    m = sm.OLS(df["y"], X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    out = {"Intercept": m.params["const"], "t(Intercept)": m.tvalues["const"],
           "R2": m.rsquared, "N": int(m.nobs)}
    for nm in regressors:
        out[nm] = m.params[nm]
        out[f"t({nm})"] = m.tvalues[nm]
    return out


def table6_predictors(total_ret: pd.DataFrame,
                      sig_total: pd.DataFrame,
                      sig_spot: pd.DataFrame,
                      sig_roll: pd.DataFrame,
                      net_spec: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Plusieurs spécifications empilées (lignes), à la Table 6 :
      (1) Full TSMOM seul,
      (2) Spot MOM + Roll MOM,
      (3) + niveau et variation des positions nettes spéculateurs (si dispo),
      (4) + interactions Spot×Δpos et Roll×Δpos (si dispo).
    Les colonnes communes sont alignées ; les cases vides = régresseur absent.
    """
    specs = {}
    specs["(1) Full TSMOM"] = _pooled_predictive(total_ret, {"FullMOM": sig_total})
    specs["(2) Spot+Roll"] = _pooled_predictive(
        total_ret, {"SpotMOM": sig_spot, "RollMOM": sig_roll})

    # Garde-fou : un historique CFTC trop court (ex. 1 ligne) ne permet pas les
    # spécifications avec positions -> on les omet plutôt que d'afficher du vide.
    if net_spec is not None and len(net_spec.dropna(how="all")) < 24:
        net_spec = None

    if net_spec is not None:
        ns = net_spec.reindex_like(sig_total)
        d_ns = ns.diff()
        specs["(3) +Spec pos"] = _pooled_predictive(
            total_ret, {"SpotMOM": sig_spot, "RollMOM": sig_roll,
                        "NetSpec": ns, "dNetSpec": d_ns})
        specs["(4) +Interactions"] = _pooled_predictive(
            total_ret, {"SpotMOM": sig_spot, "RollMOM": sig_roll,
                        "dNetSpec": d_ns,
                        "Spot×dSpec": sig_spot * d_ns,
                        "Roll×dSpec": sig_roll * d_ns})
    return pd.DataFrame(specs).T


# ---- Figure 6 Panel A : event study des rendements --------------------------

def event_study_returns(monthly_ret: pd.DataFrame,
                        k: int = 12,
                        window_before: int = 12,
                        window_after: int = 36,
                        demean: bool = True) -> pd.DataFrame:
    """
    Pour chaque (instrument, t), on regarde le signe du rendement passé sur k
    mois, puis on suit le rendement CUMULÉ moyen de -window_before à +window_after
    mois autour de l'événement, conditionnellement au signe (positif/négatif).
    Renvoie un DataFrame indexé par le décalage d'événement, colonnes
    'positive' / 'negative' (rendement cumulé moyen, base 0 au temps 0).

    CORRECTION : `demean=True` retire de chaque instrument sa moyenne plein
    échantillon AVANT de cumuler, comme le papier (« we standardize the returns
    to have a zero mean across time and across the two groups »). Sans ce
    retrait, la dérive haussière inconditionnelle des futures domine et masque
    la structure continuation-puis-réversion : les deux courbes montent et la
    branche « négative » ne descend jamais. Avec, on retrouve le motif du papier
    (poursuite ~12 mois puis réversion partielle, symétrique pour les deux signes).
    """
    R = monthly_ret.sub(monthly_ret.mean()) if demean else monthly_ret
    past = past_k_month_returns(monthly_ret, k=k)   # signe sur rendements BRUTS
    sign = np.sign(past)
    pos_paths, neg_paths = [], []
    ncols = R.shape[1]
    arr = R.values
    sgn = sign.values
    T = len(R)
    offsets = range(-window_before, window_after + 1)

    for j in range(ncols):
        col = arr[:, j]
        s = sgn[:, j]
        for t in range(window_before, T - window_after):
            if np.isnan(s[t]):
                continue
            seg = col[t - window_before: t + window_after + 1]
            if np.isnan(seg).any():
                continue
            cum = np.cumsum(seg) - np.cumsum(seg)[window_before]  # base 0 au temps 0
            (pos_paths if s[t] > 0 else neg_paths).append(cum)

    res = pd.DataFrame(index=list(offsets))
    res["positive"] = np.nanmean(np.array(pos_paths), axis=0) if pos_paths else np.nan
    res["negative"] = np.nanmean(np.array(neg_paths), axis=0) if neg_paths else np.nan
    res.index.name = "event_month"
    return res


def event_study_positions(net_spec: pd.DataFrame,
                          monthly_ret: pd.DataFrame,
                          k: int = 12,
                          window_before: int = 12,
                          window_after: int = 36) -> pd.DataFrame:
    """Figure 6 Panel B : même event study mais sur la position nette des
    spéculateurs (niveau dé-moyené), conditionnellement au signe du momentum."""
    common = [c for c in net_spec.columns if c in monthly_ret.columns]
    if not common:
        return pd.DataFrame()
    past = past_k_month_returns(monthly_ret[common], k=k)
    sign = np.sign(past).reindex(net_spec.index)
    ns = net_spec[common].sub(net_spec[common].mean())   # dé-moyenné
    pos_paths, neg_paths = [], []
    idx = ns.index
    offsets = range(-window_before, window_after + 1)
    for c in common:
        s = sign[c].reindex(idx).values
        v = ns[c].values
        for t in range(window_before, len(idx) - window_after):
            if np.isnan(s[t]):
                continue
            seg = v[t - window_before: t + window_after + 1]
            if np.isnan(seg).any():
                continue
            (pos_paths if s[t] > 0 else neg_paths).append(seg)
    res = pd.DataFrame(index=list(offsets))
    res["positive"] = np.nanmean(np.array(pos_paths), axis=0) if pos_paths else np.nan
    res["negative"] = np.nanmean(np.array(neg_paths), axis=0) if neg_paths else np.nan
    res.index.name = "event_month"
    return res


# ---- Figure 7 : réponse impulsionnelle (VAR returns × positions) ------------

def _bivariate_irf_from_Ys(Ys: list, lags: int, horizon: int) -> np.ndarray | None:
    """Estime le VAR(p) bivarié à coefficients communs sur la liste de séries
    `Ys` (chacune un array T×2 [ret, pos]), et renvoie l'IRF cumulée à un choc
    de +1σ sur le rendement (Cholesky, rendement ordonné en premier).
    Renvoie un array (horizon+1)×2 [cum_return, cum_position] ou None."""
    k = 2
    X_list, Z_list = [], []
    for Y in Ys:
        T = len(Y)
        for t in range(lags, T):
            row = [1.0]
            for L in range(1, lags + 1):
                row.extend(Y[t - L])
            X_list.append(row)
            Z_list.append(Y[t])
    if len(X_list) < (1 + k * lags) + 5:
        return None
    X = np.asarray(X_list); Z = np.asarray(Z_list)
    B, *_ = np.linalg.lstsq(X, Z, rcond=None)
    resid = Z - X @ B
    Sigma = np.cov(resid.T)
    A = [B[1 + L * k:1 + (L + 1) * k, :].T for L in range(lags)]
    try:
        P = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        return None
    shock = P[:, 0]
    psi = [np.zeros((k, k)) for _ in range(horizon + 1)]
    psi[0] = np.eye(k)
    for h in range(1, horizon + 1):
        acc = np.zeros((k, k))
        for L in range(1, lags + 1):
            if h - L >= 0:
                acc += A[L - 1] @ psi[h - L]
        psi[h] = acc
    irf = np.array([psi[h] @ shock for h in range(horizon + 1)])
    return np.cumsum(irf, axis=0)


def _univariate_irf(series: pd.Series, lags: int, horizon: int) -> np.ndarray | None:
    """IRF cumulée d'un AR(p) sur une série unique (repli sans positions)."""
    try:
        from statsmodels.tsa.ar_model import AutoReg
    except Exception:
        return None
    s = series.dropna()
    if len(s) < lags + 12:
        return None
    phi = AutoReg(s, lags=lags, old_names=False).fit().params[1:1 + lags].values
    resp = np.zeros(horizon + 1); resp[0] = 1.0
    for h in range(1, horizon + 1):
        resp[h] = sum(phi[i] * resp[h - 1 - i] for i in range(lags) if h - 1 - i >= 0)
    return np.cumsum(resp)


def impulse_response(monthly_ret: pd.DataFrame,
                     net_spec: pd.DataFrame | None = None,
                     instrument: str | None = None,
                     horizon: int = 36,
                     lags: int = 24,
                     diff_positions: bool = True,
                     n_boot: int = 0,
                     ci: float = 0.90,
                     seed: int = 0) -> pd.DataFrame | None:
    """
    Réponse impulsionnelle cumulée à un choc de +1 écart-type sur le rendement.

    Conforme à MOP (2012, §6.2) : VAR mensuel bivarié [rendement, *variation* de
    la position nette des spéculateurs] avec 24 retards, décomposition de Cholesky
    rendement ordonné en premier. Estimation en PANEL À COEFFICIENTS COMMUNS :
    retards bornés À L'INTÉRIEUR de chaque instrument, puis empilés (MCO).

    Bandes de confiance (n_boot > 0) : CLUSTER BOOTSTRAP par instrument — on
    ré-échantillonne AVEC REMISE l'ensemble des instruments (l'unité de
    dépendance est l'instrument ; les retards restent bornés à chaque série),
    on ré-estime le VAR et on recalcule l'IRF. On renvoie les quantiles
    (1-ci)/2 et (1+ci)/2. Pour le repli univarié, bootstrap par blocs mobiles
    de la série moyenne. Colonnes *_lo / *_hi ajoutées si n_boot > 0.

    - net_spec fourni  -> VAR bivarié : cum_return + cum_position (+ bandes).
    - sinon            -> AR(p) univarié : cum_return (+ bandes).
    """
    rng = np.random.default_rng(seed)

    if net_spec is not None:
        cols = ([instrument] if instrument else
                [c for c in net_spec.columns if c in monthly_ret.columns])
        Ys = []
        for c in cols:
            pos = net_spec[c].diff() if diff_positions else net_spec[c]
            d = pd.concat([monthly_ret[c].rename("ret"),
                           pos.rename("pos")], axis=1).dropna()
            if len(d) > lags + 12:
                Ys.append(d.values)
        if Ys:
            cum = _bivariate_irf_from_Ys(Ys, lags, horizon)
            if cum is not None:
                out = pd.DataFrame({"cum_return": cum[:, 0],
                                    "cum_position": cum[:, 1]})
                out.index.name = "horizon"
                if n_boot and len(Ys) >= 3:
                    boots_r, boots_p = [], []
                    nI = len(Ys)
                    for _ in range(n_boot):
                        idx = rng.integers(0, nI, nI)            # instruments avec remise
                        bc = _bivariate_irf_from_Ys([Ys[i] for i in idx], lags, horizon)
                        if bc is not None:
                            boots_r.append(bc[:, 0]); boots_p.append(bc[:, 1])
                    if len(boots_r) >= max(20, n_boot // 5):
                        lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
                        R = np.array(boots_r); Pp = np.array(boots_p)
                        out["cum_return_lo"] = np.percentile(R, lo, axis=0)
                        out["cum_return_hi"] = np.percentile(R, hi, axis=0)
                        out["cum_position_lo"] = np.percentile(Pp, lo, axis=0)
                        out["cum_position_hi"] = np.percentile(Pp, hi, axis=0)
                return out
        # net_spec inexploitable -> repli univarié

    # ---- repli univarié : AR(p) sur le rendement moyen ----
    series = (monthly_ret[instrument].dropna() if instrument
              else monthly_ret.mean(axis=1).dropna())
    base = _univariate_irf(series, lags, horizon)
    if base is None:
        return None
    out = pd.DataFrame({"cum_return": base}); out.index.name = "horizon"
    if n_boot:
        # bootstrap par BLOCS MOBILES (préserve l'autocorrélation de la série)
        s = series.values; T = len(s); blk = max(lags, 12)
        boots = []
        for _ in range(n_boot):
            chunks, acc = [], 0
            while acc < T:
                start = rng.integers(0, T - blk + 1)
                chunks.append(s[start:start + blk]); acc += blk
            bs = pd.Series(np.concatenate(chunks)[:T])
            bi = _univariate_irf(bs, lags, horizon)
            if bi is not None:
                boots.append(bi)
        if len(boots) >= max(20, n_boot // 5):
            lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
            Bm = np.array(boots)
            out["cum_return_lo"] = np.percentile(Bm, lo, axis=0)
            out["cum_return_hi"] = np.percentile(Bm, hi, axis=0)
    return out