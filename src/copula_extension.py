"""
Extension A — Le « straddle » de TSMOM comme DÉPENDANCE DE QUEUE.

Idée : le test paramétrique du papier (régression sur MKT²) est symétrique,
global et dilue l'effet. On le remplace par une mesure non-paramétrique et
focalisée sur les extrêmes : la dépendance de queue entre les rendements TSMOM
et les rendements du marché.

Procédure rigoureuse en 2 étapes (IFM / méthode des marges) — c'est ce qui
sépare une vraie application d'un placage naïf :
  (1) Filtrer CHAQUE marge par AR(1)-GARCH(1,1) à innovations Student-t
      -> résidus standardisés approximativement iid.
  (2) PIT non-paramétrique (rang/(n+1)) -> pseudo-observations uniformes.
  (3) Ajuster les copules sur ces pseudo-observations, sélection AIC/BIC,
      test d'adéquation (Cramér-von Mises) et report explicite de λ_L, λ_U.

Prédiction testable du straddle : λ_L > 0 ET λ_U > 0 (co-mouvement fort aux
DEUX extrêmes), avec asymétrie possible (protection plus forte dans les krachs).
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy import stats, optimize
from arch import arch_model
from statsmodels.distributions.copula.api import (
    GaussianCopula, StudentTCopula, ClaytonCopula, GumbelCopula, FrankCopula,
)

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# (1) Filtrage des marges : AR(1)-GARCH(1,1) à innovations Student-t
# ----------------------------------------------------------------------
def filter_marginal(series: pd.Series) -> tuple[pd.Series, dict]:
    """Renvoie les résidus standardisés iid et un résumé du modèle.
    arch travaille mieux en échelle ~%, on multiplie par 100 puis on garde
    les résidus standardisés (sans unité, donc l'échelle est neutralisée)."""
    s = series.dropna() * 100.0
    am = arch_model(s, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="t")
    res = am.fit(disp="off")
    std_resid = (res.resid / res.conditional_volatility).dropna()
    # Ljung-Box sur résidus² : reste-t-il du clustering ?
    lb = stats.kstest((std_resid - std_resid.mean()) / std_resid.std(),
                      "norm").pvalue
    info = {
        "nu (innov.)": float(res.params.get("nu", np.nan)),
        "alpha+beta (persist.)": float(res.params.get("alpha[1]", np.nan)
                                        + res.params.get("beta[1]", np.nan)),
        "KS resid p": lb,
    }
    return std_resid, info


def to_pseudo_obs(*series: pd.Series) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Aligne les séries, applique la PIT empirique rang/(n+1) -> uniformes."""
    df = pd.concat(series, axis=1).dropna()
    u = df.rank().to_numpy() / (len(df) + 1.0)
    return u, df.index


# ----------------------------------------------------------------------
# Coefficients de dépendance de queue ANALYTIQUES par famille
# ----------------------------------------------------------------------
def tail_dependence(family: str, params: dict) -> tuple[float, float]:
    if family == "Gaussian":
        return 0.0, 0.0
    if family == "Student-t":
        rho, nu = params["rho"], params["df"]
        x = -np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
        lam = 2 * stats.t.cdf(x, df=nu + 1)
        return lam, lam                       # symétrique
    if family == "Clayton":
        th = params["theta"]
        return 2.0 ** (-1.0 / th), 0.0        # queue BASSE seulement
    if family == "Gumbel":
        th = params["theta"]
        return 0.0, 2.0 - 2.0 ** (1.0 / th)   # queue HAUTE seulement
    if family == "Frank":
        return 0.0, 0.0                        # pas de dépendance de queue
    return np.nan, np.nan


# ----------------------------------------------------------------------
# (3) Estimation des copules par MLE + AIC/BIC
# ----------------------------------------------------------------------
def _archimedean_nll(theta, u, family):
    cls = {"Clayton": ClaytonCopula, "Gumbel": GumbelCopula, "Frank": FrankCopula}[family]
    try:
        lp = cls(theta=theta).logpdf(u)
        if not np.all(np.isfinite(lp)):
            return 1e10
        return -lp.sum()
    except Exception:
        return 1e10


def fit_all_copulas(u: np.ndarray) -> pd.DataFrame:
    n = len(u)
    out = {}

    def record(name, ll, k, params):
        lL, lU = tail_dependence(name, params)
        out[name] = {
            "logL": ll, "AIC": -2 * ll + 2 * k, "BIC": -2 * ll + k * np.log(n),
            "lambda_L": lL, "lambda_U": lU, "param": _fmt(params), "N": n,
        }

    # Gaussienne (H0 : λ = 0)
    rho = float(np.ravel(GaussianCopula().fit_corr_param(u))[0])
    llG = GaussianCopula(corr=rho).logpdf(u).sum()
    record("Gaussian", llG, 1, {"rho": rho})

    # Student-t : MLE sur (rho, nu) — dépendance de queue SYMÉTRIQUE
    def nll_t(p):
        r, nu = p
        if not (-0.999 < r < 0.999) or nu <= 2.1:
            return 1e10
        try:
            lp = StudentTCopula(corr=r, df=nu).logpdf(u)
            return -lp.sum() if np.all(np.isfinite(lp)) else 1e10
        except Exception:
            return 1e10
    rt = optimize.minimize(nll_t, x0=[rho, 8.0], method="Nelder-Mead")
    record("Student-t", -rt.fun, 2, {"rho": rt.x[0], "df": rt.x[1]})

    # Archimédiennes : Clayton (queue basse), Gumbel (queue haute), Frank (aucune)
    for fam, b in [("Clayton", (1e-3, 30)), ("Gumbel", (1.0001, 30)), ("Frank", (1e-3, 50))]:
        r = optimize.minimize_scalar(_archimedean_nll, bounds=b, args=(u, fam),
                                     method="bounded")
        record(fam, -r.fun, 1, {"theta": r.x})

    df = pd.DataFrame(out).T
    return df[["logL", "AIC", "BIC", "lambda_L", "lambda_U", "param", "N"]]


def _fmt(p):
    return ", ".join(f"{k}={v:.3f}" for k, v in p.items())


# ----------------------------------------------------------------------
# Dépendance de queue NON-PARAMÉTRIQUE (model-free) + asymétrie
# ----------------------------------------------------------------------
def empirical_tail_dependence(u: np.ndarray, q: float = 0.10) -> dict:
    """Estimateurs empiriques :
        λ_L(q) = C_n(q,q)/q          (co-krach)
        λ_U(q) = (1-2q+C_n(1-q,1-q))/q  (co-boom)
    """
    U, V = u[:, 0], u[:, 1]
    Cqq = np.mean((U <= q) & (V <= q))
    Cuu = np.mean((U <= 1 - q) & (V <= 1 - q))
    lamL = Cqq / q
    lamU = (1 - 2 * (1 - q) + Cuu) / q
    return {"q": q, "lambda_L_emp": lamL, "lambda_U_emp": lamU,
            "asymmetry_L_minus_U": lamL - lamU}


def exceedance_correlation(x: pd.Series, y: pd.Series,
                           thresholds=np.arange(0.0, 1.4, 0.2)) -> pd.DataFrame:
    """Corrélations d'exceedance (Longin-Solnik / Ang-Chen) sur résidus
    standardisés. Pour chaque seuil c :
      - côté négatif : corr(x,y | x<-c & y<-c)  (krachs joints)
      - côté positif : corr(x,y | x>+c & y>+c)  (booms joints)
    Un straddle ASYMÉTRIQUE => corrélation négative-extrême > positive-extrême.
    """
    xs = (x - x.mean()) / x.std()
    ys = (y - y.mean()) / y.std()
    rows = []
    for c in thresholds:
        neg = (xs < -c) & (ys < -c)
        pos = (xs > c) & (ys > c)
        rn = np.corrcoef(xs[neg], ys[neg])[0, 1] if neg.sum() > 5 else np.nan
        rp = np.corrcoef(xs[pos], ys[pos])[0, 1] if pos.sum() > 5 else np.nan
        rows.append({"threshold": c, "corr_neg": rn, "n_neg": int(neg.sum()),
                     "corr_pos": rp, "n_pos": int(pos.sum())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Test d'adéquation Cramér-von Mises (Sn) + bootstrap paramétrique
# ----------------------------------------------------------------------
def _emp_copula(u, points):
    U, V = u[:, 0], u[:, 1]
    return np.array([np.mean((U <= a) & (V <= b)) for a, b in points])


def cvm_gof(u: np.ndarray, family: str, fitted: dict, n_boot: int = 200,
            seed: int = 12345) -> dict:
    """Statistique Sn = somme (C_n - C_theta)² aux pseudo-obs, p-value par
    bootstrap paramétrique. p-value haute => on NE rejette PAS la copule."""
    rng = np.random.default_rng(seed)
    n = len(u)
    pts = u

    def cdf_of(family, par, P):
        if family == "Gaussian":
            return GaussianCopula(corr=par["rho"]).cdf(P)
        if family == "Student-t":
            # CDF non close-form -> approximation Monte-Carlo (échantillon fixe)
            S = StudentTCopula(corr=par["rho"], df=par["df"]).rvs(40000, random_state=rng)
            A, B = S[:, 0][None, :], S[:, 1][None, :]
            return np.mean((A <= P[:, 0][:, None]) & (B <= P[:, 1][:, None]), axis=1)
        cls = {"Clayton": ClaytonCopula, "Gumbel": GumbelCopula, "Frank": FrankCopula}[family]
        return cls(theta=par["theta"]).cdf(P)

    Cn = _emp_copula(u, pts)
    Cth = cdf_of(family, fitted, pts)
    Sn = np.sum((Cn - Cth) ** 2)

    def sampler(par):
        if family == "Gaussian":
            return GaussianCopula(corr=par["rho"]).rvs(n, random_state=rng)
        if family == "Student-t":
            return StudentTCopula(corr=par["rho"], df=par["df"]).rvs(n, random_state=rng)
        cls = {"Clayton": ClaytonCopula, "Gumbel": GumbelCopula, "Frank": FrankCopula}[family]
        return cls(theta=par["theta"]).rvs(n, random_state=rng)

    count = 0
    for _ in range(n_boot):
        ub = sampler(fitted)
        ub = stats.rankdata(ub, axis=0) / (n + 1.0)
        Cnb = _emp_copula(ub, ub)
        Cthb = cdf_of(family, fitted, ub)
        if np.sum((Cnb - Cthb) ** 2) >= Sn:
            count += 1
    return {"family": family, "Sn": Sn, "p_value": (count + 0.5) / (n_boot + 1)}
