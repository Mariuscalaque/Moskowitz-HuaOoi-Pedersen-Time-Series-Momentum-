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

CORRECTIONS (revue méthodologique) :
  - filter_marginal : le diagnostic résiduel est désormais un VRAI Ljung-Box sur
    les résidus standardisés AU CARRÉ (test d'ARCH résiduel), et non un test KS
    contre la normale qui était à la fois mal nommé et non pertinent.
  - exceedance_correlation : ajout du BENCHMARK GAUSSIEN (Boyer-Gibson-Loretan).
    Conditionner sur deux grands mouvements gonfle mécaniquement la corrélation,
    même sous une normale bivariée à corrélation constante. On simule donc une
    normale de même ρ inconditionnel et on lui applique le même conditionnement :
    seul l'ÉCART empirique − gaussien est interprétable comme dépendance de queue.
  - cvm_gof : l'échantillon Monte-Carlo de référence Student-t est tiré UNE fois
    (et non à chaque itération bootstrap) -> beaucoup plus rapide.
  - rolling_copula : fenêtre par défaut portée à 60 mois (la dépendance de queue
    est ininterprétable sur 36 obs) ; option `refilter` pour refiltrer les marges
    DANS chaque fenêtre (zéro fuite d'information future) en plus du filtrage
    global standard (type Patton 2006), descriptif et assumé.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy import stats, optimize
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.distributions.copula.api import (
    GaussianCopula, StudentTCopula, ClaytonCopula, GumbelCopula, FrankCopula,
)

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# (1) Filtrage des marges : AR(1)-GARCH(1,1) à innovations Student-t
# ----------------------------------------------------------------------
def filter_marginal(series: pd.Series, lb_lags: int = 10) -> tuple[pd.Series, dict]:
    """Renvoie les résidus standardisés iid et un résumé du modèle.

    arch travaille mieux en échelle ~%, on multiplie par 100 puis on garde
    les résidus standardisés (sans unité, donc l'échelle est neutralisée).

    Diagnostic : Ljung-Box sur les résidus standardisés AU CARRÉ. H0 = pas
    d'autocorrélation des carrés = pas d'ARCH résiduel. Une p-value ÉLEVÉE
    (> 0.05) indique donc que le GARCH a bien capté le clustering de volatilité.
    """
    s = series.dropna() * 100.0
    am = arch_model(s, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="t")
    res = am.fit(disp="off")
    std_resid = (res.resid / res.conditional_volatility).dropna()

    # Ljung-Box sur les résidus standardisés au carré -> ARCH résiduel
    try:
        lb = acorr_ljungbox(std_resid ** 2, lags=[lb_lags], return_df=True)
        lb_p = float(lb["lb_pvalue"].iloc[0])
    except Exception:
        lb_p = np.nan

    info = {
        "nu (innov.)": float(res.params.get("nu", np.nan)),
        "alpha+beta (persist.)": float(res.params.get("alpha[1]", np.nan)
                                        + res.params.get("beta[1]", np.nan)),
        f"LB resid^2 p (lag{lb_lags})": lb_p,
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
                           thresholds=np.arange(0.0, 1.4, 0.2),
                           n_sim: int = 200_000, seed: int = 7) -> pd.DataFrame:
    """Corrélations d'exceedance (Longin-Solnik / Ang-Chen) sur résidus
    standardisés, AVEC benchmark gaussien (Boyer-Gibson-Loretan).

    Pour chaque seuil c :
      - côté négatif : corr(x,y | x<-c & y<-c)  (krachs joints)
      - côté positif : corr(x,y | x>+c & y>+c)  (booms joints)
    Conditionner sur de grands mouvements gonfle MÉCANIQUEMENT la corrélation,
    même sous une normale bivariée à corrélation constante. On simule donc une
    normale de même ρ inconditionnel et on lui applique le même conditionnement.
    Colonnes *_gauss = ce que produirait l'absence de dépendance de queue ;
    seul l'écart (corr_neg − corr_neg_gauss) est interprétable comme un straddle.
    Un straddle ASYMÉTRIQUE => excès côté négatif > excès côté positif.
    """
    xs = ((x - x.mean()) / x.std()).to_numpy()
    ys = ((y - y.mean()) / y.std()).to_numpy()
    rho = float(np.corrcoef(xs, ys)[0, 1])

    # Échantillon normal bivarié de même corrélation -> benchmark sans tail-dep
    rng = np.random.default_rng(seed)
    g = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n_sim)
    gx, gy = g[:, 0], g[:, 1]

    def _cond_corr(a, b, mask):
        return float(np.corrcoef(a[mask], b[mask])[0, 1]) if mask.sum() > 5 else np.nan

    rows = []
    for c in thresholds:
        neg = (xs < -c) & (ys < -c)
        pos = (xs > c) & (ys > c)
        gneg = (gx < -c) & (gy < -c)
        gpos = (gx > c) & (gy > c)
        rows.append({
            "threshold": c,
            "corr_neg": _cond_corr(xs, ys, neg), "n_neg": int(neg.sum()),
            "corr_pos": _cond_corr(xs, ys, pos), "n_pos": int(pos.sum()),
            "corr_neg_gauss": _cond_corr(gx, gy, gneg),
            "corr_pos_gauss": _cond_corr(gx, gy, gpos),
            "rho": rho,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Test d'adéquation Cramér-von Mises (Sn) + bootstrap paramétrique
# ----------------------------------------------------------------------
def _emp_copula(u, points):
    U, V = u[:, 0], u[:, 1]
    return np.array([np.mean((U <= a) & (V <= b)) for a, b in points])


def cvm_gof(u: np.ndarray, family: str, fitted: dict, n_boot: int = 200,
            seed: int = 12345, mc_ref: int = 60_000) -> dict:
    """Statistique Sn = somme (C_n - C_theta)² aux pseudo-obs, p-value par
    bootstrap paramétrique. p-value haute => on NE rejette PAS la copule.

    Perf : pour la Student-t (CDF sans forme fermée) l'échantillon Monte-Carlo
    de référence est tiré UNE seule fois (la copule de référence est fixe), au
    lieu d'être régénéré à chaque itération bootstrap.
    """
    rng = np.random.default_rng(seed)
    n = len(u)
    pts = u

    # Référence Monte-Carlo pour la CDF Student-t (copule de H0 fixe) — tirée 1x
    studt_ref = None
    if family == "Student-t":
        studt_ref = StudentTCopula(corr=fitted["rho"], df=fitted["df"]).rvs(
            mc_ref, random_state=rng)

    def cdf_of(fam, par, P):
        if fam == "Gaussian":
            return GaussianCopula(corr=par["rho"]).cdf(P)
        if fam == "Student-t":
            A, B = studt_ref[:, 0][None, :], studt_ref[:, 1][None, :]
            return np.mean((A <= P[:, 0][:, None]) & (B <= P[:, 1][:, None]), axis=1)
        cls = {"Clayton": ClaytonCopula, "Gumbel": GumbelCopula, "Frank": FrankCopula}[fam]
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


# ----------------------------------------------------------------------
# Extension B — Copule roulante (rolling window)
# ----------------------------------------------------------------------
def rolling_copula(z1: pd.Series, z2: pd.Series, window: int = 60,
                   refilter: bool = False) -> pd.DataFrame:
    """Estime la meilleure copule sur des fenêtres roulantes de `window` mois.

    DEUX modes (cf. revue méthodologique) :

      refilter=False (DÉFAUT) — z1/z2 sont des RÉSIDUS déjà filtrés AR-GARCH-t
        sur TOUT l'échantillon (filtrage marginal global, standard type
        Patton 2006). Seule la copule roule. C'est un exercice DESCRIPTIF a
        posteriori : le GARCH global utilise l'information plein-échantillon,
        ce qui est assumé (aucune prétention « temps réel »). Plus stable car
        le GARCH n'est jamais estimé sur une poignée de points.

      refilter=True — z1/z2 sont les SÉRIES BRUTES ; on refiltre AR(1)-GARCH(1,1)-t
        DANS chaque fenêtre. AUCUNE fuite d'information future, au prix d'un GARCH
        estimé sur `window` points seulement (plus bruité) et d'un calcul plus lent.
        En cas de non-convergence du GARCH sur une fenêtre, repli sur une
        standardisation simple (centrage/réduction) de la fenêtre.

    `window` >= 60 est recommandé : la dépendance de queue est ininterprétable
    sur ~36 obs (3-4 points par queue à 10 %).

    Retourne un DataFrame indexé par la DATE DE FIN de chaque fenêtre avec :
      best_family, lambda_L/U (copule retenue), lambda_L_t (Student-t),
      lambda_L_cl (Clayton), AIC_best, AIC_gauss.
    """
    df = pd.concat([z1, z2], axis=1).dropna()
    n_total = len(df)
    records = []

    def _window_pseudo_obs(sub: pd.DataFrame) -> np.ndarray:
        if not refilter:
            return sub.rank().to_numpy() / (len(sub) + 1.0)
        # refiltrage par fenêtre (zéro look-ahead)
        cols = []
        for c in sub.columns:
            try:
                r, _ = filter_marginal(sub[c])
            except Exception:
                v = sub[c]
                r = (v - v.mean()) / v.std()        # repli si GARCH échoue
            cols.append(r.rename(c))
        sub2 = pd.concat(cols, axis=1).dropna()
        return sub2.rank().to_numpy() / (len(sub2) + 1.0)

    for i in range(window, n_total + 1):
        sub = df.iloc[i - window: i]
        u = _window_pseudo_obs(sub)
        fit = fit_all_copulas(u)
        best = fit["AIC"].astype(float).idxmin()
        records.append({
            "date": sub.index[-1],
            "best_family": best,
            "lambda_L": float(fit.loc[best, "lambda_L"]),
            "lambda_U": float(fit.loc[best, "lambda_U"]),
            "lambda_L_t": float(fit.loc["Student-t", "lambda_L"]),
            "lambda_L_cl": float(fit.loc["Clayton", "lambda_L"]),
            "AIC_best": float(fit.loc[best, "AIC"]),
            "AIC_gauss": float(fit.loc["Gaussian", "AIC"]),
        })
        if (i - window) % 24 == 0:
            pct = (i - window) / max(n_total - window, 1) * 100
            print(f"  rolling copula : {pct:5.1f}%  ({sub.index[-1]:%Y-%m})", flush=True)
    result = pd.DataFrame(records).set_index("date")
    print("  rolling copula : 100.0%  — terminé")
    return result
