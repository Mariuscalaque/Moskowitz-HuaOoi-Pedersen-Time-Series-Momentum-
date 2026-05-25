"""Exécute l'Extension A sur les données réelles du projet."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import copula_extension as ce

PROJ = "/mnt/project"

# ---- 1. Charger TSMOM diversifié (déjà produit par ta réplication) ----
tsmom = (pd.read_csv(f"{PROJ}/diversified_tsmom_series.csv", parse_dates=["date"])
         .set_index("date")["TSMOM"].dropna())

# ---- 2. Marché en excès = MXWO - RF, MÊME définition que ton test du smile ----
mx = (pd.read_csv(f"{PROJ}/data_monthly_returns.csv", parse_dates=["date"])
      .set_index("date")["MXWO Index"])
_raw = open(f"{PROJ}/FF_Research_Data_Factors.csv").read().replace("\r", "\n")
_lines = [ln for ln in _raw.split("\n") if ln.strip()]
_hdr = next(i for i, ln in enumerate(_lines) if ln.lstrip().startswith(",Mkt-RF"))
import io
ff = pd.read_csv(io.StringIO("\n".join(_lines[_hdr:])), skipinitialspace=True)
ff.columns = [str(c).strip() for c in ff.columns]
ff = ff.rename(columns={ff.columns[0]: "ym"})
ff = ff[ff["ym"].astype(str).str.match(r"^\s*\d{6}\s*$", na=False)].copy()
ff["date"] = pd.to_datetime(ff["ym"].astype(str).str.strip(), format="%Y%m") \
               .dt.to_period("M").dt.to_timestamp("M")
ff = ff.set_index("date")
rf = pd.to_numeric(ff["RF"], errors="coerce") / 100.0
mkt = (mx - rf.reindex(mx.index)).dropna().rename("MKT")

# ---- Filtrage AR(1)-GARCH(1,1)-t des deux marges ----
print("=== (1) Filtrage des marges AR(1)-GARCH(1,1)-t ===")
z_ts, info_ts = ce.filter_marginal(tsmom)
z_mk, info_mk = ce.filter_marginal(mkt)
print("TSMOM :", {k: round(v, 3) for k, v in info_ts.items()})
print("MKT   :", {k: round(v, 3) for k, v in info_mk.items()})

# ---- (2) PIT -> pseudo-observations ----
u, idx = ce.to_pseudo_obs(z_ts.rename("TS"), z_mk.rename("MK"))
print(f"\nÉchantillon joint : N = {len(u)} mois ({idx.min():%Y-%m} → {idx.max():%Y-%m})")

# ---- (3) Ajustement des copules ----
print("\n=== (3) Sélection des copules (AIC/BIC) et dépendance de queue ===")
fit = ce.fit_all_copulas(u)
fit_show = fit.copy()
for c in ["logL", "AIC", "BIC", "lambda_L", "lambda_U"]:
    fit_show[c] = fit_show[c].astype(float).round(3)
print(fit_show.to_string())
best_aic = fit["AIC"].astype(float).idxmin()
best_bic = fit["BIC"].astype(float).idxmin()
print(f"\nMeilleure copule  AIC: {best_aic}   |   BIC: {best_bic}")

# ---- Dépendance de queue NON-PARAMÉTRIQUE ----
print("\n=== Dépendance de queue empirique (model-free) ===")
emp10 = ce.empirical_tail_dependence(u, q=0.10)
emp15 = ce.empirical_tail_dependence(u, q=0.15)
for e in (emp10, emp15):
    print({k: round(v, 3) for k, v in e.items()})

# ---- Test d'adéquation CvM sur la gaussienne (H0) et la meilleure famille ----
print("\n=== Test d'adéquation Cramér-von Mises (bootstrap paramétrique) ===")
def parse_param(row):
    d = {}
    for kv in row["param"].split(", "):
        k, v = kv.split("=")
        d[{"rho": "rho", "df": "df", "theta": "theta"}[k]] = float(v)
    return d
gof_rows = []
for fam in ["Gaussian", best_aic]:
    par = parse_param(fit.loc[fam])
    g = ce.cvm_gof(u, fam, par, n_boot=250)
    gof_rows.append(g)
    print(f"  {fam:10s}  Sn={g['Sn']:.4f}  p-value={g['p_value']:.3f}")

# ---- Corrélations d'exceedance (asymétrie krach vs boom) ----
exc = ce.exceedance_correlation(z_ts.reindex(idx), z_mk.reindex(idx))
print("\n=== Corrélations d'exceedance (résidus standardisés) ===")
print(exc.round(3).to_string(index=False))

# ======================= SORTIES ========================
# Tableau principal au format des autres tables du projet
res = fit.copy()
res.insert(0, "Family", res.index)
res.to_csv("ext_A_copula_selection.csv", index=False)

# Résumé synthétique
summary = pd.DataFrame({
    "Best copula (AIC)": [best_aic],
    "Best copula (BIC)": [best_bic],
    "lambda_L (best)": [float(fit.loc[best_aic, "lambda_L"])],
    "lambda_U (best)": [float(fit.loc[best_aic, "lambda_U"])],
    "lambda_L emp (q=.10)": [emp10["lambda_L_emp"]],
    "lambda_U emp (q=.10)": [emp10["lambda_U_emp"]],
    "Gaussian rejected? (p<.05)": [gof_rows[0]["p_value"] < 0.05],
    "Best not rejected? (p>.05)": [gof_rows[1]["p_value"] > 0.05],
    "N": [len(u)],
}).T.rename(columns={0: "value"})
summary.to_csv("ext_A_summary.csv")
print("\n=== RÉSUMÉ ===")
print(summary.to_string())

# Figure : (a) corrélations d'exceedance  (b) nuage des pseudo-obs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
ax1.plot(exc["threshold"], exc["corr_neg"], "o-", color="#c0392b",
         label="Krachs joints (queue basse)")
ax1.plot(exc["threshold"], exc["corr_pos"], "s-", color="#2980b9",
         label="Booms joints (queue haute)")
ax1.axhline(0, color="grey", lw=0.7)
ax1.set_xlabel("Seuil |z| (écarts-types)")
ax1.set_ylabel("Corrélation d'exceedance TSMOM–Marché")
ax1.set_title("(a) Le straddle dans les queues")
ax1.legend(fontsize=8)
ax2.scatter(u[:, 0], u[:, 1], s=10, alpha=0.5, color="#34495e")
ax2.set_xlabel("u = F(résidu TSMOM)")
ax2.set_ylabel("v = F(résidu Marché)")
ax2.set_title("(b) Pseudo-observations (copule empirique)")
plt.tight_layout()
plt.savefig("fig8_copula_tail_dependence.png", dpi=130)
print("\nFichiers écrits : ext_A_copula_selection.csv, ext_A_summary.csv, fig8_copula_tail_dependence.png")
