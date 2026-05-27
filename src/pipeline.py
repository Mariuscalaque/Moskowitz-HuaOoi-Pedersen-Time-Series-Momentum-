"""
pipeline.py — Orchestrateur de la réplication COMPLÈTE de Moskowitz, Hua Ooi &
Pedersen (2012) : produit les 6 Tables et les 7 Figures.

Usage :
    from src.pipeline import run
    res = run(start="1985-01-01", end="2009-12-31", use_external=True)

Tout ce qui est réalisable avec data.xlsx seul est toujours produit. Les
sorties qui dépendent de données externes (VME → Table 3B ; liquidité/sentiment
→ Table 3C ; positions CFTC → Table 6, Fig 5/6B/7 ; indices hedge funds →
Table 5C) sont produites si `external_data` parvient à les récupérer, et sinon
proprement signalées comme « SKIPPED (données externes manquantes) ».

Les CSV des tables vont dans outputs/tables/, les PNG des figures dans
outputs/figures/.
"""

import numpy as np
import pandas as pd

from . import tables, plotting, analysis
from .config import (TABLES_DIR, FIGURES_DIR, TARGET_VOL, LOOKBACK_MONTHS,
                     PAPER_START, PAPER_END)
from .data_loader import load_raw
from .returns import build_daily_excess_returns, daily_to_monthly_returns
from .volatility import ewma_ex_ante_vol, vol_for_signal
from .strategy import (tsmom_instrument_returns, diversified_tsmom,
                       tsmom_by_asset_class, passive_long)
from .crosssectional import xsmom_by_asset_class, decomposition_by_asset_class
from .rollyield import spot_roll_monthly, momentum_signals_12m
from .analysis import (pooled_lag_regression_size, pooled_lag_regression_sign,
                       table2_grid, table3_full, sharpe_by_instrument,
                       build_vme_factor_matrix, build_extremes_factor_matrix,
                       table4_within_class, table4_across_class,
                       table5_tsmom_on_xsmom, table5_what_tsmom_explains,
                       table6_predictors, event_study_returns,
                       event_study_positions, impulse_response)
from .factors import fetch_ff_factors, build_table3_factors


def _mret(prices, col):
    return prices[col].resample("ME").last().pct_change()


def _ensure_dirs():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def run(start: str = PAPER_START, end: str = PAPER_END,
        use_external: bool = True, verbose: bool = True) -> dict:
    _ensure_dirs()
    status = {}

    def log(msg):
        if verbose:
            print(msg)

    # ---------- Données de base ----------
    prices = load_raw()
    daily = build_daily_excess_returns(prices)
    monthly = daily_to_monthly_returns(daily)
    dvol = ewma_ex_ante_vol(daily)
    mvol = vol_for_signal(dvol, monthly.index)

    m = (monthly.index >= start) & (monthly.index <= end)
    mret, mvolp = monthly.loc[m], mvol.loc[m]

    # Stratégies
    inst = tsmom_instrument_returns(monthly, mvol, k=LOOKBACK_MONTHS)
    tsmom = diversified_tsmom(inst)
    tsmom_ac = tsmom_by_asset_class(inst)
    passive = passive_long(monthly, mvol)
    inst_passive = (TARGET_VOL / mvol).replace([np.inf, -np.inf], np.nan).shift(1) * monthly
    # passive par classe (équipondéré)
    from .config import asset_class_of
    pac = {}
    for ac in ("Commodity", "Equity", "Bond", "Currency"):
        cols = [c for c in inst_passive.columns if asset_class_of(c) == ac]
        if cols:
            pac[ac] = inst_passive[cols].mean(axis=1)
    passive_ac = pd.DataFrame(pac)

    xsmom_ac = xsmom_by_asset_class(monthly, k=LOOKBACK_MONTHS)

    # Facteurs marché
    mkt = _mret(prices, "MXWO Index")
    gsci = _mret(prices, "SPGSCI Index")
    bond = _mret(prices, "LBUSTRUU Index")
    rf = (prices["US0001M Index"] / 100 / 12).resample("ME").last()
    mkt_excess = (mkt - rf)

    # Table 2 : modèle COMPLET de l'Eq. (4) — MKT, BOND, GSCI, SMB, HML, UMD.
    # (Auparavant seuls MKT/GSCI/BOND étaient inclus, ce qui n'est pas l'Eq. 4.)
    # Repli automatique sur 3 facteurs si les facteurs Fama-French sont indisponibles.
    factors_t2 = pd.DataFrame({"MKT": mkt_excess, "BOND": bond - rf,
                               "GSCI": gsci - rf}).loc[m]
    try:
        _ff_t2 = fetch_ff_factors(start=start, end=end, source="auto")
        factors_t2 = factors_t2.join(_ff_t2[["SMB", "HML", "UMD"]], how="left")
        factors_t2 = factors_t2.dropna()
    except Exception as _e:
        log(f"[Table 2] FF indisponibles, repli 3 facteurs : {str(_e)[:50]}")

    tsmom_p = tsmom.loc[m]

    # ---------- Données externes (optionnel) ----------
    ext = {}
    if use_external:
        try:
            from .external_data import fetch_all
            ext = fetch_all()
        except Exception as e:
            log(f"[external] indisponible : {str(e)[:80]}")

    # ============================================================
    # TABLE 1
    # ============================================================
    tables.table1_summary_stats(mret, daily.loc[(daily.index >= start) & (daily.index <= end)])
    status["Table 1"] = "OK"; log("[OK]   Table 1 — summary statistics")

    # ============================================================
    # FIGURE 1
    # ============================================================
    reg_size = pooled_lag_regression_size(mret, mvolp, max_lag=60)
    reg_sign = pooled_lag_regression_sign(mret, mvolp, max_lag=60)
    plotting.figure1_panelA(reg_size); plotting.figure1_panelB(reg_sign)
    status["Figure 1"] = "OK"; log("[OK]   Figure 1 — predictability by lag")

    # ============================================================
    # TABLE 2
    # ============================================================
    grid = table2_grid(mret, mvolp, factors_t2)
    tables.table2_save(grid, "panelA_all")
    status["Table 2"] = "OK"; log("[OK]   Table 2 — (k,h) alpha t-stats")

    # ============================================================
    # FIGURE 2
    # ============================================================
    sharpes = sharpe_by_instrument(inst.loc[m])
    plotting.figure2_sharpe_by_instrument(sharpes)
    status["Figure 2"] = "OK"; log("[OK]   Figure 2 — Sharpe by instrument")

    # ============================================================
    # TABLE 3 — Panel A (FF), B (VME), C (extremes)
    # ============================================================
    # Panel A
    try:
        ff = fetch_ff_factors(start=start, end=end, source="auto")
        bench_m = prices[["MXWO Index", "LBUSTRUU Index", "SPGSCI Index"]].resample("ME").last().pct_change()
        X6 = build_table3_factors(bench_m, ff)
        X4 = X6[["MKT", "SMB", "HML", "UMD"]]
        t3a = table3_full(tsmom_p, X4)
        tables.table3_panel_save(t3a, "table3_panelA_ff")
        status["Table 3A"] = "OK"; log("[OK]   Table 3 Panel A — Fama-French factors")
    except Exception as e:
        status["Table 3A"] = f"SKIP ({str(e)[:50]})"; log(f"[SKIP] Table 3 Panel A : {str(e)[:60]}")

    # Panel B — VME
    if "aqr_vme" in ext:
        try:
            Xvme = build_vme_factor_matrix(ext["aqr_vme"], mkt_excess)
            t3b = table3_full(tsmom_p, Xvme.loc[Xvme.index.isin(tsmom_p.index)])
            tables.table3_panel_save(t3b, "table3_panelB_vme")
            status["Table 3B"] = "OK"; log("[OK]   Table 3 Panel B — VME factors")
        except Exception as e:
            status["Table 3B"] = f"SKIP ({str(e)[:50]})"; log(f"[SKIP] Table 3 Panel B : {str(e)[:60]}")
    else:
        status["Table 3B"] = "SKIP (VME externe manquant)"; log("[SKIP] Table 3 Panel B — VME absent")

    # Panel C — extrêmes : régressions SÉPARÉES, en TRIMESTRIEL (fidélité au papier)
    try:
        from .analysis import table3_smile_quarterly, table3_extremes_blocks
        # (1) Straddle / smile : TSMOM ~ MKT + MKT² en trimestriel (résultat phare)
        t3c_smile = table3_smile_quarterly(tsmom_p, mkt_excess.loc[m])
        tables.table3_panel_save(t3c_smile, "table3_panelC_smile")

        # (2) Blocs extrêmes, chacun dans sa propre régression trimestrielle.
        #     Le papier utilise le NIVEAU du VIX (pas ΔVIX).
        vix = prices["VIX Index"].resample("ME").last()
        ted = prices[".TEDSP Index"].resample("ME").last()
        ps = ext.get("pastor_stambaugh")
        bw = ext.get("baker_wurgler")
        ps_s = ps["innov_liq"] if (ps is not None and "innov_liq" in ps) else None
        sent_lvl = (bw.iloc[:, 0] if bw is not None and bw.shape[1] else None)
        sent_chg = sent_lvl.diff() if sent_lvl is not None else None
        t3c = table3_extremes_blocks(tsmom_p, vix_level=vix.loc[m], ted=ted.loc[m],
                                     liq=ps_s, sent_level=sent_lvl, sent_change=sent_chg)
        if len(t3c):
            tables.table3_panel_save(t3c, "table3_panelC_extremes")
        note = "" if (ps_s is not None and sent_lvl is not None) else " (VIX/TED only — PS/BW manquants)"
        status["Table 3C"] = "OK" + note; log(f"[OK]   Table 3 Panel C — smile + extrêmes séparés{note}")
    except Exception as e:
        status["Table 3C"] = f"SKIP ({str(e)[:50]})"; log(f"[SKIP] Table 3 Panel C : {str(e)[:60]}")

    # ============================================================
    # FIGURE 3 & 4
    # ============================================================
    plotting.figure3_cumulative(tsmom_p, passive.loc[m])
    status["Figure 3"] = "OK"; log("[OK]   Figure 3 — cumulative TSMOM vs passive")
    sp_excess = _mret(prices, "SP1 Index").loc[m]
    plotting.figure4_smile(tsmom_p, sp_excess)
    status["Figure 4"] = "OK"; log("[OK]   Figure 4 — TSMOM smile")

    # ============================================================
    # TABLE 4 — corrélations
    # ============================================================
    within = table4_within_class(inst.loc[m], inst_passive.loc[m])
    across = table4_across_class(tsmom_ac.loc[m], passive_ac.loc[m])
    tables.table4_save(within, across)
    status["Table 4"] = "OK"; log("[OK]   Table 4 — correlations within/across classes")

    # ============================================================
    # TABLE 5 — A (TSMOM~XSMOM), B (decomposition), C (what TSMOM explains)
    # ============================================================
    t5a = table5_tsmom_on_xsmom(tsmom_ac.loc[m], xsmom_ac.loc[m],
                                tsmom_all=tsmom_p,
                                xsmom_all=xsmom_ac["ALL"].loc[m] if "ALL" in xsmom_ac else None)
    # Décomposition Lo-MacKinlay sur rendements VOL-SCALÉS (comme tout le papier) :
    # sans cela, bonds/FX (faible vol) contribuent ~0 et la ligne ALL s'effondre.
    # En vol-scalé, les classes redeviennent homogènes (Equity XS_Auto ≈ 0.75% ≈ papier).
    scaled_ret = (TARGET_VOL / mvol).replace([np.inf, -np.inf], np.nan).shift(1) * monthly
    t5b = decomposition_by_asset_class(scaled_ret.loc[m], lookback=LOOKBACK_MONTHS)
    t5c = pd.DataFrame()
    targets = {}
    # Panel C du papier : les PREMIÈRES lignes régressent XSMOM (ALL + par classe)
    # sur TSMOM (TSMOM « explique-t-il » le momentum cross-sectionnel ?).
    for ac in ("ALL", "Commodity", "Equity", "Bond", "Currency"):
        if ac in xsmom_ac.columns:
            targets[f"XSMOM {ac}"] = xsmom_ac[ac].loc[m]
    # facteurs FF + indices hedge funds externes si dispo
    try:
        ff = fetch_ff_factors(start=start, end=end, source="auto")
        for col in ("UMD", "HML", "SMB"):
            if col in ff:
                targets[f"FF {col}"] = ff[col]
    except Exception:
        pass
    if "hedge_funds" in ext:
        for c in ext["hedge_funds"].columns:
            targets[f"HF {c}"] = ext["hedge_funds"][c]
    else:
        # repli : CSV hedge funds déposé manuellement (données sous licence)
        try:
            from .factors import load_hedge_fund_indices
            hf = load_hedge_fund_indices()
            if hf is not None:
                for c in hf.columns:
                    targets[f"HF {c}"] = hf[c]
        except Exception:
            pass
    if targets:
        t5c = table5_what_tsmom_explains(targets, tsmom_p)
    tables.table5_save(t5a, t5b, t5c if len(t5c) else None)
    status["Table 5"] = "OK" + ("" if len(t5c) else " (Panel C limité : pas d'indices HF)")
    log(f"[OK]   Table 5 — A/B" + (" /C" if len(t5c) else " (C limité)"))

    # ============================================================
    # TABLE 6 — prédicteurs spot/roll/positions
    # ============================================================
    try:
        comps = spot_roll_monthly(prices)
        sigs = momentum_signals_12m(comps, k=LOOKBACK_MONTHS)
        total = comps["total"].loc[comps["total"].index.isin(monthly.index)]
        net_spec = ext.get("cftc_cot")
        t6 = table6_predictors(total.loc[m], sigs["total"].loc[m],
                               sigs["spot"].loc[m], sigs["roll"].loc[m],
                               net_spec.loc[net_spec.index.isin(total.index)] if net_spec is not None else None)
        tables.table6_save(t6)
        note = "" if net_spec is not None else " (spot/roll only — CFTC manquant)"
        status["Table 6"] = "OK" + note; log(f"[OK]   Table 6 — predictors{note}")
    except Exception as e:
        status["Table 6"] = f"SKIP ({str(e)[:50]})"; log(f"[SKIP] Table 6 : {str(e)[:60]}")

    # ============================================================
    # FIGURE 5 — positions spéculateurs
    # ============================================================
    net_spec = ext.get("cftc_cot")
    if net_spec is not None:
        plotting.figure5_net_speculator(net_spec)
        status["Figure 5"] = "OK"; log("[OK]   Figure 5 — net speculator positions")
    else:
        status["Figure 5"] = "SKIP (CFTC manquant)"; log("[SKIP] Figure 5 — CFTC absent")

    # ============================================================
    # FIGURE 6 — event study
    # ============================================================
    es_ret = event_study_returns(monthly, k=LOOKBACK_MONTHS)
    es_pos = (event_study_positions(net_spec, monthly, k=LOOKBACK_MONTHS)
              if net_spec is not None else None)
    plotting.figure6_event_study(es_ret, es_pos)
    status["Figure 6"] = "OK" + ("" if es_pos is not None else " (Panel A only — CFTC manquant)")
    log("[OK]   Figure 6 — event study" + ("" if es_pos is not None else " (A only)"))

    # ============================================================
    # FIGURE 7 — réponse impulsionnelle
    # ============================================================
    irf = impulse_response(monthly, net_spec=net_spec, horizon=36, lags=24,
                           n_boot=300, ci=0.90, seed=0)
    if irf is not None:
        plotting.figure7_impulse_response(irf)
        status["Figure 7"] = "OK" + ("" if net_spec is not None else " (univarié — CFTC manquant)")
        log("[OK]   Figure 7 — impulse response" + ("" if net_spec is not None else " (univariate)"))
    else:
        status["Figure 7"] = "SKIP (statsmodels VAR indisponible)"; log("[SKIP] Figure 7")

    # ============================================================
    # RÉSUMÉ DE PERFORMANCE + SÉRIE TSMOM  (source UNIQUE de vérité)
    # Auparavant générés à part par le notebook -> risque d'outputs périmés.
    # On les produit ici pour qu'ils soient TOUJOURS cohérents avec les tables.
    # ============================================================
    def _perf(series, ppy=12):
        r = series.dropna()
        if len(r) == 0:
            return {k: np.nan for k in ("N months", "Ann. mean", "Ann. vol",
                    "Sharpe", "Skew", "Excess kurt", "Max DD", "CAGR")}
        mean_a, vol_a = r.mean() * ppy, r.std() * np.sqrt(ppy)
        cum = (1 + r).cumprod()
        dd = (cum / cum.cummax() - 1).min()
        cagr = cum.iloc[-1] ** (ppy / len(r)) - 1
        return {"N months": len(r), "Ann. mean": mean_a, "Ann. vol": vol_a,
                "Sharpe": mean_a / vol_a if vol_a > 0 else np.nan,
                "Skew": r.skew(), "Excess kurt": r.kurtosis(),
                "Max DD": dd, "CAGR": cagr}

    perf_rows = {
        "TSMOM diversifié (1985-2009)": _perf(tsmom_p),
        "Passive long (1985-2009)": _perf(passive.loc[m]),
        "MSCI World (1985-2009)": _perf(mkt_excess.loc[m]),
    }
    for ac in ("Commodity", "Equity", "Bond", "Currency"):
        if ac in tsmom_ac.columns:
            perf_rows[f"TSMOM {ac} (1985-2009)"] = _perf(tsmom_ac[ac].loc[m])
    perf_df = pd.DataFrame(perf_rows).T
    perf_df.index.name = "Series"
    perf_df.to_csv(TABLES_DIR / "performance_summary.csv", float_format="%.4f")
    (tsmom_p.rename("TSMOM").to_frame()
        .to_csv(TABLES_DIR / "diversified_tsmom_series.csv", index_label="date"))
    status["Perf summary"] = "OK"
    log("[OK]   performance_summary + diversified_tsmom_series (cohérents)")

    # ---------- Récapitulatif ----------
    if verbose:
        print("\n" + "=" * 60 + "\nRÉCAPITULATIF\n" + "=" * 60)
        for k, v in status.items():
            print(f"  {k:12s} : {v}")
        print(f"\nTables  -> {TABLES_DIR}\nFigures -> {FIGURES_DIR}")

    return {"status": status, "tsmom": tsmom, "tsmom_ac": tsmom_ac,
            "xsmom_ac": xsmom_ac, "monthly": monthly, "prices": prices}


if __name__ == "__main__":
    run()