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
    factors_t2 = pd.DataFrame({"MKT": mkt_excess, "GSCI": gsci, "BOND": bond}).loc[m]

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

    # Panel C — extrêmes (VIX/TED in-dataset + PS/BW externes)
    try:
        vix = prices["VIX Index"].resample("ME").last()
        ted = prices[".TEDSP Index"].resample("ME").last()
        ps = ext.get("pastor_stambaugh")
        bw = ext.get("baker_wurgler")
        ps_s = ps["innov_liq"] if (ps is not None and "innov_liq" in ps) else None
        bw_s = (bw.iloc[:, 0] if bw is not None and bw.shape[1] else None)
        Xc = build_extremes_factor_matrix(mkt_excess.loc[m], vix.loc[m], ted.loc[m], ps_s, bw_s)
        t3c = table3_full(tsmom_p, Xc.loc[Xc.index.isin(tsmom_p.index)])
        tables.table3_panel_save(t3c, "table3_panelC_extremes")
        note = "" if (ps_s is not None and bw_s is not None) else " (VIX/TED only — PS/BW manquants)"
        status["Table 3C"] = "OK" + note; log(f"[OK]   Table 3 Panel C — extremes{note}")
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
    t5b = decomposition_by_asset_class(mret, lookback=LOOKBACK_MONTHS)
    t5c = pd.DataFrame()
    targets = {}
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
    irf = impulse_response(monthly, net_spec=net_spec, horizon=36, lags=2)
    if irf is not None:
        plotting.figure7_impulse_response(irf)
        status["Figure 7"] = "OK" + ("" if net_spec is not None else " (univarié — CFTC manquant)")
        log("[OK]   Figure 7 — impulse response" + ("" if net_spec is not None else " (univariate)"))
    else:
        status["Figure 7"] = "SKIP (statsmodels VAR indisponible)"; log("[SKIP] Figure 7")

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
