"""
Figure plotting for the replication.
Mimics Figs. 1, 2, 3, 4 in Moskowitz, Hua Ooi, Pedersen (2012).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from .config import (
    FIGURES_DIR,
    ASSET_CLASS_COLORS,
    asset_class_of,
    pretty_name,
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def _save(fig: plt.Figure, name: str):
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    # Ferme la figure : sinon, sous %matplotlib inline, Jupyter l'affiche
    # automatiquement EN PLUS du display(Image(...)) -> graphique en double.
    plt.close(fig)
    return path


# -----------------------------------------------------------------------
# Figure 1 — t-statistics by month lag
# -----------------------------------------------------------------------

def figure1_panelA(reg_size: pd.DataFrame, save_name: str = "fig1_panelA"):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = ["#2ca02c" if t > 0 else "#d62728" for t in reg_size["tstat"]]
    ax.bar(reg_size.index, reg_size["tstat"], color=colors, alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(1.96, color="gray", ls="--", lw=0.6)
    ax.axhline(-1.96, color="gray", ls="--", lw=0.6)
    ax.set_xlabel("Month lag $h$")
    ax.set_ylabel("$t$-statistic")
    ax.set_title("Fig. 1 (Panel A) — Pooled regression of vol-scaled returns on lagged vol-scaled returns\n"
                 r"$r^s_t / \sigma^s_{t-1} = \alpha + \beta_h \cdot r^s_{t-h}/\sigma^s_{t-h-1} + \varepsilon$",
                 loc="left")
    ax.set_xticks(np.arange(0, len(reg_size) + 1, 6))
    return _save(fig, save_name)


def figure1_panelB(reg_sign: pd.DataFrame, save_name: str = "fig1_panelB"):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = ["#2ca02c" if t > 0 else "#d62728" for t in reg_sign["tstat"]]
    ax.bar(reg_sign.index, reg_sign["tstat"], color=colors, alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(1.96, color="gray", ls="--", lw=0.6)
    ax.axhline(-1.96, color="gray", ls="--", lw=0.6)
    ax.set_xlabel("Month lag $h$")
    ax.set_ylabel("$t$-statistic")
    ax.set_title("Fig. 1 (Panel B) — Pooled sign regression\n"
                 r"$r^s_t / \sigma^s_{t-1} = \alpha + \beta_h \cdot \mathrm{sign}(r^s_{t-h}) + \varepsilon$",
                 loc="left")
    ax.set_xticks(np.arange(0, len(reg_sign) + 1, 6))
    return _save(fig, save_name)


def figure1_panelC(reg_by_class: dict, save_name: str = "fig1_panelC"):
    """
    Per-asset-class t-stats by lag from sign regression (Panel C of paper Fig. 1).
    `reg_by_class` is a dict {asset_class: DataFrame indexed by lag, with tstat column}.
    """
    classes = ["Commodity", "Equity", "Bond", "Currency"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for ax, ac in zip(axes.ravel(), classes):
        df = reg_by_class.get(ac)
        if df is None or df.empty:
            ax.set_visible(False); continue
        col = ASSET_CLASS_COLORS[ac]
        colors = [col if t > 0 else "lightgray" for t in df["tstat"]]
        ax.bar(df.index, df["tstat"], color=colors, alpha=0.9, edgecolor="black", linewidth=0.4)
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(1.96, color="gray", ls="--", lw=0.5)
        ax.axhline(-1.96, color="gray", ls="--", lw=0.5)
        ax.set_title(f"{ac}")
        ax.set_ylabel("$t$-stat")
    for ax in axes[-1]:
        ax.set_xlabel("Month lag")
    fig.suptitle("Fig. 1 (Panel C) — Sign regression $t$-stats by asset class", y=1.02)
    fig.tight_layout()
    return _save(fig, save_name)


# -----------------------------------------------------------------------
# Figure 2 — Sharpe ratio by instrument
# -----------------------------------------------------------------------

def figure2_sharpe_by_instrument(sharpes: pd.Series, save_name: str = "fig2_sharpe_by_instrument"):
    """
    Bar chart of annualized Sharpe ratio for each instrument's TSMOM strategy,
    colored by asset class. Sorted within each asset class.
    """
    # Group/sort by asset class, then by Sharpe descending within
    info = pd.DataFrame({
        "sharpe": sharpes,
        "ac": [asset_class_of(t) for t in sharpes.index],
        "name": [pretty_name(t) for t in sharpes.index],
    })
    info = info.dropna(subset=["sharpe"])
    order = ["Commodity", "Equity", "Bond", "Currency"]
    info["ac"] = pd.Categorical(info["ac"], categories=order, ordered=True)
    info = info.sort_values(["ac", "sharpe"], ascending=[True, False])

    fig, ax = plt.subplots(figsize=(15, 6))
    colors = [ASSET_CLASS_COLORS[a] for a in info["ac"]]
    bars = ax.bar(range(len(info)), info["sharpe"].values,
                  color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(info)))
    ax.set_xticklabels(info["name"], rotation=90, fontsize=8)
    ax.set_ylabel("Annualized Sharpe ratio (gross)")
    ax.set_title("Fig. 2 — Sharpe ratio of 12-month TSMOM by instrument (sample-wide)", loc="left")
    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=ASSET_CLASS_COLORS[c], label=c) for c in order]
    ax.legend(handles=handles, loc="upper right", frameon=False)
    return _save(fig, save_name)


# -----------------------------------------------------------------------
# Figure 3 — Cumulative TSMOM vs passive long
# -----------------------------------------------------------------------

def figure3_cumulative(tsmom: pd.Series, passive: pd.Series,
                       save_name: str = "fig3_cumulative"):
    """Growth of $100 invested in TSMOM vs passive long (log scale)."""
    df = pd.concat([tsmom.rename("TSMOM"), passive.rename("Passive long")], axis=1).dropna()
    growth = (1.0 + df).cumprod() * 100.0

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(growth.index, growth["TSMOM"], color="#1f3a93", lw=1.8, label="Time Series Momentum")
    ax.plot(growth.index, growth["Passive long"], color="#7f7f7f", lw=1.4, label="Passive long (vol-matched)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${int(x):,}"))
    ax.set_ylabel("Growth of $100 (log scale)")
    ax.set_title("Fig. 3 — Cumulative excess return of the diversified TSMOM strategy", loc="left")
    ax.legend(loc="upper left", frameon=False)
    return _save(fig, save_name)


# -----------------------------------------------------------------------
# Figure 4 — TSMOM vs market: the "smile"
# -----------------------------------------------------------------------

def figure4_smile(tsmom: pd.Series, market: pd.Series, save_name: str = "fig4_smile"):
    """TSMOM returns plotted against market returns; expect a U / smile shape."""
    df = pd.concat([market.rename("MKT"), tsmom.rename("TSMOM")], axis=1).dropna()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["MKT"], df["TSMOM"], s=22, color="#1f3a93", alpha=0.55, edgecolor="none")

    # Quadratic fit
    x = df["MKT"].values; y = df["TSMOM"].values
    coefs = np.polyfit(x, y, 2)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, np.polyval(coefs, xs), color="#d62728", lw=2, label=f"Quadratic fit ($\\beta_2$ = {coefs[0]:.2f})")
    ax.axhline(0, color="black", lw=0.6); ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("MSCI World monthly excess return")
    ax.set_ylabel("TSMOM monthly excess return")
    ax.set_title("Fig. 4 — TSMOM 'smile': payoff vs market", loc="left")
    ax.legend(loc="upper left", frameon=False)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    return _save(fig, save_name)


# -----------------------------------------------------------------------
# Bonus — drawdown chart
# -----------------------------------------------------------------------

def figure_drawdown(tsmom: pd.Series, save_name: str = "fig5_drawdown"):
    df = tsmom.dropna()
    eq = (1.0 + df).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.55)
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_title("TSMOM drawdown over time", loc="left")
    return _save(fig, save_name)


# -----------------------------------------------------------------------
# Figure 5 — Positions nettes des spéculateurs
# -----------------------------------------------------------------------

def figure5_net_speculator(net_spec: pd.DataFrame, save_name: str = "fig5_net_speculator",
                           min_periods: int = 24):
    """Position nette moyenne des spéculateurs (% open interest) par instrument,
    moyennée sur l'échantillon, colorée par classe d'actifs.

    Renvoie None si l'historique CFTC est trop court (< min_periods dates) : un
    fichier d'une seule ligne donnerait une « moyenne » = instantané trompeur."""
    if net_spec is None or len(net_spec.dropna(how="all")) < min_periods:
        return None
    means = net_spec.mean().dropna().sort_values()
    if means.empty:
        return None
    colors = [ASSET_CLASS_COLORS.get(asset_class_of(t), "#888") for t in means.index]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(means)), means.values, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels([pretty_name(t) for t in means.index], rotation=90, fontsize=8)
    ax.set_ylabel("Net speculator position (avg, % open interest)")
    ax.set_title("Fig. 5 — Average net speculator positions by instrument", loc="left")
    return _save(fig, save_name)


# -----------------------------------------------------------------------
# Figure 6 — Event study (rendements conditionnels au signe du momentum)
# -----------------------------------------------------------------------

def figure6_event_study(returns_paths: pd.DataFrame,
                        positions_paths: pd.DataFrame | None = None,
                        save_name: str = "fig6_event_study"):
    """Panel A : rendement cumulé moyen autour d'événements TSMOM +/-.
    Panel B (si fourni) : position nette moyenne des spéculateurs."""
    has_pos = (positions_paths is not None and not positions_paths.empty
               and positions_paths[["positive", "negative"]].notna().any().any())
    n = 2 if has_pos else 1
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    ax.plot(returns_paths.index, returns_paths["positive"], color="#2ca02c",
            lw=2, label="Past 12m return > 0")
    ax.plot(returns_paths.index, returns_paths["negative"], color="#d62728",
            lw=2, label="Past 12m return < 0")
    ax.axvline(0, color="black", lw=0.7, ls="--")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Event month (0 = signal date)")
    ax.set_ylabel("Cumulative excess return")
    ax.set_title("Fig. 6A — Returns around TSMOM events", loc="left")
    ax.legend(frameon=False)

    if n == 2:
        ax2 = axes[1]
        ax2.plot(positions_paths.index, positions_paths["positive"], color="#2ca02c", lw=2)
        ax2.plot(positions_paths.index, positions_paths["negative"], color="#d62728", lw=2)
        ax2.axvline(0, color="black", lw=0.7, ls="--")
        ax2.axhline(0, color="black", lw=0.6)
        ax2.set_xlabel("Event month")
        ax2.set_ylabel("Net speculator position (de-meaned)")
        ax2.set_title("Fig. 6B — Speculator positions around events", loc="left")
    fig.tight_layout()
    return _save(fig, save_name)


# -----------------------------------------------------------------------
# Figure 7 — Réponse impulsionnelle à un choc de rendement
# -----------------------------------------------------------------------

def figure7_impulse_response(irf: pd.DataFrame, save_name: str = "fig7_impulse_response"):
    """Réponse cumulée à un choc de +1σ sur le rendement : prix (et position
    spéculateurs si VAR bivarié)."""
    has_pos = "cum_position" in irf.columns
    n = 2 if has_pos else 1
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 4.5))
    axes = np.atleast_1d(axes)
    axes[0].plot(irf.index, irf["cum_return"], color="#1f3a93", lw=2)
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set_xlabel("Months after shock")
    axes[0].set_ylabel("Cumulative return response")
    axes[0].set_title("Fig. 7A — Price response to a +1σ return shock", loc="left")
    if has_pos:
        axes[1].plot(irf.index, irf["cum_position"], color="#d62728", lw=2)
        axes[1].axhline(0, color="black", lw=0.6)
        axes[1].set_xlabel("Months after shock")
        axes[1].set_ylabel("Cumulative position response")
        axes[1].set_title("Fig. 7B — Speculator position response", loc="left")
    fig.tight_layout()
    return _save(fig, save_name)
