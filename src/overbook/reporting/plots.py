"""Report figures.

House rules, applied everywhere in this module:

* One y-axis per panel. Two measures on different scales become two stacked
  panels sharing an x-axis, never a twin axis.
* Categorical hues are assigned in fixed slot order and never cycled; the
  palette below was checked for colour-vision separation before use.
* A legend whenever there is more than one series; direct labels where they
  replace a lookup rather than decorate.
* Grid and axes recede; the marks carry the ink.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical slots (blue, orange, aqua) on a light chart surface.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8b8a85"
GRID = "#e3e2de"
GOOD = "#008300"
BAD = "#e34948"


def _style(ax, *, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=10)


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def _spread(values: list[float], lo: float, hi: float, min_gap: float) -> list[float]:
    """Nudge label positions apart, preserving their order.

    End-of-line labels collide whenever two series finish at a similar value.
    Rather than drop the labels (the legend alone makes the reader look twice)
    the positions are pushed apart by at least ``min_gap`` and clamped to the
    axis.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = list(values)
    prev = None
    for i in order:
        v = out[i] if prev is None else max(out[i], prev + min_gap)
        out[i] = min(max(v, lo), hi)
        prev = out[i]
    return out


def fig_reliability(before: pd.DataFrame, after: pd.DataFrame, path: Path) -> Path:
    """Predicted vs observed cancellation rate, before and after calibration."""
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot(
        [0, 1],
        [0, 1],
        color=INK_MUTED,
        linewidth=1.2,
        linestyle=(0, (4, 3)),
        label="perfect calibration",
        zorder=1,
    )
    ax.plot(
        before["mean_predicted"],
        before["observed_rate"],
        color=SERIES[0],
        linewidth=2,
        marker="o",
        markersize=5,
        label="raw model",
        zorder=2,
    )
    ax.plot(
        after["mean_predicted"],
        after["observed_rate"],
        color=SERIES[1],
        linewidth=2,
        marker="s",
        markersize=5,
        label="after isotonic calibration",
        zorder=3,
    )
    _style(
        ax,
        title="Do the probabilities mean what they say?",
        xlabel="predicted P(cancel)",
        ylabel="observed cancellation rate",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK_SOFT)
    return _save(fig, path)


def fig_policy_profit(summary: pd.DataFrame, path: Path, baseline: str = "no_overbook") -> Path:
    """Profit per room-night by policy, with the perfect-foresight bound."""
    order = summary.sort_values("profit_per_room_night")
    labels = order.index.tolist()
    values = order["profit_per_room_night"].to_numpy()

    def group(name: str) -> int:
        return 2 if name == "oracle" else (1 if name == "distributional" else 0)

    colors = [(INK_MUTED if group(n) == 2 else SERIES[group(n)]) for n in labels]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    bars = ax.barh(labels, values, color=colors, height=0.62)
    for bar, v in zip(bars, values, strict=True):
        ax.text(
            v + max(values) * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{v:,.1f}",
            va="center",
            ha="left",
            fontsize=9,
            color=INK_SOFT,
        )
    base = summary.loc[baseline, "profit_per_room_night"]
    ax.axvline(base, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=0)
    _style(
        ax,
        title="Profit per available room-night, held-out arrivals",
        xlabel="€ contribution per room-night",
    )
    ax.set_xlim(0, max(values) * 1.12)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=SERIES[0]),
        plt.Rectangle((0, 0), 1, 1, color=SERIES[1]),
        plt.Rectangle((0, 0), 1, 1, color=INK_MUTED),
    ]
    leg = ax.legend(
        handles,
        ["baseline policies", "this project", "perfect foresight (bound)"],
        frameon=False,
        fontsize=9,
        loc="lower right",
    )
    for t in leg.get_texts():
        t.set_color(INK_SOFT)
    return _save(fig, path)


def fig_decision_anatomy(diag: dict, path: Path, marks: dict[str, int] | None = None) -> Path:
    """One night, in full: the profit curve above, the arrival risk below.

    The panels do NOT share an x-axis. "Bookings authorised" runs to the size
    of the queue while "guests arriving" lives in a much narrower band, and
    forcing them onto one scale crushes the distribution into a spike.
    """
    curve = np.asarray(diag["profit_curve"], dtype=float)
    pmf = np.asarray(diag["arrival_pmf"], dtype=float)
    cap = diag["capacity"]
    chosen = diag["authorised"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.8, 6.8), gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.42}
    )

    # -- upper: expected profit against authorisation level -----------------
    x = np.arange(curve.size)
    ax1.plot(x, curve, color=SERIES[0], linewidth=2, zorder=2)
    ax1.plot(
        [chosen],
        [curve[chosen]],
        marker="o",
        markersize=9,
        color=SERIES[1],
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        zorder=4,
    )
    _style(
        ax1,
        title=f"{diag['hotel']}, arrivals on {diag['arrival_date']}",
        xlabel="bookings authorised",
        ylabel="expected profit (€)",
    )
    ax1.set_xlim(-curve.size * 0.02, curve.size * 1.02)
    span = curve.max() - min(curve.min(), 0)
    ax1.set_ylim(min(curve.min(), 0) - span * 0.06, curve.max() + span * 0.22)

    # Anchor the callout inside the axes, flipping side near the right edge.
    right_side = chosen > curve.size * 0.6
    ax1.annotate(
        f"authorise {chosen}  (+{diag['overbook_pct']:.0f}% over capacity)",
        xy=(chosen, curve[chosen]),
        xytext=(-10 if right_side else 10, 16),
        textcoords="offset points",
        fontsize=9.5,
        color=INK,
        ha="right" if right_side else "left",
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.9},
    )
    # Where the other rules would have stopped. Their labels go in one block
    # rather than beside each mark: several rules land within a booking or two
    # of each other, so per-mark labels collide however they are nudged.
    other = {k: v for k, v in (marks or {}).items() if k not in {"distributional", "oracle"}}
    if other:
        for a in other.values():
            ax1.axvline(
                min(a, curve.size - 1), color=GRID, linewidth=1.1, linestyle=(0, (3, 3)), zorder=1
            )
        # Bottom-right: the one part of this panel the curve never reaches.
        legend_text = "\n".join(f"{name.replace('_', ' ')} → {a}" for name, a in other.items())
        ax1.annotate(
            f"where other rules stop\n{legend_text}",
            xy=(0.985, 0.04),
            xycoords="axes fraction",
            fontsize=8.5,
            color=INK_MUTED,
            ha="right",
            va="bottom",
            linespacing=1.45,
        )

    # -- lower: the arrival distribution, on its own scale ------------------
    support = np.flatnonzero(pmf > 1e-4)
    lo = max(0, int(support.min()) - 3) if support.size else 0
    hi = min(pmf.size - 1, int(support.max()) + 3) if support.size else pmf.size - 1
    hi = max(hi, cap + 3)
    ks = np.arange(lo, hi + 1)
    colors = [BAD if k > cap else SERIES[2] for k in ks]
    ax2.bar(ks, pmf[lo : hi + 1], color=colors, width=0.88)
    ax2.axvline(cap, color=INK, linewidth=1.4, linestyle=(0, (4, 3)), zorder=3)
    _style(ax2, xlabel="guests arriving", ylabel="probability")
    ax2.set_xlim(lo - 0.6, hi + 0.6)
    top = pmf[lo : hi + 1].max()
    ax2.set_ylim(0, top * 1.38)
    ax2.annotate(
        f"capacity {cap}",
        xy=(cap, top * 1.30),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=9,
        color=INK,
        va="center",
    )
    ax2.annotate(
        f"P(oversell) = {diag['p_oversell']:.1%}\nexpected walks {diag['expected_walks']:.2f}",
        xy=(0.985, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
        color=INK_SOFT,
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[2]), plt.Rectangle((0, 0), 1, 1, color=BAD)]
    leg = ax2.legend(
        handles,
        ["within capacity", "oversold — guests walked"],
        frameon=False,
        fontsize=9,
        loc="upper left",
    )
    for t in leg.get_texts():
        t.set_color(INK_SOFT)
    return _save(fig, path)


def fig_monthly_stability(df: pd.DataFrame, path: Path, split_at: str | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x = np.arange(len(df))
    ax.plot(
        x,
        df["observed"],
        color=SERIES[0],
        linewidth=2,
        marker="o",
        markersize=5,
        label="observed cancellation rate",
    )
    ax.plot(
        x,
        df["predicted"],
        color=SERIES[1],
        linewidth=2,
        marker="s",
        markersize=5,
        label="model prediction",
    )
    if split_at is not None and split_at in set(df["month"]):
        i = df.index[df["month"] == split_at][0]
        ax.axvline(i - 0.5, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.annotate(
            "held-out period begins",
            xy=(i - 0.4, ax.get_ylim()[1]),
            fontsize=9,
            color=INK_SOFT,
            va="top",
        )
    ax.set_xticks(x[::2])
    ax.set_xticklabels(df["month"][::2], rotation=45, ha="right")
    _style(ax, title="Calibration holds as the season turns", ylabel="cancellation rate")
    leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK_SOFT)
    return _save(fig, path)


def fig_lead_time_risk(df: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(
        df["lead_time_mid"],
        df["cancel_rate"],
        color=SERIES[0],
        linewidth=2,
        marker="o",
        markersize=5,
    )
    for _, r in df.iloc[::2].iterrows():
        ax.annotate(
            f"{r['cancel_rate']:.0%}",
            xy=(r["lead_time_mid"], r["cancel_rate"]),
            xytext=(0, 8),
            textcoords="offset points",
            fontsize=8,
            color=INK_SOFT,
            ha="center",
        )
    _style(
        ax,
        title="Cancellation risk rises with how far ahead a guest books",
        xlabel="lead time (days, bucket midpoint)",
        ylabel="cancellation rate",
    )
    ax.set_ylim(0, max(df["cancel_rate"]) * 1.25)
    return _save(fig, path)


def fig_importance(importance: pd.Series, path: Path, top_n: int = 18) -> Path:
    s = importance.sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(s) + 1.4))
    ax.barh(s.index, s.to_numpy(), color=SERIES[0], height=0.62)
    _style(ax, title="What the model leans on", xlabel="mean |SHAP value| (log-odds)")
    for y, v in zip(range(len(s)), s.to_numpy(), strict=True):
        ax.text(v * 1.02, y, f"{v:.3f}", va="center", fontsize=8, color=INK_SOFT)
    ax.set_xlim(0, s.max() * 1.16)
    return _save(fig, path)


def fig_walk_response(df: pd.DataFrame, path: Path) -> Path:
    """Does the policy react to the cost of failure?

    Two panels, not two y-axes: euros and walk counts share an x-axis and
    nothing else. The lower panel carries the finding — a mean-based rule
    walks the same number of guests whatever you tell it a walk costs.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 6.4), sharex=True)
    order = sorted(df["policy"].unique(), key=lambda n: (n != "mean_rate", n))
    for i, name in enumerate(order):
        g = df[df["policy"] == name].sort_values("walk_cost_multiplier")
        color = SERIES[i % len(SERIES)]
        label = name.replace("_", " ")
        ax1.plot(
            g["walk_cost_multiplier"],
            g["uplift_per_room_night"],
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            label=label,
        )
        ax2.plot(
            g["walk_cost_multiplier"],
            g["walks_per_100_nights"],
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            label=label,
        )
        last = g.iloc[-1]
        ax2.annotate(
            label,
            xy=(last["walk_cost_multiplier"], last["walks_per_100_nights"]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8.5,
            color=INK_SOFT,
            va="center",
        )
    _style(ax1, title="Reacting to the cost of walking a guest", ylabel="€ uplift per room-night")
    leg = ax1.legend(frameon=False, fontsize=9, loc="lower left")
    for t in leg.get_texts():
        t.set_color(INK_SOFT)
    _style(ax2, xlabel="assumed cost of a walk (× ADR)", ylabel="guests walked per 100 nights")
    ax2.set_xlim(df["walk_cost_multiplier"].min() - 0.2, df["walk_cost_multiplier"].max() * 1.42)
    return _save(fig, path)


def fig_regimes(df: pd.DataFrame, path: Path, champion: str = "distributional_shock") -> Path:
    """How much of the achievable gain each policy captures, by how tight capacity is."""
    wanted = [champion, "expected_arrivals", "mean_rate", "accept_all"]
    colors = {
        champion: SERIES[0],
        "expected_arrivals": SERIES[1],
        "mean_rate": SERIES[2],
        "accept_all": INK_MUTED,
    }
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ends: list[tuple[str, float, float]] = []
    for name in wanted:
        g = df[df["policy"] == name].sort_values("binds_pct_of_nights")
        if g.empty:
            continue
        ax.plot(
            g["binds_pct_of_nights"],
            g["pct_of_oracle_gap_closed"],
            color=colors[name],
            linewidth=2,
            marker="o",
            markersize=5,
            label=name.replace("_", " "),
        )
        ends.append(
            (
                name,
                float(g.iloc[-1]["binds_pct_of_nights"]),
                float(g.iloc[-1]["pct_of_oracle_gap_closed"]),
            )
        )
    ax.axhline(100, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate(
        "perfect foresight",
        xy=(df["binds_pct_of_nights"].min(), 100),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=8.5,
        color=INK_SOFT,
    )
    _style(
        ax,
        title="The decision layer earns its keep when rooms are scarce",
        xlabel="share of nights where demand exceeds capacity (%)",
        ylabel="% of the achievable gain captured",
    )
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, max(hi, 112))
    ax.set_xlim(df["binds_pct_of_nights"].min() - 3, df["binds_pct_of_nights"].max() * 1.30)
    ys = _spread([e[2] for e in ends], *ax.get_ylim(), min_gap=(hi - lo) * 0.075)
    for (name, xe, _), y in zip(ends, ys, strict=True):
        ax.annotate(
            name.replace("_", " "),
            xy=(xe, y),
            xytext=(7, 0),
            textcoords="offset points",
            fontsize=8.5,
            color=INK_SOFT,
            va="center",
        )
    leg = ax.legend(frameon=False, fontsize=9, loc="lower left")
    for t in leg.get_texts():
        t.set_color(INK_SOFT)
    return _save(fig, path)
