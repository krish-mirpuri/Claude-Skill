"""Report figures.

House rules: one y-axis per panel, categorical hues assigned in fixed order
and never cycled, a legend whenever there is more than one series, recessive
grid and axes. The palette is the colour-vision-checked set also used by the
sibling project in this repository, so the two read as one body of work.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8b8a85"
GRID = "#e3e2de"


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


def _legend(ax, handles=None, labels=None, **kw):
    leg = (
        ax.legend(handles, labels, frameon=False, fontsize=9, **kw)
        if handles
        else ax.legend(frameon=False, fontsize=9, **kw)
    )
    for t in leg.get_texts():
        t.set_color(INK_SOFT)
    return leg


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_per_label(table: pd.DataFrame, path: Path) -> Path:
    """Per-label F1 with its bootstrap interval — the chart the headline hides.

    Sorted by support, so the widening intervals down the page are the whole
    argument: the rare labels are not scored badly, they are barely scored.
    """
    t = table.sort_values("support").reset_index(drop=True)
    y = np.arange(len(t))
    colors = [SERIES[0] if m else SERIES[1] for m in t["measurable"]]

    fig, ax = plt.subplots(figsize=(8.0, 0.30 * len(t) + 1.8))
    for yi, (_, r), c in zip(y, t.iterrows(), colors, strict=True):
        ax.plot(
            [r["f1_ci_low"], r["f1_ci_high"]],
            [yi, yi],
            color=c,
            linewidth=2.4,
            solid_capstyle="round",
            alpha=0.45,
        )
        ax.plot(
            [r["f1"]],
            [yi],
            marker="o",
            markersize=7,
            color=c,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.label}  (n={int(r.support)})" for r in t.itertuples()])
    _style(ax, title="Per-label F1 on the held-out set, with 95% bootstrap intervals", xlabel="F1")
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.8, len(t) - 0.2)
    handles = [
        plt.Line2D(
            [],
            [],
            color=SERIES[0],
            marker="o",
            linewidth=2.4,
            markersize=7,
            label="enough test positives to measure",
        ),
        plt.Line2D(
            [],
            [],
            color=SERIES[1],
            marker="o",
            linewidth=2.4,
            markersize=7,
            label="too few to measure",
        ),
    ]
    # Upper left: the frequent labels at the top of the chart all score well
    # above 0.25, so this corner is the one the marks never reach.
    _legend(ax, handles, [h.get_label() for h in handles], loc="upper left")
    return _save(fig, path)


def fig_decoders(df: pd.DataFrame, path: Path) -> Path:
    """Macro-F1 by threshold strategy, with intervals.

    Dots rather than bars, deliberately. The differences here are a fraction
    of a point, so a bar chart would have to truncate its baseline to show
    them at all — and a truncated bar makes a rounding error look like a
    result, which is the opposite of what this chart is for. Dots carry no
    baseline promise, so the axis can zoom in honestly.
    """
    order = ["flat", "global_tuned", "per_label_dev", "per_label_oof", "flat_plus_argmax_fallback"]
    d = (
        df.set_index("strategy")
        .reindex([o for o in order if o in set(df["strategy"])])
        .reset_index()
    )
    y = np.arange(len(d))[::-1]
    point = d["macro_f1"].astype(float).to_numpy()
    lo = d["macro_f1_ci_low"].astype(float).to_numpy()
    hi = d["macro_f1_ci_high"].astype(float).to_numpy()
    base = float(d.loc[d["strategy"] == "flat", "macro_f1"].iloc[0])

    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    ax.axvline(base, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    for yi, p_, l_, h_ in zip(y, point, lo, hi, strict=True):
        colour = SERIES[0] if p_ >= base else SERIES[1]
        ax.plot(
            [l_, h_],
            [yi, yi],
            color=colour,
            linewidth=2.6,
            alpha=0.45,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(
            [p_],
            [yi],
            marker="o",
            markersize=8,
            color=colour,
            markeredgecolor=SURFACE,
            markeredgewidth=1.6,
            zorder=3,
        )
        ax.annotate(
            f"{p_:.4f}",
            xy=(h_, yi),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK_SOFT,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("_", " ") for s in d["strategy"]], fontsize=9.5)
    _style(
        ax, title="Tuning the thresholds changes almost nothing — and can hurt", xlabel="macro-F1"
    )
    span = hi.max() - lo.min()
    ax.set_xlim(lo.min() - span * 0.08, hi.max() + span * 0.30)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.annotate(
        "the conventional 0.5",
        xy=(base, len(d) - 0.55),
        xytext=(-6, 0),
        textcoords="offset points",
        fontsize=8.5,
        color=INK_MUTED,
        ha="right",
        va="center",
    )
    return _save(fig, path)


def fig_coarse(coarse: dict, fine: dict, path: Path) -> Path:
    """The same predictions, scored against three taxonomies."""
    rows = [
        ("27 emotions + neutral", fine["macro_f1"], fine["n_labels"], fine.get("min_support", 0))
    ]
    for view in ("ekman", "sentiment"):
        c = coarse[view]
        label = {
            "ekman": "6 Ekman families + neutral",
            "sentiment": "positive / negative / ambiguous / neutral",
        }[view]
        rows.append((label, c["macro_f1"], c["n_labels"], c["min_support"]))

    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    y = np.arange(len(rows))[::-1]
    vals = [r[1] for r in rows]
    ax.barh(y, vals, color=SERIES[0], height=0.5)
    for yi, r in zip(y, rows, strict=True):
        ax.text(r[1] + 0.012, yi, f"{r[1]:.3f}", va="center", fontsize=9.5, color=INK)
        ax.text(
            0.012,
            yi - 0.30,
            f"{r[2]} labels · rarest has {r[3]} test examples",
            va="center",
            fontsize=8.5,
            color=INK_MUTED,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    _style(ax, title="The model does not change. Only the question does.", xlabel="macro-F1")
    ax.set_xlim(0, max(vals) * 1.25)
    return _save(fig, path)


def fig_confusion(pairs: pd.DataFrame, taxonomy, path: Path, top_n: int = 14) -> Path:
    """Most common substitutions, split by whether they stay in the family."""
    p = pairs.head(top_n).iloc[::-1].reset_index(drop=True)
    same = [
        taxonomy.family_of(a) == taxonomy.family_of(b)
        for a, b in zip(p["true"], p["predicted_instead"], strict=True)
    ]
    colors = [SERIES[2] if s else SERIES[1] for s in same]

    fig, ax = plt.subplots(figsize=(8.0, 0.36 * len(p) + 1.8))
    y = np.arange(len(p))
    ax.barh(y, p["count"], color=colors, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{a} → {b}" for a, b in zip(p["true"], p["predicted_instead"], strict=True)], fontsize=9
    )
    for yi, v in zip(y, p["count"], strict=True):
        ax.text(
            v + max(p["count"]) * 0.012, yi, str(int(v)), va="center", fontsize=8.5, color=INK_SOFT
        )
    _style(ax, title="What it says instead", xlabel="times the substitution was made")
    ax.set_xlim(0, max(p["count"]) * 1.12)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=SERIES[2]),
        plt.Rectangle((0, 0), 1, 1, color=SERIES[1]),
    ]
    _legend(
        ax,
        handles,
        ["same Ekman family — a shade, not a miss", "different family — a real error"],
        loc="lower right",
    )
    return _save(fig, path)


def fig_noise(rates: pd.DataFrame, path: Path, top_n: int = 16) -> Path:
    r = rates.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 0.33 * len(r) + 1.8))
    y = np.arange(len(r))
    ax.barh(y, r["disputed_share_of_positives"], color=SERIES[0], height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{lab}  (n={int(n)})" for lab, n in zip(r["label"], r["support"], strict=True)],
        fontsize=9,
    )
    for yi, v in zip(y, r["disputed_share_of_positives"], strict=True):
        ax.text(v + 0.008, yi, f"{v:.0%}", va="center", fontsize=8.5, color=INK_SOFT)
    _style(
        ax,
        title="Assignments the model confidently disputes",
        xlabel="share of that label's training examples",
    )
    ax.set_xlim(0, min(1.0, r["disputed_share_of_positives"].max() * 1.2))
    return _save(fig, path)


def fig_transfer(table: pd.DataFrame, path: Path) -> Path:
    """Accuracy by model and evaluation set, including the leaky version."""
    t = table.copy()
    t["setting"] = np.where(
        t["evaluated_on"].str.startswith("Reddit"),
        "Reddit (in-domain)",
        np.where(
            t["emoticons"].eq("left in (leaky)"),
            "Twitter, emoticons left in",
            "Twitter (out-of-domain)",
        ),
    )
    settings = ["Reddit (in-domain)", "Twitter (out-of-domain)", "Twitter, emoticons left in"]
    models = list(dict.fromkeys(t["model"]))
    x = np.arange(len(models))
    width = 0.26

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for i, setting in enumerate(settings):
        vals = [
            float(t[(t["model"] == m) & (t["setting"] == setting)]["accuracy"].mean())
            if ((t["model"] == m) & (t["setting"] == setting)).any()
            else np.nan
            for m in models
        ]
        pos = x + (i - 1) * width
        ax.bar(pos, vals, width=width - 0.02, color=SERIES[i], label=setting)
        for xi, v in zip(pos, vals, strict=True):
            if np.isfinite(v):
                ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5, color=INK_SOFT)
    ax.axhline(0.5, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" (", "\n(") for m in models], fontsize=9)
    _style(ax, title="Leaving Reddit costs more than any modelling choice", ylabel="accuracy")
    ax.set_ylim(0, 1.16)
    # The final group has no leaky bar, so its right-hand slot is free for the
    # reference label and the legend sits above it.
    ax.set_xlim(-0.55, len(models) - 0.5 + 2 * width + 0.12)
    ax.annotate(
        "coin flip",
        xy=(len(models) - 1 + 1.5 * width, 0.5),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=8.5,
        color=INK_MUTED,
        ha="center",
    )
    _legend(ax, loc="upper right", ncols=1)
    return _save(fig, path)


def fig_active(curve: pd.DataFrame, path: Path) -> Path:
    """Mean learning curve per strategy, with the spread across seeds."""
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for i, (strategy, g) in enumerate(curve.groupby("strategy")):
        agg = g.groupby("budget")["macro_f1"].agg(["mean", "min", "max"])
        ax.fill_between(agg.index, agg["min"], agg["max"], color=SERIES[i], alpha=0.16, linewidth=0)
        ax.plot(
            agg.index,
            agg["mean"],
            color=SERIES[i],
            linewidth=2,
            marker="o",
            markersize=5,
            label=strategy,
        )
    full = curve.groupby("budget")["macro_f1"].mean().sort_index()
    ax.axhline(full.iloc[-1] * 0.95, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate(
        "95% of the full-data score",
        xy=(curve["budget"].min(), full.iloc[-1] * 0.95),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=8.5,
        color=INK_MUTED,
    )
    ax.set_xscale("log")
    _style(
        ax,
        title="How much of the annotation budget bought anything",
        xlabel="comments labelled (log scale)",
        ylabel="macro-F1 on the held-out set",
    )
    _legend(ax, loc="lower right")
    return _save(fig, path)
