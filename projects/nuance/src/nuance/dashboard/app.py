"""Explorer for the model and, more to the point, for its scoreboard.

streamlit run src/nuance/dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from nuance.config import load_config
from nuance.features.text import normalise

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MUTED = "#8b8a85"

st.set_page_config(page_title="nuance", page_icon="🎭", layout="wide")
alt.data_transformers.disable_max_rows()


@st.cache_resource(show_spinner=False)
def _model():
    cfg = load_config()
    d = cfg.resolve(cfg.paths.models_dir)
    return joblib.load(d / "champion.joblib"), joblib.load(d / "decoders.joblib")


@st.cache_data(show_spinner=False)
def _reports():
    cfg = load_config()
    r = cfg.resolve(cfg.paths.reports_dir)

    def maybe_csv(name):
        p = r / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    def maybe_json(name):
        p = r / name
        return json.loads(p.read_text()) if p.exists() else {}

    card = json.loads((cfg.resolve(cfg.paths.models_dir) / "model_card.json").read_text())
    return (
        cfg,
        card,
        maybe_csv("per_label.csv"),
        maybe_json("model_metrics.json"),
        maybe_csv("suspected_label_errors.csv"),
        maybe_csv("noise_rates.csv"),
        maybe_csv("transfer.csv"),
        maybe_json("transfer_metrics.json"),
        maybe_csv("confusion_pairs.csv"),
    )


cfg, card, per_label, metrics, suspects, noise, transfer, transfer_meta, confusions = _reports()
model, decoders = _model()
names = card["labels"]


def fig_path(name: str) -> Path:
    return cfg.resolve(cfg.paths.figures_dir) / name


st.title("nuance")
head = metrics.get("ladder", {}).get(card["model"], {})
if head:
    st.caption(
        f"27 emotions + neutral on {head['n']:,} held-out Reddit comments · "
        f"macro-F1 **{head['macro_f1']:.3f}** "
        f"(95% CI {head['macro_f1_ci_low']:.3f}–{head['macro_f1_ci_high']:.3f}) · "
        f"{head['n_labels_measurable']} of {head['n_labels']} labels have enough "
        f"held-out examples to score"
    )

tab_try, tab_score, tab_labels, tab_travel = st.tabs(
    ["Try it", "The scoreboard", "Are the labels right?", "Does it travel?"]
)

# --------------------------------------------------------------------------
with tab_try:
    default = "thank you so much, this genuinely made my week"
    text = st.text_area("A comment", value=default, height=90)
    strategy = st.selectbox(
        "Threshold strategy",
        sorted(decoders),
        index=sorted(decoders).index(cfg.thresholds.report),
        help="the decision rule, which is a fitted object and not a constant",
    )
    if text.strip():
        decoder = decoders[strategy]
        norm = normalise(text)
        P = model.predict_proba([norm])[0]
        B = decoder.decode(P.reshape(1, -1))[0]
        chosen = [names[j] for j in np.nonzero(B)[0]]

        st.markdown("**Predicted:** " + (", ".join(f"`{c}`" for c in chosen) or "_nothing_"))
        top = (
            pd.DataFrame(
                {
                    "label": names,
                    "score": P,
                    "threshold": decoder.thresholds,
                    "selected": np.where(B.astype(bool), "predicted", "below threshold"),
                }
            )
            .sort_values("score", ascending=False)
            .head(10)
        )
        st.altair_chart(
            alt.Chart(top)
            .mark_bar()
            .encode(
                y=alt.Y("label:N", sort="-x", title=None),
                x=alt.X("score:Q", title="score", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "selected:N",
                    scale=alt.Scale(domain=["predicted", "below threshold"], range=[BLUE, MUTED]),
                    legend=alt.Legend(orient="top", title=None),
                ),
                tooltip=[
                    "label",
                    alt.Tooltip("score:Q", format=".3f"),
                    alt.Tooltip("threshold:Q", format=".2f"),
                ],
            )
            .properties(height=360),
            use_container_width=True,
        )
        if chosen:
            j = names.index(chosen[0])
            st.markdown(f"**Why `{chosen[0]}`** — exact contributions (coefficient × tf-idf):")
            st.dataframe(
                pd.DataFrame(model.explain(norm, j, k=8)), use_container_width=True, hide_index=True
            )

# --------------------------------------------------------------------------
with tab_score:
    if head:
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "macro-F1",
            f"{head['macro_f1']:.4f}",
            f"±{head['macro_f1_ci_width'] / 2:.4f} (95% CI)",
            delta_color="off",
        )
        c2.metric(
            "micro-F1",
            f"{head['micro_f1']:.4f}",
            f"±{head['micro_f1_ci_width'] / 2:.4f}",
            delta_color="off",
        )
        c3.metric(
            "labels actually measurable",
            f"{head['n_labels_measurable']} / {head['n_labels']}",
            f"needs ≥{head['min_support_measurable']} held-out positives",
            delta_color="off",
        )
        st.caption(
            "The interval is wider than the gap between most published methods on "
            "this benchmark. That is the finding, not a caveat."
        )

    if not per_label.empty:
        t = per_label.sort_values("support")
        base = alt.Chart(t).encode(
            y=alt.Y("label:N", sort=alt.EncodingSortField("support"), title=None),
            color=alt.Color(
                "measurable:N",
                scale=alt.Scale(domain=[True, False], range=[BLUE, ORANGE]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        st.altair_chart(
            (
                base.mark_rule(strokeWidth=3, opacity=0.4).encode(
                    x=alt.X("f1_ci_low:Q", title="F1", scale=alt.Scale(domain=[0, 1])),
                    x2="f1_ci_high:Q",
                )
                + base.mark_point(size=90, filled=True).encode(
                    x="f1:Q",
                    tooltip=[
                        "label",
                        "support",
                        alt.Tooltip("f1:Q", format=".3f"),
                        alt.Tooltip("f1_ci_low:Q", format=".3f"),
                        alt.Tooltip("f1_ci_high:Q", format=".3f"),
                    ],
                )
            ).properties(height=620, title="Per-label F1 with 95% bootstrap intervals"),
            use_container_width=True,
        )

    if metrics.get("decoders"):
        st.subheader("Threshold strategies")
        d = pd.DataFrame(metrics["decoders"]).T.reset_index(names="strategy")
        st.dataframe(
            d[
                [
                    "strategy",
                    "macro_f1",
                    "macro_f1_ci_low",
                    "macro_f1_ci_high",
                    "micro_f1",
                    "fitted_on",
                ]
            ].style.format(
                {
                    c: "{:.4f}"
                    for c in ("macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "micro_f1")
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    if metrics.get("paired"):
        st.subheader("Paired comparisons")
        p = pd.DataFrame(metrics["paired"]).T.reset_index(names="comparison")
        st.dataframe(
            p[["comparison", "difference", "ci_low", "ci_high", "significant"]].style.format(
                {c: "{:+.4f}" for c in ("difference", "ci_low", "ci_high")}
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Paired on identical bootstrap resamples. Comparing two separate "
            "confidence intervals instead would throw away most of the power."
        )
    for name, cap in (
        ("taxonomy_views.png", "The same predictions, three taxonomies"),
        ("confusions.png", "What it says instead"),
    ):
        if fig_path(name).exists():
            st.image(str(fig_path(name)), caption=cap)

# --------------------------------------------------------------------------
with tab_labels:
    audit = (
        json.loads((cfg.resolve(cfg.paths.reports_dir) / "audit_metrics.json").read_text())
        if (cfg.resolve(cfg.paths.reports_dir) / "audit_metrics.json").exists()
        else {}
    )
    if audit:
        fam = audit.get("family_breakdown", {})
        c1, c2 = st.columns(2)
        c1.metric("assignments flagged", f"{audit['flagged']:,}")
        if fam:
            c2.metric(
                "flags inside the same emotion family",
                f"{fam['share_same_family']:.0%}",
                "a shade, not an error",
                delta_color="off",
            )
    if not noise.empty:
        st.altair_chart(
            alt.Chart(noise.head(16))
            .mark_bar(color=BLUE)
            .encode(
                y=alt.Y("label:N", sort="-x", title=None),
                x=alt.X(
                    "disputed_share_of_positives:Q",
                    title="share disputed",
                    axis=alt.Axis(format="%"),
                ),
                tooltip=["label", "support", "spurious", "missing"],
            )
            .properties(height=420, title="Assignments the model confidently disputes"),
            use_container_width=True,
        )
    if not suspects.empty:
        st.subheader("Browse the flags")
        kind = st.radio("Kind", sorted(suspects["kind"].unique()), horizontal=True)
        st.dataframe(
            suspects[suspects["kind"] == kind][
                ["text", "given_labels", "label", "p", "threshold", "confidence"]
            ]
            .head(60)
            .style.format({c: "{:.3f}" for c in ("p", "threshold", "confidence")}),
            use_container_width=True,
            hide_index=True,
        )

# --------------------------------------------------------------------------
with tab_travel:
    if transfer_meta.get("leakage"):
        lk = transfer_meta["leakage"]
        c1, c2 = st.columns(2)
        c1.metric(
            "tweets decidable by emoticon alone", f"{lk['share_decidable_by_emoticon_alone']:.1%}"
        )
        c2.metric(
            "accuracy of that rule where it fires",
            f"{lk['accuracy_of_emoticon_rule_where_it_fires']:.1%}",
        )
        st.caption(
            "NLTK's tweet labels come from emoticons that are still in the text. "
            "Evaluate without stripping them and you are measuring emoticon "
            "detection — which is why the leaky row below looks so good."
        )
    if not transfer.empty:
        st.dataframe(
            transfer[["model", "evaluated_on", "emoticons", "n", "accuracy", "f1"]].style.format(
                {"accuracy": "{:.3f}", "f1": "{:.3f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )
    if fig_path("transfer.png").exists():
        st.image(str(fig_path("transfer.png")))
    if fig_path("active_learning.png").exists():
        st.subheader("Annotation budget")
        st.image(str(fig_path("active_learning.png")))
