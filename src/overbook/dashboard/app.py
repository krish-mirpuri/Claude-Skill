"""Revenue-management console.

    streamlit run src/overbook/dashboard/app.py

Three views, in the order a revenue manager would use them: the decision for
one night, how the policies compare over the held-out period, and whether the
model underneath can still be trusted.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from overbook.config import load_config
from overbook.decision import backtest as bt
from overbook.decision.policy import (
    Economics,
    ExpectedArrivals,
    MeanRate,
    NoOverbook,
    Oracle,
    diagnose,
    realised_profit,
)
from overbook.pipeline import load_predictions

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MUTED, RED = "#8b8a85", "#e34948"

st.set_page_config(page_title="overbook", page_icon="🏨", layout="wide")
alt.data_transformers.disable_max_rows()


@st.cache_data(show_spinner=False)
def _load():
    cfg = load_config()
    preds = load_predictions(cfg)
    reports = cfg.resolve(cfg.paths.reports_dir)
    nights = pd.read_parquet(reports / "backtest_nights.parquet")
    summary = pd.read_csv(reports / "policy_summary.csv")
    meta = json.loads((reports / "backtest_metrics.json").read_text())
    model = json.loads((reports / "model_metrics.json").read_text())
    stability = pd.read_csv(reports / "monthly_stability.csv")
    drift = pd.read_csv(reports / "drift.csv")
    return cfg, preds, nights, summary, meta, model, stability, drift


def _fig_path(cfg, name: str) -> Path:
    return cfg.resolve(cfg.paths.figures_dir) / name


cfg, preds, nights, summary, meta, model_metrics, stability, drift = _load()
econ_default = meta["economics"]
tau = float(meta.get("shared_shock_tau") or 0.0)

st.title("overbook")
st.caption(
    f"Calibrated cancellation risk and revenue-optimal authorisation limits · "
    f"held-out arrivals {cfg.split.test[0]} to {cfg.split.test[1]} · "
    f"model ROC-AUC {model_metrics['calibration']['test_calibrated']['roc_auc']:.3f}, "
    f"ECE {model_metrics['calibration']['test_calibrated']['ece']:.3f}"
)

tab_night, tab_policy, tab_model = st.tabs(
    ["Tonight's decision", "Policy comparison", "Is the model still honest?"]
)

# --------------------------------------------------------------------------
with tab_night:
    test = preds[preds["split"] == "test"]
    left, right = st.columns([1, 2])
    with left:
        hotel = st.selectbox("Hotel", sorted(test["hotel"].unique()))
        sub = test[test["hotel"] == hotel]
        dates = sorted(sub["arrival_date"].dt.date.unique())
        busiest = sub.groupby(sub["arrival_date"].dt.date).size().idxmax()
        date = st.selectbox(
            "Arrival date",
            dates,
            index=dates.index(busiest),
            help="defaults to the busiest night in the held-out period",
        )
        capacity = st.number_input(
            "Rooms available",
            min_value=1,
            max_value=500,
            value=int(meta["capacity"][hotel]),
            help="inferred from training arrivals; override to explore",
        )
        walk_mult = st.slider(
            "Cost of walking a guest (× ADR)",
            0.5,
            8.0,
            float(econ_default["walk_cost_multiplier"]),
            0.5,
            help="the asymmetry that makes this a decision rather than a forecast",
        )
        margin_ratio = st.slider(
            "Variable cost (share of ADR)",
            0.0,
            0.6,
            float(econ_default["variable_cost_ratio"]),
            0.05,
        )

    econ = Economics(margin_ratio, walk_mult)
    night = sub[sub["arrival_date"].dt.date == date]
    state = bt.build_states(night, night["p_cal"].to_numpy(), {hotel: int(capacity)})[0]
    diag = diagnose(state, econ, tau=tau)

    with right:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("On the books", state.n_bookings)
        c2.metric("Authorise", diag["authorised"], f"{diag['overbook_pct']:+.0f}% vs capacity")
        c3.metric("P(oversell)", f"{diag['p_oversell']:.1%}")
        c4.metric("Expected walks", f"{diag['expected_walks']:.2f}")

        pmf = np.asarray(diag["arrival_pmf"])
        pmf_df = pd.DataFrame({"arrivals": np.arange(pmf.size), "probability": pmf}).query(
            "probability > 1e-5"
        )
        pmf_df["status"] = np.where(
            pmf_df["arrivals"] > capacity, "oversold — guests walked", "within capacity"
        )
        st.altair_chart(
            alt.Chart(pmf_df)
            .mark_bar()
            .encode(
                x=alt.X("arrivals:Q", title="guests arriving"),
                y=alt.Y(
                    "probability:Q", title="probability", axis=alt.Axis(tickCount=4, format=".3f")
                ),
                color=alt.Color(
                    "status:N",
                    scale=alt.Scale(
                        domain=["within capacity", "oversold — guests walked"],
                        range=[AQUA, RED],
                    ),
                    legend=alt.Legend(orient="top", title=None),
                ),
                tooltip=["arrivals", alt.Tooltip("probability:Q", format=".4f"), "status"],
            )
            .properties(height=260, title=f"How many actually turn up — {hotel}, {date}"),
            use_container_width=True,
        )

        curve = np.asarray(diag["profit_curve"])
        curve_df = pd.DataFrame({"authorised": np.arange(curve.size), "expected_profit": curve})
        base = (
            alt.Chart(curve_df)
            .mark_line(color=BLUE, strokeWidth=2)
            .encode(
                x=alt.X("authorised:Q", title="bookings authorised"),
                y=alt.Y(
                    "expected_profit:Q", title="expected profit (€)", scale=alt.Scale(zero=False)
                ),
                tooltip=["authorised", alt.Tooltip("expected_profit:Q", format=",.0f")],
            )
        )
        picked = (
            alt.Chart(curve_df.iloc[[diag["authorised"]]])
            .mark_point(size=140, color=ORANGE, filled=True)
            .encode(x="authorised:Q", y="expected_profit:Q")
        )
        st.altair_chart(
            (base + picked).properties(height=240, title="Expected profit by authorisation level"),
            use_container_width=True,
        )

    st.subheader("What each rule would have done on this night")
    pooled = float(meta["pooled_train_cancel_rate"])
    rows = []
    for pol in (NoOverbook(), MeanRate(pooled), ExpectedArrivals(), Oracle()):
        a = pol.limit(state)
        rows.append(
            {
                "rule": pol.name,
                "authorised": a,
                "realised profit (€)": realised_profit(state, a, econ),
            }
        )
    rows.append(
        {
            "rule": "distributional (this project)",
            "authorised": diag["authorised"],
            "realised profit (€)": realised_profit(state, diag["authorised"], econ),
        }
    )
    st.dataframe(
        pd.DataFrame(rows).style.format({"realised profit (€)": "{:,.0f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"Actual arrivals that night: **{int(state.showed.sum())}** against "
        f"{capacity} rooms. `oracle` is hindsight, not a policy."
    )

# --------------------------------------------------------------------------
with tab_policy:
    st.subheader("Held-out performance")
    show = summary.copy()
    st.dataframe(
        show[
            [
                "policy",
                "mean_authorised",
                "occupancy",
                "walks_per_100_nights",
                "profit_per_room_night",
                "uplift_pct",
                "pct_of_oracle_gap_closed",
            ]
        ].style.format(
            {
                "mean_authorised": "{:.1f}",
                "occupancy": "{:.1%}",
                "walks_per_100_nights": "{:.1f}",
                "profit_per_room_night": "{:,.2f}",
                "uplift_pct": "{:+.1f}%",
                "pct_of_oracle_gap_closed": "{:.0f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    h2h = meta["head_to_head"]
    cols = st.columns(len(h2h))
    for col, (name, b) in zip(cols, h2h.items(), strict=True):
        col.metric(
            name.replace("vs_", "vs "),
            f"{b['mean_uplift_per_night']:+,.0f} €/night",
            f"95% CI {b['ci_low']:+,.0f} … {b['ci_high']:+,.0f}",
            delta_color="off",
        )
    st.caption(
        "The negative row is real and kept deliberately: against the same model with "
        "the decision collapsed to a mean, the distributional policy is not ahead on "
        "profit at the default walk cost. What it buys is a far lower walk rate and "
        "the ability to respond when that cost changes."
    )
    for name, caption in (
        ("policy_profit.png", "Profit per available room-night by policy"),
        ("walk_response.png", "Only the distributional policy reacts to the cost of a walk"),
        ("capacity_regimes.png", "The decision layer earns its keep when rooms are scarce"),
    ):
        p = _fig_path(cfg, name)
        if p.exists():
            st.image(str(p), caption=caption, use_container_width=True)

# --------------------------------------------------------------------------
with tab_model:
    st.subheader("Calibration")
    c1, c2 = st.columns(2)
    cal = model_metrics["calibration"]
    with c1:
        st.metric(
            "Brier (test)",
            f"{cal['test_calibrated']['brier']:.4f}",
            f"{cal['test_calibrated']['brier'] - cal['test_raw']['brier']:+.4f} vs raw",
            delta_color="inverse",
        )
    with c2:
        st.metric(
            "Calibration error (ECE)",
            f"{cal['test_calibrated']['ece']:.4f}",
            f"{cal['test_calibrated']['ece'] - cal['test_raw']['ece']:+.4f} vs raw",
            delta_color="inverse",
        )
    p = _fig_path(cfg, "reliability.png")
    if p.exists():
        st.image(str(p), use_container_width=False)

    st.subheader("Does it hold as the season turns?")
    melt = stability.melt("month", ["observed", "predicted"], var_name="series", value_name="rate")
    st.altair_chart(
        alt.Chart(melt)
        .mark_line(strokeWidth=2, point=True)
        .encode(
            x=alt.X("month:N", title=None),
            y=alt.Y("rate:Q", title="cancellation rate"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(range=[BLUE, ORANGE]),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=["month", "series", alt.Tooltip("rate:Q", format=".3f")],
        )
        .properties(height=260),
        use_container_width=True,
    )
    st.caption(
        "The cancellation rate climbs into the held-out period. A frozen calibrator "
        "drifts with it, which is why the decision layer refits monthly on arrivals "
        "that have already resolved."
    )

    st.subheader("Feature drift, train → held-out")
    st.dataframe(
        drift.head(12)[["feature", "kind", "psi", "band"]].style.format({"psi": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )
