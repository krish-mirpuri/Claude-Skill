"""End-to-end pipeline stages.

Each stage reads from disk and writes to disk, so any stage can be re-run on
its own and every reported number has a file behind it.

    prepare  ->  train  ->  backtest  ->  report
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from overbook.config import Config, load_config
from overbook.data.clean import build_processed, load_processed
from overbook.decision import backtest as bt
from overbook.decision.overdispersion import estimate_shared_shock
from overbook.decision.policy import (
    Distributional,
    Economics,
    ExpectedArrivals,
    MeanRate,
    NoOverbook,
    Oracle,
    diagnose,
)
from overbook.features.build import build_features
from overbook.models.calibrate import Calibrator, rolling_recalibrate
from overbook.models.evaluate import (
    classification_metrics,
    reliability_table,
    segment_metrics,
)
from overbook.models.train import (
    cross_validate_lightgbm,
    save_artifacts,
    train_lightgbm,
    train_logistic,
    train_prior,
)
from overbook.monitoring.drift import drift_report, monthly_stability
from overbook.reporting import plots

log = logging.getLogger(__name__)

PREDICTIONS_FILE = "predictions.parquet"


@dataclass
class Stage:
    name: str
    seconds: float


def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_jsonable))


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return str(o)


# --------------------------------------------------------------------------
# stage 1: prepare
# --------------------------------------------------------------------------
def run_prepare(cfg: Config) -> pd.DataFrame:
    log.info("=== prepare ===")
    return build_processed(cfg)


# --------------------------------------------------------------------------
# stage 2: train
# --------------------------------------------------------------------------
def run_train(cfg: Config) -> dict:
    log.info("=== train ===")
    cfg.ensure_dirs()
    df = load_processed(cfg)
    reports = cfg.resolve(cfg.paths.reports_dir)

    results: dict = {"views": {}, "models": {}}

    # --- the value of waiting: same model, two information sets -----------
    for view in ("booking_time", "arrival_eve"):
        ff, fb = build_features(df, cfg, view=view)
        model = train_lightgbm(ff.split("train"), cfg)
        view_metrics = {}
        for split in ("valid", "test"):
            s = ff.split(split)
            p = model.predict_proba(s.X)
            view_metrics[split] = classification_metrics(s.y.to_numpy(), p)
        results["views"][view] = view_metrics
        log.info(
            "view=%-13s test AUC=%.4f  Brier=%.4f",
            view,
            view_metrics["test"]["roc_auc"],
            view_metrics["test"]["brier"],
        )
        if view == cfg.features.view:
            primary = (ff, fb, model)

    ff, fb, lgbm = primary
    tr, va, te = ff.split("train"), ff.split("valid"), ff.split("test")

    # --- the model ladder --------------------------------------------------
    for name, model in (
        ("prior", train_prior(tr)),
        ("logistic", train_logistic(tr, cfg)),
        ("lightgbm", lgbm),
    ):
        results["models"][name] = {
            split_name: classification_metrics(s.y.to_numpy(), model.predict_proba(s.X))
            for split_name, s in (("valid", va), ("test", te))
        }
        log.info(
            "%-9s test AUC=%.4f  Brier=%.4f  ECE=%.4f",
            name,
            results["models"][name]["test"]["roc_auc"],
            results["models"][name]["test"]["brier"],
            results["models"][name]["test"]["ece"],
        )

    # --- rolling-origin CV inside the training window ----------------------
    cv = cross_validate_lightgbm(tr, cfg)
    cv.to_csv(reports / "cv_folds.csv", index=False)
    results["cv"] = {
        "folds": int(len(cv)),
        "roc_auc_mean": float(cv["roc_auc"].mean()),
        "roc_auc_std": float(cv["roc_auc"].std(ddof=1)),
        "brier_mean": float(cv["brier"].mean()),
    }

    # --- calibration, fitted on valid only ---------------------------------
    p_valid_raw = lgbm.predict_proba(va.X)
    calibrator = Calibrator(cfg.model.calibration).fit(p_valid_raw, va.y.to_numpy())

    p_test_raw = lgbm.predict_proba(te.X)
    p_test_cal = calibrator.predict(p_test_raw)
    results["calibration"] = {
        "method": cfg.model.calibration,
        "fitted_on": f"{cfg.split.valid[0]}..{cfg.split.valid[1]}",
        "test_raw": classification_metrics(te.y.to_numpy(), p_test_raw),
        "test_calibrated": classification_metrics(te.y.to_numpy(), p_test_cal),
    }
    log.info(
        "calibration on test: Brier %.5f -> %.5f | ECE %.4f -> %.4f",
        results["calibration"]["test_raw"]["brier"],
        results["calibration"]["test_calibrated"]["brier"],
        results["calibration"]["test_raw"]["ece"],
        results["calibration"]["test_calibrated"]["ece"],
    )

    # --- persist artifacts and predictions ---------------------------------
    models_dir = cfg.resolve(cfg.paths.models_dir)
    save_artifacts(
        lgbm,
        fb,
        cfg,
        extra={
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "train_window": [str(d) for d in cfg.split.train],
            "valid_window": [str(d) for d in cfg.split.valid],
            "test_window": [str(d) for d in cfg.split.test],
            "test_metrics": results["calibration"]["test_calibrated"],
        },
    )
    joblib.dump(calibrator, models_dir / "calibrator.joblib")

    p_all_raw = lgbm.predict_proba(ff.X)
    preds = ff.keys.copy()
    preds["p_raw"] = p_all_raw
    preds["p_cal"] = calibrator.predict(p_all_raw)
    preds.to_parquet(cfg.resolve(cfg.data.processed_path).parent / PREDICTIONS_FILE, index=False)

    # --- diagnostics and figures -------------------------------------------
    rel_before = reliability_table(te.y.to_numpy(), p_test_raw)
    rel_after = reliability_table(te.y.to_numpy(), p_test_cal)
    rel_before.to_csv(reports / "reliability_raw.csv", index=False)
    rel_after.to_csv(reports / "reliability_calibrated.csv", index=False)

    figures = cfg.resolve(cfg.paths.figures_dir)
    plots.fig_reliability(rel_before, rel_after, figures / "reliability.png")

    # Aggregate metrics can hide a segment the decision layer relies on, so
    # every meaningful slice is scored separately.
    slices = df.set_index("booking_id").loc[te.keys["booking_id"]].reset_index()
    seg = pd.concat(
        [
            segment_metrics(te.y.to_numpy(), p_test_cal, slices[col]).assign(dimension=col)
            for col in ("hotel", "market_segment", "deposit_type", "customer_type")
        ],
        ignore_index=True,
    )
    seg.to_csv(reports / "segment_metrics.csv", index=False)
    results["worst_segment_ece"] = (
        seg.sort_values("ece")
        .tail(3)[["dimension", "segment", "n", "ece", "bias_ratio"]]
        .to_dict(orient="records")
    )

    stability = monthly_stability(
        pd.concat([va.keys, te.keys], ignore_index=True),
        np.concatenate([calibrator.predict(p_valid_raw), p_test_cal]),
    )
    stability.to_csv(reports / "monthly_stability.csv", index=False)
    plots.fig_monthly_stability(
        stability,
        figures / "monthly_stability.png",
        split_at=str(pd.Timestamp(cfg.split.test[0]).to_period("M")),
    )

    drift = drift_report(tr.X, te.X, ff.categorical)
    drift.to_csv(reports / "drift.csv", index=False)
    results["drift_top"] = drift.head(8).to_dict(orient="records")

    # --- what the model leans on -------------------------------------------
    try:
        from overbook.explain.shap_report import global_importance

        imp = global_importance(lgbm.booster, te.X, sample=4000)
        imp.to_csv(reports / "shap_importance.csv")
        plots.fig_importance(imp, figures / "importance.png")
        results["top_features"] = imp.head(10).round(4).to_dict()
    except Exception as exc:  # shap is optional; never fail the pipeline for it
        log.warning("SHAP step skipped: %s", exc)

    _lead_time_figure(df, cfg, figures)

    # Snapshot the state the API needs but a request cannot carry.
    from overbook.api.context import build_serving_context

    build_serving_context(cfg)

    _write_json(results, reports / "model_metrics.json")
    return results


def _lead_time_figure(df: pd.DataFrame, cfg: Config, figures: Path) -> None:
    edges = [0, 7, 14, 30, 60, 90, 150, 240, 400, 800]
    b = pd.cut(df["lead_time"], edges, right=True, include_lowest=True)
    g = df.groupby(b, observed=True)["is_canceled"].agg(["size", "mean"])
    g = g[g["size"] >= 200]
    tbl = pd.DataFrame(
        {
            "lead_time_mid": [iv.mid for iv in g.index],
            "cancel_rate": g["mean"].to_numpy(),
            "n": g["size"].to_numpy(),
        }
    )
    plots.fig_lead_time_risk(tbl, figures / "lead_time_risk.png")


# --------------------------------------------------------------------------
# stage 3: backtest
# --------------------------------------------------------------------------
def load_predictions(cfg: Config) -> pd.DataFrame:
    path = cfg.resolve(cfg.data.processed_path).parent / PREDICTIONS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `overbook train` first.")
    return pd.read_parquet(path)


def _estimate_tau(cfg: Config, preds: pd.DataFrame, capacity: dict) -> float:
    """Estimate the shared nightly shock on the VALIDATION window only."""
    if not cfg.decision.overdispersion:
        return 0.0
    va = preds[preds["split"] == "valid"].reset_index(drop=True)
    states = bt.build_states(va, va["p_cal"].to_numpy(), capacity)
    mu = np.array([st.show_probs.sum() for st in states])
    pbv = np.array([(st.show_probs * (1 - st.show_probs)).sum() for st in states])
    act = np.array([st.showed.sum() for st in states])
    return estimate_shared_shock(mu, pbv, act)


def _decision_probabilities(cfg: Config, preds: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Probabilities the decision layer will use, plus a static/rolling comparison."""
    test = preds[preds["split"] == "test"].copy()
    comparison = {
        "static": classification_metrics(test["is_canceled"].to_numpy(), test["p_cal"].to_numpy())
    }
    if cfg.model.calibration_refresh:
        rolled = rolling_recalibrate(
            preds[preds["split"] == "valid"],
            test,
            period=cfg.model.calibration_refresh,
            method=cfg.model.calibration,
        )
        test = test.loc[rolled.index]
        test["p_decision"] = rolled.to_numpy()
        comparison["rolling"] = classification_metrics(
            test["is_canceled"].to_numpy(), test["p_decision"].to_numpy()
        )
    else:
        test["p_decision"] = test["p_cal"]
    return test.reset_index(drop=True), comparison


def _policies(cfg: Config, econ: Economics, pooled: float, tau: float) -> list:
    policies = bt.default_policies(cfg, econ, pooled)
    if tau > 0:
        # Insert the shock variant next to its independence-assuming twin.
        policies.insert(-1, Distributional(econ, tau=tau))
    return policies


def run_backtest(cfg: Config) -> dict:
    log.info("=== backtest ===")
    cfg.ensure_dirs()
    reports = cfg.resolve(cfg.paths.reports_dir)
    figures = cfg.resolve(cfg.paths.figures_dir)

    preds = load_predictions(cfg)
    train_keys = preds[preds["split"] == "train"]
    pooled_rate = float(train_keys["is_canceled"].mean())
    capacity = bt.estimate_capacity(train_keys, cfg.economics.capacity_quantile)
    econ = Economics(
        variable_cost_ratio=cfg.economics.variable_cost_ratio,
        walk_cost_multiplier=cfg.economics.walk_cost_multiplier,
    )

    test_keys, calibration_compare = _decision_probabilities(cfg, preds)
    tau = _estimate_tau(cfg, preds, capacity)

    states = bt.build_states(test_keys, test_keys["p_decision"].to_numpy(), capacity)
    binding = float(np.mean([st.showed.sum() > st.capacity for st in states]))
    log.info("capacity binds on %.1f%% of held-out nights", 100 * binding)

    policies = _policies(cfg, econ, pooled_rate, tau)
    results = bt.run_backtest(states, policies, econ)
    results.to_parquet(reports / "backtest_nights.parquet", index=False)

    summary = bt.summarise(results)
    summary.to_csv(reports / "policy_summary.csv")
    plots.fig_policy_profit(summary, figures / "policy_profit.png")

    champion = "distributional_shock" if tau > 0 else "distributional"
    n_boot, seed = cfg.backtest.bootstrap_samples, cfg.seed
    boot = [
        bt.bootstrap_uplift(results, p, "no_overbook", n_boot, seed)
        for p in summary.index
        if p != "no_overbook"
    ]
    pd.DataFrame(boot).to_csv(reports / "bootstrap_uplift.csv", index=False)

    # The comparisons that actually decide whether the method earns its keep.
    head_to_head = {
        "vs_mean_rate": bt.bootstrap_uplift(results, champion, "mean_rate", n_boot, seed),
        "vs_expected_arrivals": bt.bootstrap_uplift(
            results, champion, "expected_arrivals", n_boot, seed
        ),
        "vs_accept_all": bt.bootstrap_uplift(results, champion, "accept_all", n_boot, seed),
    }
    pd.DataFrame(head_to_head.values()).to_csv(reports / "head_to_head.csv", index=False)

    # --- one night, explained ----------------------------------------------
    # A representative *binding* night, not the single busiest one: the busiest
    # night in this data holds 291 bookings against 68 rooms, which produces a
    # true but ridiculous-looking "+309% over capacity" headline.
    contested = [st for st in states if st.showed.sum() > st.capacity] or states
    contested = sorted(contested, key=lambda st: st.n_bookings / max(st.capacity, 1))
    busiest = contested[len(contested) // 2]
    diag = diagnose(busiest, econ, tau=tau)
    marks = {
        pol.name: pol.limit(busiest)
        for pol in (NoOverbook(), MeanRate(pooled_rate), ExpectedArrivals(), Oracle())
    }
    plots.fig_decision_anatomy(diag, figures / "decision_anatomy.png", marks=marks)
    _write_json(
        {
            **{k: v for k, v in diag.items() if k not in ("profit_curve", "arrival_pmf")},
            "policy_limits": marks,
        },
        reports / "example_night.json",
    )

    sens = _sensitivity(cfg, test_keys, capacity, pooled_rate, tau)
    sens.to_csv(reports / "sensitivity.csv", index=False)
    plots.fig_walk_response(
        sens[sens["policy"].isin([champion, "expected_arrivals", "mean_rate"])],
        figures / "walk_response.png",
    )

    regimes = _capacity_sensitivity(cfg, train_keys, test_keys, pooled_rate, econ, tau)
    regimes.to_csv(reports / "sensitivity_capacity.csv", index=False)
    plots.fig_regimes(regimes, figures / "capacity_regimes.png", champion=champion)

    out = {
        "capacity": capacity,
        "capacity_binds_pct_of_nights": 100 * binding,
        "pooled_train_cancel_rate": pooled_rate,
        "shared_shock_tau": tau,
        "champion": champion,
        "calibration_for_decisions": calibration_compare,
        "economics": {
            "variable_cost_ratio": econ.variable_cost_ratio,
            "walk_cost_multiplier": econ.walk_cost_multiplier,
            "capacity_quantile": cfg.economics.capacity_quantile,
        },
        "nights": int(len(states)),
        "summary": summary.reset_index().to_dict(orient="records"),
        "bootstrap_vs_no_overbook": boot,
        "head_to_head": head_to_head,
        "example_night": {
            k: v for k, v in diag.items() if k not in ("profit_curve", "arrival_pmf")
        },
    }
    # The API must widen the arrival distribution exactly as the backtest did,
    # so the estimate travels with the model artifacts.
    card_path = cfg.resolve(cfg.paths.models_dir) / "model_card.json"
    if card_path.exists():
        card = json.loads(card_path.read_text())
        card["shared_shock_tau"] = tau
        card["capacity"] = capacity
        card_path.write_text(json.dumps(card, indent=2, default=_jsonable))

    _write_json(out, reports / "backtest_metrics.json")
    for name, b in head_to_head.items():
        log.info(
            "%s %s: %+.2f EUR/night (95%% CI %.2f..%.2f, P(>0)=%.2f)",
            champion,
            name,
            b["mean_uplift_per_night"],
            b["ci_low"],
            b["ci_high"],
            b["p_positive"],
        )
    return out


def _sensitivity(
    cfg: Config, test_keys: pd.DataFrame, capacity: dict, pooled_rate: float, tau: float
) -> pd.DataFrame:
    """Sweep the walk-cost assumption.

    The interesting column is ``walks_per_100_nights``: a mean-based rule
    cannot react to the cost of a walk at all, so its walk rate is flat across
    the sweep, while the distributional policy trades occupancy away as the
    penalty grows.
    """
    rows = []
    states = bt.build_states(test_keys, test_keys["p_decision"].to_numpy(), capacity)
    for mult in cfg.economics.sensitivity_walk_multipliers:
        econ = Economics(cfg.economics.variable_cost_ratio, mult)
        res = bt.run_backtest(states, _policies(cfg, econ, pooled_rate, tau), econ)
        s = bt.summarise(res)
        base = s.loc["no_overbook", "profit_per_room_night"]
        for policy in s.index:
            rows.append(
                {
                    "walk_cost_multiplier": mult,
                    "policy": policy,
                    "profit_per_room_night": float(s.loc[policy, "profit_per_room_night"]),
                    "uplift_per_room_night": float(s.loc[policy, "profit_per_room_night"] - base),
                    "walks_per_100_nights": float(s.loc[policy, "walks_per_100_nights"]),
                    "occupancy": float(s.loc[policy, "occupancy"]),
                }
            )
    return pd.DataFrame(rows)


def _capacity_sensitivity(
    cfg: Config,
    train_keys: pd.DataFrame,
    test_keys: pd.DataFrame,
    pooled_rate: float,
    econ: Economics,
    tau: float,
) -> pd.DataFrame:
    """Sweep capacity — it is inferred, and it sets how hard the problem is.

    At a loose capacity the constraint almost never binds and every policy
    converges on "accept everything"; the decision layer only has work to do
    when rooms are genuinely scarce.
    """
    rows = []
    for q in cfg.backtest.capacity_quantiles:
        cap = bt.estimate_capacity(train_keys, q)
        states = bt.build_states(test_keys, test_keys["p_decision"].to_numpy(), cap)
        binding = float(np.mean([st.showed.sum() > st.capacity for st in states]))
        s = bt.summarise(bt.run_backtest(states, _policies(cfg, econ, pooled_rate, tau), econ))
        base = s.loc["no_overbook", "profit_per_room_night"]
        for policy in s.index:
            rows.append(
                {
                    "capacity_quantile": q,
                    **{f"capacity_{k.replace(' ', '_').lower()}": v for k, v in cap.items()},
                    "binds_pct_of_nights": 100 * binding,
                    "policy": policy,
                    "profit_per_room_night": float(s.loc[policy, "profit_per_room_night"]),
                    "uplift_per_room_night": float(s.loc[policy, "profit_per_room_night"] - base),
                    "walks_per_100_nights": float(s.loc[policy, "walks_per_100_nights"]),
                    "pct_of_oracle_gap_closed": float(s.loc[policy, "pct_of_oracle_gap_closed"]),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# stage 4: report
# --------------------------------------------------------------------------
def run_report(cfg: Config) -> Path:
    from overbook.reporting.summary import write_results_markdown

    return write_results_markdown(cfg)


def run_all(cfg: Config | None = None) -> None:
    cfg = cfg or load_config()
    stages = []
    for name, fn in (
        ("prepare", run_prepare),
        ("train", run_train),
        ("backtest", run_backtest),
        ("report", run_report),
    ):
        t0 = time.time()
        fn(cfg)
        stages.append(Stage(name, time.time() - t0))
        log.info("stage %s finished in %.1fs", name, stages[-1].seconds)
    log.info("pipeline complete: %s", {s.name: round(s.seconds, 1) for s in stages})
