"""Pipeline stages. Each reads from disk and writes to disk.

download -> train -> audit -> transfer -> active -> report
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nuance.active.simulate import budget_to_reach, run_simulation
from nuance.config import Config, load_config
from nuance.data.goemotions import load_corpus
from nuance.data.twitter import leakage_summary, load_transfer_set
from nuance.evaluation.confusion import coarse_gain, confusion_pairs, family_error_split
from nuance.evaluation.metrics import headline, paired_comparison, per_label_table
from nuance.models.decode import build_decoders
from nuance.models.train import build_model, out_of_fold_predictions
from nuance.noise.confident_learning import (
    estimated_noise_rates,
    family_breakdown,
    rows_to_drop,
    suspect_labels,
)
from nuance.reporting import plots

log = logging.getLogger(__name__)

LADDER = (
    "prior",
    "most_frequent",
    "word_logistic",
    "wordchar_logistic",
    "wordchar_logistic_unweighted",
    "wordchar_svm",
)
CHAMPION = "wordchar_logistic"


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_jsonable))


# --------------------------------------------------------------------------
# stage 1: train and evaluate
# --------------------------------------------------------------------------
def run_train(cfg: Config) -> dict:
    log.info("=== train ===")
    cfg.ensure_dirs()
    rng = np.random.default_rng(cfg.seed)
    reports = cfg.resolve(cfg.paths.reports_dir)
    figures = cfg.resolve(cfg.paths.figures_dir)
    models_dir = cfg.resolve(cfg.paths.models_dir)

    corpus = load_corpus(cfg)
    names = corpus.taxonomy.fine
    train, dev, test = corpus.train, corpus.dev, corpus.test
    corpus.support("fine").to_csv(reports / "label_support.csv")

    boot = cfg.evaluation.bootstrap_samples
    min_sup = cfg.evaluation.min_support_measurable
    results: dict = {"ladder": {}, "decoders": {}, "paired": {}}

    # --- the model ladder, all at the conventional 0.5 -------------------
    fitted, probabilities = {}, {}
    for name in LADDER:
        t0 = time.time()
        model = build_model(name, cfg).fit(train.text, train.Y)
        P_test = model.predict_proba(test.text)
        fitted[name] = model
        probabilities[name] = {"dev": model.predict_proba(dev.text), "test": P_test}
        B = (P_test >= 0.5).astype(np.int8)
        results["ladder"][name] = headline(test.Y, B, min_support=min_sup, n_boot=boot, rng=rng) | {
            "seconds": round(time.time() - t0, 1)
        }
        log.info(
            "%-30s macro %.4f [%.4f, %.4f]  micro %.4f  (%.0fs)",
            name,
            results["ladder"][name]["macro_f1"],
            results["ladder"][name]["macro_f1_ci_low"],
            results["ladder"][name]["macro_f1_ci_high"],
            results["ladder"][name]["micro_f1"],
            results["ladder"][name]["seconds"],
        )

    champion = fitted[CHAMPION]
    P_dev, P_test = probabilities[CHAMPION]["dev"], probabilities[CHAMPION]["test"]

    # --- out-of-fold predictions: thresholds and the audit both need them -
    t0 = time.time()
    P_oof = out_of_fold_predictions(train.text, train.Y, cfg, CHAMPION)
    np.save(cfg.resolve(cfg.data.processed_dir) / "oof_probabilities.npy", P_oof)
    log.info("out-of-fold predictions in %.0fs", time.time() - t0)

    # --- threshold strategies --------------------------------------------
    decoders = build_decoders(
        n_labels=len(names),
        grid=cfg.thresholds.grid,
        Y_dev=dev.Y,
        P_dev=P_dev,
        Y_oof=train.Y,
        P_oof=P_oof,
    )
    for strategy, decoder in decoders.items():
        B = decoder.decode(P_test)
        results["decoders"][strategy] = headline(
            test.Y, B, min_support=min_sup, n_boot=boot, rng=rng
        ) | {"fitted_on": decoder.fitted_on, **decoder.meta}
        log.info(
            "decoder %-14s macro %.4f  micro %.4f  (thresholds %.2f-%.2f)",
            strategy,
            results["decoders"][strategy]["macro_f1"],
            results["decoders"][strategy]["micro_f1"],
            decoder.meta["min"],
            decoder.meta["max"],
        )

    # The empty-prediction fallback, on top of the reported strategy.
    report_decoder = decoders[cfg.thresholds.report]
    fallback = type(report_decoder)(
        report_decoder.thresholds,
        report_decoder.strategy,
        True,
        report_decoder.fitted_on,
        dict(report_decoder.meta),
    )
    results["decoders"]["flat_plus_argmax_fallback"] = headline(
        test.Y, fallback.decode(P_test), min_support=min_sup, n_boot=boot, rng=rng
    ) | {"fitted_on": "nothing (every comment has >=1 label by construction)"}

    # --- paired comparisons: what is actually distinguishable ------------
    B_report = report_decoder.decode(P_test)
    for other in ("most_frequent", "word_logistic", "wordchar_svm", "wordchar_logistic_unweighted"):
        results["paired"][f"{CHAMPION}_vs_{other}"] = paired_comparison(
            test.Y,
            B_report,
            (probabilities[other]["test"] >= 0.5).astype(np.int8),
            n_boot=boot,
            rng=rng,
        )
    for strategy in ("per_label_dev", "per_label_oof", "global_tuned"):
        results["paired"][f"{strategy}_vs_flat"] = paired_comparison(
            test.Y,
            decoders[strategy].decode(P_test),
            decoders["flat"].decode(P_test),
            n_boot=boot,
            rng=rng,
        )
    for name, cmp in results["paired"].items():
        log.info(
            "%-45s %+.4f [%+.4f, %+.4f] %s",
            name,
            cmp["difference"],
            cmp["ci_low"],
            cmp["ci_high"],
            "significant" if cmp["significant"] else "not distinguishable",
        )

    # --- per-label detail --------------------------------------------------
    table = per_label_table(test.Y, B_report, names, min_support=min_sup, n_boot=boot, rng=rng)
    table.to_csv(reports / "per_label.csv", index=False)
    results["per_label_summary"] = {
        "labels": int(len(table)),
        "measurable": int(table["measurable"].sum()),
        "median_ci_width": float(table["f1_ci_width"].median()),
        "median_ci_width_rare": float(table.loc[~table["measurable"], "f1_ci_width"].median()),
        "median_ci_width_common": float(table.loc[table["measurable"], "f1_ci_width"].median()),
    }

    # --- where the errors land --------------------------------------------
    results["errors"] = {
        "family_split": family_error_split(test.Y, B_report, names, corpus.taxonomy),
        "coarse": {
            v: coarse_gain(test.Y, B_report, corpus.taxonomy, v) for v in ("ekman", "sentiment")
        },
    }
    pairs = confusion_pairs(test.Y, B_report, names, top_k=25)
    pairs.to_csv(reports / "confusion_pairs.csv", index=False)

    # --- figures ------------------------------------------------------------
    plots.fig_per_label(table, figures / "per_label_f1.png")
    plots.fig_decoders(
        pd.DataFrame(results["decoders"]).T.reset_index(names="strategy"),
        figures / "threshold_strategies.png",
    )
    plots.fig_coarse(
        results["errors"]["coarse"], results["ladder"][CHAMPION], figures / "taxonomy_views.png"
    )
    plots.fig_confusion(pairs, corpus.taxonomy, figures / "confusions.png")

    # --- artifacts ----------------------------------------------------------
    joblib.dump(champion, models_dir / "champion.joblib")
    joblib.dump(decoders, models_dir / "decoders.joblib")
    np.save(cfg.resolve(cfg.data.processed_dir) / "test_probabilities.npy", P_test)
    _write_json(
        {
            "model": CHAMPION,
            "labels": names,
            "decoder": cfg.thresholds.report,
            "thresholds": report_decoder.thresholds.tolist(),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test": results["ladder"][CHAMPION],
        },
        models_dir / "model_card.json",
    )
    _write_json(results, reports / "model_metrics.json")
    return results


# --------------------------------------------------------------------------
# stage 2: label-noise audit
# --------------------------------------------------------------------------
def run_audit(cfg: Config) -> dict:
    log.info("=== audit ===")
    reports = cfg.resolve(cfg.paths.reports_dir)
    corpus = load_corpus(cfg)
    names = corpus.taxonomy.fine
    P_oof = np.load(cfg.resolve(cfg.data.processed_dir) / "oof_probabilities.npy")

    rates = estimated_noise_rates(corpus.train.Y, P_oof, names)
    rates.to_csv(reports / "noise_rates.csv", index=False)

    suspects = suspect_labels(
        corpus.train.Y, P_oof, corpus.train.raw_text, names, top_k=cfg.noise.top_k
    )
    suspects.to_csv(reports / "suspected_label_errors.csv", index=False)

    out = {
        "flagged": int(len(suspects)),
        "family_breakdown": family_breakdown(suspects, corpus.taxonomy),
        "worst_labels": rates.head(6).to_dict(orient="records"),
        "examples": suspects.head(12).to_dict(orient="records"),
    }

    # Does acting on the audit help? Drop the implicated training rows, refit,
    # and score on the untouched test set.
    drop = rows_to_drop(corpus.train.Y, P_oof)
    keep = np.setdiff1d(np.arange(len(corpus.train.Y)), drop)
    model = build_model(CHAMPION, cfg).fit(corpus.train.text.to_numpy()[keep], corpus.train.Y[keep])
    B = (model.predict_proba(corpus.test.text) >= 0.5).astype(np.int8)
    rng = np.random.default_rng(cfg.seed)
    cleaned = headline(
        corpus.test.Y,
        B,
        min_support=cfg.evaluation.min_support_measurable,
        n_boot=cfg.evaluation.bootstrap_samples,
        rng=rng,
    )
    baseline = json.loads((reports / "model_metrics.json").read_text())["ladder"][CHAMPION]
    P_full = np.load(cfg.resolve(cfg.data.processed_dir) / "test_probabilities.npy")
    out["retrain_without_suspects"] = {
        "rows_dropped": int(len(drop)),
        "share_of_training_set": float(len(drop) / len(corpus.train.Y)),
        "macro_f1_before": baseline["macro_f1"],
        "macro_f1_after": cleaned["macro_f1"],
        "paired": paired_comparison(
            corpus.test.Y,
            B,
            (P_full >= 0.5).astype(np.int8),
            n_boot=cfg.evaluation.bootstrap_samples,
            rng=rng,
        ),
    }
    log.info(
        "dropping %d suspected rows: macro %.4f -> %.4f (%s)",
        len(drop),
        baseline["macro_f1"],
        cleaned["macro_f1"],
        "significant"
        if out["retrain_without_suspects"]["paired"]["significant"]
        else "not distinguishable",
    )
    plots.fig_noise(rates, cfg.resolve(cfg.paths.figures_dir) / "label_noise.png")
    _write_json(out, reports / "audit_metrics.json")
    return out


# --------------------------------------------------------------------------
# stage 3: domain transfer
# --------------------------------------------------------------------------
def run_transfer(cfg: Config) -> dict:
    from nuance.transfer.sentiment import run_transfer as _run

    log.info("=== transfer ===")
    reports = cfg.resolve(cfg.paths.reports_dir)
    corpus = load_corpus(cfg)
    tweets = load_transfer_set(cfg)

    table = _run(cfg, corpus, tweets)
    table.to_csv(reports / "transfer.csv", index=False)
    out = {"leakage": leakage_summary(tweets), "results": table.to_dict(orient="records")}
    _write_json(out, reports / "transfer_metrics.json")
    plots.fig_transfer(table, cfg.resolve(cfg.paths.figures_dir) / "transfer.png")
    for r in out["results"]:
        log.info(
            "%-40s %-26s %-16s acc %.3f",
            r["model"],
            r["evaluated_on"],
            r["emoticons"],
            r["accuracy"],
        )
    return out


# --------------------------------------------------------------------------
# stage 4: labelling budget
# --------------------------------------------------------------------------
def run_active(cfg: Config) -> dict:
    log.info("=== active ===")
    reports = cfg.resolve(cfg.paths.reports_dir)
    corpus = load_corpus(cfg)
    curve = run_simulation(
        cfg,
        corpus.train.text.to_numpy(),
        corpus.train.Y,
        corpus.test.text.to_numpy(),
        corpus.test.Y,
    )
    curve.to_csv(reports / "active_learning.csv", index=False)
    reach = budget_to_reach(curve)
    reach.to_csv(reports / "active_budget.csv", index=False)
    plots.fig_active(curve, cfg.resolve(cfg.paths.figures_dir) / "active_learning.png")
    out = {"budget_to_reach_95pct": reach.to_dict(orient="records")}
    _write_json(out, reports / "active_metrics.json")
    return out


# --------------------------------------------------------------------------
# stage 5: report
# --------------------------------------------------------------------------
def run_report(cfg: Config) -> Path:
    from nuance.reporting.summary import write_results_markdown

    return write_results_markdown(cfg)


def run_all(cfg: Config | None = None) -> None:
    cfg = cfg or load_config()
    timings = {}
    for name, fn in (
        ("train", run_train),
        ("audit", run_audit),
        ("transfer", run_transfer),
        ("active", run_active),
        ("report", run_report),
    ):
        t0 = time.time()
        fn(cfg)
        timings[name] = round(time.time() - t0, 1)
        log.info("stage %s finished in %.1fs", name, timings[name])
    log.info("pipeline complete: %s", timings)
