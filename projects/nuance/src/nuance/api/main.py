"""Scoring service.

``POST /classify`` returns, for each comment: the labels the decoder actually
selects, every label's calibrated score, and — because the model is linear —
the *exact* n-gram decomposition of the winning scores rather than an
approximation of it.

The response deliberately carries the decoder's provenance and the label's
test support. A caller acting on ``grief`` deserves to know that the estimate
behind it rests on six held-out examples.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException

from nuance.api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    CommentPrediction,
    HealthResponse,
    LabelScore,
)
from nuance.config import load_config

log = logging.getLogger(__name__)

app = FastAPI(
    title="nuance",
    version="0.1.0",
    summary="Multi-label emotion classification, scored honestly.",
)


@lru_cache(maxsize=1)
def _bundle():
    import joblib

    cfg = load_config()
    models_dir = cfg.resolve(cfg.paths.models_dir)
    card_path = models_dir / "model_card.json"
    if not card_path.exists():
        raise FileNotFoundError(f"{card_path} not found — run `nuance train` first.")
    card = json.loads(card_path.read_text())
    model = joblib.load(models_dir / "champion.joblib")
    decoders = joblib.load(models_dir / "decoders.joblib")

    support = {}
    per_label = cfg.resolve(cfg.paths.reports_dir) / "per_label.csv"
    if per_label.exists():
        import pandas as pd

        df = pd.read_csv(per_label)
        support = {
            r.label: {
                "test_support": int(r.support),
                "f1": float(r.f1),
                "f1_ci_low": float(r.f1_ci_low),
                "f1_ci_high": float(r.f1_ci_high),
                "measurable": bool(r.measurable),
            }
            for r in df.itertuples()
        }
    return cfg, model, decoders, card, support


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        cfg, _model, decoders, card, _ = _bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    test = card.get("test", {})
    return HealthResponse(
        status="ok",
        model=card.get("model", "unknown"),
        trained_at=card.get("trained_at"),
        n_labels=len(card.get("labels", [])),
        decoders=sorted(decoders),
        default_decoder=cfg.thresholds.report,
        macro_f1=test.get("macro_f1"),
        macro_f1_ci=[test.get("macro_f1_ci_low"), test.get("macro_f1_ci_high")]
        if test.get("macro_f1_ci_low") is not None
        else None,
    )


@app.get("/labels")
def labels() -> dict:
    """The taxonomy, with how well each label is actually known."""
    _cfg, _model, _dec, card, support = _bundle()
    return {
        "labels": [{"label": name, **support.get(name, {})} for name in card["labels"]],
        "note": "labels flagged measurable=false have too few held-out examples "
        "for their F1 to be a measurement",
    }


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    cfg, model, decoders, card, support = _bundle()
    strategy = req.decoder or cfg.thresholds.report
    if strategy not in decoders:
        raise HTTPException(
            status_code=422,
            detail=f"unknown decoder {strategy!r}; available: {sorted(decoders)}",
        )
    decoder = decoders[strategy]
    names = card["labels"]

    from nuance.features.text import normalise

    texts = [normalise(t) for t in req.texts]
    try:
        P = model.predict_proba(texts)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"scoring failed: {exc}") from exc
    B = decoder.decode(P)

    predictions = []
    for i, text in enumerate(texts):
        selected = np.nonzero(B[i])[0]
        order = np.argsort(P[i])[::-1][: req.top_k]
        predictions.append(
            CommentPrediction(
                text=req.texts[i],
                normalised=text,
                labels=[names[j] for j in selected],
                scores=[
                    LabelScore(
                        label=names[j],
                        score=float(P[i, j]),
                        threshold=float(decoder.thresholds[j]),
                        selected=bool(B[i, j]),
                        **(
                            {
                                "test_support": support[names[j]]["test_support"],
                                "measurable": support[names[j]]["measurable"],
                            }
                            if names[j] in support
                            else {}
                        ),
                    )
                    for j in order
                ],
                drivers=(
                    model.explain(text, int(selected[0]), k=req.top_k)
                    if len(selected) and req.explain
                    else []
                ),
            )
        )
    return ClassifyResponse(decoder=strategy, predictions=predictions)
