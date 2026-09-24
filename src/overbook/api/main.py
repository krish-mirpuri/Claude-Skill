"""Scoring service.

Two endpoints, matching the two questions a revenue manager actually asks:

``POST /score``
    "How likely is each of these bookings to cancel, and why?"

``POST /authorize``
    "Given those bookings and this many rooms, how many should I hold?"

Bookings are scored as a **set for one arrival date**, not one at a time, for
a substantive reason: several features describe a booking's position in that
night's queue, so a booking's risk is not independent of the night it belongs
to. Scoring one booking in isolation would silently feed the model a night
with exactly one reservation on the books.
"""

from __future__ import annotations

import datetime as dt
import logging
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from overbook.api.context import load_serving_context
from overbook.api.schemas import (
    AuthorizeRequest,
    AuthorizeResponse,
    BookingScore,
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
)
from overbook.config import load_config
from overbook.decision.overdispersion import arrival_pmf
from overbook.decision.policy import Economics, expected_profit_curve
from overbook.models.train import load_artifacts

log = logging.getLogger(__name__)

app = FastAPI(
    title="overbook",
    version="0.1.0",
    summary="Calibrated cancellation risk and revenue-optimal overbooking limits.",
)


@lru_cache(maxsize=1)
def _bundle():
    """Load model artifacts once per process."""
    cfg = load_config()
    predictor, feature_builder, calibrator = load_artifacts(cfg)
    try:
        context = load_serving_context(cfg)
    except FileNotFoundError:
        log.warning("no serving context; falling back to training-mean rates")
        context = {"hotels": {}, "fallback": {}, "capacity": {}, "as_of": None}
    import json

    card_path = cfg.resolve(cfg.paths.models_dir) / "model_card.json"
    card = json.loads(card_path.read_text()) if card_path.exists() else {}
    return cfg, predictor, feature_builder, calibrator, context, card


def _frame(hotel: str, arrival: dt.date, bookings: list) -> pd.DataFrame:
    """Turn a request into the canonical booking table the builder expects."""
    rows = []
    arrival_ts = pd.Timestamp(arrival)
    for seq, b in enumerate(bookings):
        d = b.model_dump()
        d.pop("reference", None)
        d.update(
            hotel=hotel,
            arrival_date=arrival_ts,
            booking_date=arrival_ts - pd.Timedelta(days=int(d["lead_time"])),
            booking_seq=seq,
            booking_id=seq,
            # Never read by the feature builder for serving; present because
            # the canonical table carries it.
            is_canceled=0,
            split="serve",
        )
        d["total_guests"] = d["adults"] + d["children"] + d["babies"]
        d["total_nights"] = d["stays_in_week_nights"] + d["stays_in_weekend_nights"]
        rows.append(d)
    return pd.DataFrame(rows)


def _score(hotel: str, arrival: dt.date, bookings: list) -> tuple[np.ndarray, pd.DataFrame]:
    cfg, predictor, fb, calibrator, context, _ = _bundle()
    overrides = context["hotels"].get(hotel) or context.get("fallback") or {}
    df = _frame(hotel, arrival, bookings)
    try:
        ff = fb.transform(df, rolling_overrides=overrides)
    except Exception as exc:  # unknown category, bad type, missing column
        raise HTTPException(status_code=422, detail=f"could not build features: {exc}") from exc
    p = predictor.predict_proba(ff.X)
    if calibrator is not None:
        p = calibrator.predict(p)
    return np.asarray(p, dtype=float), ff.X


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        _cfg, _p, _fb, _cal, context, card = _bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"model not loaded: {exc}") from exc
    return HealthResponse(
        status="ok",
        model={
            "name": card.get("model", "unknown"),
            "feature_view": card.get("feature_view"),
            "trained_at": card.get("trained_at"),
            "test_roc_auc": (card.get("test_metrics") or {}).get("roc_auc"),
            "test_brier": (card.get("test_metrics") or {}).get("brier"),
        },
        context_as_of=context.get("as_of"),
    )


@app.get("/model-card")
def model_card() -> dict:
    return _bundle()[5]


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    p, X = _score(req.hotel, req.arrival_date, req.bookings)

    drivers: list[list[dict]] = [[] for _ in req.bookings]
    _, predictor, _, _, _, _ = _bundle()
    booster = getattr(predictor, "booster", None)
    if booster is not None and len(req.bookings) <= 50:
        # SHAP is not free; skip it on large batches rather than stall the call.
        try:
            from overbook.explain.shap_report import explain_one

            drivers = [explain_one(booster, X.iloc[[i]]) for i in range(len(req.bookings))]
        except Exception as exc:  # explanations are a nicety, not the contract
            log.warning("explanation unavailable: %s", exc)

    return ScoreResponse(
        hotel=req.hotel,
        arrival_date=req.arrival_date,
        n_bookings=len(req.bookings),
        expected_arrivals=float((1.0 - p).sum()),
        scores=[
            BookingScore(reference=b.reference, p_cancel=float(pi), drivers=d)
            for b, pi, d in zip(req.bookings, p, drivers, strict=True)
        ],
    )


@app.post("/authorize", response_model=AuthorizeResponse)
def authorize(req: AuthorizeRequest) -> AuthorizeResponse:
    cfg, _predictor, _fb, _cal, context, card = _bundle()
    capacity = req.capacity or context.get("capacity", {}).get(req.hotel)
    if not capacity:
        raise HTTPException(
            status_code=422,
            detail=f"no capacity given and none known for hotel {req.hotel!r}",
        )

    econ = Economics(
        variable_cost_ratio=(
            req.variable_cost_ratio
            if req.variable_cost_ratio is not None
            else cfg.economics.variable_cost_ratio
        ),
        walk_cost_multiplier=(
            req.walk_cost_multiplier
            if req.walk_cost_multiplier is not None
            else cfg.economics.walk_cost_multiplier
        ),
    )
    tau = float(card.get("shared_shock_tau") or 0.0)

    p, _X = _score(req.hotel, req.arrival_date, req.bookings)
    show = 1.0 - p
    adr = float(np.mean([b.adr for b in req.bookings]))

    curve = expected_profit_curve(show, capacity, econ.margin(adr), econ.walk_cost(adr), tau=tau)
    a = int(np.argmax(curve))
    pmf = arrival_pmf(show[:a], tau=tau)
    k = np.arange(pmf.size)

    return AuthorizeResponse(
        hotel=req.hotel,
        arrival_date=req.arrival_date,
        capacity=int(capacity),
        bookings_on_books=len(req.bookings),
        authorised=a,
        overbook_pct=100.0 * (a - capacity) / capacity,
        expected_arrivals=float(np.dot(k, pmf)),
        p_oversell=float(pmf[k > capacity].sum()),
        expected_walks=float(np.dot(np.maximum(k - capacity, 0), pmf)),
        expected_unsold=float(np.dot(np.maximum(capacity - k, 0), pmf)),
        expected_profit=float(curve[a]),
        arrival_pmf=pmf.tolist(),
        assumptions={
            "margin_per_room": econ.margin(adr),
            "walk_cost": econ.walk_cost(adr),
            "mean_adr": adr,
            "shared_shock_tau": tau,
        },
    )
