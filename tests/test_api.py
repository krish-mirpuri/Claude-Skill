"""Service contract tests.

These need trained artifacts and skip without them, so the unit suite still
runs on a clean checkout.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client(artifacts):
    from overbook.api.main import app

    return TestClient(app)


def _booking(**over):
    base = {
        "lead_time": 45,
        "adr": 120.0,
        "adults": 2,
        "market_segment": "Online TA",
        "deposit_type": "No Deposit",
    }
    base.update(over)
    return base


def _night(n=30, **over):
    payload = {
        "hotel": "City Hotel",
        "arrival_date": "2017-09-15",
        "bookings": [_booking(lead_time=10 * i, reference=f"B{i}") for i in range(n)],
    }
    payload.update(over)
    return payload


def test_health_reports_a_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"]["name"]
    assert 0.5 < body["model"]["test_roc_auc"] <= 1.0


def test_model_card_is_served(client):
    card = client.get("/model-card").json()
    assert card["feature_view"] in {"arrival_eve", "booking_time"}


def test_score_returns_a_probability_per_booking(client):
    body = client.post("/score", json=_night(6)).json()
    assert body["n_bookings"] == 6
    assert len(body["scores"]) == 6
    assert all(0.0 <= s["p_cancel"] <= 1.0 for s in body["scores"])
    assert [s["reference"] for s in body["scores"]] == [f"B{i}" for i in range(6)]


def test_expected_arrivals_is_consistent_with_the_scores(client):
    body = client.post("/score", json=_night(8)).json()
    implied = sum(1 - s["p_cancel"] for s in body["scores"])
    assert body["expected_arrivals"] == pytest.approx(implied, rel=1e-9)


def test_non_refundable_bookings_score_as_high_risk(client):
    """Non-refundable bookings cancel 99% of the time in this data.

    The assertion is the *contrast* between two otherwise identical bookings
    rather than an absolute threshold, which would only encode whatever the
    current model happens to output.
    """
    refundable = _night(4)
    non_refundable = _night(4)
    non_refundable["bookings"][0]["deposit_type"] = "Non Refund"

    low = client.post("/score", json=refundable).json()["scores"][0]["p_cancel"]
    high = client.post("/score", json=non_refundable).json()["scores"][0]["p_cancel"]
    assert high > low + 0.3
    assert high > 0.6


def test_authorize_returns_a_feasible_and_coherent_plan(client):
    body = client.post("/authorize", json=_night(40, capacity=25)).json()
    assert body["capacity"] == 25
    assert 0 <= body["authorised"] <= body["bookings_on_books"]
    assert 0.0 <= body["p_oversell"] <= 1.0
    assert body["expected_walks"] >= 0 and body["expected_unsold"] >= 0
    assert sum(body["arrival_pmf"]) == pytest.approx(1.0, abs=1e-6)


def test_a_costlier_walk_makes_the_service_more_cautious(client):
    cheap = client.post("/authorize", json=_night(40, capacity=25, walk_cost_multiplier=0.5)).json()
    dear = client.post("/authorize", json=_night(40, capacity=25, walk_cost_multiplier=12.0)).json()
    assert dear["authorised"] <= cheap["authorised"]
    assert dear["p_oversell"] <= cheap["p_oversell"] + 1e-9


def test_capacity_defaults_to_the_trained_estimate(client):
    body = client.post("/authorize", json=_night(20)).json()
    assert body["capacity"] > 0


def test_unknown_hotel_without_capacity_is_rejected(client):
    r = client.post("/authorize", json=_night(5, hotel="Hotel California"))
    assert r.status_code == 422


def test_empty_booking_list_is_rejected(client):
    assert client.post("/score", json=_night(0)).status_code == 422


def test_a_booking_with_no_guests_is_rejected(client):
    payload = _night(3)
    payload["bookings"][0].update(adults=0, children=0, babies=0)
    assert client.post("/score", json=payload).status_code == 422


def test_negative_lead_time_is_rejected(client):
    payload = _night(3)
    payload["bookings"][0]["lead_time"] = -1
    assert client.post("/score", json=payload).status_code == 422
