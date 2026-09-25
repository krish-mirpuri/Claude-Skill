"""Service contract tests. These need trained artifacts and skip without them."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client(artifacts):
    from nuance.api.main import app

    return TestClient(app)


def test_health_reports_a_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["n_labels"] == 28
    assert body["default_decoder"] in body["decoders"]
    assert 0.0 < body["macro_f1"] < 1.0
    lo, hi = body["macro_f1_ci"]
    assert lo <= body["macro_f1"] <= hi


def test_labels_endpoint_admits_what_is_unmeasurable(client):
    labels = client.get("/labels").json()["labels"]
    assert len(labels) == 28
    unmeasurable = [entry for entry in labels if entry.get("measurable") is False]
    assert unmeasurable, "some labels have too little support to score"
    assert all(entry["test_support"] < 50 for entry in unmeasurable)


def test_classify_returns_one_prediction_per_text(client):
    body = client.post("/classify", json={"texts": ["thank you so much", "i am furious"]}).json()
    assert len(body["predictions"]) == 2
    assert body["decoder"]


def test_obvious_gratitude_is_recognised(client):
    body = client.post(
        "/classify", json={"texts": ["thank you so much, this genuinely helped"]}
    ).json()
    scores = {s["label"]: s["score"] for s in body["predictions"][0]["scores"]}
    assert "gratitude" in body["predictions"][0]["labels"] or scores.get("gratitude", 0) > 0.5


def test_scores_are_probabilities_and_carry_their_threshold(client):
    body = client.post("/classify", json={"texts": ["this is fine"], "top_k": 10}).json()
    for s in body["predictions"][0]["scores"]:
        assert 0.0 <= s["score"] <= 1.0
        assert 0.0 < s["threshold"] < 1.0
        assert s["selected"] == (s["score"] >= s["threshold"])


def test_scores_come_back_ranked(client):
    body = client.post("/classify", json={"texts": ["what a day"], "top_k": 8}).json()
    scores = [s["score"] for s in body["predictions"][0]["scores"]]
    assert scores == sorted(scores, reverse=True)


def test_drivers_explain_the_chosen_label(client):
    body = client.post("/classify", json={"texts": ["thank you so much"], "explain": True}).json()
    drivers = body["predictions"][0]["drivers"]
    if body["predictions"][0]["labels"]:
        assert drivers and {"term", "contribution"} <= set(drivers[0])


def test_explanations_can_be_turned_off(client):
    body = client.post("/classify", json={"texts": ["thank you so much"], "explain": False}).json()
    assert body["predictions"][0]["drivers"] == []


def test_decoder_can_be_chosen(client):
    available = client.get("/health").json()["decoders"]
    body = client.post("/classify", json={"texts": ["ok"], "decoder": available[0]}).json()
    assert body["decoder"] == available[0]


def test_unknown_decoder_is_rejected(client):
    r = client.post("/classify", json={"texts": ["ok"], "decoder": "vibes"})
    assert r.status_code == 422


def test_text_is_normalised_before_scoring(client):
    body = client.post(
        "/classify", json={"texts": ["thanks @someone see https://example.com"]}
    ).json()
    normalised = body["predictions"][0]["normalised"]
    assert "[NAME]" in normalised and "[URL]" in normalised


def test_blank_and_empty_input_is_rejected(client):
    assert client.post("/classify", json={"texts": ["   "]}).status_code == 422
    assert client.post("/classify", json={"texts": []}).status_code == 422
