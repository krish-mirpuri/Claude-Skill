"""Request and response models for the scoring service."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ClassifyRequest(BaseModel):
    texts: list[str] = Field(
        min_length=1, max_length=256, examples=[["thank you so much, this made my day"]]
    )
    decoder: str | None = Field(
        default=None,
        description="threshold strategy; defaults to the one the reports use",
    )
    top_k: int = Field(default=5, ge=1, le=28)
    explain: bool = True

    @field_validator("texts")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if any(not str(t).strip() for t in v):
            raise ValueError("texts must not be blank")
        return v


class LabelScore(BaseModel):
    label: str
    score: float
    threshold: float
    selected: bool
    test_support: int | None = None
    measurable: bool | None = None


class CommentPrediction(BaseModel):
    text: str
    normalised: str
    labels: list[str]
    scores: list[LabelScore]
    drivers: list[dict] = Field(default_factory=list)


class ClassifyResponse(BaseModel):
    decoder: str
    predictions: list[CommentPrediction]


class HealthResponse(BaseModel):
    status: str
    model: str
    trained_at: str | None = None
    n_labels: int
    decoders: list[str]
    default_decoder: str
    macro_f1: float | None = None
    macro_f1_ci: list[float] | None = None
