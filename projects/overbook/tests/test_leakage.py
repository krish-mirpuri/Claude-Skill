"""The leakage guarantees, enforced rather than asserted in prose.

These are the tests that matter most in this project. A hotel-cancellation
model that quietly learns from ``reservation_status`` scores about 0.99 AUC
and is worth nothing, and the failure is invisible in every conventional
metric. So the ban is checked at the schema, at the feature matrix, and at the
fitted vocabulary.
"""

from __future__ import annotations

import pandas as pd
import pytest

from overbook.data.schema import ACCRUAL_COLUMNS, LEAKY_COLUMNS, SchemaError, assert_no_leakage
from overbook.features.build import FeatureBuilder, build_features


def test_leaky_columns_never_reach_the_design_matrix(clean_df, cfg):
    ff, _ = build_features(clean_df, cfg, view="arrival_eve")
    for col in LEAKY_COLUMNS:
        assert col not in ff.X.columns


def test_assert_no_leakage_raises_and_names_the_column():
    with pytest.raises(SchemaError, match="reservation_status"):
        assert_no_leakage(["lead_time", "reservation_status"])


def test_assert_no_leakage_passes_on_clean_columns():
    assert assert_no_leakage(["lead_time", "adr", "market_segment"]) is None


def test_booking_time_view_drops_post_booking_accruals(clean_df, cfg):
    early, _ = build_features(clean_df, cfg, view="booking_time")
    late, _ = build_features(clean_df, cfg, view="arrival_eve")
    for col in ACCRUAL_COLUMNS:
        assert col not in early.X.columns, f"{col} is not known when a booking is made"
        assert col in late.X.columns


def test_category_vocabulary_is_fitted_on_train_only(clean_df, cfg):
    """A category seen only after the training window must not enter the vocabulary."""
    df = clean_df.copy()
    df.loc[df["split"] == "test", "market_segment"] = "Martian TA"
    fb = FeatureBuilder(cfg).fit(df[df["split"] == "train"])
    assert "Martian TA" not in fb.vocab_["market_segment"]

    out = fb.transform(df)
    unseen = out.X.loc[(df["split"] == "test").to_numpy(), "market_segment"]
    assert set(unseen.astype(str)) == {"__OTHER__"}


def test_rolling_cancel_rate_uses_only_earlier_arrivals(clean_df, cfg):
    """Corrupting the future must not change a past booking's features."""
    ff_a, _ = build_features(clean_df, cfg, view="arrival_eve")

    tampered = clean_df.copy()
    last_month = tampered["arrival_date"] >= tampered["arrival_date"].max() - pd.Timedelta(days=30)
    tampered.loc[last_month, "is_canceled"] = 1 - tampered.loc[last_month, "is_canceled"]
    ff_b, _ = build_features(tampered, cfg, view="arrival_eve")

    untouched = (~last_month).to_numpy()
    for col in ("hotel_cancel_rate_28d", "hotel_cancel_rate_90d"):
        pd.testing.assert_series_equal(
            ff_a.X.loc[untouched, col], ff_b.X.loc[untouched, col], check_names=False
        )
