"""Feature construction: determinism, shape, and the serving path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from overbook.features.build import KEY_COLUMNS, FeatureBuilder, build_features


def test_feature_build_is_deterministic(clean_df, cfg):
    a, _ = build_features(clean_df, cfg)
    b, _ = build_features(clean_df, cfg)
    pd.testing.assert_frame_equal(a.X, b.X)


def test_keys_are_carried_but_never_modelled(clean_df, cfg):
    ff, _ = build_features(clean_df, cfg)
    assert list(ff.keys.columns) == list(KEY_COLUMNS)
    assert not set(ff.X.columns) & {"booking_id", "split", "is_canceled"}


def test_target_is_aligned_with_the_design_matrix(clean_df, cfg):
    ff, _ = build_features(clean_df, cfg)
    assert len(ff.X) == len(ff.y) == len(ff.keys) == len(clean_df)
    np.testing.assert_array_equal(ff.y.to_numpy(), ff.keys["is_canceled"].to_numpy())


def test_arrival_eve_view_adds_features_over_booking_time(clean_df, cfg):
    early, _ = build_features(clean_df, cfg, view="booking_time")
    late, _ = build_features(clean_df, cfg, view="arrival_eve")
    assert set(early.X.columns) < set(late.X.columns)


def test_unknown_view_is_rejected(cfg):
    with pytest.raises(ValueError):
        FeatureBuilder(cfg, view="crystal_ball")


def test_transform_before_fit_is_an_error(clean_df, cfg):
    with pytest.raises(RuntimeError):
        FeatureBuilder(cfg).transform(clean_df)


def test_splits_partition_the_frame(clean_df, cfg):
    ff, _ = build_features(clean_df, cfg)
    total = sum(len(ff.split(s)) for s in ("train", "valid", "test"))
    assert total == int((clean_df["split"] != "unused").sum())


def test_categoricals_are_typed_for_the_booster(clean_df, cfg):
    ff, _ = build_features(clean_df, cfg)
    for col in ff.categorical:
        assert str(ff.X[col].dtype) == "category"


def test_serving_overrides_replace_the_rolling_features(clean_df, cfg):
    """At serve time the rolling rates come from a snapshot, not the request."""
    fb = FeatureBuilder(cfg).fit(clean_df[clean_df["split"] == "train"])
    one_night = clean_df[clean_df["split"] == "test"].head(5).copy()
    overrides = {"hotel_cancel_rate_28d": 0.42, "hotel_cancel_rate_90d": 0.38}
    out = fb.transform(one_night, rolling_overrides=overrides)
    assert (out.X["hotel_cancel_rate_28d"] == 0.42).all()
    assert (out.X["hotel_cancel_rate_90d"] == 0.38).all()


def test_on_books_features_describe_the_night(clean_df, cfg):
    ff, _ = build_features(clean_df, cfg)
    counts = clean_df.groupby(["hotel", "arrival_date"])["booking_id"].transform("size")
    np.testing.assert_array_equal(ff.X["bookings_on_books"].to_numpy(), counts.to_numpy())
    assert ff.X["booking_rank_frac"].between(0, 1).all()
