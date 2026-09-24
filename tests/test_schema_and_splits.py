"""Schema validation and the temporal split discipline."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from overbook.config import SplitConfig
from overbook.data.clean import assign_split
from overbook.data.schema import SchemaError, validate_raw
from overbook.models.train import rolling_origin_folds


def test_validate_raw_accepts_a_well_formed_extract(raw_df):
    validate_raw(raw_df)


def test_validate_raw_rejects_missing_columns(raw_df):
    with pytest.raises(SchemaError, match="missing columns"):
        validate_raw(raw_df.drop(columns=["adr"]))


def test_validate_raw_rejects_impossible_values(raw_df):
    bad = raw_df.copy()
    bad.loc[bad.index[0], "lead_time"] = -5
    with pytest.raises(SchemaError, match="negative lead_time"):
        validate_raw(bad)


def test_validate_raw_rejects_unparseable_months(raw_df):
    bad = raw_df.copy()
    bad.loc[bad.index[0], "arrival_date_month"] = "Smarch"
    with pytest.raises(SchemaError, match="arrival_date_month"):
        validate_raw(bad)


def test_split_windows_must_not_overlap():
    overlapping = SplitConfig(
        train=("2015-01-01", "2016-06-30"),
        valid=("2016-06-01", "2016-09-30"),
        test=("2016-10-01", "2016-12-31"),
    )
    with pytest.raises(ValueError, match="overlaps"):
        overlapping.check_ordering()


def test_splits_are_ordered_in_time_and_disjoint(clean_df, cfg):
    cfg.split.check_ordering()
    bounds = {
        name: (
            clean_df.loc[clean_df["split"] == name, "arrival_date"].min(),
            clean_df.loc[clean_df["split"] == name, "arrival_date"].max(),
        )
        for name in ("train", "valid", "test")
    }
    assert bounds["train"][1] < bounds["valid"][0] < bounds["valid"][1] < bounds["test"][0]


def test_assign_split_labels_by_arrival_date(cfg):
    s = pd.Series(pd.to_datetime(["2015-08-01", "2017-02-01", "2017-06-01", "2020-01-01"]))
    out = assign_split(s, cfg)
    assert out.tolist() == ["train", "valid", "test", "unused"]


def test_rolling_origin_folds_never_train_on_the_future():
    dates = pd.Series(pd.to_datetime(np.repeat(pd.date_range("2016-01-01", periods=100), 3)))
    folds = rolling_origin_folds(dates, n_splits=4)
    assert len(folds) == 4
    for train_idx, valid_idx in folds:
        assert dates.iloc[train_idx].max() < dates.iloc[valid_idx].min()
        assert not set(train_idx) & set(valid_idx)


def test_rolling_origin_folds_expand():
    dates = pd.Series(pd.to_datetime(np.repeat(pd.date_range("2016-01-01", periods=60), 2)))
    sizes = [len(t) for t, _ in rolling_origin_folds(dates, 4)]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]


def test_rolling_origin_folds_reject_impossible_requests():
    dates = pd.Series(pd.to_datetime(pd.date_range("2016-01-01", periods=3)))
    with pytest.raises(ValueError):
        rolling_origin_folds(dates, n_splits=10)


def test_clean_is_deterministic(raw_df, cfg):
    from overbook.data.clean import clean

    a, b = clean(raw_df, cfg), clean(raw_df, cfg)
    pd.testing.assert_frame_equal(a, b)


def test_booking_date_reconstruction(clean_df):
    delta = (clean_df["arrival_date"] - clean_df["booking_date"]).dt.days
    assert (delta == clean_df["lead_time"]).all()


def test_booking_seq_is_a_dense_queue_per_night(clean_df):
    g = clean_df.groupby(["hotel", "arrival_date"])["booking_seq"]
    assert (g.min() == 0).all()
    assert (g.max() + 1 == g.size()).all()


def test_booking_queue_is_ordered_by_booking_date(clean_df):
    ordered = clean_df.sort_values(["hotel", "arrival_date", "booking_seq"])
    diffs = ordered.groupby(["hotel", "arrival_date"])["booking_date"].diff().dropna()
    assert (diffs >= pd.Timedelta(0)).all()


def test_split_config_parses_dates():
    sc = SplitConfig(
        train=("2015-01-01", "2015-12-31"),
        valid=("2016-01-01", "2016-03-31"),
        test=("2016-04-01", "2016-12-31"),
    )
    assert sc.train[0] == dt.date(2015, 1, 1)
