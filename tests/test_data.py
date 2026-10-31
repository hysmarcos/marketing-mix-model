from pathlib import Path

import pandas as pd
import pytest

from mmm.data import RAW_PATH, DataError, load_raw, prepare_features

BASE_CONTROLS = {"newsletter", "competitor_sales_B", "holiday", "trend"}


def write_variant(tmp_path: Path, change) -> Path:
    df = pd.read_csv(RAW_PATH)
    df = change(df)
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    return path


def test_load_raw_returns_208_weeks():
    df = load_raw()
    assert len(df) == 208
    assert df["DATE"].min() == pd.Timestamp("2015-11-23")


def test_missing_column_raises(tmp_path):
    bad_csv = write_variant(tmp_path, lambda df: df.drop(columns="tv_S"))
    with pytest.raises(DataError, match="tv_S"):
        load_raw(bad_csv)


def test_missing_week_raises_with_date(tmp_path):
    bad_csv = write_variant(tmp_path, lambda df: df[df["DATE"] != "2016-01-04"])
    with pytest.raises(DataError, match="2016-01-04"):
        load_raw(bad_csv)


def test_duplicate_week_raises_with_date(tmp_path):
    def duplicate(df):
        row = df[df["DATE"] == "2016-01-04"]
        return pd.concat([df, row]).sort_values("DATE")

    bad_csv = write_variant(tmp_path, duplicate)
    with pytest.raises(DataError, match="2016-01-04"):
        load_raw(bad_csv)


def test_negative_spend_raises(tmp_path):
    def negative(df):
        df.loc[5, "facebook_S"] = -1
        return df

    bad_csv = write_variant(tmp_path, negative)
    with pytest.raises(DataError, match="facebook_S"):
        load_raw(bad_csv)


def test_prepare_features_adds_controls():
    df, controls = prepare_features(load_raw())
    assert BASE_CONTROLS <= set(controls)
    assert all(c.startswith("event_") for c in controls if c not in BASE_CONTROLS)
    assert df["holiday"].isin([0, 1]).all()
    assert df.loc[df["DATE"] == "2015-12-21", "holiday"].item() == 1  # Navidad en Alemania
    assert df["trend"].iloc[0] == 0 and df["trend"].iloc[-1] == 1


def test_continuous_controls_are_scaled_to_max_1():
    df, _ = prepare_features(load_raw())
    for column in ("newsletter", "competitor_sales_B"):
        assert df[column].max() == pytest.approx(1.0)
        assert df[column].min() >= 0
