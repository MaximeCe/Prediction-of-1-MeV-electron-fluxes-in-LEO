"""Reading the two hourly sources that the preprocessing pipeline starts from."""
from __future__ import annotations

import pandas as pd
import pytest

from preprocessing import io


def test_load_noaa_hourly_returns_long_table_indexed_by_time():
    df = io.load_noaa_hourly()
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "time"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert {"Lbin", "p6_sum", "n", "omni_p1_sum", "omni_p1_n"} <= set(df.columns)
    assert len(df) > 0


def test_load_noaa_hourly_covers_the_project_period():
    df = io.load_noaa_hourly()
    assert df.index.min() >= pd.Timestamp("2020-01-01")
    assert df.index.max() < pd.Timestamp("2025-01-01")


def test_load_omni_returns_hourly_parameters():
    df = io.load_omni()
    assert df.index.name == "time"
    assert set(io.OMNI_COLUMNS) <= set(df.columns)
    assert len(df) > 0


def test_missing_file_raises_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        io.load_noaa_hourly(tmp_path / "nope.parquet")


def test_available_lbins_are_the_quarter_step_grid_plus_the_polar_bin():
    df = io.load_noaa_hourly()
    lbins = sorted(df["Lbin"].unique())
    assert io.POLAR_LBIN == 10.0
    assert 10.0 in lbins
    belt = [x for x in lbins if x < io.POLAR_LBIN]
    steps = {round(belt[i + 1] - belt[i], 6) for i in range(len(belt) - 1)}
    assert steps == {0.25}
