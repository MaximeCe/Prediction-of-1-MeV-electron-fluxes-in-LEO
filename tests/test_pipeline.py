"""End-to-end: the pipeline turns the two hourly sources into the modelling table."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from preprocessing import pipeline
from preprocessing.config import Config

from conftest import HOURS


@pytest.fixture
def synthetic_paths(tmp_path, tiny_noaa, tiny_omni):
    noaa_path = tmp_path / "noaa.parquet"
    omni_path = tmp_path / "omni.parquet"
    tiny_noaa.to_parquet(noaa_path, index=False)
    tiny_omni.to_parquet(omni_path)
    return Config(noaa_path=str(noaa_path), omni_path=str(omni_path),
                  out_path=str(tmp_path / "table_12h.parquet"))


def test_run_returns_the_table_the_report_and_the_threshold(synthetic_paths):
    result = pipeline.run(synthetic_paths)
    assert {"table", "wide", "report", "summary"} <= set(result)


def test_run_writes_a_parquet_that_can_be_read_back(synthetic_paths):
    pipeline.run(synthetic_paths)
    out = pd.read_parquet(synthetic_paths.out_path)
    assert len(out) == 6  # 3 L bins x 2 intervals
    assert {"p6_flux", "log10_p6", "n_samples", "coverage", "V_kms"} <= set(out.columns)


def test_the_written_table_is_long_and_indexed_by_l_bin_and_interval(synthetic_paths):
    pipeline.run(synthetic_paths)
    out = pd.read_parquet(synthetic_paths.out_path)
    assert {"Lbin", "time_bin"} <= set(out.columns) or set(out.index.names) >= {"Lbin", "time_bin"}


def test_the_table_only_contains_the_selected_l_range(synthetic_paths):
    result = pipeline.run(synthetic_paths)
    assert set(result["table"].index.get_level_values("Lbin").unique()) == {3.0, 4.5, 6.0}


def test_run_is_idempotent(synthetic_paths):
    first = pipeline.run(synthetic_paths)["table"]
    second = pipeline.run(synthetic_paths)["table"]
    pd.testing.assert_frame_equal(first, second)


def test_the_report_is_json_serialisable(synthetic_paths):
    report = pipeline.run(synthetic_paths)["report"]
    json.dumps(report)


def test_the_report_keeps_the_cleaning_trace(synthetic_paths):
    report = pipeline.run(synthetic_paths)["report"]
    assert report["cleaning"]["dropped_no_measurement"] == 1
    assert report["cleaning"]["n_kept"] < report["cleaning"]["n_input"]


def test_the_summary_holds_the_event_threshold(synthetic_paths):
    result = pipeline.run(synthetic_paths)
    assert result["summary"]["event_threshold"] == pytest.approx(1000.0)
