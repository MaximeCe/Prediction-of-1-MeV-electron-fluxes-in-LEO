"""Quality report and the 95th-percentile event threshold."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sp519 import aggregate, clean, qc
from sp519.config import Config


def test_event_threshold_is_the_requested_percentile():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert qc.event_threshold(s, 50.0)["threshold"] == pytest.approx(3.0)
    assert qc.event_threshold(s, 100.0)["threshold"] == pytest.approx(5.0)


def test_event_threshold_reports_the_log_value_and_the_event_rate():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = qc.event_threshold(s, 80.0)
    assert out["threshold"] == pytest.approx(4.2)
    assert out["log10"] == pytest.approx(np.log10(4.2))
    assert out["n"] == 5
    assert out["n_events"] == 1
    assert out["fraction_events"] == pytest.approx(0.2)


def test_event_threshold_ignores_missing_values():
    s = pd.Series([1.0, np.nan, 3.0])
    out = qc.event_threshold(s, 100.0)
    assert out["n"] == 2
    assert out["threshold"] == pytest.approx(3.0)


def test_coverage_report_lists_every_l_bin(tiny_noaa, tiny_omni):
    noaa, _ = clean.clean_noaa(tiny_noaa, Config())
    omni, _ = clean.clean_omni(tiny_omni, Config())
    long = aggregate.to_long_table(noaa, omni, Config())
    rep = qc.coverage_report(long)
    assert set(rep.index) == {3.0, 4.5, 6.0}
    assert {"n_intervals", "median_coverage", "min_coverage", "median_samples"} <= set(rep.columns)


def test_the_summary_reports_the_counts_and_the_threshold(tiny_noaa, tiny_omni):
    noaa, _ = clean.clean_noaa(tiny_noaa, Config())
    omni, _ = clean.clean_omni(tiny_omni, Config())
    long = aggregate.to_long_table(noaa, omni, Config())
    summary = qc.summary(long, Config())
    assert summary["n_intervals"] == 2
    assert summary["n_l_bins"] == 3
    assert summary["n_rows"] == 6
    assert summary["event_threshold"] == pytest.approx(1000.0)
    # internal consistency: an event is a strictly-above-threshold value
    assert summary["n_events"] == int((long["p6_flux"] > summary["event_threshold"]).sum())


def test_the_summary_is_json_serialisable(tiny_noaa, tiny_omni):
    import json

    noaa, _ = clean.clean_noaa(tiny_noaa, Config())
    omni, _ = clean.clean_omni(tiny_omni, Config())
    long = aggregate.to_long_table(noaa, omni, Config())
    json.dumps(qc.summary(long, Config()))  # must not raise
