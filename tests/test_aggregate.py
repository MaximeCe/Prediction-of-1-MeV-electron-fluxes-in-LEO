"""12 h binning and the synchronization of the two sources into the modelling table."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from preprocessing import aggregate, clean
from preprocessing.config import Config

from conftest import HOURS

BIN_A = pd.Timestamp("2020-01-01 00:00")
BIN_B = pd.Timestamp("2020-01-01 12:00")


@pytest.fixture
def tables(tiny_noaa, tiny_omni):
    noaa, _ = clean.clean_noaa(tiny_noaa, Config())
    omni, _ = clean.clean_omni(tiny_omni, Config())
    return noaa, omni


def test_binning_averages_on_epoch_aligned_12h_intervals(tables):
    noaa, omni = tables
    out = aggregate.to_long_table(noaa, omni, Config())
    assert list(out.index.get_level_values("time_bin").unique()) == [BIN_A, BIN_B]


def test_flux_is_the_mean_over_the_interval(tables):
    noaa, omni = tables
    out = aggregate.to_long_table(noaa, omni, Config())
    assert out.loc[(4.5, BIN_A), "p6_flux"] == pytest.approx(1000.0)
    assert out.loc[(3.0, BIN_A), "p6_flux"] == pytest.approx(100.0)
    assert out.loc[(6.0, BIN_B), "p6_flux"] == pytest.approx(400.0)


def test_an_hour_without_measurement_does_not_bias_the_mean(tables):
    # 11 valid hours out of 12, all equal to 1000 -> the mean stays 1000
    noaa, omni = tables
    out = aggregate.to_long_table(noaa, omni, Config())
    assert out.loc[(4.5, BIN_A), "n_hours"] == 11
    assert out.loc[(4.5, BIN_B), "n_hours"] == 12


def test_sample_count_is_the_number_of_raw_measurements(tables):
    noaa, omni = tables
    out = aggregate.to_long_table(noaa, omni, Config())
    assert out.loc[(4.5, BIN_A), "n_samples"] == 110  # 11 hours x 10 samples
    assert out.loc[(4.5, BIN_B), "n_samples"] == 120


def test_coverage_is_the_fraction_of_the_interval_that_was_observed(tables):
    noaa, omni = tables
    out = aggregate.to_long_table(noaa, omni, Config())
    assert out.loc[(4.5, BIN_A), "coverage"] == pytest.approx(11 / 12)
    assert out.loc[(4.5, BIN_B), "coverage"] == pytest.approx(1.0)


def test_log_flux_keeps_the_physical_mean_but_applies_a_half_count_floor(tables):
    noaa, omni = tables
    cfg = Config()
    out = aggregate.to_long_table(noaa, omni, cfg)
    # the 12 h mean stays in physical units...
    assert out.loc[(3.0, BIN_A), "p6_flux"] == pytest.approx(100.0)
    # ... and it can legitimately sit below one count, where the log column applies its floor
    assert out.loc[(3.0, BIN_A), "p6_flux"] < cfg.flux_floor
    assert out.loc[(3.0, BIN_A), "log10_p6"] == pytest.approx(np.log10(cfg.flux_floor))
    # a bin well above the floor is left untouched
    assert out.loc[(4.5, BIN_A), "log10_p6"] == pytest.approx(3.0)
    assert np.isfinite(out["log10_p6"]).all()


def test_the_solar_parameters_are_carried_onto_every_bin(tables):
    noaa, omni = tables
    out = aggregate.to_long_table(noaa, omni, Config())
    assert out.loc[(4.5, BIN_A), "V_kms"] == pytest.approx(400.0)
    assert out.loc[(6.0, BIN_B), "Dst_nT"] == pytest.approx(-10.0)


def test_the_long_table_has_one_row_per_l_bin_and_interval(tables):
    noaa, omni = tables
    out = aggregate.to_long_table(noaa, omni, Config())
    assert len(out) == 3 * 2  # 3 L bins x 2 intervals
    assert out.index.names == ["Lbin", "time_bin"]


def test_the_wide_table_has_one_row_per_interval(tables):
    noaa, omni = tables
    long = aggregate.to_long_table(noaa, omni, Config())
    wide = aggregate.to_wide_table(long)
    assert len(wide) == 2
    assert wide.index.name == "time_bin"
    assert {"p6_L3", "p6_L4.5", "p6_L6"} <= set(wide.columns)


def test_the_wide_table_carries_the_l_bins_as_columns(tables):
    noaa, omni = tables
    wide = aggregate.to_wide_table(aggregate.to_long_table(noaa, omni, Config()))
    assert wide.loc[BIN_A, "p6_L4.5"] == pytest.approx(1000.0)
    assert wide.loc[BIN_B, "p6_L6"] == pytest.approx(400.0)


def test_the_contamination_fraction_is_reported_per_interval(tables):
    noaa, omni = tables
    out = aggregate.to_long_table(noaa, omni, Config())
    assert out.loc[(4.5, BIN_B), "contaminated_frac"] == pytest.approx(1 / 12)


def test_bins_below_the_coverage_floor_are_dropped(tiny_noaa, tiny_omni):
    noaa, _ = clean.clean_noaa(tiny_noaa, Config())
    omni, _ = clean.clean_omni(tiny_omni, Config())
    out = aggregate.to_long_table(noaa, omni, Config(min_coverage=0.99))
    assert (4.5, BIN_A) not in out.index  # 11/12 hours observed -> dropped
    assert (4.5, BIN_B) in out.index
    assert (3.0, BIN_A) in out.index  # fully observed -> kept
