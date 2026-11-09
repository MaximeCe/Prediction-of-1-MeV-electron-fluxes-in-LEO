"""Cleaning rules: what a valid measurement is, and what must be dropped."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sp519 import clean
from sp519.config import Config

from conftest import FLUX_BY_LBIN, HOURS, LBINS, SPE_HOUR


# --------------------------------------------------------------------------- flux
def test_hourly_flux_is_the_mean_of_the_hour(tiny_noaa):
    flux = clean.hourly_flux(tiny_noaa)
    assert flux.loc[(HOURS[0], 4.5)] == pytest.approx(1000.0)
    assert flux.loc[(HOURS[0], 3.0)] == pytest.approx(100.0)
    assert flux.loc[(HOURS[0], 6.0)] == pytest.approx(400.0)


def test_an_hour_without_measurement_gives_nan(tiny_noaa):
    flux = clean.hourly_flux(tiny_noaa)
    assert np.isnan(flux.loc[(HOURS[3], 4.5)])


def test_zero_counts_is_a_real_value_not_a_gap(tiny_noaa):
    # a bin with p6_sum = 0 is a genuine "0 count" measurement, it must stay 0
    flux = clean.hourly_flux(tiny_noaa)
    assert flux.loc[(HOURS[0], 2.0)] == 0.0


# --------------------------------------------------------------------------- protons
def test_proton_flux_is_computed_like_the_electron_flux(tiny_noaa):
    p1 = clean.proton_flux(tiny_noaa)
    assert p1.loc[(HOURS[0], 4.5)] == pytest.approx(0.05)
    assert p1.loc[(SPE_HOUR, 4.5)] == pytest.approx(5.0)


def test_contamination_threshold_is_a_percentile_of_the_belt_protons(tiny_noaa):
    cfg = Config(spe_percentile=99.5)
    thr = clean.contamination_threshold(tiny_noaa, cfg)
    assert 0.05 < thr <= 5.0


def test_flag_contamination_marks_only_the_event(tiny_noaa):
    flag = clean.flag_contamination(clean.proton_flux(tiny_noaa), threshold=1.0)
    assert bool(flag.loc[(SPE_HOUR, 4.5)])
    assert not bool(flag.loc[(HOURS[0], 4.5)])


# --------------------------------------------------------------------------- selection
def test_clean_noaa_keeps_only_the_justified_l_range(tiny_noaa):
    out, _ = clean.clean_noaa(tiny_noaa, Config())
    assert set(out.index.get_level_values("Lbin")) == set(FLUX_BY_LBIN)


def test_clean_noaa_drops_the_polar_bin_and_out_of_range_l(tiny_noaa):
    out, _ = clean.clean_noaa(tiny_noaa, Config())
    lbins = set(out.index.get_level_values("Lbin"))
    assert 10.0 not in lbins  # polar cap
    assert 7.0 not in lbins
    assert 2.0 not in lbins


def test_clean_noaa_drops_hours_without_measurement(tiny_noaa):
    out, _ = clean.clean_noaa(tiny_noaa, Config())
    assert (HOURS[3], 4.5) not in out.index
    assert (HOURS[0], 4.5) in out.index


def test_clean_noaa_adds_the_flux_and_the_contamination_flag(tiny_noaa):
    out, _ = clean.clean_noaa(tiny_noaa, Config())
    assert {"flux_hourly", "proton_flux", "contaminated"} <= set(out.columns)
    assert bool(out.loc[(SPE_HOUR, 4.5), "contaminated"])


def test_clean_noaa_report_counts_what_was_removed(tiny_noaa):
    _, rep = clean.clean_noaa(tiny_noaa, Config())
    assert rep["n_input"] == len(tiny_noaa)
    assert rep["dropped_outside_l_range"] > 0
    assert rep["dropped_no_measurement"] == 1
    assert rep["n_kept"] < rep["n_input"]


def test_clean_noaa_never_produces_non_finite_flux(tiny_noaa):
    out, _ = clean.clean_noaa(tiny_noaa, Config())
    assert np.isfinite(out["flux_hourly"]).all()


# --------------------------------------------------------------------------- omni
def test_clean_omni_keeps_the_parameters_and_the_index(tiny_omni):
    out, rep = clean.clean_omni(tiny_omni, Config())
    assert list(out.columns) == list(tiny_omni.columns)
    assert out.index.equals(tiny_omni.index)
    assert rep["n_input"] == len(tiny_omni)
