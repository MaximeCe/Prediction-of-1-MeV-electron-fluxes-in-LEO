"""The project's reasoning choices live in one place and must be self-consistent."""
from __future__ import annotations

import pandas as pd

from preprocessing import config as C


def test_l_range_is_the_justified_choice():
    # Outer radiation belt: below L=3 the P6 channel is dominated by protons of the inner
    # belt / South Atlantic Anomaly, above L=6 the orbit barely samples the belt.
    assert C.L_RANGE == (3.0, 6.0)


def test_l_range_falls_on_the_bin_grid():
    lo, hi = C.L_RANGE
    assert round((lo - C.L_BIN_STEP) % C.L_BIN_STEP, 9) in (0.0, C.L_BIN_STEP), "lo off grid"
    assert round((hi - lo) % C.L_BIN_STEP, 9) == 0.0, "range not a whole number of bins"


def test_frequency_divides_the_forecast_horizons():
    for h in (12, 24, 36, 48):
        assert pd.Timedelta(hours=h) % pd.Timedelta(C.FREQ) == pd.Timedelta(0)


def test_flux_floor_is_positive_and_below_one_count():
    assert 0 < C.FLUX_FLOOR < C.P6_COUNT_STEP


def test_default_config_matches_module_constants():
    cfg = C.Config()
    assert cfg.l_range == C.L_RANGE
    assert cfg.freq == C.FREQ
    assert cfg.event_percentile == C.EVENT_PERCENTILE


def test_horizons_are_the_four_required_ones():
    assert C.HORIZONS_H == (12, 24, 36, 48)
