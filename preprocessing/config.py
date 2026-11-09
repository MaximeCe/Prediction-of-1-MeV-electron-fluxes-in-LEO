"""Every reasoning choice of the preprocessing, in one place, with its justification.

Nothing in the pipeline hard-codes a number: it all comes from here, so that the report can
point at a single file and so that a choice can be re-tested by changing one value.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------
# The region of the radiation belts that is predicted
# --------------------------------------------------------------------------------------
# Justification (Step 1, "select the region of the radiation belts you predict"):
#   * L < 3.0  the P6 channel is dominated by trapped protons (inner belt, South Atlantic
#              Anomaly): the electron flux is no longer what the channel measures.
#   * L > 6.0  NOAA-15 (810 km, polar) spends only a few minutes per orbit there and the
#              flux drops by two orders of magnitude -> too few samples per bin.
#   * 3.0-6.0  contains the slot region and the whole outer belt maximum, which the
#              hourly table samples ~13-31 times per hour per 0.25 L bin.
# This is the range most of the literature uses (2.8-6.0 in PreMevE 2.0 and in
# Kieokaew et al. 2026), trimmed to the bins that are actually well sampled here.
L_RANGE = (3.0, 6.0)
L_BIN_STEP = 0.25  # step of the L bins in the raw table

# --------------------------------------------------------------------------------------
# Time resolution
# --------------------------------------------------------------------------------------
FREQ = "12h"  # Step 1: "average all parameters over 12-hour intervals"
HORIZONS_H = (12, 24, 36, 48)  # prediction horizons of Step 2, in hours

# --------------------------------------------------------------------------------------
# The target channel
# --------------------------------------------------------------------------------------
# mep_pro_tel90_flux_p6: the P6 channel of the 90 deg telescope. It is a *proton* channel
# (>6.9 MeV) but in the belts its counts are dominated by electrons above ~1 MeV, which is
# why it is used as the >1 MeV electron flux. During a Solar Proton Event it counts protons
# again, hence the contamination flag handled in sp519.clean.
TARGET = "mep_pro_tel90_flux_p6"
P6_COUNT_STEP = 243.9024  # flux of a single count [#/cm2/s/sr]
FLUX_FLOOR = P6_COUNT_STEP / 2  # value used for a 0-count bin before log10

# --------------------------------------------------------------------------------------
# Cleaning thresholds
# --------------------------------------------------------------------------------------
# A 0-count hour is a real measurement (p6_sum = 0), not a gap: only n == 0 is a gap.
MIN_HOURLY_SAMPLES = 1
# Coverage = fraction of the 12 hours of an interval that were really observed. 0.0 keeps
# every interval that has at least one hour; raise it to reject poorly sampled intervals.
MIN_COVERAGE = 0.0
# Proton contamination: an hour is flagged when the >25 MeV proton flux (omni_p1) exceeds
# this percentile of its own distribution inside L_RANGE. Data-driven on purpose: it adapts
# to the solar cycle instead of assuming a fixed proton flux.
SPE_PERCENTILE = 99.5
# The flag is reported and kept by default; set to True to also drop the flagged hours.
DROP_CONTAMINATED = False

# --------------------------------------------------------------------------------------
# The event that Step 2 has to classify
# --------------------------------------------------------------------------------------
EVENT_PERCENTILE = 95.0  # Step 1: "determine the 95th-percentile flux threshold"

# --------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------
NOAA_PATH = "data/noaa15_p6_hourly_Lbins.parquet"  # hourly P6 flux per L bin (versioned)
OMNI_PATH = "data/omni_hourly_2020_2024.parquet"  # hourly solar wind / indices (versioned)
OUT_PATH = "data/table_12h_2020_2024.parquet"  # output of Step 1, input of Step 2
REPORT_PATH = "data/table_12h_2020_2024.report.json"


@dataclass(frozen=True)
class Config:
    """All the knobs of the preprocessing. Frozen: a run is fully described by its Config."""

    # selection
    l_range: tuple[float, float] = L_RANGE
    l_bin_step: float = L_BIN_STEP
    freq: str = FREQ
    # cleaning
    flux_floor: float = FLUX_FLOOR
    min_hourly_samples: int = MIN_HOURLY_SAMPLES
    min_coverage: float = MIN_COVERAGE
    spe_percentile: float = SPE_PERCENTILE
    drop_contaminated: bool = DROP_CONTAMINATED
    # event definition
    event_percentile: float = EVENT_PERCENTILE
    # files
    noaa_path: str = NOAA_PATH
    omni_path: str = OMNI_PATH
    out_path: str = OUT_PATH
    report_path: str | None = REPORT_PATH
    figures_dir: str | None = field(default=None)

    @property
    def expected_hours_per_bin(self) -> int:
        """Number of hourly samples a full interval of ``freq`` should contain."""
        import pandas as pd

        return int(pd.Timedelta(self.freq) / pd.Timedelta(hours=1))

    @property
    def l_bins(self) -> list[float]:
        """The L bins of the selected range, on the raw table's grid."""
        n = int(round((self.l_range[1] - self.l_range[0]) / self.l_bin_step))
        return [round(self.l_range[0] + i * self.l_bin_step, 10) for i in range(n + 1)]
