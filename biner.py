"""
biner.py
========
Step 1, bullet "Average all parameters over 12-hour intervals, which also defines the time
resolution of the predictions."

Module built on the previous ones: input handling comes from ``cleaner.as_dataframe``; the data
to average is the cleaned dataset (``loader.load_cleaned_data``). This file only does the averaging.

    bin_mean(data, freq="12h")   mean of every parameter per interval (+ number of samples)

Intervals are aligned on the UTC epoch, so with 12 h they always start at 00:00 and 12:00 UTC,
whatever the first time stamp of the data. Each row is labelled with the start of its interval.
NaN are ignored by the mean; an interval without any sample gives NaN and ``n_samples = 0``.
"""
from __future__ import annotations

import pandas as pd

from cleaner import as_dataframe

FREQ = "12h"  # time resolution of the project (and of the predictions)


def bin_mean(data, freq: str = FREQ, columns=None) -> pd.DataFrame:
    """Average the parameters over regular ``freq`` intervals.

    ``data``    : anything ``cleaner.as_dataframe`` accepts (ideally the output of ``cleaner.clean``
                  or ``cleaner.synchronize``, possibly several chunks concatenated)
    ``columns`` : parameters to average (default: all columns)
    Returns one row per interval with the mean of each parameter and ``n_samples``.
    """
    df = as_dataframe(data)
    if columns is not None:
        df = df[list(columns)]
    res = df.resample(freq, origin="epoch")
    out = res.mean()
    out["n_samples"] = res.size()
    return out


# --------------------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    from loader import load_cleaned_data

    cleaned = load_cleaned_data()  # data/cleaned/cleaned.parquet, already in L_RANGE (cleaner.py)
    binned = bin_mean(cleaned)  # 12 h means of every column (flux, L, solar parameters...)
    print(f"{len(cleaned)} cleaned samples -> {len(binned)} intervals of {FREQ}")
    print(binned.head())