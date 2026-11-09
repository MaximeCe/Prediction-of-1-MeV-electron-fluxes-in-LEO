"""SP519 preprocessing package.

Turn the two hourly sources (NOAA-15 P6 flux binned in L, and the OMNI solar-wind
parameters) into the single modelling table used by Step 2 of the project.

See ``sp519.pipeline.run`` for the one-call entry point, and ``sp519.config`` for the
choices that the report has to justify.
"""
from __future__ import annotations

__all__ = ["config", "io", "clean", "aggregate", "qc", "plots", "pipeline"]
__version__ = "1.0.0"
