"""
Shared helpers used across the pipeline stages.
"""
from pathlib import Path

# Project paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "reports" / "figures"

for p in (DATA_INTERIM, DATA_PROCESSED, FIGURES):
    p.mkdir(parents=True, exist_ok=True)

# Columns where 0 is a SENTINEL meaning "not recorded", not a true zero.
# See Stage 2 for the per-column treatment rules.
SENTINEL_ZERO_COLS = [
    "metacritic_score", "user_score",
    "positive", "negative",
    "achievements", "recommendations",
    "average_playtime_forever", "median_playtime_forever",
    "average_playtime_2weeks", "median_playtime_2weeks",
    "peak_ccu",
]

# Columns that are strong candidates for log1p transformation.
# Final decision is made in Stage 4 after measuring skew.
LOG_CANDIDATES = [
    "peak_ccu", "recommendations", "positive", "negative",
    "achievements", "average_playtime_forever",
    "median_playtime_forever", "price",
]