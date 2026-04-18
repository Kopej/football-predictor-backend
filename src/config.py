from __future__ import annotations

from pathlib import Path
from typing import Dict, List

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Other directories
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Ensure important directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Expected master files in data/raw
LEAGUE_FILES: Dict[str, str] = {
    "EPL": "EPL_MASTER LIST.xlsx",
    "LALIGA": "LALIGA_MASTER LIST.xlsx",
    "SERIEA": "SERIEA_MASTER.xlsx",
    "BUNDESLIGA": "BUNDESLIGA_MASTER.xlsx",
}

# Final standardized columns expected in all master files
EXPECTED_COLUMNS: List[str] = [
    "date",
    "season",
    "league",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
    "ht_home_goals",
    "ht_away_goals",
    "ht_result",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_corners",
    "away_corners",
    "home_fouls",
    "away_fouls",
    "home_yellow_cards",
    "away_yellow_cards",
    "home_red_cards",
    "away_red_cards",
    "odds_home_win",
    "odds_draw",
    "odds_away_win",
]

# Core numeric columns
NUMERIC_COLUMNS: List[str] = [
    "home_goals",
    "away_goals",
    "ht_home_goals",
    "ht_away_goals",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_corners",
    "away_corners",
    "home_fouls",
    "away_fouls",
    "home_yellow_cards",
    "away_yellow_cards",
    "home_red_cards",
    "away_red_cards",
    "odds_home_win",
    "odds_draw",
    "odds_away_win",
]

# Categorical columns
CATEGORICAL_COLUMNS: List[str] = [
    "season",
    "league",
    "home_team",
    "away_team",
    "result",
    "ht_result",
]

# Valid result labels
VALID_RESULTS = {"H", "D", "A"}

# Output file names
COMBINED_CSV_PATH = PROCESSED_DATA_DIR / "combined_matches.csv"
COMBINED_PARQUET_PATH = PROCESSED_DATA_DIR / "combined_matches.parquet"