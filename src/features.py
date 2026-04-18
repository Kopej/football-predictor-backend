from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import COMBINED_CSV_PATH, PROCESSED_DATA_DIR


TRAINING_DATASET_CSV_PATH = PROCESSED_DATA_DIR / "training_features.csv"
TRAINING_DATASET_PARQUET_PATH = PROCESSED_DATA_DIR / "training_features.parquet"


ROLLING_WINDOW = 5


def load_combined_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load the combined historical match dataset.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Combined dataset not found: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["date"])
    if df.empty:
        raise ValueError("Combined dataset is empty.")

    return df


def add_implied_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert odds into normalized implied probabilities.
    """
    temp = df.copy()

    temp["raw_home_prob"] = 1 / temp["odds_home_win"]
    temp["raw_draw_prob"] = 1 / temp["odds_draw"]
    temp["raw_away_prob"] = 1 / temp["odds_away_win"]

    total = temp["raw_home_prob"] + temp["raw_draw_prob"] + temp["raw_away_prob"]

    temp["implied_home_prob"] = temp["raw_home_prob"] / total
    temp["implied_draw_prob"] = temp["raw_draw_prob"] / total
    temp["implied_away_prob"] = temp["raw_away_prob"] / total

    temp = temp.drop(columns=["raw_home_prob", "raw_draw_prob", "raw_away_prob"])

    return temp


def build_team_match_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a team-centric match history table where each row is one team's view of a match.
    """
    home_df = pd.DataFrame({
        "date": df["date"],
        "season": df["season"],
        "league": df["league"],
        "team": df["home_team"],
        "opponent": df["away_team"],
        "is_home": 1,
        "goals_for": df["home_goals"],
        "goals_against": df["away_goals"],
        "shots": df["home_shots"],
        "shots_on_target": df["home_shots_on_target"],
        "corners": df["home_corners"],
        "fouls": df["home_fouls"],
        "yellow_cards": df["home_yellow_cards"],
        "red_cards": df["home_red_cards"],
        "result": df["result"],
    })

    away_df = pd.DataFrame({
        "date": df["date"],
        "season": df["season"],
        "league": df["league"],
        "team": df["away_team"],
        "opponent": df["home_team"],
        "is_home": 0,
        "goals_for": df["away_goals"],
        "goals_against": df["home_goals"],
        "shots": df["away_shots"],
        "shots_on_target": df["away_shots_on_target"],
        "corners": df["away_corners"],
        "fouls": df["away_fouls"],
        "yellow_cards": df["away_yellow_cards"],
        "red_cards": df["away_red_cards"],
        "result": df["result"].map({"H": "A", "A": "H", "D": "D"}),
    })

    team_history = pd.concat([home_df, away_df], ignore_index=True)
    team_history = team_history.sort_values(["league", "team", "date"]).reset_index(drop=True)

    return team_history


def add_points(team_history: pd.DataFrame) -> pd.DataFrame:
    """
    Add points from the team perspective.
    """
    temp = team_history.copy()
    temp["points"] = temp["result"].map({"H": 3, "D": 1, "A": 0})
    return temp


def compute_rolling_features(team_history: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Compute rolling averages from previous matches only.
    """
    temp = team_history.copy()

    rolling_cols = [
        "points",
        "goals_for",
        "goals_against",
        "shots",
        "shots_on_target",
        "corners",
        "fouls",
        "yellow_cards",
        "red_cards",
    ]

    for col in rolling_cols:
        temp[f"{col}_avg_last{window}"] = (
            temp.groupby(["league", "team"])[col]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )

    temp[f"matches_played_before_last{window}"] = (
        temp.groupby(["league", "team"])
        .cumcount()
    )

    return temp


def split_home_away_features(team_history_features: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join team rolling features back to the original match rows for home and away teams.
    """
    feature_cols = [
        "date",
        "league",
        "team",
        "points_avg_last5",
        "goals_for_avg_last5",
        "goals_against_avg_last5",
        "shots_avg_last5",
        "shots_on_target_avg_last5",
        "corners_avg_last5",
        "fouls_avg_last5",
        "yellow_cards_avg_last5",
        "red_cards_avg_last5",
        "matches_played_before_last5",
    ]

    home_features = team_history_features[feature_cols].copy()
    away_features = team_history_features[feature_cols].copy()

    home_features = home_features.rename(columns={
        "team": "home_team",
        "points_avg_last5": "home_points_last5",
        "goals_for_avg_last5": "home_goals_for_avg_last5",
        "goals_against_avg_last5": "home_goals_against_avg_last5",
        "shots_avg_last5": "home_shots_avg_last5",
        "shots_on_target_avg_last5": "home_shots_on_target_avg_last5",
        "corners_avg_last5": "home_corners_avg_last5",
        "fouls_avg_last5": "home_fouls_avg_last5",
        "yellow_cards_avg_last5": "home_yellow_avg_last5",
        "red_cards_avg_last5": "home_red_avg_last5",
        "matches_played_before_last5": "home_matches_played_before",
    })

    away_features = away_features.rename(columns={
        "team": "away_team",
        "points_avg_last5": "away_points_last5",
        "goals_for_avg_last5": "away_goals_for_avg_last5",
        "goals_against_avg_last5": "away_goals_against_avg_last5",
        "shots_avg_last5": "away_shots_avg_last5",
        "shots_on_target_avg_last5": "away_shots_on_target_avg_last5",
        "corners_avg_last5": "away_corners_avg_last5",
        "fouls_avg_last5": "away_fouls_avg_last5",
        "yellow_cards_avg_last5": "away_yellow_avg_last5",
        "red_cards_avg_last5": "away_red_avg_last5",
        "matches_played_before_last5": "away_matches_played_before",
    })

    merged = original_df.merge(
        home_features,
        on=["date", "league", "home_team"],
        how="left",
    )

    merged = merged.merge(
        away_features,
        on=["date", "league", "away_team"],
        how="left",
    )

    return merged


def add_difference_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add home-away differential features.
    """
    temp = df.copy()

    temp["points_diff_last5"] = temp["home_points_last5"] - temp["away_points_last5"]
    temp["goals_for_diff_last5"] = temp["home_goals_for_avg_last5"] - temp["away_goals_for_avg_last5"]
    temp["goals_against_diff_last5"] = temp["home_goals_against_avg_last5"] - temp["away_goals_against_avg_last5"]
    temp["shots_diff_last5"] = temp["home_shots_avg_last5"] - temp["away_shots_avg_last5"]
    temp["shots_on_target_diff_last5"] = temp["home_shots_on_target_avg_last5"] - temp["away_shots_on_target_avg_last5"]

    return temp


def select_training_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the core columns needed for model training.
    """
    columns = [
        "date",
        "season",
        "league",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
        "odds_home_win",
        "odds_draw",
        "odds_away_win",
        "implied_home_prob",
        "implied_draw_prob",
        "implied_away_prob",
        "home_points_last5",
        "home_goals_for_avg_last5",
        "home_goals_against_avg_last5",
        "home_shots_avg_last5",
        "home_shots_on_target_avg_last5",
        "home_corners_avg_last5",
        "home_fouls_avg_last5",
        "home_yellow_avg_last5",
        "home_red_avg_last5",
        "home_matches_played_before",
        "away_points_last5",
        "away_goals_for_avg_last5",
        "away_goals_against_avg_last5",
        "away_shots_avg_last5",
        "away_shots_on_target_avg_last5",
        "away_corners_avg_last5",
        "away_fouls_avg_last5",
        "away_yellow_avg_last5",
        "away_red_avg_last5",
        "away_matches_played_before",
        "points_diff_last5",
        "goals_for_diff_last5",
        "goals_against_diff_last5",
        "shots_diff_last5",
        "shots_on_target_diff_last5",
    ]

    return df[columns].copy()


def build_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature-building pipeline.
    """
    temp = df.copy()
    temp = temp.sort_values(["date", "league", "home_team", "away_team"]).reset_index(drop=True)

    temp = add_implied_probabilities(temp)

    team_history = build_team_match_history(temp)
    team_history = add_points(team_history)
    team_history = compute_rolling_features(team_history, window=ROLLING_WINDOW)

    temp = split_home_away_features(team_history, temp)
    temp = add_difference_features(temp)
    temp = select_training_columns(temp)

    return temp


def save_training_dataset(df: pd.DataFrame) -> None:
    """
    Save training features dataset.
    """
    df.to_csv(TRAINING_DATASET_CSV_PATH, index=False)
    df.to_parquet(TRAINING_DATASET_PARQUET_PATH, index=False)


def main() -> None:
    df = load_combined_dataset(COMBINED_CSV_PATH)
    training_df = build_training_features(df)
    save_training_dataset(training_df)

    print("Training features dataset created successfully.")
    print(f"Saved CSV: {TRAINING_DATASET_CSV_PATH}")
    print(f"Saved Parquet: {TRAINING_DATASET_PARQUET_PATH}")
    print(f"Total rows: {len(training_df):,}")
    print(f"Total columns: {len(training_df.columns)}")
    print("\nColumns:")
    print(list(training_df.columns))
    print("\nPreview:")
    print(training_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()