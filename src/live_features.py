from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from src.config import COMBINED_CSV_PATH


ROLLING_WINDOW = 5


def load_historical_matches(file_path: Path = COMBINED_CSV_PATH) -> pd.DataFrame:
    """
    Load the combined historical dataset used to build live features.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Combined historical dataset not found: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["date"])
    if df.empty:
        raise ValueError("Combined historical dataset is empty.")

    return df


def normalize_team_name(team_name: str) -> str:
    """
    Basic normalization for team names.
    """
    return " ".join(str(team_name).strip().split())


def implied_probabilities_from_odds(
    odds_home_win: float,
    odds_draw: float,
    odds_away_win: float,
) -> Dict[str, float]:
    """
    Convert decimal odds to normalized implied probabilities.
    """
    raw_home = 1 / odds_home_win
    raw_draw = 1 / odds_draw
    raw_away = 1 / odds_away_win

    total = raw_home + raw_draw + raw_away

    return {
        "implied_home_prob": raw_home / total,
        "implied_draw_prob": raw_draw / total,
        "implied_away_prob": raw_away / total,
    }


def points_from_result(result: str, is_home_team_perspective: bool) -> int:
    """
    Convert match result to team points from the team's perspective.
    """
    if result == "D":
        return 1

    if is_home_team_perspective:
        return 3 if result == "H" else 0

    return 3 if result == "A" else 0


def build_team_history_view(df: pd.DataFrame, team_name: str, league: str) -> pd.DataFrame:
    """
    Create a team-centric historical dataframe for one team within one league.
    """
    team_name = normalize_team_name(team_name)
    league = league.upper()

    home_matches = df[
        (df["league"].str.upper() == league)
        & (df["home_team"] == team_name)
    ].copy()

    away_matches = df[
        (df["league"].str.upper() == league)
        & (df["away_team"] == team_name)
    ].copy()

    home_view = pd.DataFrame({
        "date": home_matches["date"],
        "team": home_matches["home_team"],
        "opponent": home_matches["away_team"],
        "goals_for": home_matches["home_goals"],
        "goals_against": home_matches["away_goals"],
        "shots": home_matches["home_shots"],
        "shots_on_target": home_matches["home_shots_on_target"],
        "corners": home_matches["home_corners"],
        "fouls": home_matches["home_fouls"],
        "yellow_cards": home_matches["home_yellow_cards"],
        "red_cards": home_matches["home_red_cards"],
        "result": home_matches["result"],
        "is_home": 1,
    })

    away_view = pd.DataFrame({
        "date": away_matches["date"],
        "team": away_matches["away_team"],
        "opponent": away_matches["home_team"],
        "goals_for": away_matches["away_goals"],
        "goals_against": away_matches["home_goals"],
        "shots": away_matches["away_shots"],
        "shots_on_target": away_matches["away_shots_on_target"],
        "corners": away_matches["away_corners"],
        "fouls": away_matches["away_fouls"],
        "yellow_cards": away_matches["away_yellow_cards"],
        "red_cards": away_matches["away_red_cards"],
        "result": away_matches["result"],
        "is_home": 0,
    })

    history = pd.concat([home_view, away_view], ignore_index=True)
    history = history.sort_values("date").reset_index(drop=True)

    if history.empty:
        return history

    history["points"] = history.apply(
        lambda row: points_from_result(
            row["result"],
            is_home_team_perspective=bool(row["is_home"]),
        ),
        axis=1,
    )

    return history


def get_recent_team_form(
    df: pd.DataFrame,
    team_name: str,
    league: str,
    before_date: pd.Timestamp,
    window: int = ROLLING_WINDOW,
) -> Dict[str, Optional[float]]:
    """
    Compute rolling historical stats for one team before a fixture date.
    """
    history = build_team_history_view(df, team_name, league)

    if history.empty:
        return {
            "points_last5": None,
            "goals_for_avg_last5": None,
            "goals_against_avg_last5": None,
            "shots_avg_last5": None,
            "shots_on_target_avg_last5": None,
            "corners_avg_last5": None,
            "fouls_avg_last5": None,
            "yellow_avg_last5": None,
            "red_avg_last5": None,
            "matches_played_before": 0,
        }

    prior_matches = history[history["date"] < before_date].sort_values("date")

    matches_played_before = len(prior_matches)
    recent = prior_matches.tail(window)

    if recent.empty:
        return {
            "points_last5": None,
            "goals_for_avg_last5": None,
            "goals_against_avg_last5": None,
            "shots_avg_last5": None,
            "shots_on_target_avg_last5": None,
            "corners_avg_last5": None,
            "fouls_avg_last5": None,
            "yellow_avg_last5": None,
            "red_avg_last5": None,
            "matches_played_before": 0,
        }

    return {
        "points_last5": recent["points"].mean(),
        "goals_for_avg_last5": recent["goals_for"].mean(),
        "goals_against_avg_last5": recent["goals_against"].mean(),
        "shots_avg_last5": recent["shots"].mean(),
        "shots_on_target_avg_last5": recent["shots_on_target"].mean(),
        "corners_avg_last5": recent["corners"].mean(),
        "fouls_avg_last5": recent["fouls"].mean(),
        "yellow_avg_last5": recent["yellow_cards"].mean(),
        "red_avg_last5": recent["red_cards"].mean(),
        "matches_played_before": matches_played_before,
    }


def confidence_note_from_history(
    home_matches_played_before: int,
    away_matches_played_before: int,
) -> Optional[str]:
    """
    Return an advisory note for promoted/new teams or thin history cases.
    """
    min_history = min(home_matches_played_before, away_matches_played_before)

    if min_history < 3:
        return "Very limited same-league history for one or both teams; prediction confidence should be treated cautiously."
    if min_history < 5:
        return "Limited same-league history for one or both teams; confidence may be slightly reduced."
    return None


def build_live_feature_row(
    historical_df: pd.DataFrame,
    date: str,
    league: str,
    home_team: str,
    away_team: str,
    odds_home_win: float,
    odds_draw: float,
    odds_away_win: float,
) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    """
    Build a single model-ready feature row for an upcoming unseen fixture.

    Returns:
        row_df: dataframe with one row matching training feature schema
        metadata: extra info such as history warning note
    """
    fixture_date = pd.to_datetime(date, errors="coerce", utc=True)

    if pd.isna(fixture_date):
        raise ValueError("Invalid fixture date. Use YYYY-MM-DD or ISO-compatible format.")

    # Convert to timezone-naive UTC so it matches historical dataframe dates
    fixture_date = fixture_date.tz_localize(None)

    league = league.upper()
    home_team = normalize_team_name(home_team)
    away_team = normalize_team_name(away_team)

    home_form = get_recent_team_form(
        df=historical_df,
        team_name=home_team,
        league=league,
        before_date=fixture_date,
        window=ROLLING_WINDOW,
    )

    away_form = get_recent_team_form(
        df=historical_df,
        team_name=away_team,
        league=league,
        before_date=fixture_date,
        window=ROLLING_WINDOW,
    )

    implied = implied_probabilities_from_odds(
        odds_home_win=odds_home_win,
        odds_draw=odds_draw,
        odds_away_win=odds_away_win,
    )

    row = {
        "date": fixture_date,
        "season": None,
        "league": league,
        "home_team": home_team,
        "away_team": away_team,
        "home_goals": None,
        "away_goals": None,
        "result": None,
        "odds_home_win": odds_home_win,
        "odds_draw": odds_draw,
        "odds_away_win": odds_away_win,
        "implied_home_prob": implied["implied_home_prob"],
        "implied_draw_prob": implied["implied_draw_prob"],
        "implied_away_prob": implied["implied_away_prob"],
        "home_points_last5": home_form["points_last5"],
        "home_goals_for_avg_last5": home_form["goals_for_avg_last5"],
        "home_goals_against_avg_last5": home_form["goals_against_avg_last5"],
        "home_shots_avg_last5": home_form["shots_avg_last5"],
        "home_shots_on_target_avg_last5": home_form["shots_on_target_avg_last5"],
        "home_corners_avg_last5": home_form["corners_avg_last5"],
        "home_fouls_avg_last5": home_form["fouls_avg_last5"],
        "home_yellow_avg_last5": home_form["yellow_avg_last5"],
        "home_red_avg_last5": home_form["red_avg_last5"],
        "home_matches_played_before": home_form["matches_played_before"],
        "away_points_last5": away_form["points_last5"],
        "away_goals_for_avg_last5": away_form["goals_for_avg_last5"],
        "away_goals_against_avg_last5": away_form["goals_against_avg_last5"],
        "away_shots_avg_last5": away_form["shots_avg_last5"],
        "away_shots_on_target_avg_last5": away_form["shots_on_target_avg_last5"],
        "away_corners_avg_last5": away_form["corners_avg_last5"],
        "away_fouls_avg_last5": away_form["fouls_avg_last5"],
        "away_yellow_avg_last5": away_form["yellow_avg_last5"],
        "away_red_avg_last5": away_form["red_avg_last5"],
        "away_matches_played_before": away_form["matches_played_before"],
    }

    row["points_diff_last5"] = (
        row["home_points_last5"] - row["away_points_last5"]
        if row["home_points_last5"] is not None and row["away_points_last5"] is not None
        else None
    )
    row["goals_for_diff_last5"] = (
        row["home_goals_for_avg_last5"] - row["away_goals_for_avg_last5"]
        if row["home_goals_for_avg_last5"] is not None and row["away_goals_for_avg_last5"] is not None
        else None
    )
    row["goals_against_diff_last5"] = (
        row["home_goals_against_avg_last5"] - row["away_goals_against_avg_last5"]
        if row["home_goals_against_avg_last5"] is not None and row["away_goals_against_avg_last5"] is not None
        else None
    )
    row["shots_diff_last5"] = (
        row["home_shots_avg_last5"] - row["away_shots_avg_last5"]
        if row["home_shots_avg_last5"] is not None and row["away_shots_avg_last5"] is not None
        else None
    )
    row["shots_on_target_diff_last5"] = (
        row["home_shots_on_target_avg_last5"] - row["away_shots_on_target_avg_last5"]
        if row["home_shots_on_target_avg_last5"] is not None and row["away_shots_on_target_avg_last5"] is not None
        else None
    )

    metadata = {
        "history_note": confidence_note_from_history(
            home_matches_played_before=home_form["matches_played_before"],
            away_matches_played_before=away_form["matches_played_before"],
        )
    }

    row_df = pd.DataFrame([row])

    return row_df, metadata


def demo_live_feature_build() -> None:
    """
    Small local demo for feature generation of one hypothetical upcoming fixture.
    """
    historical_df = load_historical_matches()

    row_df, metadata = build_live_feature_row(
        historical_df=historical_df,
        date="2026-08-15",
        league="EPL",
        home_team="Arsenal",
        away_team="Chelsea",
        odds_home_win=2.10,
        odds_draw=3.40,
        odds_away_win=3.20,
    )

    print("=" * 100)
    print("LIVE FEATURE DEMO")
    print("=" * 100)
    print(row_df.to_string(index=False))
    print("\nMetadata:")
    print(metadata)


if __name__ == "__main__":
    demo_live_feature_build()