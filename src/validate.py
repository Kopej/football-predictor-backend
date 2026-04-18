from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.config import (
    COMBINED_CSV_PATH,
    EXPECTED_COLUMNS,
    VALID_RESULTS,
)


def load_combined_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load the combined dataset from CSV.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Combined dataset not found: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["date"])
    if df.empty:
        raise ValueError("Combined dataset is empty.")

    return df


def check_required_columns(df: pd.DataFrame, expected_columns: List[str]) -> None:
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def check_duplicate_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify exact duplicate fixtures by date + league + teams.
    """
    duplicate_mask = df.duplicated(
        subset=["date", "league", "home_team", "away_team"],
        keep=False,
    )
    return df.loc[duplicate_mask].sort_values(
        ["date", "league", "home_team", "away_team"]
    )


def derive_result_from_goals(home_goals: float, away_goals: float) -> str | None:
    if pd.isna(home_goals) or pd.isna(away_goals):
        return None
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def check_full_time_result_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare stored result with result implied by full-time goals.
    """
    temp = df.copy()
    temp["derived_result"] = temp.apply(
        lambda row: derive_result_from_goals(row["home_goals"], row["away_goals"]),
        axis=1,
    )

    mismatches = temp[
        temp["derived_result"].notna()
        & temp["result"].notna()
        & (temp["derived_result"] != temp["result"])
    ].copy()

    return mismatches


def check_half_time_result_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare stored ht_result with result implied by half-time goals.
    """
    temp = df.copy()
    temp["derived_ht_result"] = temp.apply(
        lambda row: derive_result_from_goals(row["ht_home_goals"], row["ht_away_goals"]),
        axis=1,
    )

    mismatches = temp[
        temp["derived_ht_result"].notna()
        & temp["ht_result"].notna()
        & (temp["derived_ht_result"] != temp["ht_result"])
    ].copy()

    return mismatches


def check_result_labels(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Check invalid result labels for result and ht_result.
    """
    issues = {}
    for col in ["result", "ht_result"]:
        values = df[col].dropna().unique().tolist()
        invalid = sorted([v for v in values if v not in VALID_RESULTS])
        if invalid:
            issues[col] = invalid
    return issues


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return missing counts and percentages by column.
    """
    total_rows = len(df)
    report = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
    })
    report["missing_pct"] = (report["missing_count"] / total_rows * 100).round(2)
    return report.sort_values(["missing_pct", "column"], ascending=[False, True]).reset_index(drop=True)


def odds_sanity_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find suspicious odds values.
    Odds should generally be > 1.
    """
    odds_cols = ["odds_home_win", "odds_draw", "odds_away_win"]

    suspicious = df[
        (
            (df["odds_home_win"].notna() & (df["odds_home_win"] <= 1))
            | (df["odds_draw"].notna() & (df["odds_draw"] <= 1))
            | (df["odds_away_win"].notna() & (df["odds_away_win"] <= 1))
        )
    ].copy()

    return suspicious


def league_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize row counts and date ranges by league.
    """
    summary = (
        df.groupby("league")
        .agg(
            rows=("league", "size"),
            min_date=("date", "min"),
            max_date=("date", "max"),
            seasons=("season", "nunique"),
            teams_home=("home_team", "nunique"),
            teams_away=("away_team", "nunique"),
        )
        .reset_index()
    )
    return summary.sort_values("league").reset_index(drop=True)


def print_report(df: pd.DataFrame) -> None:
    """
    Print a validation summary to console.
    """
    print("=" * 80)
    print("VALIDATION REPORT")
    print("=" * 80)

    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print()

    print("League summary:")
    print(league_summary(df).to_string(index=False))
    print()

    invalid_result_labels = check_result_labels(df)
    if invalid_result_labels:
        print("Invalid result labels found:")
        for col, values in invalid_result_labels.items():
            print(f"  - {col}: {values}")
    else:
        print("Result label check: PASSED")
    print()

    duplicates = check_duplicate_fixtures(df)
    print(f"Duplicate fixtures found: {len(duplicates):,}")
    if not duplicates.empty:
        print(duplicates.head(10).to_string(index=False))
    print()

    ft_mismatches = check_full_time_result_consistency(df)
    print(f"Full-time result mismatches: {len(ft_mismatches):,}")
    if not ft_mismatches.empty:
        print(
            ft_mismatches[
                ["date", "league", "home_team", "away_team", "home_goals", "away_goals", "result", "derived_result"]
            ]
            .head(10)
            .to_string(index=False)
        )
    print()

    ht_mismatches = check_half_time_result_consistency(df)
    print(f"Half-time result mismatches: {len(ht_mismatches):,}")
    if not ht_mismatches.empty:
        print(
            ht_mismatches[
                ["date", "league", "home_team", "away_team", "ht_home_goals", "ht_away_goals", "ht_result", "derived_ht_result"]
            ]
            .head(10)
            .to_string(index=False)
        )
    print()

    suspicious_odds = odds_sanity_check(df)
    print(f"Suspicious odds rows (<= 1): {len(suspicious_odds):,}")
    if not suspicious_odds.empty:
        print(
            suspicious_odds[
                ["date", "league", "home_team", "away_team", "odds_home_win", "odds_draw", "odds_away_win"]
            ]
            .head(10)
            .to_string(index=False)
        )
    print()

    print("Top missing-value columns:")
    print(missing_value_report(df).head(15).to_string(index=False))
    print()

    print("=" * 80)
    print("END OF REPORT")
    print("=" * 80)


def main() -> None:
    df = load_combined_dataset(COMBINED_CSV_PATH)
    check_required_columns(df, EXPECTED_COLUMNS)
    print_report(df)


if __name__ == "__main__":
    main()