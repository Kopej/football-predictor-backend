from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from src.config import (
    COMBINED_CSV_PATH,
    COMBINED_PARQUET_PATH,
    EXPECTED_COLUMNS,
    LEAGUE_FILES,
    NUMERIC_COLUMNS,
    RAW_DATA_DIR,
    VALID_RESULTS,
)


def read_master_file(file_path: str | pd.io.common.FilePathOrBuffer) -> pd.DataFrame:
    """
    Read a master Excel file into a DataFrame.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is empty.
    """
    df = pd.read_excel(file_path)

    if df.empty:
        raise ValueError(f"File is empty: {file_path}")

    return df


def normalize_text_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Strip whitespace while preserving missing values.
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Normalize result fields to uppercase while preserving NaN
    for result_col in ["result", "ht_result"]:
        if result_col in df.columns:
            df[result_col] = df[result_col].apply(
                lambda x: x.upper().strip() if isinstance(x, str) else x
            )

    # Clean team names lightly
    for team_col in ["home_team", "away_team"]:
        if team_col in df.columns:
            df[team_col] = df[team_col].apply(
                lambda x: " ".join(x.split()) if isinstance(x, str) else x
            )

    return df


def coerce_numeric_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Convert numeric columns to numeric dtype, coercing invalid values to NaN.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the date column and standardize it to pandas datetime.
    Handles both ISO dates and football-data style day-first dates.
    """
    if "date" not in df.columns:
        raise ValueError("Missing required column: date")

    sample = df["date"].dropna().astype(str).head(10)

    if sample.str.match(r"^\d{4}-\d{2}-\d{2}$").all():
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

    if df["date"].isna().all():
        raise ValueError("All dates failed to parse in dataset.")

    return df


def validate_required_columns(df: pd.DataFrame, expected_columns: List[str]) -> None:
    """
    Ensure all expected columns exist.
    """
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_result_values(df: pd.DataFrame) -> None:
    """
    Ensure result and ht_result values are limited to H/D/A where non-null.
    """
    for col in ["result", "ht_result"]:
        if col in df.columns:
            non_null = df[col].dropna()
            invalid = non_null[~non_null.isin(VALID_RESULTS)].unique()
            if len(invalid) > 0:
                raise ValueError(f"Invalid values found in {col}: {invalid}")


def ensure_league_label(df: pd.DataFrame, expected_league: str) -> pd.DataFrame:
    """
    Ensure league column exists and matches expected league key.
    """
    if "league" not in df.columns:
        df["league"] = expected_league

    df["league"] = df["league"].astype(str).str.strip().str.upper()

    if not (df["league"] == expected_league).all():
        # Force standard label for consistency
        df["league"] = expected_league

    return df


def reorder_columns(df: pd.DataFrame, expected_columns: List[str]) -> pd.DataFrame:
    """
    Reorder DataFrame columns to the standard schema.
    """
    return df[expected_columns].copy()


def standardize_master_dataframe(df: pd.DataFrame, expected_league: str) -> pd.DataFrame:
    """
    Apply all standardization steps to one league master DataFrame.
    """
    validate_required_columns(df, EXPECTED_COLUMNS)

    df = ensure_league_label(df, expected_league)
    df = normalize_text_columns(
        df,
        columns=["season", "league", "home_team", "away_team", "result", "ht_result"],
    )
    df = coerce_numeric_columns(df, NUMERIC_COLUMNS)
    df = parse_dates(df)
    validate_result_values(df)

    df = reorder_columns(df, EXPECTED_COLUMNS)
    return df


def load_all_leagues() -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load and standardize all configured league master files.

    Returns:
        combined_df: Combined dataframe across leagues
        row_counts: Row counts by league
    """
    league_frames: List[pd.DataFrame] = []
    row_counts: Dict[str, int] = {}

    for league, filename in LEAGUE_FILES.items():
        file_path = RAW_DATA_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Expected master file not found for {league}: {file_path}"
            )

        df = read_master_file(file_path)
        df = standardize_master_dataframe(df, expected_league=league)

        row_counts[league] = len(df)
        league_frames.append(df)

    combined_df = pd.concat(league_frames, ignore_index=True)
    combined_df = combined_df.sort_values(["date", "league", "home_team", "away_team"]).reset_index(drop=True)

    return combined_df, row_counts


def save_combined_dataset(df: pd.DataFrame) -> None:
    """
    Save combined dataset to CSV and Parquet.
    """
    df.to_csv(COMBINED_CSV_PATH, index=False)
    df.to_parquet(COMBINED_PARQUET_PATH, index=False)


def main() -> None:
    """
    Load, standardize, combine, and save all league master datasets.
    """
    combined_df, row_counts = load_all_leagues()
    save_combined_dataset(combined_df)

    print("Combined dataset created successfully.")
    print(f"Saved CSV: {COMBINED_CSV_PATH}")
    print(f"Saved Parquet: {COMBINED_PARQUET_PATH}")
    print("Row counts by league:")
    for league, count in row_counts.items():
        print(f"  - {league}: {count:,}")
    print(f"Total rows: {len(combined_df):,}")
    print("\nColumns:")
    print(list(combined_df.columns))


if __name__ == "__main__":
    main()