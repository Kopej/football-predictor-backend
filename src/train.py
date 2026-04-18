from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

from src.config import MODELS_DIR, PROCESSED_DATA_DIR


TRAINING_FEATURES_CSV_PATH = PROCESSED_DATA_DIR / "training_features.csv"
MODEL_OUTPUT_PATH = MODELS_DIR / "xgb_match_result_model.joblib"
LABEL_ENCODER_OUTPUT_PATH = MODELS_DIR / "label_encoder.joblib"
FEATURE_COLUMNS_OUTPUT_PATH = MODELS_DIR / "model_feature_columns.joblib"

TARGET_COLUMN = "result"

NUMERIC_FEATURES: List[str] = [
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

CATEGORICAL_FEATURES: List[str] = [
    "league",
]

ALL_FEATURES: List[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_training_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load engineered training features.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Training features file not found: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["date"])
    if df.empty:
        raise ValueError("Training features dataset is empty.")

    return df


def sort_dataset_for_time_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort dataset chronologically for train/test split.
    """
    return df.sort_values(["date", "league", "home_team", "away_team"]).reset_index(drop=True)


def build_train_test_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a chronological split instead of a random split.
    This is important for time-series-like sports data.
    """
    split_index = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    return train_df, test_df


def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target.
    """
    missing_features = [col for col in ALL_FEATURES if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required training features: {missing_features}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Build preprocessing pipeline for numeric and categorical features.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def build_model() -> XGBClassifier:
    """
    Create the V1 XGBoost classifier for 1X2 prediction.
    """
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    return model


def build_training_pipeline() -> Pipeline:
    """
    Full training pipeline = preprocessing + model.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", build_model()),
        ]
    )
    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test_encoded: np.ndarray,
    label_encoder: LabelEncoder,
) -> Dict[str, float]:
    """
    Evaluate trained model on holdout set.
    """
    y_pred_encoded = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)

    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    logloss = log_loss(y_test_encoded, y_pred_proba)

    target_names = list(label_encoder.classes_)
    report = classification_report(
        y_test_encoded,
        y_pred_encoded,
        target_names=target_names,
        digits=4,
    )

    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print("\nClassification Report:")
    print(report)

    return {
        "accuracy": accuracy,
        "log_loss": logloss,
    }


def save_artifacts(
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
    feature_columns: List[str],
) -> None:
    """
    Save model and metadata for inference.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_OUTPUT_PATH)
    joblib.dump(feature_columns, FEATURE_COLUMNS_OUTPUT_PATH)

    print("\nSaved artifacts:")
    print(f"  - Model: {MODEL_OUTPUT_PATH}")
    print(f"  - Label encoder: {LABEL_ENCODER_OUTPUT_PATH}")
    print(f"  - Feature columns: {FEATURE_COLUMNS_OUTPUT_PATH}")


def main() -> None:
    df = load_training_dataset(TRAINING_FEATURES_CSV_PATH)
    df = sort_dataset_for_time_split(df)

    train_df, test_df = build_train_test_split(df, test_size=0.2)

    print(f"Total rows: {len(df):,}")
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Train date range: {train_df['date'].min()} -> {train_df['date'].max()}")
    print(f"Test date range: {test_df['date'].min()} -> {test_df['date'].max()}")

    X_train, y_train = prepare_xy(train_df)
    X_test, y_test = prepare_xy(test_df)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    print(f"\nTarget classes: {list(label_encoder.classes_)}")

    pipeline = build_training_pipeline()
    pipeline.fit(X_train, y_train_encoded)

    evaluate_model(pipeline, X_test, y_test_encoded, label_encoder)
    save_artifacts(pipeline, label_encoder, ALL_FEATURES)


if __name__ == "__main__":
    main()