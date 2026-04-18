from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

from src.config import MODELS_DIR, PROCESSED_DATA_DIR


TRAINING_FEATURES_CSV_PATH = PROCESSED_DATA_DIR / "training_features.csv"

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

MARKET_CONFIG = {
    "btts": {
        "target_col": "target_btts",
        "model_path": MODELS_DIR / "xgb_btts_model.joblib",
        "label_encoder_path": MODELS_DIR / "label_encoder_btts.joblib",
        "feature_columns_path": MODELS_DIR / "model_feature_columns_btts.joblib",
        "positive_label": "YES",
        "negative_label": "NO",
    },
    "over_2_5": {
        "target_col": "target_over_2_5",
        "model_path": MODELS_DIR / "xgb_over_2_5_model.joblib",
        "label_encoder_path": MODELS_DIR / "label_encoder_over_2_5.joblib",
        "feature_columns_path": MODELS_DIR / "model_feature_columns_over_2_5.joblib",
        "positive_label": "YES",
        "negative_label": "NO",
    },
    "under_4_5": {
        "target_col": "target_under_4_5",
        "model_path": MODELS_DIR / "xgb_under_4_5_model.joblib",
        "label_encoder_path": MODELS_DIR / "label_encoder_under_4_5.joblib",
        "feature_columns_path": MODELS_DIR / "model_feature_columns_under_4_5.joblib",
        "positive_label": "YES",
        "negative_label": "NO",
    },
}


def load_training_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load engineered training features and base match columns.
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


def add_market_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target columns for additional betting markets.
    """
    temp = df.copy()

    if "home_goals" not in temp.columns or "away_goals" not in temp.columns:
        raise ValueError(
            "training_features.csv must contain home_goals and away_goals columns. "
            "Update features.py to retain them before training market models."
        )

    total_goals = temp["home_goals"] + temp["away_goals"]

    temp["target_btts"] = (
        ((temp["home_goals"] > 0) & (temp["away_goals"] > 0))
        .map({True: "YES", False: "NO"})
    )

    temp["target_over_2_5"] = (
        (total_goals > 2.5)
        .map({True: "YES", False: "NO"})
    )

    temp["target_under_4_5"] = (
        (total_goals < 4.5)
        .map({True: "YES", False: "NO"})
    )

    return temp


def build_train_test_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform chronological split.
    """
    split_index = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def prepare_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and selected target.
    """
    missing_features = [col for col in ALL_FEATURES if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required training features: {missing_features}")

    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    X = df[ALL_FEATURES].copy()
    y = df[target_col].copy()

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
    Create binary XGBoost classifier.
    """
    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
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
    market_name: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test_encoded,
    label_encoder: LabelEncoder,
) -> Dict[str, float]:
    """
    Evaluate binary market model on holdout set.
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
    print(f"MODEL EVALUATION: {market_name.upper()}")
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
    model_name: str,
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
    feature_columns: List[str],
) -> None:
    """
    Save model and metadata for one market.
    """
    cfg = MARKET_CONFIG[model_name]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, cfg["model_path"])
    joblib.dump(label_encoder, cfg["label_encoder_path"])
    joblib.dump(feature_columns, cfg["feature_columns_path"])

    print("\nSaved artifacts:")
    print(f"  - Model: {cfg['model_path']}")
    print(f"  - Label encoder: {cfg['label_encoder_path']}")
    print(f"  - Feature columns: {cfg['feature_columns_path']}")


def train_single_market(
    df: pd.DataFrame,
    market_name: str,
    test_size: float = 0.2,
) -> Dict[str, float]:
    """
    Train and evaluate a single market model.
    """
    cfg = MARKET_CONFIG[market_name]
    target_col = cfg["target_col"]

    train_df, test_df = build_train_test_split(df, test_size=test_size)

    X_train, y_train = prepare_xy(train_df, target_col)
    X_test, y_test = prepare_xy(test_df, target_col)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    print("\n" + "-" * 80)
    print(f"Training market: {market_name}")
    print(f"Target classes: {list(label_encoder.classes_)}")
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Train date range: {train_df['date'].min()} -> {train_df['date'].max()}")
    print(f"Test date range: {test_df['date'].min()} -> {test_df['date'].max()}")

    pipeline = build_training_pipeline()
    pipeline.fit(X_train, y_train_encoded)

    metrics = evaluate_model(
        market_name=market_name,
        pipeline=pipeline,
        X_test=X_test,
        y_test_encoded=y_test_encoded,
        label_encoder=label_encoder,
    )

    save_artifacts(
        model_name=market_name,
        pipeline=pipeline,
        label_encoder=label_encoder,
        feature_columns=ALL_FEATURES,
    )

    return metrics


def main() -> None:
    df = load_training_dataset(TRAINING_FEATURES_CSV_PATH)
    df = sort_dataset_for_time_split(df)
    df = add_market_targets(df)

    print(f"Total rows: {len(df):,}")
    print(f"Total columns after market targets: {len(df.columns)}")

    summary = {}
    for market_name in MARKET_CONFIG.keys():
        summary[market_name] = train_single_market(df, market_name)

    print("\n" + "=" * 80)
    print("SUMMARY OF MARKET MODELS")
    print("=" * 80)
    for market_name, metrics in summary.items():
        print(
            f"{market_name}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"log_loss={metrics['log_loss']:.4f}"
        )


if __name__ == "__main__":
    main()