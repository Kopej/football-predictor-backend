from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.live_features import load_historical_matches, build_live_feature_row
from src.live_fixtures import fetch_upcoming_fixtures, filter_fixtures_with_complete_odds


TRAINING_FEATURES_PATH = PROCESSED_DATA_DIR / "training_features.csv"

RESULT_MODEL_PATH = MODELS_DIR / "xgb_match_result_model.joblib"
RESULT_LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
RESULT_FEATURE_COLUMNS_PATH = MODELS_DIR / "model_feature_columns.joblib"

BTTS_MODEL_PATH = MODELS_DIR / "xgb_btts_model.joblib"
BTTS_LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder_btts.joblib"
BTTS_FEATURE_COLUMNS_PATH = MODELS_DIR / "model_feature_columns_btts.joblib"

OVER25_MODEL_PATH = MODELS_DIR / "xgb_over_2_5_model.joblib"
OVER25_LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder_over_2_5.joblib"
OVER25_FEATURE_COLUMNS_PATH = MODELS_DIR / "model_feature_columns_over_2_5.joblib"

UNDER45_MODEL_PATH = MODELS_DIR / "xgb_under_4_5_model.joblib"
UNDER45_LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder_under_4_5.joblib"
UNDER45_FEATURE_COLUMNS_PATH = MODELS_DIR / "model_feature_columns_under_4_5.joblib"


def load_joblib_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return joblib.load(path)


def load_all_artifacts() -> Dict:
    """
    Load all trained models, encoders, and feature lists.
    """
    artifacts = {
        "result": {
            "model": load_joblib_file(RESULT_MODEL_PATH),
            "label_encoder": load_joblib_file(RESULT_LABEL_ENCODER_PATH),
            "feature_columns": load_joblib_file(RESULT_FEATURE_COLUMNS_PATH),
        },
        "btts": {
            "model": load_joblib_file(BTTS_MODEL_PATH),
            "label_encoder": load_joblib_file(BTTS_LABEL_ENCODER_PATH),
            "feature_columns": load_joblib_file(BTTS_FEATURE_COLUMNS_PATH),
        },
        "over_2_5": {
            "model": load_joblib_file(OVER25_MODEL_PATH),
            "label_encoder": load_joblib_file(OVER25_LABEL_ENCODER_PATH),
            "feature_columns": load_joblib_file(OVER25_FEATURE_COLUMNS_PATH),
        },
        "under_4_5": {
            "model": load_joblib_file(UNDER45_MODEL_PATH),
            "label_encoder": load_joblib_file(UNDER45_LABEL_ENCODER_PATH),
            "feature_columns": load_joblib_file(UNDER45_FEATURE_COLUMNS_PATH),
        },
    }
    return artifacts


def load_training_features() -> pd.DataFrame:
    """
    Load engineered training feature dataset for demo inference.
    """
    if not TRAINING_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Training features file not found: {TRAINING_FEATURES_PATH}")

    df = pd.read_csv(TRAINING_FEATURES_PATH, parse_dates=["date"])
    if df.empty:
        raise ValueError("Training features dataset is empty.")

    return df


def predict_single_model(
    row_df: pd.DataFrame,
    model,
    label_encoder,
    feature_columns: List[str],
) -> Tuple[str, Dict[str, float]]:
    """
    Predict one model and return decoded label + probability map.
    """
    X = row_df[feature_columns].copy()

    pred_encoded = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    class_labels = list(label_encoder.classes_)
    probability_map = {
        class_labels[i]: float(pred_proba[i]) for i in range(len(class_labels))
    }

    return pred_label, probability_map


def derive_double_chance(probabilities_1x2: Dict[str, float]) -> Dict[str, float]:
    """
    Derive double chance probabilities from 1X2 probabilities.
    """
    prob_home = probabilities_1x2.get("H", 0.0)
    prob_draw = probabilities_1x2.get("D", 0.0)
    prob_away = probabilities_1x2.get("A", 0.0)

    return {
        "1X": prob_home + prob_draw,
        "12": prob_home + prob_away,
        "X2": prob_draw + prob_away,
    }


def get_confidence_band(confidence: float) -> str:
    """
    Convert numeric confidence into a frontend-friendly band.
    """
    if confidence >= 0.80:
        return "Strong"
    if confidence >= 0.65:
        return "Medium"
    return "Low"


def build_ranked_predictions(
    result_label: str,
    result_probs: Dict[str, float],
    btts_label: str,
    btts_probs: Dict[str, float],
    over25_label: str,
    over25_probs: Dict[str, float],
    under45_label: str,
    under45_probs: Dict[str, float],
) -> List[Dict]:
    """
    Build a ranked list of market predictions across all supported markets.
    """
    result_selection_map = {
        "H": "Home Win",
        "D": "Draw",
        "A": "Away Win",
    }

    double_chance_probs = derive_double_chance(result_probs)
    best_double_chance = max(double_chance_probs, key=double_chance_probs.get)

    ranked_predictions = [
        {
            "market_key": "match_result",
            "market": "Match Result",
            "selection": result_selection_map.get(result_label, result_label),
            "raw_selection": result_label,
            "confidence": result_probs.get(result_label, 0.0),
        },
        {
            "market_key": "double_chance",
            "market": "Double Chance",
            "selection": best_double_chance,
            "raw_selection": best_double_chance,
            "confidence": double_chance_probs.get(best_double_chance, 0.0),
        },
        {
            "market_key": "btts",
            "market": "BTTS",
            "selection": "Yes" if btts_label == "YES" else "No",
            "raw_selection": btts_label,
            "confidence": btts_probs.get(btts_label, 0.0),
        },
        {
            "market_key": "over_2_5",
            "market": "Over 2.5 Goals",
            "selection": "Yes" if over25_label == "YES" else "No",
            "raw_selection": over25_label,
            "confidence": over25_probs.get(over25_label, 0.0),
        },
        {
            "market_key": "under_4_5",
            "market": "Under 4.5 Goals",
            "selection": "Yes" if under45_label == "YES" else "No",
            "raw_selection": under45_label,
            "confidence": under45_probs.get(under45_label, 0.0),
        },
    ]

    for item in ranked_predictions:
        item["confidence_band"] = get_confidence_band(item["confidence"])

    ranked_predictions = sorted(
        ranked_predictions,
        key=lambda x: x["confidence"],
        reverse=True,
    )

    return ranked_predictions


def select_top_predictions(ranked_predictions: List[Dict], top_n: int = 2) -> List[Dict]:
    """
    Select top predictions with a light diversity rule:
    prefer not to return duplicate market types, which is already guaranteed here,
    and simply keep the top N by confidence.
    """
    return ranked_predictions[:top_n]


def build_prediction_output(row: pd.Series, artifacts: Dict) -> Dict:
    """
    Generate all market predictions for one match row, plus ranked top picks.
    """
    row_df = pd.DataFrame([row])

    # 1X2
    result_label, result_probs = predict_single_model(
        row_df=row_df,
        model=artifacts["result"]["model"],
        label_encoder=artifacts["result"]["label_encoder"],
        feature_columns=artifacts["result"]["feature_columns"],
    )

    result_map = {
        "H": "Home Win",
        "D": "Draw",
        "A": "Away Win",
    }

    # BTTS
    btts_label, btts_probs = predict_single_model(
        row_df=row_df,
        model=artifacts["btts"]["model"],
        label_encoder=artifacts["btts"]["label_encoder"],
        feature_columns=artifacts["btts"]["feature_columns"],
    )

    # Over 2.5
    over25_label, over25_probs = predict_single_model(
        row_df=row_df,
        model=artifacts["over_2_5"]["model"],
        label_encoder=artifacts["over_2_5"]["label_encoder"],
        feature_columns=artifacts["over_2_5"]["feature_columns"],
    )

    # Under 4.5
    under45_label, under45_probs = predict_single_model(
        row_df=row_df,
        model=artifacts["under_4_5"]["model"],
        label_encoder=artifacts["under_4_5"]["label_encoder"],
        feature_columns=artifacts["under_4_5"]["feature_columns"],
    )

    double_chance_probs = derive_double_chance(result_probs)
    best_double_chance = max(double_chance_probs, key=double_chance_probs.get)

    ranked_predictions = build_ranked_predictions(
        result_label=result_label,
        result_probs=result_probs,
        btts_label=btts_label,
        btts_probs=btts_probs,
        over25_label=over25_label,
        over25_probs=over25_probs,
        under45_label=under45_label,
        under45_probs=under45_probs,
    )

    top_predictions = select_top_predictions(ranked_predictions, top_n=2)
    primary_prediction = top_predictions[0] if len(top_predictions) >= 1 else None
    secondary_prediction = top_predictions[1] if len(top_predictions) >= 2 else None

    return {
        "match": {
            "date": str(row["date"].date()) if pd.notna(row["date"]) else None,
            "league": row["league"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "actual_result": row.get("result"),
        },
        "primary_prediction": primary_prediction,
        "secondary_prediction": secondary_prediction,
        "top_predictions": top_predictions,
        "all_ranked_predictions": ranked_predictions,
        "all_markets": {
            "match_result": {
                "predicted_class": result_label,
                "predicted_result": result_map.get(result_label, result_label),
                "probabilities": {
                    "home_win": result_probs.get("H", 0.0),
                    "draw": result_probs.get("D", 0.0),
                    "away_win": result_probs.get("A", 0.0),
                },
            },
            "double_chance": {
                "probabilities": double_chance_probs,
                "best_market": best_double_chance,
            },
            "btts": {
                "predicted": btts_label,
                "probabilities": {
                    "yes": btts_probs.get("YES", 0.0),
                    "no": btts_probs.get("NO", 0.0),
                },
            },
            "over_2_5": {
                "predicted": over25_label,
                "probabilities": {
                    "yes": over25_probs.get("YES", 0.0),
                    "no": over25_probs.get("NO", 0.0),
                },
            },
            "under_4_5": {
                "predicted": under45_label,
                "probabilities": {
                    "yes": under45_probs.get("YES", 0.0),
                    "no": under45_probs.get("NO", 0.0),
                },
            },
        },
    }

def build_live_prediction_output(
    fixture: Dict,
    historical_df: pd.DataFrame,
    artifacts: Dict,
) -> Dict:
    """
    Build predictions for one upcoming live fixture.
    """
    row_df, metadata = build_live_feature_row(
        historical_df=historical_df,
        date=fixture["date"],
        league=fixture["league"],
        home_team=fixture["home_team"],
        away_team=fixture["away_team"],
        odds_home_win=fixture["odds_home_win"],
        odds_draw=fixture["odds_draw"],
        odds_away_win=fixture["odds_away_win"],
    )

    row = row_df.iloc[0]
    output = build_prediction_output(row, artifacts)

    # attach live odds + history note
    output["match"]["odds"] = {
        "home_win": fixture["odds_home_win"],
        "draw": fixture["odds_draw"],
        "away_win": fixture["odds_away_win"],
    }

    output["prediction_note"] = metadata.get("history_note")

    return output


def predict_upcoming_fixtures_for_league(
    league: str,
    limit: int = 10,
) -> List[Dict]:
    """
    Fetch upcoming fixtures for a league, generate live features,
    run predictions, and return ranked outputs.
    """
    artifacts = load_all_artifacts()
    historical_df = load_historical_matches()

    fixtures = fetch_upcoming_fixtures(league)
    fixtures = filter_fixtures_with_complete_odds(fixtures)
    fixtures = fixtures[:limit]

    outputs = []
    for fixture in fixtures:
        try:
            prediction = build_live_prediction_output(
                fixture=fixture,
                historical_df=historical_df,
                artifacts=artifacts,
            )
            outputs.append(prediction)
        except Exception as exc:
            outputs.append(
                {
                    "match": fixture,
                    "error": str(exc),
                }
            )

    return outputs

def demo_predict_latest_matches(n: int = 10) -> None:
    """
    Run demo predictions on the latest n matches from training_features.csv
    to verify inference works end-to-end.
    """
    artifacts = load_all_artifacts()
    df = load_training_features()

    df = df.sort_values("date", ascending=False).head(n).copy()

    print("=" * 110)
    print(f"DEMO PREDICTIONS FOR LATEST {n} MATCHES")
    print("=" * 110)

    for _, row in df.iterrows():
        output = build_prediction_output(row, artifacts)

        match = output["match"]
        primary = output["primary_prediction"]
        secondary = output["secondary_prediction"]
        result = output["all_markets"]["match_result"]
        dc = output["all_markets"]["double_chance"]
        btts = output["all_markets"]["btts"]
        over25 = output["all_markets"]["over_2_5"]
        under45 = output["all_markets"]["under_4_5"]

        print(f"{match['date']} | {match['league']} | {match['home_team']} vs {match['away_team']}")
        print(f"Actual result: {match['actual_result']}")

        if primary:
            print(
                f"Primary Prediction: {primary['market']} - {primary['selection']} "
                f"({primary['confidence']:.3f}, {primary['confidence_band']})"
            )
        if secondary:
            print(
                f"Secondary Prediction: {secondary['market']} - {secondary['selection']} "
                f"({secondary['confidence']:.3f}, {secondary['confidence_band']})"
            )

        print(
            f"1X2 -> Home={result['probabilities']['home_win']:.3f}, "
            f"Draw={result['probabilities']['draw']:.3f}, "
            f"Away={result['probabilities']['away_win']:.3f}"
        )
        print(
            f"Double Chance -> 1X={dc['probabilities']['1X']:.3f}, "
            f"12={dc['probabilities']['12']:.3f}, "
            f"X2={dc['probabilities']['X2']:.3f}"
        )
        print(
            f"BTTS -> Yes={btts['probabilities']['yes']:.3f}, "
            f"No={btts['probabilities']['no']:.3f}"
        )
        print(
            f"Over 2.5 -> Yes={over25['probabilities']['yes']:.3f}, "
            f"No={over25['probabilities']['no']:.3f}"
        )
        print(
            f"Under 4.5 -> Yes={under45['probabilities']['yes']:.3f}, "
            f"No={under45['probabilities']['no']:.3f}"
        )
        print("-" * 110)

def demo_predict_upcoming(league: str = "EPL", limit: int = 5) -> None:
    """
    Demo upcoming fixture predictions for one league.
    """
    outputs = predict_upcoming_fixtures_for_league(league=league, limit=limit)

    print("=" * 110)
    print(f"UPCOMING MATCH PREDICTIONS - {league.upper()}")
    print("=" * 110)

    for output in outputs:
        if "error" in output:
            print(f"ERROR for fixture: {output['match']}")
            print(output["error"])
            print("-" * 110)
            continue

        match = output["match"]
        primary = output["primary_prediction"]
        secondary = output["secondary_prediction"]

        print(f"{match['date']} | {match['league']} | {match['home_team']} vs {match['away_team']}")
        print(
            f"Odds -> Home={match['odds']['home_win']}, "
            f"Draw={match['odds']['draw']}, "
            f"Away={match['odds']['away_win']}"
        )

        if primary:
            print(
                f"Primary Prediction: {primary['market']} - {primary['selection']} "
                f"({primary['confidence']:.3f}, {primary['confidence_band']})"
            )
        if secondary:
            print(
                f"Secondary Prediction: {secondary['market']} - {secondary['selection']} "
                f"({secondary['confidence']:.3f}, {secondary['confidence_band']})"
            )

        if output.get("prediction_note"):
            print(f"Note: {output['prediction_note']}")

        print("-" * 110)

def main() -> None:
    demo_predict_upcoming(league="EPL", limit=5)


if __name__ == "__main__":
    main()