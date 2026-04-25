from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.telegram_sender import send_league_predictions_to_telegram

from src.predict import (
    build_prediction_output,
    load_all_artifacts,
    load_training_features,
    predict_upcoming_fixtures_for_league,
)

app = FastAPI(
    title="Football Predictor API",
    version="1.0.0",
    description="Multi-market football prediction API for EPL, LaLiga, Serie A, and Bundesliga.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later for production frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS = None
TRAINING_FEATURES_DF = None


class MatchPredictionResponse(BaseModel):
    match: Dict[str, Any]
    primary_prediction: Optional[Dict[str, Any]]
    secondary_prediction: Optional[Dict[str, Any]]
    top_predictions: List[Dict[str, Any]]
    all_ranked_predictions: List[Dict[str, Any]]
    all_markets: Dict[str, Any]


@app.on_event("startup")
def startup_event() -> None:
    global ARTIFACTS, TRAINING_FEATURES_DF
    ARTIFACTS = load_all_artifacts()
    TRAINING_FEATURES_DF = load_training_features()

@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Football Predictor API is running."}

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/leagues")
def get_leagues() -> Dict[str, List[str]]:
    global TRAINING_FEATURES_DF
    if TRAINING_FEATURES_DF is None:
        raise HTTPException(status_code=500, detail="Training dataset not loaded.")

    leagues = sorted(TRAINING_FEATURES_DF["league"].dropna().unique().tolist())
    return {"leagues": leagues}


@app.get("/latest-matches")
def latest_matches(
    league: Optional[str] = Query(default=None, description="League code, e.g. EPL"),
    limit: int = Query(default=20, ge=1, le=100),
) -> Dict[str, Any]:
    global TRAINING_FEATURES_DF
    if TRAINING_FEATURES_DF is None:
        raise HTTPException(status_code=500, detail="Training dataset not loaded.")

    df = TRAINING_FEATURES_DF.copy()

    if league:
        df = df[df["league"].str.upper() == league.upper()]

    df = df.sort_values("date", ascending=False).head(limit)

    matches = []
    for _, row in df.iterrows():
        matches.append(
            {
                "date": str(row["date"].date()) if pd.notna(row["date"]) else None,
                "league": row["league"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "result": row.get("result"),
            }
        )

    return {
        "count": len(matches),
        "matches": matches,
    }


@app.get("/predict-latest", response_model=List[MatchPredictionResponse])
def predict_latest(
    league: Optional[str] = Query(default=None, description="League code, e.g. EPL"),
    limit: int = Query(default=10, ge=1, le=50),
) -> List[Dict[str, Any]]:
    global TRAINING_FEATURES_DF, ARTIFACTS

    if TRAINING_FEATURES_DF is None or ARTIFACTS is None:
        raise HTTPException(status_code=500, detail="Prediction resources not loaded.")

    df = TRAINING_FEATURES_DF.copy()

    if league:
        df = df[df["league"].str.upper() == league.upper()]

    if df.empty:
        raise HTTPException(status_code=404, detail="No matches found for the requested league.")

    df = df.sort_values("date", ascending=False).head(limit)

    outputs = []
    for _, row in df.iterrows():
        outputs.append(build_prediction_output(row, ARTIFACTS))

    return outputs


@app.get("/predict-match", response_model=MatchPredictionResponse)
def predict_match(
    date: str = Query(..., description="Match date in YYYY-MM-DD format"),
    league: str = Query(..., description="League code, e.g. EPL"),
    home_team: str = Query(..., description="Home team name"),
    away_team: str = Query(..., description="Away team name"),
) -> Dict[str, Any]:
    global TRAINING_FEATURES_DF, ARTIFACTS

    if TRAINING_FEATURES_DF is None or ARTIFACTS is None:
        raise HTTPException(status_code=500, detail="Prediction resources not loaded.")

    df = TRAINING_FEATURES_DF.copy()
    target_date = pd.to_datetime(date, errors="coerce")

    if pd.isna(target_date):
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    mask = (
        (df["date"] == target_date)
        & (df["league"].str.upper() == league.upper())
        & (df["home_team"].str.lower() == home_team.lower())
        & (df["away_team"].str.lower() == away_team.lower())
    )

    match_df = df.loc[mask]

    if match_df.empty:
        raise HTTPException(
            status_code=404,
            detail="Match not found in training features dataset.",
        )

    row = match_df.iloc[0]
    return build_prediction_output(row, ARTIFACTS)

@app.get("/predict-upcoming")
def predict_upcoming(
    league: str = Query(..., description="League code, e.g. EPL"),
    limit: int = Query(default=10, ge=1, le=50),
) -> List[Dict[str, Any]]:
    outputs = predict_upcoming_fixtures_for_league(league=league, limit=limit)
    return outputs

import os
from fastapi import HTTPException

TELEGRAM_SECRET_KEY = os.getenv("TELEGRAM_SECRET_KEY")


@app.post("/telegram/send-league")
def send_telegram_league_predictions(
    league: str = Query(default="EPL"),
    limit: int = Query(default=5),
    secret: str = Query(...)
):
    if secret != TELEGRAM_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

    result = send_league_predictions_to_telegram(
        league=league,
        limit=limit
    )

    return {
        "status": "sent",
        "league": league,
        "result": result
    }