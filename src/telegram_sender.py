from __future__ import annotations

import html
import os
from datetime import datetime
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

from src.predict import predict_upcoming_fixtures_for_league


load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SUPPORTED_LEAGUES = ["EPL", "LALIGA", "SERIEA", "BUNDESLIGA"]

LEAGUE_DISPLAY_NAMES = {
    "EPL": "Premier League",
    "LALIGA": "LaLiga",
    "SERIEA": "Serie A",
    "BUNDESLIGA": "Bundesliga",
}


def escape_html(value: object) -> str:
    """
    Escape text safely for Telegram HTML parse mode.
    """
    return html.escape(str(value), quote=False)


def format_percentage(value: Optional[float]) -> str:
    """
    Convert decimal confidence to clean percentage.
    """
    if value is None:
        return "N/A"
    return f"{round(float(value) * 100)}%"


def confidence_emoji(confidence_band: Optional[str]) -> str:
    """
    Return emoji based on confidence band.
    """
    if confidence_band == "Strong":
        return "🔥"
    if confidence_band == "Medium":
        return "✅"
    return "⚠️"


def format_match_time(date_value: Optional[str]) -> str:
    """
    Format ISO date string into readable UTC time.
    """
    if not date_value:
        return "Time TBC"

    try:
        dt = datetime.fromisoformat(str(date_value).replace("Z", "+00:00"))
        return dt.strftime("%a, %d %b • %H:%M UTC")
    except Exception:
        return str(date_value)


def format_prediction_line(label: str, prediction: Optional[Dict]) -> str:
    """
    Format one prediction line for Telegram.
    """
    if not prediction:
        return ""

    market = escape_html(prediction.get("market", "Prediction"))
    selection = escape_html(prediction.get("selection", "N/A"))
    confidence = format_percentage(prediction.get("confidence"))
    band = escape_html(prediction.get("confidence_band", "N/A"))
    emoji = confidence_emoji(prediction.get("confidence_band"))

    return (
        f"{emoji} <b>{escape_html(label)}:</b> "
        f"{market} — <b>{selection}</b> "
        f"({confidence}, {band})"
    )


def format_single_match(prediction: Dict, index: int) -> str:
    """
    Format one match prediction block.
    """
    match = prediction.get("match", {})
    primary = prediction.get("primary_prediction")
    secondary = prediction.get("secondary_prediction")
    note = prediction.get("prediction_note")

    home_team = escape_html(match.get("home_team", "Home Team"))
    away_team = escape_html(match.get("away_team", "Away Team"))
    league = escape_html(match.get("league", ""))
    date_text = escape_html(format_match_time(match.get("date")))

    odds = match.get("odds", {}) or {}
    home_odds = odds.get("home_win", "N/A")
    draw_odds = odds.get("draw", "N/A")
    away_odds = odds.get("away_win", "N/A")

    lines = [
        f"<b>{index}. {home_team} vs {away_team}</b>",
        f"🏆 {league} | 🕒 {date_text}",
        f"📊 Odds: H {escape_html(home_odds)} | D {escape_html(draw_odds)} | A {escape_html(away_odds)}",
        format_prediction_line("Best Bet", primary),
        format_prediction_line("Second Pick", secondary),
    ]

    if note:
        lines.append(f"ℹ️ <i>{escape_html(note)}</i>")

    return "\n".join([line for line in lines if line])


def build_telegram_message(
    league: str,
    predictions: List[Dict],
    limit: int,
) -> str:
    """
    Build the full Telegram message for one league.
    """
    league_upper = league.upper()
    league_name = LEAGUE_DISPLAY_NAMES.get(league_upper, league_upper)

    header = [
        "⚽ <b>World Football Predictor</b>",
        f"🔥 <b>{escape_html(league_name)} Top Predictions</b>",
        "",
        f"Showing top {min(limit, len(predictions))} upcoming fixtures.",
        "",
    ]

    body = []

    valid_predictions = [item for item in predictions if "error" not in item]

    for index, prediction in enumerate(valid_predictions[:limit], start=1):
        body.append(format_single_match(prediction, index))
        body.append("")

    footer = [
        "🔗 Join for daily football predictions.",
        "⚠️ <i>18+ only. Gamble responsibly. Predictions are informational only.</i>",
    ]

    return "\n".join(header + body + footer).strip()


def send_telegram_message(message: str) -> Dict:
    """
    Send a message to the configured Telegram channel.
    """
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("Missing TELEGRAM_BOT_TOKEN environment variable.")

    if not TELEGRAM_CHAT_ID:
        raise ValueError("Missing TELEGRAM_CHAT_ID environment variable.")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    response = requests.post(url, json=payload, timeout=30)

    if not response.ok:
        raise RuntimeError(
            f"Telegram send failed: {response.status_code} - {response.text}"
        )

    return response.json()


def send_league_predictions_to_telegram(
    league: str = "EPL",
    limit: int = 5,
) -> Dict:
    """
    Fetch upcoming predictions for one league and send them to Telegram.
    """
    league = league.upper()

    if league not in SUPPORTED_LEAGUES:
        raise ValueError(
            f"Unsupported league '{league}'. Supported leagues: {SUPPORTED_LEAGUES}"
        )

    predictions = predict_upcoming_fixtures_for_league(
        league=league,
        limit=limit,
    )

    message = build_telegram_message(
        league=league,
        predictions=predictions,
        limit=limit,
    )

    return send_telegram_message(message)


def send_all_leagues_predictions_to_telegram(limit_per_league: int = 3) -> List[Dict]:
    """
    Send one Telegram message per supported league.
    """
    results = []

    for league in SUPPORTED_LEAGUES:
        result = send_league_predictions_to_telegram(
            league=league,
            limit=limit_per_league,
        )
        results.append(
            {
                "league": league,
                "telegram_response": result,
            }
        )

    return results


def demo_build_message_only(league: str = "EPL", limit: int = 3) -> None:
    """
    Local demo: build message but do not send.
    """
    predictions = predict_upcoming_fixtures_for_league(
        league=league,
        limit=limit,
    )

    message = build_telegram_message(
        league=league,
        predictions=predictions,
        limit=limit,
    )

    print("=" * 100)
    print("TELEGRAM MESSAGE PREVIEW")
    print("=" * 100)
    print(message)


if __name__ == "__main__":
    demo_build_message_only(league="EPL", limit=3)