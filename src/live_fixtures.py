from __future__ import annotations

import os
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv


load_dotenv()

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4/sports"

LEAGUE_ENDPOINTS = {
    "EPL": "soccer_epl/odds",
    "LALIGA": "soccer_spain_la_liga/odds",
    "SERIEA": "soccer_italy_serie_a/odds",
    "BUNDESLIGA": "soccer_germany_bundesliga/odds",
}


def normalize_team_name(team_name: str) -> str:
    """
    Basic team-name normalization.
    """
    return " ".join(str(team_name).strip().split())


def get_league_endpoint(league: str) -> str:
    """
    Resolve league code to Odds API endpoint path.
    """
    league_upper = league.upper()
    if league_upper not in LEAGUE_ENDPOINTS:
        raise ValueError(
            f"Unsupported league '{league}'. Supported leagues: {list(LEAGUE_ENDPOINTS.keys())}"
        )
    return LEAGUE_ENDPOINTS[league_upper]


def extract_h2h_odds(match: Dict) -> Dict[str, Optional[float]]:
    """
    Extract Home / Draw / Away odds from the first available bookmaker market.
    """
    home_team = match.get("home_team")
    away_team = match.get("away_team")

    default_output = {
        "odds_home_win": None,
        "odds_draw": None,
        "odds_away_win": None,
    }

    bookmakers = match.get("bookmakers", [])
    if not bookmakers:
        return default_output

    first_bookmaker = bookmakers[0]
    markets = first_bookmaker.get("markets", [])
    if not markets:
        return default_output

    h2h_market = None
    for market in markets:
        if market.get("key") == "h2h":
            h2h_market = market
            break

    if h2h_market is None:
        return default_output

    outcomes = h2h_market.get("outcomes", [])
    odds_map = {}
    for outcome in outcomes:
        name = outcome.get("name")
        price = outcome.get("price")
        odds_map[name] = price

    return {
        "odds_home_win": odds_map.get(home_team),
        "odds_draw": odds_map.get("Draw"),
        "odds_away_win": odds_map.get(away_team),
    }


def fetch_upcoming_fixtures(league: str) -> List[Dict]:
    """
    Fetch upcoming fixtures for one league from The Odds API.
    """
    if not ODDS_API_KEY:
        raise ValueError(
            "Missing ODDS_API_KEY. Add it to your environment or .env file."
        )

    endpoint = get_league_endpoint(league)
    url = f"{ODDS_API_BASE_URL}/{endpoint}"

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h",
        "dateFormat": "iso",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    fixtures = []
    for match in data:
        odds = extract_h2h_odds(match)

        fixture = {
            "date": match.get("commence_time"),
            "league": league.upper(),
            "home_team": normalize_team_name(match.get("home_team")),
            "away_team": normalize_team_name(match.get("away_team")),
            "odds_home_win": odds["odds_home_win"],
            "odds_draw": odds["odds_draw"],
            "odds_away_win": odds["odds_away_win"],
        }

        fixtures.append(fixture)

    return fixtures


def filter_fixtures_with_complete_odds(fixtures: List[Dict]) -> List[Dict]:
    """
    Keep only fixtures where Home/Draw/Away odds are all present.
    """
    cleaned = []
    for fixture in fixtures:
        if (
            fixture["odds_home_win"] is not None
            and fixture["odds_draw"] is not None
            and fixture["odds_away_win"] is not None
        ):
            cleaned.append(fixture)
    return cleaned


def demo_fetch_upcoming(league: str = "EPL", limit: int = 5) -> None:
    """
    Demo fetch for one league.
    """
    fixtures = fetch_upcoming_fixtures(league)
    fixtures = filter_fixtures_with_complete_odds(fixtures)

    print("=" * 100)
    print(f"UPCOMING FIXTURES DEMO - {league.upper()}")
    print("=" * 100)
    print(f"Fetched fixtures: {len(fixtures)}")
    print()

    for fixture in fixtures[:limit]:
        print(
            f"{fixture['date']} | {fixture['league']} | "
            f"{fixture['home_team']} vs {fixture['away_team']} | "
            f"Home={fixture['odds_home_win']}, Draw={fixture['odds_draw']}, Away={fixture['odds_away_win']}"
        )


if __name__ == "__main__":
    demo_fetch_upcoming("EPL", limit=5)