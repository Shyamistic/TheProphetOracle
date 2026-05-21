"""Sports Data Fetcher — ESPN Public API integration.

Fetches live scores, odds, and game results from ESPN's free public API.
No API key required. Covers NBA, NFL, MLB, NHL, soccer, cricket, tennis.

This gives us DETERMINISTIC data for sports events instead of relying on
LLM guesses. If we know Team A is up 3-1 in a series, we can assign
much higher probability than an LLM that might not know the current state.

ESPN API format:
  http://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

ESPN_BASE = "http://site.api.espn.com/apis/site/v2/sports"

# Map of sport/league to ESPN path
SPORT_LEAGUES = {
    "nba": ("basketball", "nba"),
    "nfl": ("football", "nfl"),
    "mlb": ("baseball", "mlb"),
    "nhl": ("hockey", "nhl"),
    "soccer_epl": ("soccer", "eng.1"),
    "soccer_laliga": ("soccer", "esp.1"),
    "soccer_seriea": ("soccer", "ita.1"),
    "soccer_ligue1": ("soccer", "fra.1"),
    "soccer_bundesliga": ("soccer", "ger.1"),
    "soccer_mls": ("soccer", "usa.1"),
    "cricket_test": ("cricket", "test"),
    "tennis_atp": ("tennis", "atp"),
    "tennis_wta": ("tennis", "wta"),
}

# Keywords to detect sport type from event title
SPORT_KEYWORDS = {
    "nba": ["nba", "lakers", "celtics", "warriors", "bucks", "nuggets", "thunder", "knicks", "76ers", "heat", "playoff"],
    "nfl": ["nfl", "super bowl", "chiefs", "eagles", "49ers", "cowboys", "ravens"],
    "mlb": ["mlb", "yankees", "dodgers", "braves", "astros", "world series"],
    "nhl": ["nhl", "stanley cup", "calder", "oilers", "panthers", "rangers"],
    "soccer_epl": ["premier league", "epl", "arsenal", "manchester", "liverpool", "chelsea"],
    "soccer_laliga": ["la liga", "barcelona", "real madrid", "atletico"],
    "soccer_seriea": ["serie a", "inter", "juventus", "napoli", "milan"],
    "soccer_ligue1": ["ligue 1", "psg", "marseille", "lyon"],
    "cricket_test": ["cricket", "test match", "ashes"],
    "tennis_atp": ["atp", "tennis"],
    "tennis_wta": ["wta"],
}


def detect_sport(title: str, context: str = "") -> Optional[str]:
    """Detect which sport/league an event is about from its title.
    
    Returns the sport key (e.g., 'nba', 'soccer_epl') or None.
    """
    combined = (title + " " + context).lower()
    
    for sport_key, keywords in SPORT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in combined:
                return sport_key
    return None


async def fetch_espn_scoreboard(sport_key: str) -> Optional[List[Dict]]:
    """Fetch current scoreboard from ESPN for a given sport.
    
    Returns list of event dicts with teams, scores, status, odds.
    """
    if sport_key not in SPORT_LEAGUES:
        return None
    
    sport, league = SPORT_LEAGUES[sport_key]
    url = f"{ESPN_BASE}/{sport}/{league}/scoreboard"
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            if response.status_code != 200:
                logger.debug(f"ESPN API returned {response.status_code} for {sport_key}")
                return None
            
            data = response.json()
            events = data.get("events", [])
            
            parsed = []
            for event in events:
                parsed_event = _parse_espn_event(event)
                if parsed_event:
                    parsed.append(parsed_event)
            
            return parsed if parsed else None
            
    except Exception as e:
        logger.debug(f"ESPN fetch failed for {sport_key}: {e}")
        return None


def _parse_espn_event(event: Dict) -> Optional[Dict]:
    """Parse a single ESPN event into a simplified format."""
    try:
        name = event.get("name", "")
        status_obj = event.get("status", {})
        status_type = status_obj.get("type", {}).get("name", "")  # STATUS_SCHEDULED, STATUS_IN_PROGRESS, STATUS_FINAL
        
        competitions = event.get("competitions", [])
        if not competitions:
            return None
        
        comp = competitions[0]
        competitors = comp.get("competitors", [])
        
        teams = []
        for c in competitors:
            team_info = {
                "name": c.get("team", {}).get("displayName", c.get("team", {}).get("name", "")),
                "abbreviation": c.get("team", {}).get("abbreviation", ""),
                "score": c.get("score", "0"),
                "winner": c.get("winner", False),
                "home_away": c.get("homeAway", ""),
            }
            teams.append(team_info)
        
        # Extract odds if available
        odds = None
        odds_data = comp.get("odds", [])
        if odds_data:
            odds = {
                "spread": odds_data[0].get("details", ""),
                "overUnder": odds_data[0].get("overUnder", None),
            }
            # Try to get moneyline/win probability
            away_ml = odds_data[0].get("awayTeamOdds", {})
            home_ml = odds_data[0].get("homeTeamOdds", {})
            if away_ml.get("winPercentage"):
                odds["away_win_pct"] = away_ml["winPercentage"]
            if home_ml.get("winPercentage"):
                odds["home_win_pct"] = home_ml["winPercentage"]
        
        # Series info (for playoffs)
        series = None
        series_data = comp.get("series", {})
        if series_data:
            series = {
                "summary": series_data.get("summary", ""),
                "completed": series_data.get("completed", False),
            }
        
        return {
            "name": name,
            "status": status_type,
            "teams": teams,
            "odds": odds,
            "series": series,
            "date": event.get("date", ""),
        }
    except Exception:
        return None


async def get_sports_context(title: str, context: str = "") -> Optional[str]:
    """Get relevant sports data for an event.
    
    Detects the sport, fetches ESPN data, and returns a formatted
    context string that can be injected into the research evidence.
    
    Returns None if no relevant sports data found.
    """
    sport_key = detect_sport(title, context)
    if not sport_key:
        return None
    
    events = await fetch_espn_scoreboard(sport_key)
    if not events:
        return None
    
    # Try to find a matching event
    title_lower = title.lower()
    
    # Extract team names from our event title
    best_match = None
    best_score = 0
    
    for espn_event in events:
        event_name_lower = espn_event["name"].lower()
        team_names = [t["name"].lower() for t in espn_event["teams"]]
        team_abbrevs = [t["abbreviation"].lower() for t in espn_event["teams"]]
        
        # Score based on word overlap
        score = 0
        for team in team_names + team_abbrevs:
            if team and team in title_lower:
                score += 2
        
        # Also check event name overlap
        title_words = set(title_lower.split())
        event_words = set(event_name_lower.split())
        word_overlap = len(title_words & event_words)
        score += word_overlap
        
        if score > best_score:
            best_score = score
            best_match = espn_event
    
    if not best_match or best_score < 2:
        # No good match — return general scoreboard summary
        return _format_scoreboard_summary(events, sport_key)
    
    return _format_matched_event(best_match)


def _format_matched_event(event: Dict) -> str:
    """Format a matched ESPN event into context string."""
    parts = [f"[ESPN LIVE DATA] {event['name']}"]
    parts.append(f"Status: {event['status']}")
    
    for team in event["teams"]:
        marker = " ★" if team["winner"] else ""
        parts.append(f"  {team['name']}: {team['score']}{marker}")
    
    if event.get("odds"):
        odds = event["odds"]
        if odds.get("home_win_pct"):
            parts.append(f"Win probability: Home {odds['home_win_pct']:.1%}, Away {odds.get('away_win_pct', 0):.1%}")
        if odds.get("spread"):
            parts.append(f"Spread: {odds['spread']}")
    
    if event.get("series"):
        parts.append(f"Series: {event['series']['summary']}")
    
    return " | ".join(parts)


def _format_scoreboard_summary(events: List[Dict], sport_key: str) -> str:
    """Format a general scoreboard summary."""
    if not events:
        return None
    
    parts = [f"[ESPN {sport_key.upper()} SCOREBOARD]"]
    for event in events[:5]:  # Max 5 events
        status = event["status"]
        teams_str = " vs ".join(t["name"] for t in event["teams"])
        scores_str = "-".join(t["score"] for t in event["teams"])
        parts.append(f"{teams_str}: {scores_str} ({status})")
    
    return " | ".join(parts)
