"""
Prophet Forecasting Agent - STRESS TEST
========================================
Simulates the HARDEST events the agent will face during the 14-day
evaluation period (May 17 - May 31, 2026).

Categories tested:
- Sports (NBA playoffs, tennis, football, cricket)
- Entertainment (TV show outcomes, awards, Netflix)
- Economics (Fed decisions, CPI, oil prices, crypto)
- Politics (elections, primaries, policy decisions)
- Multi-outcome (10-30 outcomes like "who wins X award")

Each event is designed to be GENUINELY HARD:
- Obscure events with scarce information
- Multi-outcome events with 5-20+ options
- Events closing in 2 days vs 2 weeks
- Requires real research and reasoning

Usage:
    venv\\Scripts\\python.exe backtest\\run_stress_test.py
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Fix Windows encoding issues
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import httpx

SERVER_URL = "http://127.0.0.1:8888"
PREDICT_ENDPOINT = f"{SERVER_URL}/predict"
TIMEOUT_SECONDS = 300  # 5 minutes per event (stress test allows more time)
DELAY_BETWEEN_EVENTS = 5  # seconds between events to avoid overload
RESULTS_DIR = Path(__file__).parent / "results"


# ═══════════════════════════════════════════════════════════════════════
# STRESS TEST EVENTS — 20 genuinely hard events across 5 categories
# ═══════════════════════════════════════════════════════════════════════


def get_stress_test_events() -> List[Dict[str, Any]]:
    """
    20 events designed to stress-test the agent on the hardest scenarios
    it will face during the May 17-31, 2026 evaluation window.

    Each event uses the EXACT Prophet Arena format:
    - task_id, title, outcomes, predict_by, source, context
    - metadata with category info
    """
    events = []

    # ─────────────────────────────────────────────────────────────────
    # CATEGORY 1: SPORTS (4 events)
    # Hard because: live results, obscure matches, playoff uncertainty
    # ─────────────────────────────────────────────────────────────────

    # 1. NBA Playoffs — Conference Finals (multi-outcome: who wins series)
    events.append({
        "task_id": "STRESS-NBA-ECF-2026",
        "title": "Which team will win the 2026 NBA Eastern Conference Finals?",
        "outcomes": [
            "Boston Celtics", "Cleveland Cavaliers", "New York Knicks",
            "Milwaukee Bucks", "Indiana Pacers", "Orlando Magic"
        ],
        "predict_by": "2026-05-28T03:59:00+00:00",
        "source": "STRESS-NBA-ECF-2026",
        "context": "The 2026 NBA Eastern Conference Finals are underway. Which team will advance to the NBA Finals? Resolves to the team that wins the series.",
        "metadata": {"category": "Sports", "difficulty": "hard"}
    })

    # 2. Tennis — Obscure ATP 250 match (scarce info)
    events.append({
        "task_id": "STRESS-ATP-LYON-QF-2026",
        "title": "Who will win the Lyon Open 2026 Men's Singles quarterfinal: Fils vs Cerundolo?",
        "outcomes": ["Arthur Fils", "Francisco Cerundolo"],
        "predict_by": "2026-05-22T14:00:00+00:00",
        "source": "STRESS-ATP-LYON-QF-2026",
        "context": "ATP 250 Lyon Open 2026 quarterfinal match between Arthur Fils (FRA) and Francisco Cerundolo (ARG) on clay court. Resolves to the winner of the match.",
        "metadata": {"category": "Sports", "difficulty": "very_hard"}
    })

    # 3. Cricket — IPL Playoff match (niche for Western models)
    events.append({
        "task_id": "STRESS-IPL-QUALIFIER-2026",
        "title": "Which team will win IPL 2026 Qualifier 1?",
        "outcomes": [
            "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bengaluru",
            "Kolkata Knight Riders", "Gujarat Titans"
        ],
        "predict_by": "2026-05-20T18:00:00+00:00",
        "source": "STRESS-IPL-QUALIFIER-2026",
        "context": "IPL 2026 Qualifier 1 between the top two teams from the league stage. The winner advances directly to the final. Resolves to the winning team.",
        "metadata": {"category": "Sports", "difficulty": "hard"}
    })

    # 4. Football — UEFA Champions League Final scorer (multi-outcome)
    events.append({
        "task_id": "STRESS-UCL-FINAL-SCORER-2026",
        "title": "Who will score the first goal in the 2026 UEFA Champions League Final?",
        "outcomes": [
            "Erling Haaland", "Kylian Mbappe", "Vinicius Jr", "Jude Bellingham",
            "Rodrygo", "Phil Foden", "Kevin De Bruyne", "Bukayo Saka",
            "Own Goal", "No goals scored in regulation", "Other player"
        ],
        "predict_by": "2026-05-31T19:00:00+00:00",
        "source": "STRESS-UCL-FINAL-SCORER-2026",
        "context": "The 2026 UEFA Champions League Final is scheduled for May 31, 2026. Which player will score the first goal? Resolves to the player credited with the first goal.",
        "metadata": {"category": "Sports", "difficulty": "very_hard"}
    })


    # ─────────────────────────────────────────────────────────────────
    # CATEGORY 2: ENTERTAINMENT (4 events)
    # Hard because: subjective outcomes, many candidates, niche awards
    # ─────────────────────────────────────────────────────────────────

    # 5. Crunchyroll Anime Awards — Best Anime of the Year (many outcomes)
    events.append({
        "task_id": "STRESS-ANIME-AOTY-2026",
        "title": "Which show will win Anime of the Year at the 2026 Crunchyroll Anime Awards?",
        "outcomes": [
            "DAN DA DAN Season 2", "Solo Leveling Season 2 -Arise from the Shadow-",
            "Demon Slayer: Kimetsu no Yaiba Infinity Castle",
            "Chainsaw Man - The Movie: Reze Arc", "ONE PIECE",
            "My Hero Academia FINAL SEASON", "The Apothecary Diaries",
            "Gachiakuta", "Kaiju No. 8 Season 2", "Blue Box",
            "SAKAMOTO DAYS", "Takopi's Original Sin", "Tie"
        ],
        "predict_by": "2026-05-24T03:59:00+00:00",
        "source": "STRESS-ANIME-AOTY-2026",
        "context": "If DAN DA DAN Season 2 has won Anime of the Year at the 2026 Crunchyroll Anime Awards, then the market resolves to Yes.",
        "metadata": {"category": "Entertainment", "difficulty": "hard"}
    })

    # 6. Netflix #1 show viewership (requires current data)
    events.append({
        "task_id": "STRESS-NETFLIX-VIEWS-MAY25",
        "title": "How many views will the #1 show on Netflix have on the chart published May 26, 2026?",
        "outcomes": [
            "At least 6 million", "At least 9 million", "At least 12 million",
            "At least 15 million", "At least 18 million", "At least 21 million",
            "At least 25 million", "At least 30 million", "At least 35 million",
            "At least 40 million", "At least 50 million"
        ],
        "predict_by": "2026-05-26T03:59:00+00:00",
        "source": "STRESS-NETFLIX-VIEWS-MAY25",
        "context": "If the #1 Show on Netflix has at least 6 million views on the chart published on May 26, 2026, then the market resolves to Yes.",
        "metadata": {"category": "Entertainment", "difficulty": "hard"}
    })

    # 7. Emmy Awards prediction — Outstanding Drama (far out, many nominees)
    events.append({
        "task_id": "STRESS-EMMY-DRAMA-2026",
        "title": "Which show will win Outstanding Drama Series at the 78th Emmy Awards (Sep 2026)?",
        "outcomes": [
            "The White Lotus", "Severance", "The Last of Us",
            "Squid Game", "Shogun", "The Diplomat",
            "Slow Horses", "The Bear", "Yellowjackets",
            "House of the Dragon", "Tie"
        ],
        "predict_by": "2026-05-30T03:59:00+00:00",
        "source": "STRESS-EMMY-DRAMA-2026",
        "context": "Which show will win Outstanding Drama Series at the 78th Primetime Emmy Awards? Resolves to the winner announced at the ceremony in September 2026.",
        "metadata": {"category": "Entertainment", "difficulty": "very_hard"}
    })

    # 8. Billboard Hot 100 #1 — who will be #1 next week
    events.append({
        "task_id": "STRESS-BILLBOARD-HOT100-MAY26",
        "title": "Which artist will have the #1 song on the Billboard Hot 100 chart dated May 31, 2026?",
        "outcomes": [
            "Taylor Swift", "Drake", "Kendrick Lamar", "Sabrina Carpenter",
            "Billie Eilish", "Post Malone", "The Weeknd", "Doja Cat",
            "Travis Scott", "Chappell Roan", "Other artist"
        ],
        "predict_by": "2026-05-27T03:59:00+00:00",
        "source": "STRESS-BILLBOARD-HOT100-MAY26",
        "context": "Which artist will hold the #1 position on the Billboard Hot 100 chart dated May 31, 2026? Resolves to the artist credited on the #1 song.",
        "metadata": {"category": "Entertainment", "difficulty": "very_hard"}
    })


    # ─────────────────────────────────────────────────────────────────
    # CATEGORY 3: ECONOMICS (4 events)
    # Hard because: precise numeric outcomes, many thresholds, macro uncertainty
    # ─────────────────────────────────────────────────────────────────

    # 9. Fed Funds Rate after June FOMC (many threshold outcomes)
    events.append({
        "task_id": "STRESS-FED-JUNE-2026",
        "title": "Where will the upper bound of the US federal funds target rate sit after the June 2026 FOMC meeting?",
        "outcomes": [
            "Above 3.00%", "Above 3.25%", "Above 3.50%", "Above 3.75%",
            "Above 4.00%", "Above 4.25%", "Above 4.50%", "Above 4.75%",
            "Above 5.00%", "Above 5.25%"
        ],
        "predict_by": "2026-06-18T17:55:00+00:00",
        "source": "STRESS-FED-JUNE-2026",
        "context": "If the upper bound of the target federal funds rate published on the Federal Reserve's official website is greater than 3.00% following the Federal Reserve's Jun 18, 2026 meeting, then the market resolves to Yes.",
        "metadata": {"category": "Economics", "difficulty": "hard"}
    })

    # 10. S&P 500 precise level (many thresholds, closing soon)
    events.append({
        "task_id": "STRESS-SP500-MAY30-2026",
        "title": "Where will the S&P 500 close on May 30, 2026?",
        "outcomes": [
            "Above 5000", "Above 5100", "Above 5200", "Above 5300",
            "Above 5400", "Above 5500", "Above 5600", "Above 5700",
            "Above 5800", "Above 5900", "Above 6000", "Above 6100"
        ],
        "predict_by": "2026-05-30T19:59:00+00:00",
        "source": "STRESS-SP500-MAY30-2026",
        "context": "If the S&P 500 index closes above 5000 on May 30, 2026, then the market resolves to Yes. Based on the official closing price.",
        "metadata": {"category": "Economics", "difficulty": "hard"}
    })

    # 11. Bitcoin price range (crypto volatility, many outcomes)
    events.append({
        "task_id": "STRESS-BTC-MAY31-2026",
        "title": "What range will Bitcoin's price be in at midnight UTC on May 31, 2026?",
        "outcomes": [
            "Below $60,000", "$60,000-$70,000", "$70,000-$80,000",
            "$80,000-$90,000", "$90,000-$100,000", "$100,000-$110,000",
            "$110,000-$120,000", "$120,000-$130,000", "Above $130,000"
        ],
        "predict_by": "2026-05-31T00:00:00+00:00",
        "source": "STRESS-BTC-MAY31-2026",
        "context": "What will Bitcoin's spot price be at midnight UTC on May 31, 2026? Resolves to the price range containing the CoinGecko reported BTC/USD price.",
        "metadata": {"category": "Economics", "difficulty": "hard"}
    })

    # 12. WTI Crude Oil — precise level with many thresholds
    events.append({
        "task_id": "STRESS-OIL-WTI-MAY30-2026",
        "title": "Where will WTI crude oil settle on May 30, 2026?",
        "outcomes": [
            "Above $55", "Above $58", "Above $61", "Above $64",
            "Above $67", "Above $70", "Above $73", "Above $76",
            "Above $79", "Above $82", "Above $85"
        ],
        "predict_by": "2026-05-30T20:30:00+00:00",
        "source": "STRESS-OIL-WTI-MAY30-2026",
        "context": "If WTI crude oil front-month futures settle above $55 per barrel on May 30, 2026, then the market resolves to Yes. Based on NYMEX settlement price.",
        "metadata": {"category": "Economics", "difficulty": "hard"}
    })


    # ─────────────────────────────────────────────────────────────────
    # CATEGORY 4: POLITICS (4 events)
    # Hard because: policy uncertainty, local elections, geopolitics
    # ─────────────────────────────────────────────────────────────────

    # 13. US Congressional special election (obscure, local)
    events.append({
        "task_id": "STRESS-US-SPECIAL-ELECTION-2026",
        "title": "Which party will win the US House special election in Florida's 6th Congressional District (May 2026)?",
        "outcomes": ["Republican", "Democrat", "Other/Independent"],
        "predict_by": "2026-05-20T23:59:00+00:00",
        "source": "STRESS-US-SPECIAL-ELECTION-2026",
        "context": "A special election is being held in Florida's 6th Congressional District following the resignation of the incumbent. Resolves to the party of the winning candidate.",
        "metadata": {"category": "Politics", "difficulty": "hard"}
    })

    # 14. UK local elections — council control (very niche)
    events.append({
        "task_id": "STRESS-UK-LOCAL-COUNCIL-2026",
        "title": "Will Labour retain control of Birmingham City Council after the May 2026 local elections?",
        "outcomes": ["Yes", "No"],
        "predict_by": "2026-05-22T22:00:00+00:00",
        "source": "STRESS-UK-LOCAL-COUNCIL-2026",
        "context": "UK local elections are held in May 2026. Will the Labour Party retain majority control of Birmingham City Council? Resolves Yes if Labour holds a majority of seats after all results are declared.",
        "metadata": {"category": "Politics", "difficulty": "very_hard"}
    })

    # 15. Trump executive order — policy prediction
    events.append({
        "task_id": "STRESS-TRUMP-EO-TARIFF-MAY2026",
        "title": "Will President Trump sign an executive order imposing new tariffs on EU goods before June 1, 2026?",
        "outcomes": ["Yes", "No"],
        "predict_by": "2026-06-01T04:59:00+00:00",
        "source": "STRESS-TRUMP-EO-TARIFF-MAY2026",
        "context": "Will President Trump sign an executive order specifically imposing new tariffs on European Union goods before June 1, 2026? Resolves Yes if such an EO is signed and published in the Federal Register.",
        "metadata": {"category": "Politics", "difficulty": "hard"}
    })

    # 16. Canadian provincial election — multi-outcome
    events.append({
        "task_id": "STRESS-CANADA-ONTARIO-PREMIER-2026",
        "title": "Who will be Premier of Ontario after the next provincial election?",
        "outcomes": [
            "Doug Ford (PC)", "Marit Stiles (NDP)", "Bonnie Crombie (Liberal)",
            "Mike Schreiner (Green)", "Other candidate"
        ],
        "predict_by": "2026-05-29T03:59:00+00:00",
        "source": "STRESS-CANADA-ONTARIO-PREMIER-2026",
        "context": "Ontario's next provincial election determines the Premier. Resolves to the leader of the party that wins enough seats to form government.",
        "metadata": {"category": "Politics", "difficulty": "hard"}
    })


    # ─────────────────────────────────────────────────────────────────
    # CATEGORY 5: MULTI-OUTCOME (4 events with 5-35 outcomes)
    # Hard because: many outcomes, uniform prior, requires deep research
    # ─────────────────────────────────────────────────────────────────

    # 17. Eurovision 2026 Winner (35 countries!)
    events.append({
        "task_id": "STRESS-EUROVISION-WINNER-2026",
        "title": "Which country will win the Eurovision Song Contest 2026 Grand Final?",
        "outcomes": [
            "Albania", "Armenia", "Australia", "Austria", "Azerbaijan",
            "Belgium", "Croatia", "Cyprus", "Czechia", "Denmark",
            "Estonia", "Finland", "France", "Georgia", "Germany",
            "Greece", "Israel", "Italy", "Latvia", "Lithuania",
            "Luxembourg", "Malta", "Moldova", "Montenegro", "Norway",
            "Poland", "Portugal", "Romania", "San Marino", "Serbia",
            "Sweden", "Switzerland", "Ukraine", "United Kingdom", "Spain"
        ],
        "predict_by": "2026-05-17T19:00:00+00:00",
        "source": "STRESS-EUROVISION-WINNER-2026",
        "context": "Which country will win the Eurovision Song Contest 2026 Grand Final? Resolves to the country whose act receives the most combined jury and televote points.",
        "metadata": {"category": "Entertainment", "difficulty": "extreme"}
    })

    # 18. French Open Men's Singles Winner (16 possible winners)
    events.append({
        "task_id": "STRESS-FRENCH-OPEN-WINNER-2026",
        "title": "Who will win the 2026 French Open Men's Singles title?",
        "outcomes": [
            "Jannik Sinner", "Carlos Alcaraz", "Novak Djokovic",
            "Alexander Zverev", "Daniil Medvedev", "Casper Ruud",
            "Stefanos Tsitsipas", "Andrey Rublev", "Holger Rune",
            "Taylor Fritz", "Alex de Minaur", "Tommy Paul",
            "Frances Tiafoe", "Ben Shelton", "Grigor Dimitrov",
            "Other player"
        ],
        "predict_by": "2026-06-08T15:00:00+00:00",
        "source": "STRESS-FRENCH-OPEN-WINNER-2026",
        "context": "Who will win the 2026 Roland Garros Men's Singles championship? Resolves to the player who wins the final match.",
        "metadata": {"category": "Sports", "difficulty": "hard"}
    })

    # 19. US CPI exact reading (16 precise outcomes)
    events.append({
        "task_id": "STRESS-CPI-YOY-MAY2026",
        "title": "What will US CPI year-over-year be for April 2026 (released May 13, 2026)?",
        "outcomes": [
            "Exactly 2.0%", "Exactly 2.1%", "Exactly 2.2%", "Exactly 2.3%",
            "Exactly 2.4%", "Exactly 2.5%", "Exactly 2.6%", "Exactly 2.7%",
            "Exactly 2.8%", "Exactly 2.9%", "Exactly 3.0%", "Exactly 3.1%",
            "Exactly 3.2%", "Exactly 3.3%", "Exactly 3.4%", "Exactly 3.5%"
        ],
        "predict_by": "2026-05-13T12:29:00+00:00",
        "source": "STRESS-CPI-YOY-MAY2026",
        "context": "If the CPI year-over-year is exactly 2.0% in April 2026, then the market resolves to Yes. Based on the Bureau of Labor Statistics release.",
        "metadata": {"category": "Economics", "difficulty": "hard"}
    })

    # 20. NBA MVP Award (multi-outcome, 10+ candidates)
    events.append({
        "task_id": "STRESS-NBA-MVP-2026",
        "title": "Who will win the 2025-26 NBA Most Valuable Player award?",
        "outcomes": [
            "Nikola Jokic", "Shai Gilgeous-Alexander", "Luka Doncic",
            "Giannis Antetokounmpo", "Jayson Tatum", "Anthony Edwards",
            "Victor Wembanyama", "Kevin Durant", "LeBron James",
            "Donovan Mitchell", "Jalen Brunson", "Other player"
        ],
        "predict_by": "2026-05-20T03:59:00+00:00",
        "source": "STRESS-NBA-MVP-2026",
        "context": "Who will be named the 2025-26 NBA Most Valuable Player? Resolves to the player announced as MVP by the NBA.",
        "metadata": {"category": "Sports", "difficulty": "hard"}
    })

    return events


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════


def compute_decisiveness(probabilities: Dict[str, float]) -> float:
    """
    Compute how decisive a prediction is (how far from uniform distribution).

    Returns a score from 0.0 (perfectly uniform/indecisive) to 1.0 (all mass on one outcome).

    Uses KL-divergence-inspired metric:
    decisiveness = 1 - (entropy / max_entropy)
    """
    import math

    n = len(probabilities)
    if n <= 1:
        return 1.0

    values = list(probabilities.values())
    total = sum(values)
    if total == 0:
        return 0.0

    # Normalize
    probs = [v / total for v in values]

    # Compute entropy
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)

    # Max entropy for n outcomes
    max_entropy = math.log2(n)

    if max_entropy == 0:
        return 1.0

    decisiveness = 1.0 - (entropy / max_entropy)
    return max(0.0, min(1.0, decisiveness))


def get_category(event: Dict) -> str:
    """Extract category from event metadata."""
    if "metadata" in event and isinstance(event["metadata"], dict):
        return event["metadata"].get("category", "Unknown")
    return "Unknown"


def analyze_prediction(event: Dict, response_data: Dict, duration: float) -> Dict[str, Any]:
    """Analyze a single prediction for quality metrics."""
    probs = {}
    if "probabilities" in response_data:
        for entry in response_data["probabilities"]:
            probs[entry["market"]] = entry["probability"]

    n_outcomes = len(event["outcomes"])
    n_returned = len(probs)

    # Check if all outcomes are covered
    outcomes_covered = all(
        outcome in probs for outcome in event["outcomes"]
    )

    # Decisiveness
    decisiveness = compute_decisiveness(probs) if probs else 0.0

    # Top prediction
    if probs:
        max_outcome = max(probs, key=probs.get)
        max_prob = probs[max_outcome]
    else:
        max_outcome = "N/A"
        max_prob = 0.0

    # Check for uniform fallback (all probs roughly equal)
    is_uniform_fallback = False
    if probs and n_returned > 1:
        uniform_val = 1.0 / n_returned
        is_uniform_fallback = all(
            abs(p - uniform_val) < 0.02 for p in probs.values()
        )

    # Check probabilities sum to ~1.0
    prob_sum = sum(probs.values()) if probs else 0.0
    valid_sum = abs(prob_sum - 1.0) < 0.05

    return {
        "task_id": event["task_id"],
        "title": event["title"],
        "category": get_category(event),
        "n_outcomes": n_outcomes,
        "n_returned": n_returned,
        "outcomes_covered": outcomes_covered,
        "probabilities": probs,
        "top_outcome": max_outcome,
        "top_probability": max_prob,
        "decisiveness": decisiveness,
        "is_uniform_fallback": is_uniform_fallback,
        "prob_sum": prob_sum,
        "valid_sum": valid_sum,
        "duration_seconds": duration,
        "error": None,
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════


async def run_stress_test():
    """Run the full stress test against the prediction server."""
    print("=" * 75)
    print("  PROPHET FORECASTING AGENT - STRESS TEST")
    print("  Simulating the HARDEST events for May 17-31, 2026 evaluation")
    print("=" * 75)
    print()
    print("  Categories: Sports | Entertainment | Economics | Politics | Multi-Outcome")
    print("  Events: 20 total (4 per category)")
    print("  Timeout: 300s per event")
    print()

    events = get_stress_test_events()
    print(f"  Loaded {len(events)} stress test events")
    print()

    # ── Check server health ──
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{SERVER_URL}/health")
            if r.status_code != 200:
                print(f"  [FAIL] Server unhealthy: HTTP {r.status_code}")
                sys.exit(1)
            print(f"  [OK] Server is healthy at {SERVER_URL}")
    except Exception as e:
        print(f"  [FAIL] Cannot reach server at {SERVER_URL}: {e}")
        print(f"  Start with: venv\\Scripts\\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8888")
        sys.exit(1)

    print()
    print("-" * 75)

    # ── Run predictions ──
    results = []
    total_start = time.time()

    async with httpx.AsyncClient() as client:
        for i, event in enumerate(events, 1):
            category = get_category(event)
            n_outcomes = len(event["outcomes"])
            title_short = event["title"][:55]

            print(f"\n  [{i:02d}/20] [{category:<13}] ({n_outcomes:2d} outcomes) {title_short}...")

            event_start = time.time()

            try:
                r = await client.post(
                    PREDICT_ENDPOINT,
                    json=event,
                    timeout=TIMEOUT_SECONDS
                )
                duration = time.time() - event_start

                if r.status_code == 200:
                    data = r.json()
                    analysis = analyze_prediction(event, data, duration)
                    results.append(analysis)

                    # Print summary
                    dec_bar = "#" * int(analysis["decisiveness"] * 20)
                    dec_empty = "." * (20 - int(analysis["decisiveness"] * 20))
                    print(f"          -> {analysis['top_outcome']}: {analysis['top_probability']:.1%}")
                    print(f"          -> Decisiveness: [{dec_bar}{dec_empty}] {analysis['decisiveness']:.2f}")
                    print(f"          -> Time: {duration:.1f}s | Sum: {analysis['prob_sum']:.3f} | Covered: {analysis['outcomes_covered']}")

                    if analysis["is_uniform_fallback"]:
                        print(f"          -> WARNING: Uniform fallback detected!")

                else:
                    duration = time.time() - event_start
                    error_msg = f"HTTP {r.status_code}"
                    try:
                        error_body = r.json()
                        error_msg += f" - {error_body.get('error', r.text[:100])}"
                    except Exception:
                        error_msg += f" - {r.text[:100]}"

                    print(f"          -> FAILED: {error_msg}")
                    results.append({
                        "task_id": event["task_id"],
                        "title": event["title"],
                        "category": category,
                        "n_outcomes": n_outcomes,
                        "n_returned": 0,
                        "outcomes_covered": False,
                        "probabilities": {},
                        "top_outcome": "N/A",
                        "top_probability": 0.0,
                        "decisiveness": 0.0,
                        "is_uniform_fallback": False,
                        "prob_sum": 0.0,
                        "valid_sum": False,
                        "duration_seconds": duration,
                        "error": error_msg,
                    })

            except httpx.TimeoutException:
                duration = time.time() - event_start
                print(f"          -> TIMEOUT after {duration:.0f}s")
                results.append({
                    "task_id": event["task_id"],
                    "title": event["title"],
                    "category": category,
                    "n_outcomes": n_outcomes,
                    "n_returned": 0,
                    "outcomes_covered": False,
                    "probabilities": {},
                    "top_outcome": "N/A",
                    "top_probability": 0.0,
                    "decisiveness": 0.0,
                    "is_uniform_fallback": False,
                    "prob_sum": 0.0,
                    "valid_sum": False,
                    "duration_seconds": duration,
                    "error": f"Timeout after {TIMEOUT_SECONDS}s",
                })

            except Exception as e:
                duration = time.time() - event_start
                print(f"          -> ERROR: {e}")
                results.append({
                    "task_id": event["task_id"],
                    "title": event["title"],
                    "category": category,
                    "n_outcomes": n_outcomes,
                    "n_returned": 0,
                    "outcomes_covered": False,
                    "probabilities": {},
                    "top_outcome": "N/A",
                    "top_probability": 0.0,
                    "decisiveness": 0.0,
                    "is_uniform_fallback": False,
                    "prob_sum": 0.0,
                    "valid_sum": False,
                    "duration_seconds": duration,
                    "error": str(e),
                })

            # Delay between events
            if i < len(events):
                await asyncio.sleep(DELAY_BETWEEN_EVENTS)

    total_duration = time.time() - total_start


    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY REPORT
    # ═══════════════════════════════════════════════════════════════════

    print("\n")
    print("=" * 75)
    print("  STRESS TEST RESULTS SUMMARY")
    print("=" * 75)

    successful = [r for r in results if r["error"] is None]
    failed = [r for r in results if r["error"] is not None]

    print(f"\n  Overall: {len(successful)}/{len(results)} events predicted successfully")
    print(f"  Total time: {total_duration:.0f}s ({total_duration/60:.1f} min)")
    print(f"  Avg time per event: {total_duration/len(results):.1f}s")

    # ── Success rate by category ──
    print(f"\n  {'Category':<15} {'Success':<10} {'Avg Time':<10} {'Avg Decisiveness':<18} {'Uniform Fallbacks'}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*18} {'-'*17}")

    categories = ["Sports", "Entertainment", "Economics", "Politics"]
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_success = [r for r in cat_results if r["error"] is None]
        cat_uniform = [r for r in cat_success if r["is_uniform_fallback"]]

        if cat_results:
            success_rate = f"{len(cat_success)}/{len(cat_results)}"
            avg_time = sum(r["duration_seconds"] for r in cat_results) / len(cat_results)
            avg_dec = sum(r["decisiveness"] for r in cat_success) / len(cat_success) if cat_success else 0.0
            n_uniform = len(cat_uniform)
            print(f"  {cat:<15} {success_rate:<10} {avg_time:<10.1f} {avg_dec:<18.3f} {n_uniform}")

    # ── Multi-outcome analysis ──
    print(f"\n  Multi-Outcome Event Analysis (events with 5+ outcomes):")
    print(f"  {'Event':<45} {'Outcomes':<10} {'Covered':<10} {'Decisiveness':<14} {'Top Prob'}")
    print(f"  {'-'*45} {'-'*10} {'-'*10} {'-'*14} {'-'*10}")

    multi_outcome = [r for r in successful if r["n_outcomes"] >= 5]
    for r in multi_outcome:
        title_short = r["title"][:43]
        covered = "Yes" if r["outcomes_covered"] else "NO"
        print(f"  {title_short:<45} {r['n_outcomes']:<10} {covered:<10} {r['decisiveness']:<14.3f} {r['top_probability']:.1%}")

    # ── Decisiveness analysis ──
    if successful:
        avg_decisiveness = sum(r["decisiveness"] for r in successful) / len(successful)
        high_dec = [r for r in successful if r["decisiveness"] > 0.5]
        low_dec = [r for r in successful if r["decisiveness"] < 0.2]
        uniform_fallbacks = [r for r in successful if r["is_uniform_fallback"]]

        print(f"\n  Decisiveness Metrics:")
        print(f"    Average decisiveness: {avg_decisiveness:.3f}")
        print(f"    High decisiveness (>0.5): {len(high_dec)}/{len(successful)}")
        print(f"    Low decisiveness (<0.2): {len(low_dec)}/{len(successful)}")
        print(f"    Uniform fallbacks: {len(uniform_fallbacks)}/{len(successful)}")

        if avg_decisiveness < 0.15:
            print(f"    [WARN] Agent is too indecisive - predictions are near-uniform")
        elif avg_decisiveness > 0.7:
            print(f"    [WARN] Agent may be overconfident on genuinely hard events")
        else:
            print(f"    [OK] Decisiveness is in a reasonable range for hard events")

    # ── Validity checks ──
    if successful:
        valid_sums = [r for r in successful if r["valid_sum"]]
        all_covered = [r for r in successful if r["outcomes_covered"]]

        print(f"\n  Validity Checks:")
        print(f"    Probabilities sum to ~1.0: {len(valid_sums)}/{len(successful)}")
        print(f"    All outcomes covered: {len(all_covered)}/{len(successful)}")

        invalid_sum = [r for r in successful if not r["valid_sum"]]
        if invalid_sum:
            print(f"    [WARN] Events with invalid probability sums:")
            for r in invalid_sum:
                print(f"      - {r['task_id']}: sum={r['prob_sum']:.3f}")

        uncovered = [r for r in successful if not r["outcomes_covered"]]
        if uncovered:
            print(f"    [WARN] Events with missing outcomes:")
            for r in uncovered:
                print(f"      - {r['task_id']}: returned {r['n_returned']}/{r['n_outcomes']}")

    # ── Failures ──
    if failed:
        print(f"\n  Failures ({len(failed)}):")
        for r in failed:
            print(f"    - [{r['category']}] {r['task_id']}: {r['error']}")

    # ── Response time distribution ──
    if successful:
        times = sorted(r["duration_seconds"] for r in successful)
        print(f"\n  Response Time Distribution:")
        print(f"    Min: {times[0]:.1f}s")
        print(f"    Median: {times[len(times)//2]:.1f}s")
        print(f"    P90: {times[int(len(times)*0.9)]:.1f}s")
        print(f"    Max: {times[-1]:.1f}s")

        slow = [r for r in successful if r["duration_seconds"] > 120]
        if slow:
            print(f"    [WARN] {len(slow)} events took >120s:")
            for r in slow:
                print(f"      - {r['task_id']}: {r['duration_seconds']:.0f}s")

    # ── Final grade ──
    print(f"\n  {'='*60}")
    if successful:
        score = 0
        score += min(25, len(successful) * 25 // len(results))  # Success rate (25 pts)
        score += min(25, int(avg_decisiveness * 50))  # Decisiveness (25 pts)
        score += min(25, len(valid_sums) * 25 // len(successful))  # Validity (25 pts)
        score += min(25, len(all_covered) * 25 // len(successful))  # Coverage (25 pts)

        grade = "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55 else "D" if score >= 40 else "F"
        print(f"  STRESS TEST GRADE: {grade} ({score}/100)")
        print(f"    Success: {len(successful)}/{len(results)} | Decisiveness: {avg_decisiveness:.2f} | Valid: {len(valid_sums)}/{len(successful)} | Covered: {len(all_covered)}/{len(successful)}")
    else:
        print(f"  STRESS TEST GRADE: F (0/100) - No successful predictions")
    print(f"  {'='*60}")


    # ═══════════════════════════════════════════════════════════════════
    # SAVE RESULTS TO JSON
    # ═══════════════════════════════════════════════════════════════════

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "stress_test_results.json"

    # Compute summary stats for JSON
    summary = {
        "metadata": {
            "test_name": "Prophet Agent Stress Test",
            "timestamp": datetime.now().isoformat(),
            "evaluation_window": "May 17 - May 31, 2026",
            "total_events": len(results),
            "successful_predictions": len(successful),
            "failed_predictions": len(failed),
            "total_duration_seconds": total_duration,
            "avg_duration_seconds": total_duration / len(results) if results else 0,
            "server_url": SERVER_URL,
            "timeout_seconds": TIMEOUT_SECONDS,
        },
        "scores": {
            "success_rate": len(successful) / len(results) if results else 0,
            "avg_decisiveness": sum(r["decisiveness"] for r in successful) / len(successful) if successful else 0,
            "valid_sum_rate": len([r for r in successful if r["valid_sum"]]) / len(successful) if successful else 0,
            "coverage_rate": len([r for r in successful if r["outcomes_covered"]]) / len(successful) if successful else 0,
            "uniform_fallback_rate": len([r for r in successful if r["is_uniform_fallback"]]) / len(successful) if successful else 0,
        },
        "by_category": {},
        "results": results,
    }

    # Per-category stats
    for cat in ["Sports", "Entertainment", "Economics", "Politics"]:
        cat_results = [r for r in results if r["category"] == cat]
        cat_success = [r for r in cat_results if r["error"] is None]
        summary["by_category"][cat] = {
            "total": len(cat_results),
            "successful": len(cat_success),
            "success_rate": len(cat_success) / len(cat_results) if cat_results else 0,
            "avg_decisiveness": sum(r["decisiveness"] for r in cat_success) / len(cat_success) if cat_success else 0,
            "avg_duration": sum(r["duration_seconds"] for r in cat_results) / len(cat_results) if cat_results else 0,
            "uniform_fallbacks": len([r for r in cat_success if r["is_uniform_fallback"]]),
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {output_path}")
    print(f"\n  Stress test complete.")


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    asyncio.run(run_stress_test())
