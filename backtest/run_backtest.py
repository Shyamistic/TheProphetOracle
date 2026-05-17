"""
Prophet Forecasting Agent — Comprehensive Backtester

Runs 35 real-world resolved events through the live prediction pipeline,
computes Brier scores, calibration metrics, and generates a detailed report.

Usage:
    venv\\Scripts\\python.exe backtest\\run_backtest.py

Prerequisites:
    - Server running at http://127.0.0.1:8888
    - .env configured with API keys
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

# ============================================================
# CONFIGURATION
# ============================================================

SERVER_URL = "http://127.0.0.1:8888"
PREDICT_ENDPOINT = f"{SERVER_URL}/predict"
TIMEOUT_SECONDS = 120
DELAY_BETWEEN_EVENTS = 2  # seconds
RESULTS_DIR = Path(__file__).parent / "results"


# ============================================================
# RESOLVED EVENTS DATABASE
# ============================================================

def get_backtest_events() -> List[Dict]:
    """Return 35 real-world events that have definitively resolved."""
    events = [
        # ─────────────────────────────────────────────────────
        # SPORTS (6 events)
        # ─────────────────────────────────────────────────────
        {
            "event_ticker": "SPORTS-SUPERBOWL-2024",
            "market_ticker": "SUPERBOWL-LVIII-WINNER",
            "title": "Did the Kansas City Chiefs win Super Bowl LVIII in February 2024?",
            "description": "Super Bowl LVIII was played on February 11, 2024 between the Kansas City Chiefs and the San Francisco 49ers at Allegiant Stadium in Las Vegas.",
            "category": "sports",
            "rules": "Resolves YES if the Kansas City Chiefs won Super Bowl LVIII. Resolves NO otherwise.",
            "close_time": "2024-02-12T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
        {
            "event_ticker": "SPORTS-NBA-FINALS-2024",
            "market_ticker": "NBA-FINALS-2024-WINNER",
            "title": "Did the Boston Celtics win the 2024 NBA Finals?",
            "description": "The 2024 NBA Finals were played between the Boston Celtics and the Dallas Mavericks in June 2024.",
            "category": "sports",
            "rules": "Resolves YES if the Boston Celtics won the 2024 NBA Finals. Resolves NO otherwise.",
            "close_time": "2024-06-18T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
        {
            "event_ticker": "SPORTS-EURO-2024",
            "market_ticker": "UEFA-EURO-2024-WINNER",
            "title": "Did Spain win UEFA Euro 2024?",
            "description": "UEFA Euro 2024 was held in Germany from June 14 to July 14, 2024. The final was played between Spain and England.",
            "category": "sports",
            "rules": "Resolves YES if Spain won UEFA Euro 2024. Resolves NO otherwise.",
            "close_time": "2024-07-15T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
        {
            "event_ticker": "SPORTS-AUS-OPEN-2025",
            "market_ticker": "AUS-OPEN-MENS-2025",
            "title": "Did Jannik Sinner win the 2025 Australian Open Men's Singles?",
            "description": "The 2025 Australian Open Men's Singles tournament was held in January 2025 in Melbourne, Australia.",
            "category": "sports",
            "rules": "Resolves YES if Jannik Sinner won the 2025 Australian Open Men's Singles title. Resolves NO otherwise.",
            "close_time": "2025-01-27T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
        {
            "event_ticker": "SPORTS-WORLD-SERIES-2024",
            "market_ticker": "MLB-WS-2024-WINNER",
            "title": "Did the Los Angeles Dodgers win the 2024 World Series?",
            "description": "The 2024 World Series was played between the Los Angeles Dodgers and the New York Yankees in October 2024.",
            "category": "sports",
            "rules": "Resolves YES if the LA Dodgers won the 2024 World Series. Resolves NO otherwise.",
            "close_time": "2024-11-01T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
        {
            "event_ticker": "SPORTS-T20-WC-2024",
            "market_ticker": "T20-WC-2024-WINNER",
            "title": "Did India win the 2024 ICC T20 World Cup?",
            "description": "The 2024 ICC Men's T20 World Cup was held in the West Indies and the United States in June 2024.",
            "category": "sports",
            "rules": "Resolves YES if India won the 2024 ICC T20 World Cup. Resolves NO otherwise.",
            "close_time": "2024-06-30T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },

        # ─────────────────────────────────────────────────────
        # ECONOMICS (6 events)
        # ─────────────────────────────────────────────────────
        {
            "event_ticker": "ECON-FED-SEPT-2024",
            "market_ticker": "FED-RATE-CUT-SEPT-2024",
            "title": "Did the Federal Reserve cut interest rates in September 2024?",
            "description": "The Federal Reserve's FOMC meeting in September 2024 decided on monetary policy. Markets widely expected a rate cut after holding rates steady for over a year.",
            "category": "economics",
            "rules": "Resolves YES if the Fed announced a rate cut at the September 2024 FOMC meeting. Resolves NO otherwise.",
            "close_time": "2024-09-19T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
        {
            "event_ticker": "ECON-SP500-5000-2024",
            "market_ticker": "SP500-ABOVE-5000-2024",
            "title": "Did the S&P 500 close above 5000 at any point in 2024?",
            "description": "The S&P 500 index had been approaching the 5000 milestone throughout early 2024.",
            "category": "economics",
            "rules": "Resolves YES if the S&P 500 index closed above 5000 at any point during 2024. Resolves NO otherwise.",
            "close_time": "2024-12-31T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
        {
            "event_ticker": "ECON-BTC-100K-2024",
            "market_ticker": "BTC-ABOVE-100K-2024",
            "title": "Did Bitcoin exceed $100,000 in 2024?",
            "description": "Bitcoin had been rallying throughout 2024 following the approval of spot Bitcoin ETFs and the April 2024 halving event.",
            "category": "economics",
            "rules": "Resolves YES if Bitcoin's price exceeded $100,000 USD at any point during 2024. Resolves NO otherwise.",
            "close_time": "2024-12-31T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
        {
            "event_ticker": "ECON-US-GDP-Q3-2024",
            "market_ticker": "US-GDP-GROWTH-Q3-2024",
            "title": "Did US GDP grow in Q3 2024?",
            "description": "US GDP growth for Q3 2024 as reported by the Bureau of Economic Analysis.",
            "category": "economics",
            "rules": "Resolves YES if the advance estimate of US real GDP growth for Q3 2024 was positive. Resolves NO otherwise.",
            "close_time": "2024-10-31T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
        {
            "event_ticker": "ECON-FED-DEC-2024",
            "market_ticker": "FED-RATE-CUT-DEC-2024",
            "title": "Did the Federal Reserve cut interest rates in December 2024?",
            "description": "The Federal Reserve's FOMC meeting in December 2024 decided on monetary policy after cuts in September and November.",
            "category": "economics",
            "rules": "Resolves YES if the Fed announced a rate cut at the December 2024 FOMC meeting. Resolves NO otherwise.",
            "close_time": "2024-12-19T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
        {
            "event_ticker": "ECON-US-RECESSION-2024",
            "market_ticker": "US-RECESSION-2024",
            "title": "Did the US enter a recession in 2024?",
            "description": "Many economists predicted a US recession in 2024 following aggressive rate hikes in 2022-2023.",
            "category": "economics",
            "rules": "Resolves YES if the NBER declared a US recession starting in 2024, or if two consecutive quarters of negative GDP growth occurred in 2024. Resolves NO otherwise.",
            "close_time": "2024-12-31T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "No",
        },
    ]
    return events



def get_backtest_events_part2() -> List[Dict]:
    """Geopolitics and Technology events."""
    events = [
        # ─────────────────────────────────────────────────────
        # GEOPOLITICS (6 events)
        # ─────────────────────────────────────────────────────
        {
            "event_ticker": "GEO-US-ELECTION-2024",
            "market_ticker": "US-PRES-ELECTION-2024",
            "title": "Did Donald Trump win the 2024 US Presidential Election?",
            "description": "The 2024 US Presidential Election was held on November 5, 2024 between Donald Trump (Republican) and Kamala Harris (Democrat).",
            "category": "geopolitics",
            "rules": "Resolves YES if Donald Trump won the 2024 US Presidential Election. Resolves NO otherwise.",
            "close_time": "2024-11-06T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
        {
            "event_ticker": "GEO-UK-ELECTION-2024",
            "market_ticker": "UK-GENERAL-ELECTION-2024",
            "title": "Did the UK hold a general election in 2024?",
            "description": "UK Prime Minister Rishi Sunak had the option to call a general election at any point before January 2025.",
            "category": "geopolitics",
            "rules": "Resolves YES if the UK held a general election in 2024. Resolves NO otherwise.",
            "close_time": "2024-12-31T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
        {
            "event_ticker": "GEO-INDIA-ELECTION-2024",
            "market_ticker": "INDIA-GENERAL-ELECTION-2024",
            "title": "Did India hold general elections in 2024?",
            "description": "India's general elections (Lok Sabha elections) were scheduled for 2024.",
            "category": "geopolitics",
            "rules": "Resolves YES if India held its general elections in 2024. Resolves NO otherwise.",
            "close_time": "2024-06-30T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
        {
            "event_ticker": "GEO-UK-LABOUR-WIN-2024",
            "market_ticker": "UK-LABOUR-MAJORITY-2024",
            "title": "Did the Labour Party win a majority in the 2024 UK general election?",
            "description": "The UK general election was held on July 4, 2024. Polls showed Labour with a significant lead over the Conservatives.",
            "category": "geopolitics",
            "rules": "Resolves YES if the Labour Party won a majority of seats in the House of Commons in the 2024 UK general election. Resolves NO otherwise.",
            "close_time": "2024-07-05T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
        {
            "event_ticker": "GEO-TAIWAN-ELECTION-2024",
            "market_ticker": "TAIWAN-DPP-WIN-2024",
            "title": "Did the DPP win the 2024 Taiwan presidential election?",
            "description": "Taiwan held its presidential election on January 13, 2024. The Democratic Progressive Party (DPP) candidate was Lai Ching-te.",
            "category": "geopolitics",
            "rules": "Resolves YES if the DPP candidate won the 2024 Taiwan presidential election. Resolves NO otherwise.",
            "close_time": "2024-01-14T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
        {
            "event_ticker": "GEO-SWEDEN-NATO-2024",
            "market_ticker": "SWEDEN-NATO-JOIN-2024",
            "title": "Did Sweden officially join NATO in 2024?",
            "description": "Sweden applied for NATO membership in 2022 and was awaiting ratification from Turkey and Hungary.",
            "category": "geopolitics",
            "rules": "Resolves YES if Sweden officially became a NATO member in 2024. Resolves NO otherwise.",
            "close_time": "2024-12-31T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },

        # ─────────────────────────────────────────────────────
        # TECHNOLOGY (6 events)
        # ─────────────────────────────────────────────────────
        {
            "event_ticker": "TECH-GPT4O-2024",
            "market_ticker": "OPENAI-GPT4O-RELEASE-2024",
            "title": "Did OpenAI release GPT-4o in 2024?",
            "description": "OpenAI had been developing multimodal AI models. GPT-4o was rumored to be a new model combining text, vision, and audio capabilities.",
            "category": "technology",
            "rules": "Resolves YES if OpenAI publicly released a model called GPT-4o in 2024. Resolves NO otherwise.",
            "close_time": "2024-12-31T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
        {
            "event_ticker": "TECH-VISION-PRO-2024",
            "market_ticker": "APPLE-VISION-PRO-RELEASE-2024",
            "title": "Did Apple release the Vision Pro headset in 2024?",
            "description": "Apple announced the Vision Pro mixed reality headset at WWDC 2023 and indicated it would ship in early 2024.",
            "category": "technology",
            "rules": "Resolves YES if Apple released the Vision Pro for consumer purchase in 2024. Resolves NO otherwise.",
            "close_time": "2024-12-31T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
        {
            "event_ticker": "TECH-NVIDIA-3T-2024",
            "market_ticker": "NVIDIA-MARKET-CAP-3T-2024",
            "title": "Did Nvidia's market cap exceed $3 trillion in 2024?",
            "description": "Nvidia's stock had been surging due to AI demand. The company was approaching a $3 trillion market capitalization.",
            "category": "technology",
            "rules": "Resolves YES if Nvidia's market capitalization exceeded $3 trillion USD at any point in 2024. Resolves NO otherwise.",
            "close_time": "2024-12-31T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
        {
            "event_ticker": "TECH-TIKTOK-BAN-2024",
            "market_ticker": "TIKTOK-US-BAN-2024",
            "title": "Was TikTok banned in the United States in 2024?",
            "description": "The US Congress passed legislation requiring ByteDance to divest TikTok or face a ban. The deadline and enforcement were debated.",
            "category": "technology",
            "rules": "Resolves YES if TikTok was fully banned (unavailable for download/use) in the US during 2024. Resolves NO otherwise.",
            "close_time": "2024-12-31T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "No",
        },
        {
            "event_ticker": "TECH-OPENAI-SEARCH-2024",
            "market_ticker": "OPENAI-SEARCH-PRODUCT-2024",
            "title": "Did OpenAI launch a search product (SearchGPT/ChatGPT Search) in 2024?",
            "description": "OpenAI was rumored to be developing a search product to compete with Google.",
            "category": "technology",
            "rules": "Resolves YES if OpenAI publicly launched a web search product in 2024. Resolves NO otherwise.",
            "close_time": "2024-12-31T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
        {
            "event_ticker": "TECH-GOOGLE-GEMINI-ULTRA-2024",
            "market_ticker": "GOOGLE-GEMINI-ULTRA-2024",
            "title": "Did Google release Gemini Ultra in 2024?",
            "description": "Google announced Gemini (formerly Bard) with Ultra, Pro, and Nano tiers in December 2023, with Ultra planned for early 2024.",
            "category": "technology",
            "rules": "Resolves YES if Google publicly released Gemini Ultra (or Gemini Advanced with Ultra model) in 2024. Resolves NO otherwise.",
            "close_time": "2024-12-31T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
    ]
    return events



def get_backtest_events_part3() -> List[Dict]:
    """Science and General events."""
    events = [
        # ─────────────────────────────────────────────────────
        # SCIENCE (6 events)
        # ─────────────────────────────────────────────────────
        {
            "event_ticker": "SCI-STARSHIP-CATCH-2024",
            "market_ticker": "SPACEX-STARSHIP-CATCH-2024",
            "title": "Did SpaceX successfully catch a Starship booster with the launch tower in 2024?",
            "description": "SpaceX had been testing its Starship rocket system and planned to attempt catching the Super Heavy booster with mechanical arms on the launch tower.",
            "category": "science",
            "rules": "Resolves YES if SpaceX successfully caught a Starship Super Heavy booster with the tower arms during 2024. Resolves NO otherwise.",
            "close_time": "2024-12-31T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
        {
            "event_ticker": "SCI-ARTEMIS-II-2024",
            "market_ticker": "NASA-ARTEMIS-II-LAUNCH-2024",
            "title": "Did NASA launch Artemis II in 2024?",
            "description": "NASA's Artemis II mission was planned to send astronauts around the Moon. It had been delayed multiple times from its original 2024 target.",
            "category": "science",
            "rules": "Resolves YES if NASA launched the Artemis II mission in 2024. Resolves NO otherwise.",
            "close_time": "2024-12-31T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "No",
        },
        {
            "event_ticker": "SCI-EUROPA-CLIPPER-2024",
            "market_ticker": "NASA-EUROPA-CLIPPER-LAUNCH-2024",
            "title": "Did NASA launch the Europa Clipper mission in 2024?",
            "description": "NASA's Europa Clipper spacecraft was scheduled to launch in October 2024 to study Jupiter's moon Europa.",
            "category": "science",
            "rules": "Resolves YES if NASA launched the Europa Clipper mission in 2024. Resolves NO otherwise.",
            "close_time": "2024-12-31T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
        {
            "event_ticker": "SCI-NOBEL-PHYSICS-AI-2024",
            "market_ticker": "NOBEL-PHYSICS-AI-2024",
            "title": "Was the 2024 Nobel Prize in Physics awarded for work related to artificial intelligence or machine learning?",
            "description": "The 2024 Nobel Prize in Physics was announced in October 2024.",
            "category": "science",
            "rules": "Resolves YES if the 2024 Nobel Prize in Physics was awarded for work related to AI, neural networks, or machine learning. Resolves NO otherwise.",
            "close_time": "2024-10-09T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
        {
            "event_ticker": "SCI-BOEING-STARLINER-2024",
            "market_ticker": "BOEING-STARLINER-CREW-2024",
            "title": "Did Boeing Starliner complete its first crewed mission successfully in 2024?",
            "description": "Boeing's Starliner spacecraft launched its first crewed test flight (CFT) to the ISS in June 2024 with astronauts Butch Wilmore and Suni Williams.",
            "category": "science",
            "rules": "Resolves YES if Boeing Starliner successfully completed its crewed flight test (launched AND returned crew safely) in 2024. Resolves NO otherwise.",
            "close_time": "2024-12-31T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "No",
        },
        {
            "event_ticker": "SCI-POLARIS-DAWN-2024",
            "market_ticker": "POLARIS-DAWN-SPACEWALK-2024",
            "title": "Did the Polaris Dawn mission conduct the first commercial spacewalk in 2024?",
            "description": "The Polaris Dawn mission, funded by Jared Isaacman, planned to conduct the first-ever commercial spacewalk using SpaceX EVA suits.",
            "category": "science",
            "rules": "Resolves YES if the Polaris Dawn mission successfully conducted a commercial spacewalk in 2024. Resolves NO otherwise.",
            "close_time": "2024-12-31T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },

        # ─────────────────────────────────────────────────────
        # GENERAL (5 events)
        # ─────────────────────────────────────────────────────
        {
            "event_ticker": "GEN-OLYMPICS-PARIS-2024",
            "market_ticker": "PARIS-OLYMPICS-2024",
            "title": "Were the 2024 Summer Olympics held in Paris?",
            "description": "The 2024 Summer Olympics were scheduled to be held in Paris, France.",
            "category": "general",
            "rules": "Resolves YES if the 2024 Summer Olympics were held in Paris. Resolves NO otherwise.",
            "close_time": "2024-08-12T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
        {
            "event_ticker": "GEN-TAYLOR-SWIFT-ERAS-2024",
            "market_ticker": "TAYLOR-SWIFT-ERAS-TOUR-GROSS-2024",
            "title": "Did Taylor Swift's Eras Tour gross over $1 billion in 2024?",
            "description": "Taylor Swift's Eras Tour continued through 2024 and was on track to become the highest-grossing concert tour of all time.",
            "category": "general",
            "rules": "Resolves YES if Taylor Swift's Eras Tour grossed over $1 billion in total revenue during 2024 performances. Resolves NO otherwise.",
            "close_time": "2024-12-31T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
        {
            "event_ticker": "GEN-REDDIT-IPO-2024",
            "market_ticker": "REDDIT-IPO-2024",
            "title": "Did Reddit go public (IPO) in 2024?",
            "description": "Reddit had been planning an IPO for several years and filed its S-1 in early 2024.",
            "category": "general",
            "rules": "Resolves YES if Reddit completed its initial public offering in 2024. Resolves NO otherwise.",
            "close_time": "2024-12-31T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
        {
            "event_ticker": "GEN-FRANCIS-SCOTT-KEY-BRIDGE-2024",
            "market_ticker": "BALTIMORE-BRIDGE-COLLAPSE-2024",
            "title": "Did the Francis Scott Key Bridge in Baltimore collapse in 2024?",
            "description": "The Francis Scott Key Bridge in Baltimore, Maryland was struck by a container ship in March 2024.",
            "category": "general",
            "rules": "Resolves YES if the Francis Scott Key Bridge in Baltimore collapsed in 2024. Resolves NO otherwise.",
            "close_time": "2024-12-31T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
        {
            "event_ticker": "GEN-THREADS-100M-2024",
            "market_ticker": "META-THREADS-100M-2024",
            "title": "Did Meta's Threads app reach 100 million monthly active users by end of 2024?",
            "description": "Meta launched Threads in July 2023 as a competitor to X (formerly Twitter). The app saw rapid initial growth followed by a decline.",
            "category": "general",
            "rules": "Resolves YES if Meta's Threads app reached 100 million monthly active users by December 31, 2024. Resolves NO otherwise.",
            "close_time": "2024-12-31T00:00:00Z",
            "outcomes": ["Yes", "No"],
            "resolved_outcome": "Yes",
        },
    ]
    return events


def get_all_events() -> List[Dict]:
    """Combine all event parts into a single list."""
    return get_backtest_events() + get_backtest_events_part2() + get_backtest_events_part3()



# ============================================================
# BRIER SCORE COMPUTATION
# ============================================================

def compute_brier_score(predicted_probs: Dict[str, float], resolved_outcome: str) -> float:
    """
    Compute Brier score for a single prediction.

    For multi-outcome: BS = (1/N) * sum_i (f_i - o_i)^2
    where f_i is the forecast probability for outcome i,
    and o_i is 1 if outcome i resolved, 0 otherwise.

    Lower is better. Perfect = 0, worst = 2 (for binary).
    """
    n_outcomes = len(predicted_probs)
    brier = 0.0
    for outcome, prob in predicted_probs.items():
        actual = 1.0 if outcome == resolved_outcome else 0.0
        brier += (prob - actual) ** 2
    # Normalize by number of outcomes for multi-outcome
    return brier / n_outcomes


def compute_baseline_brier(n_outcomes: int, resolved_outcome: str, outcomes: List[str]) -> Dict[str, float]:
    """Compute baseline Brier scores for comparison."""
    # Uniform baseline: assign 1/N to each outcome
    uniform_prob = 1.0 / n_outcomes
    uniform_brier = 0.0
    for outcome in outcomes:
        actual = 1.0 if outcome == resolved_outcome else 0.0
        uniform_brier += (uniform_prob - actual) ** 2
    uniform_brier /= n_outcomes

    # Always 50/50 baseline (for binary)
    fifty_fifty_brier = 0.0
    for outcome in outcomes:
        actual = 1.0 if outcome == resolved_outcome else 0.0
        fifty_fifty_brier += (0.5 - actual) ** 2
    fifty_fifty_brier /= n_outcomes

    return {
        "uniform": uniform_brier,
        "fifty_fifty": fifty_fifty_brier,
    }


# ============================================================
# CALIBRATION ANALYSIS
# ============================================================

def compute_calibration(results: List[Dict]) -> Dict[str, Dict]:
    """
    Compute calibration metrics.

    Bins predictions by confidence level and checks how often
    the predicted outcome actually occurred.
    """
    bins = {
        "0.0-0.1": {"predictions": [], "outcomes": []},
        "0.1-0.2": {"predictions": [], "outcomes": []},
        "0.2-0.3": {"predictions": [], "outcomes": []},
        "0.3-0.4": {"predictions": [], "outcomes": []},
        "0.4-0.5": {"predictions": [], "outcomes": []},
        "0.5-0.6": {"predictions": [], "outcomes": []},
        "0.6-0.7": {"predictions": [], "outcomes": []},
        "0.7-0.8": {"predictions": [], "outcomes": []},
        "0.8-0.9": {"predictions": [], "outcomes": []},
        "0.9-1.0": {"predictions": [], "outcomes": []},
    }

    for result in results:
        if result.get("error"):
            continue
        predicted_probs = result["predicted_probs"]
        resolved = result["resolved_outcome"]

        # For each outcome, record the predicted probability and whether it resolved
        for outcome, prob in predicted_probs.items():
            actual = 1.0 if outcome == resolved else 0.0

            # Determine bin
            bin_idx = min(int(prob * 10), 9)
            bin_keys = list(bins.keys())
            bin_key = bin_keys[bin_idx]

            bins[bin_key]["predictions"].append(prob)
            bins[bin_key]["outcomes"].append(actual)

    # Compute calibration stats per bin
    calibration = {}
    for bin_key, data in bins.items():
        if len(data["predictions"]) > 0:
            avg_predicted = sum(data["predictions"]) / len(data["predictions"])
            avg_actual = sum(data["outcomes"]) / len(data["outcomes"])
            calibration[bin_key] = {
                "count": len(data["predictions"]),
                "avg_predicted": round(avg_predicted, 4),
                "avg_actual": round(avg_actual, 4),
                "calibration_error": round(abs(avg_predicted - avg_actual), 4),
            }
        else:
            calibration[bin_key] = {
                "count": 0,
                "avg_predicted": None,
                "avg_actual": None,
                "calibration_error": None,
            }

    return calibration


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_report(results: List[Dict], duration_seconds: float) -> str:
    """Generate a comprehensive markdown report."""
    successful = [r for r in results if not r.get("error")]
    failed = [r for r in results if r.get("error")]

    # Overall metrics
    brier_scores = [r["brier_score"] for r in successful]
    overall_brier = sum(brier_scores) / len(brier_scores) if brier_scores else float("inf")

    # Baseline comparisons
    baseline_briers = []
    for r in successful:
        baselines = compute_baseline_brier(
            len(r["predicted_probs"]),
            r["resolved_outcome"],
            list(r["predicted_probs"].keys()),
        )
        baseline_briers.append(baselines)

    avg_uniform_brier = (
        sum(b["uniform"] for b in baseline_briers) / len(baseline_briers)
        if baseline_briers else float("inf")
    )
    avg_5050_brier = (
        sum(b["fifty_fifty"] for b in baseline_briers) / len(baseline_briers)
        if baseline_briers else float("inf")
    )

    # Category breakdown
    categories = {}
    for r in successful:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r["brier_score"])

    # Calibration
    calibration = compute_calibration(successful)

    # Sort by Brier score
    sorted_results = sorted(successful, key=lambda x: x["brier_score"])
    best_5 = sorted_results[:5]
    worst_5 = sorted_results[-5:][::-1]

    # Confidence analysis
    overconfident = []
    underconfident = []
    for r in successful:
        prob_of_actual = r["predicted_probs"].get(r["resolved_outcome"], 0.5)
        if prob_of_actual < 0.3:
            overconfident.append(r)  # Wrong direction — very confident in wrong outcome
        elif prob_of_actual > 0.9:
            underconfident.append(r)  # Could be more decisive but already very confident

    # Build report
    report = []
    report.append("# Prophet Forecasting Agent — Backtest Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Duration:** {duration_seconds:.1f} seconds ({duration_seconds/60:.1f} minutes)")
    report.append(f"**Events Tested:** {len(results)} ({len(successful)} successful, {len(failed)} failed)")
    report.append("")

    # Summary
    report.append("## Summary")
    report.append("")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| **Overall Brier Score** | **{overall_brier:.4f}** |")
    report.append(f"| Uniform Baseline | {avg_uniform_brier:.4f} |")
    report.append(f"| 50/50 Baseline | {avg_5050_brier:.4f} |")
    report.append(f"| Improvement vs Uniform | {((avg_uniform_brier - overall_brier) / avg_uniform_brier * 100):.1f}% |")
    report.append(f"| Improvement vs 50/50 | {((avg_5050_brier - overall_brier) / avg_5050_brier * 100):.1f}% |")
    report.append(f"| Market Average (target) | ~0.1500 |")
    report.append("")

    if overall_brier < 0.15:
        report.append("> ✅ **Agent beats market average Brier score!**")
    elif overall_brier < 0.20:
        report.append("> ⚠️ **Agent is close to market average but needs improvement.**")
    else:
        report.append("> ❌ **Agent is underperforming market average.**")
    report.append("")

    # Category breakdown
    report.append("## Category Breakdown")
    report.append("")
    report.append("| Category | Events | Avg Brier | Min | Max |")
    report.append("|----------|--------|-----------|-----|-----|")
    for cat, scores in sorted(categories.items(), key=lambda x: sum(x[1])/len(x[1])):
        avg = sum(scores) / len(scores)
        report.append(f"| {cat.capitalize()} | {len(scores)} | {avg:.4f} | {min(scores):.4f} | {max(scores):.4f} |")
    report.append("")

    # Best predictions
    report.append("## Top 5 Best Predictions (Lowest Brier)")
    report.append("")
    for i, r in enumerate(best_5, 1):
        prob_of_actual = r["predicted_probs"].get(r["resolved_outcome"], 0)
        report.append(f"**{i}. {r['title']}**")
        report.append(f"   - Brier: {r['brier_score']:.4f}")
        report.append(f"   - Predicted prob of actual outcome ({r['resolved_outcome']}): {prob_of_actual:.2%}")
        report.append(f"   - Category: {r['category']}")
        report.append("")

    # Worst predictions
    report.append("## Top 5 Worst Predictions (Highest Brier)")
    report.append("")
    for i, r in enumerate(worst_5, 1):
        prob_of_actual = r["predicted_probs"].get(r["resolved_outcome"], 0)
        report.append(f"**{i}. {r['title']}**")
        report.append(f"   - Brier: {r['brier_score']:.4f}")
        report.append(f"   - Predicted prob of actual outcome ({r['resolved_outcome']}): {prob_of_actual:.2%}")
        report.append(f"   - Category: {r['category']}")
        report.append("")

    # Calibration
    report.append("## Calibration Analysis")
    report.append("")
    report.append("| Predicted Range | Count | Avg Predicted | Avg Actual | Calibration Error |")
    report.append("|-----------------|-------|---------------|------------|-------------------|")
    for bin_key, data in calibration.items():
        if data["count"] > 0:
            report.append(
                f"| {bin_key} | {data['count']} | {data['avg_predicted']:.4f} | "
                f"{data['avg_actual']:.4f} | {data['calibration_error']:.4f} |"
            )
    report.append("")

    # Overconfidence analysis
    total_cal_error = sum(
        d["calibration_error"] for d in calibration.values()
        if d["calibration_error"] is not None
    )
    n_bins_with_data = sum(1 for d in calibration.values() if d["count"] > 0)
    avg_cal_error = total_cal_error / n_bins_with_data if n_bins_with_data > 0 else 0

    report.append(f"**Average Calibration Error:** {avg_cal_error:.4f}")
    report.append("")
    if avg_cal_error < 0.05:
        report.append("> ✅ Well-calibrated predictions")
    elif avg_cal_error < 0.10:
        report.append("> ⚠️ Moderate calibration — room for improvement")
    else:
        report.append("> ❌ Poorly calibrated — significant over/underconfidence")
    report.append("")

    # Failed events
    if failed:
        report.append("## Failed Events")
        report.append("")
        for r in failed:
            report.append(f"- **{r['title']}**: {r['error']}")
        report.append("")

    # Recommendations
    report.append("## Recommendations")
    report.append("")

    if overall_brier > 0.15:
        report.append("1. **Improve research depth** — The agent may not be finding enough evidence for confident predictions.")
    if avg_cal_error > 0.05:
        report.append("2. **Tune calibration parameters** — Adjust shrinkage_factor and platt_coefficient in .env.")
    if len(overconfident) > 3:
        report.append("3. **Reduce overconfidence** — The agent is too confident on wrong outcomes. Consider stronger shrinkage toward base rates.")
    if len(failed) > 2:
        report.append("4. **Improve error handling** — Multiple events failed. Check timeout settings and API reliability.")

    # Category-specific recommendations
    worst_cat = max(categories.items(), key=lambda x: sum(x[1])/len(x[1])) if categories else None
    if worst_cat:
        report.append(f"5. **Focus on {worst_cat[0]}** — This category has the worst average Brier score ({sum(worst_cat[1])/len(worst_cat[1]):.4f}).")

    report.append("")
    report.append("---")
    report.append(f"*Report generated by Prophet Backtest Suite*")

    return "\n".join(report)



# ============================================================
# MAIN EXECUTION
# ============================================================

async def check_server_health() -> bool:
    """Check if the prediction server is running."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{SERVER_URL}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"  Server status: {data.get('status', 'unknown')}")
                if "budget" in data:
                    budget = data["budget"]
                    print(f"  Budget remaining: ${budget.get('budget_remaining_usd', '?'):.2f}")
                return True
            else:
                print(f"  Server returned status {response.status_code}")
                return False
    except httpx.ConnectError:
        print(f"  Cannot connect to {SERVER_URL}")
        return False
    except Exception as e:
        print(f"  Health check error: {e}")
        return False


async def predict_event(client: httpx.AsyncClient, event: Dict) -> Dict:
    """Send a single event to the prediction API and return results."""
    # Build request payload (exclude resolved_outcome)
    payload = {
        "event_ticker": event["event_ticker"],
        "market_ticker": event["market_ticker"],
        "title": event["title"],
        "description": event["description"],
        "category": event["category"],
        "rules": event["rules"],
        "close_time": event["close_time"],
        "outcomes": event["outcomes"],
    }

    start_time = time.time()
    try:
        response = await client.post(
            PREDICT_ENDPOINT,
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )
        duration = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            # Parse probabilities from response
            predicted_probs = {}
            for entry in data.get("probabilities", []):
                predicted_probs[entry["market"]] = entry["probability"]

            # Compute Brier score
            brier = compute_brier_score(predicted_probs, event["resolved_outcome"])

            return {
                "event_ticker": event["event_ticker"],
                "title": event["title"],
                "category": event["category"],
                "outcomes": event["outcomes"],
                "resolved_outcome": event["resolved_outcome"],
                "predicted_probs": predicted_probs,
                "brier_score": brier,
                "duration_seconds": duration,
                "error": None,
            }
        else:
            error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            return {
                "event_ticker": event["event_ticker"],
                "title": event["title"],
                "category": event["category"],
                "outcomes": event["outcomes"],
                "resolved_outcome": event["resolved_outcome"],
                "predicted_probs": {},
                "brier_score": None,
                "duration_seconds": time.time() - start_time,
                "error": error_msg,
            }
    except httpx.TimeoutException:
        return {
            "event_ticker": event["event_ticker"],
            "title": event["title"],
            "category": event["category"],
            "outcomes": event["outcomes"],
            "resolved_outcome": event["resolved_outcome"],
            "predicted_probs": {},
            "brier_score": None,
            "duration_seconds": time.time() - start_time,
            "error": f"Timeout after {TIMEOUT_SECONDS}s",
        }
    except Exception as e:
        return {
            "event_ticker": event["event_ticker"],
            "title": event["title"],
            "category": event["category"],
            "outcomes": event["outcomes"],
            "resolved_outcome": event["resolved_outcome"],
            "predicted_probs": {},
            "brier_score": None,
            "duration_seconds": time.time() - start_time,
            "error": str(e),
        }


async def run_backtest():
    """Main backtest execution loop."""
    print("=" * 70)
    print("  PROPHET FORECASTING AGENT — COMPREHENSIVE BACKTEST")
    print("=" * 70)
    print()

    # Load events
    events = get_all_events()
    print(f"📋 Loaded {len(events)} resolved events across categories:")
    categories = {}
    for e in events:
        cat = e["category"]
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"   • {cat.capitalize()}: {count} events")
    print()

    # Check server health
    print("🔍 Checking server health...")
    healthy = await check_server_health()
    if not healthy:
        print("\n❌ Server is not available. Please start the server first:")
        print(f"   venv\\Scripts\\python.exe -m uvicorn src.api:app --host 0.0.0.0 --port 8888")
        sys.exit(1)
    print("✅ Server is healthy!")
    print()

    # Run predictions
    print(f"🚀 Starting backtest ({len(events)} events, ~{DELAY_BETWEEN_EVENTS}s delay between each)...")
    print(f"   Estimated time: {len(events) * (15 + DELAY_BETWEEN_EVENTS) / 60:.0f}-{len(events) * (60 + DELAY_BETWEEN_EVENTS) / 60:.0f} minutes")
    print("-" * 70)

    results = []
    start_time = time.time()

    async with httpx.AsyncClient() as client:
        for i, event in enumerate(events, 1):
            print(f"\n[{i}/{len(events)}] Predicting: {event['title'][:60]}...")
            result = await predict_event(client, event)
            results.append(result)

            if result["error"]:
                print(f"   ❌ ERROR: {result['error'][:80]}")
            else:
                prob_of_actual = result["predicted_probs"].get(result["resolved_outcome"], 0)
                print(f"   ✅ Brier: {result['brier_score']:.4f} | "
                      f"P({result['resolved_outcome']}): {prob_of_actual:.2%} | "
                      f"Time: {result['duration_seconds']:.1f}s")

            # Delay between events (except last)
            if i < len(events):
                await asyncio.sleep(DELAY_BETWEEN_EVENTS)

    total_duration = time.time() - start_time
    print("\n" + "-" * 70)
    print(f"⏱️  Total time: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")

    # Compute summary stats
    successful = [r for r in results if not r.get("error")]
    if successful:
        brier_scores = [r["brier_score"] for r in successful]
        avg_brier = sum(brier_scores) / len(brier_scores)
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"   Overall Brier Score: {avg_brier:.4f}")
        print(f"   Successful: {len(successful)}/{len(results)}")
        print(f"   Best:  {min(brier_scores):.4f}")
        print(f"   Worst: {max(brier_scores):.4f}")
    else:
        print("\n❌ No successful predictions!")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save raw JSON
    json_path = RESULTS_DIR / "predictions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "total_events": len(events),
                    "successful_events": len(successful),
                    "failed_events": len(results) - len(successful),
                    "total_duration_seconds": total_duration,
                    "server_url": SERVER_URL,
                },
                "summary": {
                    "overall_brier_score": avg_brier if successful else None,
                    "min_brier": min(brier_scores) if successful else None,
                    "max_brier": max(brier_scores) if successful else None,
                    "median_brier": sorted(brier_scores)[len(brier_scores)//2] if successful else None,
                },
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\n💾 Raw results saved to: {json_path}")

    # Generate and save report
    if successful:
        report = generate_report(results, total_duration)
        report_path = RESULTS_DIR / "backtest_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"📝 Report saved to: {report_path}")

    print("\n✨ Backtest complete!")
    return results


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(run_backtest())
