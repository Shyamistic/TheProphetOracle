"""
Prophet Forecasting Agent — Live Test with Genuinely Unresolved Events

Tests the agent on events that are TRULY UNCERTAIN — things happening in
the next 1-2 weeks that no model could know the answer to. This gives a
realistic picture of how the agent will perform in the actual competition.

Unlike the historical backtest, these events:
- Have NOT resolved yet
- Are NOT in any model's training data
- Require genuine research and reasoning under uncertainty

Usage:
    venv\\Scripts\\python.exe backtest\\run_live_test.py

Prerequisites:
    - Server running at http://127.0.0.1:8888
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import httpx

SERVER_URL = "http://127.0.0.1:8888"
PREDICT_ENDPOINT = f"{SERVER_URL}/predict"
TIMEOUT_SECONDS = 120
DELAY_BETWEEN_EVENTS = 3
RESULTS_DIR = Path(__file__).parent / "results"


def get_live_events() -> List[Dict]:
    """
    Events that are genuinely unresolved as of May 17, 2026.
    
    These simulate what the Prophet Hacks evaluation will send:
    real questions about the near future that require research
    and probabilistic reasoning.
    """
    events = [
        # ─────────────────────────────────────────────────────
        # ECONOMICS — Near-term market/policy questions
        # ─────────────────────────────────────────────────────
        {
            "event_ticker": "LIVE-FED-JUNE-2026",
            "market_ticker": "FED-RATE-JUNE-2026",
            "title": "Will the Federal Reserve cut interest rates at the June 2026 FOMC meeting?",
            "description": "The Federal Reserve's next FOMC meeting is scheduled for June 17-18, 2026. Markets are watching for signals on rate policy given current economic conditions.",
            "category": "Economics",
            "rules": "Resolves YES if the Federal Reserve announces a rate cut at the June 17-18, 2026 FOMC meeting. Resolves NO if rates are held steady or raised.",
            "close_time": "2026-06-19T00:00:00Z",
            "outcomes": ["Yes", "No"],
        },
        {
            "event_ticker": "LIVE-SP500-5500-MAY",
            "market_ticker": "SP500-5500-MAY-2026",
            "title": "Will the S&P 500 close above 5,500 on May 30, 2026?",
            "description": "The S&P 500 index performance heading into the end of May 2026.",
            "category": "Economics",
            "rules": "Resolves YES if the S&P 500 index closes above 5,500 on May 30, 2026. Resolves NO otherwise.",
            "close_time": "2026-05-31T00:00:00Z",
            "outcomes": ["Yes", "No"],
        },
        {
            "event_ticker": "LIVE-OIL-80-MAY",
            "market_ticker": "WTI-CRUDE-80-MAY-2026",
            "title": "Will WTI crude oil price be above $80 per barrel on May 31, 2026?",
            "description": "WTI crude oil prices have been fluctuating due to OPEC+ decisions and global demand.",
            "category": "Economics",
            "rules": "Resolves YES if WTI crude oil spot price is above $80/barrel at market close on May 31, 2026.",
            "close_time": "2026-05-31T00:00:00Z",
            "outcomes": ["Yes", "No"],
        },

        # ─────────────────────────────────────────────────────
        # GEOPOLITICS — Current affairs
        # ─────────────────────────────────────────────────────
        {
            "event_ticker": "LIVE-UKRAINE-CEASEFIRE-MAY",
            "market_ticker": "UKRAINE-CEASEFIRE-MAY-2026",
            "title": "Will there be a formal ceasefire agreement between Russia and Ukraine before June 1, 2026?",
            "description": "Diplomatic efforts regarding the Russia-Ukraine conflict continue. Various peace proposals have been discussed.",
            "category": "Geopolitics",
            "rules": "Resolves YES if a formal ceasefire agreement is signed by both Russia and Ukraine before June 1, 2026. Resolves NO otherwise.",
            "close_time": "2026-06-01T00:00:00Z",
            "outcomes": ["Yes", "No"],
        },
        {
            "event_ticker": "LIVE-US-CHINA-TARIFFS-MAY",
            "market_ticker": "US-CHINA-TARIFF-REDUCTION-2026",
            "title": "Will the US announce a reduction in tariffs on Chinese goods before June 2026?",
            "description": "US-China trade relations and tariff policies under the current administration.",
            "category": "Geopolitics",
            "rules": "Resolves YES if the US government officially announces a reduction in tariffs on Chinese imports before June 1, 2026.",
            "close_time": "2026-06-01T00:00:00Z",
            "outcomes": ["Yes", "No"],
        },

        # ─────────────────────────────────────────────────────
        # TECHNOLOGY — Product/company events
        # ─────────────────────────────────────────────────────
        {
            "event_ticker": "LIVE-APPLE-WWDC-AI",
            "market_ticker": "APPLE-WWDC-2026-AI",
            "title": "Will Apple announce a major new AI feature at WWDC 2026?",
            "description": "Apple's WWDC 2026 is expected in June 2026. Apple has been expanding its AI capabilities.",
            "category": "Technology",
            "rules": "Resolves YES if Apple announces a significant new AI-powered feature or product at WWDC 2026 (scheduled for June 2026).",
            "close_time": "2026-06-15T00:00:00Z",
            "outcomes": ["Yes", "No"],
        },
        {
            "event_ticker": "LIVE-OPENAI-GPT5-MAY",
            "market_ticker": "OPENAI-GPT5-RELEASE-MAY-2026",
            "title": "Will OpenAI release GPT-5 before June 1, 2026?",
            "description": "OpenAI has been developing next-generation models. GPT-5 has been anticipated by the AI community.",
            "category": "Technology",
            "rules": "Resolves YES if OpenAI publicly releases a model officially named GPT-5 before June 1, 2026.",
            "close_time": "2026-06-01T00:00:00Z",
            "outcomes": ["Yes", "No"],
        },
        {
            "event_ticker": "LIVE-NVIDIA-STOCK-MAY",
            "market_ticker": "NVDA-ABOVE-150-MAY-2026",
            "title": "Will Nvidia stock (NVDA) close above $150 on May 30, 2026?",
            "description": "Nvidia stock price performance given AI demand and market conditions.",
            "category": "Technology",
            "rules": "Resolves YES if NVDA stock closes above $150 per share on May 30, 2026.",
            "close_time": "2026-05-31T00:00:00Z",
            "outcomes": ["Yes", "No"],
        },

        # ─────────────────────────────────────────────────────
        # SPORTS — Upcoming events
        # ─────────────────────────────────────────────────────
        {
            "event_ticker": "LIVE-NBA-FINALS-2026",
            "market_ticker": "NBA-FINALS-2026-TEAM",
            "title": "Will the Boston Celtics reach the 2026 NBA Finals?",
            "description": "The 2026 NBA Playoffs are underway. The Celtics are the defending champions.",
            "category": "Sports",
            "rules": "Resolves YES if the Boston Celtics reach the 2026 NBA Finals. Resolves NO otherwise.",
            "close_time": "2026-06-15T00:00:00Z",
            "outcomes": ["Yes", "No"],
        },
        {
            "event_ticker": "LIVE-FRENCH-OPEN-2026",
            "market_ticker": "FRENCH-OPEN-MENS-2026",
            "title": "Will Jannik Sinner win the 2026 French Open Men's Singles?",
            "description": "The 2026 French Open is scheduled for late May to early June at Roland Garros.",
            "category": "Sports",
            "rules": "Resolves YES if Jannik Sinner wins the 2026 French Open Men's Singles title.",
            "close_time": "2026-06-08T00:00:00Z",
            "outcomes": ["Yes", "No"],
        },

        # ─────────────────────────────────────────────────────
        # SCIENCE — Near-term launches/events
        # ─────────────────────────────────────────────────────
        {
            "event_ticker": "LIVE-SPACEX-STARSHIP-MAY",
            "market_ticker": "SPACEX-STARSHIP-LAUNCH-MAY-2026",
            "title": "Will SpaceX launch a Starship flight in May 2026?",
            "description": "SpaceX has been conducting regular Starship test flights. The cadence of launches in 2026.",
            "category": "Science",
            "rules": "Resolves YES if SpaceX launches a Starship vehicle (any flight test) during May 2026.",
            "close_time": "2026-06-01T00:00:00Z",
            "outcomes": ["Yes", "No"],
        },
        {
            "event_ticker": "LIVE-ARTEMIS-II-2026",
            "market_ticker": "NASA-ARTEMIS-II-2026",
            "title": "Will NASA launch Artemis II before July 2026?",
            "description": "NASA's Artemis II crewed lunar flyby mission has been delayed multiple times. Current schedule targets 2026.",
            "category": "Science",
            "rules": "Resolves YES if NASA launches Artemis II before July 1, 2026.",
            "close_time": "2026-07-01T00:00:00Z",
            "outcomes": ["Yes", "No"],
        },

        # ─────────────────────────────────────────────────────
        # GENERAL — Miscellaneous current events
        # ─────────────────────────────────────────────────────
        {
            "event_ticker": "LIVE-US-DEBT-CEILING-2026",
            "market_ticker": "US-DEBT-CEILING-RAISE-2026",
            "title": "Will the US raise or suspend the debt ceiling before June 2026?",
            "description": "The US debt ceiling has been a recurring political issue. Congress must act to avoid default.",
            "category": "General",
            "rules": "Resolves YES if the US Congress passes legislation to raise or suspend the debt ceiling before June 1, 2026.",
            "close_time": "2026-06-01T00:00:00Z",
            "outcomes": ["Yes", "No"],
        },
        {
            "event_ticker": "LIVE-GLOBAL-TEMP-MAY-2026",
            "market_ticker": "GLOBAL-TEMP-RECORD-MAY-2026",
            "title": "Will May 2026 be the hottest May on record globally?",
            "description": "Global temperatures have been setting records. Climate monitoring agencies track monthly averages.",
            "category": "Science",
            "rules": "Resolves YES if May 2026 is declared the hottest May on record by NASA GISS or NOAA.",
            "close_time": "2026-06-15T00:00:00Z",
            "outcomes": ["Yes", "No"],
        },
    ]
    return events


async def run_live_test():
    """Run predictions on genuinely unresolved events."""
    print("=" * 70)
    print("  PROPHET FORECASTING AGENT — LIVE TEST (UNRESOLVED EVENTS)")
    print("=" * 70)
    print()
    print("⚠️  These events have NOT resolved yet.")
    print("   This test measures the agent's behavior on genuine uncertainty.")
    print("   We cannot compute Brier scores — only analyze prediction quality.")
    print()

    events = get_live_events()
    print(f"📋 Loaded {len(events)} unresolved events")
    print()

    # Check server
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{SERVER_URL}/health")
            if r.status_code != 200:
                print(f"❌ Server unhealthy: {r.status_code}")
                sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot reach server: {e}")
        print(f"   Start it with: venv\\Scripts\\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8888")
        sys.exit(1)

    print("✅ Server is healthy")
    print("-" * 70)

    results = []
    start_time = time.time()

    async with httpx.AsyncClient() as client:
        for i, event in enumerate(events, 1):
            print(f"\n[{i}/{len(events)}] {event['title'][:65]}...")

            payload = {k: v for k, v in event.items()}  # All fields (no resolved_outcome here)

            try:
                r = await client.post(PREDICT_ENDPOINT, json=payload, timeout=TIMEOUT_SECONDS)
                duration = time.time() - start_time

                if r.status_code == 200:
                    data = r.json()
                    probs = {p["market"]: p["probability"] for p in data["probabilities"]}

                    # Analyze the prediction
                    max_outcome = max(probs, key=probs.get)
                    max_prob = probs[max_outcome]
                    confidence = "HIGH" if max_prob > 0.75 else "MEDIUM" if max_prob > 0.60 else "LOW"

                    print(f"   → {max_outcome}: {max_prob:.1%} (confidence: {confidence})")
                    for outcome, prob in probs.items():
                        if outcome != max_outcome:
                            print(f"     {outcome}: {prob:.1%}")

                    results.append({
                        "event_ticker": event["event_ticker"],
                        "title": event["title"],
                        "category": event["category"],
                        "outcomes": event["outcomes"],
                        "predicted_probs": probs,
                        "favored_outcome": max_outcome,
                        "confidence": max_prob,
                        "confidence_level": confidence,
                        "error": None,
                    })
                else:
                    print(f"   ❌ HTTP {r.status_code}")
                    results.append({
                        "event_ticker": event["event_ticker"],
                        "title": event["title"],
                        "category": event["category"],
                        "predicted_probs": {},
                        "error": f"HTTP {r.status_code}",
                    })
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append({
                    "event_ticker": event["event_ticker"],
                    "title": event["title"],
                    "category": event["category"],
                    "predicted_probs": {},
                    "error": str(e),
                })

            if i < len(events):
                await asyncio.sleep(DELAY_BETWEEN_EVENTS)

    total_duration = time.time() - start_time

    # Analysis
    print("\n" + "=" * 70)
    print("  LIVE TEST RESULTS")
    print("=" * 70)

    successful = [r for r in results if not r.get("error")]
    print(f"\n📊 {len(successful)}/{len(results)} events predicted successfully")
    print(f"⏱️  Total time: {total_duration:.0f}s ({total_duration/60:.1f} min)")

    if successful:
        # Confidence distribution
        high_conf = [r for r in successful if r["confidence"] > 0.75]
        med_conf = [r for r in successful if 0.60 < r["confidence"] <= 0.75]
        low_conf = [r for r in successful if r["confidence"] <= 0.60]

        print(f"\n📈 Confidence Distribution:")
        print(f"   HIGH (>75%):  {len(high_conf)} events")
        print(f"   MEDIUM (60-75%): {len(med_conf)} events")
        print(f"   LOW (≤60%):   {len(low_conf)} events")

        # Check for 50/50 predictions (potential fallbacks)
        uniform = [r for r in successful if abs(r["confidence"] - 0.5) < 0.01]
        if uniform:
            print(f"\n⚠️  {len(uniform)} events returned ~50/50 (possible fallback):")
            for r in uniform:
                print(f"   - {r['title'][:60]}")

        # Average confidence
        avg_conf = sum(r["confidence"] for r in successful) / len(successful)
        print(f"\n📊 Average confidence in favored outcome: {avg_conf:.1%}")

        # Ideal range for uncertain events: 55-75%
        in_ideal_range = [r for r in successful if 0.55 <= r["confidence"] <= 0.75]
        print(f"   Events in ideal uncertainty range (55-75%): {len(in_ideal_range)}/{len(successful)}")

        if avg_conf > 0.85:
            print("   ⚠️  Agent may be overconfident on genuinely uncertain events")
        elif avg_conf < 0.55:
            print("   ⚠️  Agent may be underconfident — not differentiating well")
        else:
            print("   ✅ Confidence levels look reasonable for uncertain events")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "live_test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "note": "These events are UNRESOLVED. No Brier scores can be computed.",
                "total_events": len(events),
                "successful": len(successful),
                "duration_seconds": total_duration,
            },
            "results": results,
        }, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")
    print("\n✨ Live test complete! Check back after events resolve to compute actual Brier scores.")


if __name__ == "__main__":
    asyncio.run(run_live_test())
