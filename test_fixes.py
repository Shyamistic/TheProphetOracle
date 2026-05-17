"""Test all 4 fixes with tricky events."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import httpx
import json
import time

SERVER = "http://127.0.0.1:8888"

# Test events designed to exercise each fix
events = [
    # Fix 1: Resolution analysis (ambiguous rules)
    {
        "task_id": "FIX1-RESOLUTION",
        "title": "Will the US impose new tariffs on EU goods before June 1, 2026?",
        "outcomes": ["Yes", "No"],
        "predict_by": "2026-06-01T00:00:00Z",
        "context": "Resolves YES only if the President signs an executive order or Congress passes legislation imposing NEW tariffs (increases to existing tariffs do not count). Temporary tariffs that are announced but immediately stayed by court order do not count.",
        "metadata": {"category": "Geopolitics"}
    },
    # Fix 2: Overconfidence test (uncertain event)
    {
        "task_id": "FIX2-UNCERTAIN",
        "title": "Will Bitcoin be above $110,000 on May 31, 2026?",
        "outcomes": ["Yes", "No"],
        "predict_by": "2026-05-31T00:00:00Z",
        "context": "Resolves YES if BTC/USD spot price on Coinbase is above $110,000 at midnight UTC on May 31, 2026.",
        "metadata": {"category": "Economics"}
    },
    # Fix 4: Sports event (should anchor heavily to market)
    {
        "task_id": "FIX4-SPORTS",
        "title": "Who will win the NBA Eastern Conference Finals 2026?",
        "outcomes": ["New York Knicks", "Indiana Pacers"],
        "predict_by": "2026-06-05T00:00:00Z",
        "context": "Resolves to the winner of the 2026 NBA Eastern Conference Finals series.",
        "metadata": {"category": "Sports"}
    },
]

print("=" * 60)
print("TESTING ALL 4 FIXES")
print("=" * 60)

time.sleep(5)  # Wait for server

for i, event in enumerate(events, 1):
    print(f"\n[{i}/3] {event['title'][:55]}...")
    start = time.time()
    try:
        r = httpx.post(f"{SERVER}/predict", json=event, timeout=300)
        dur = time.time() - start
        if r.status_code == 200:
            data = r.json()
            probs = {p["market"]: p["probability"] for p in data["probabilities"]}
            top = max(probs, key=probs.get)
            print(f"     OK | {dur:.1f}s | {top}: {probs[top]:.1%}")
            print(f"     All: {probs}")
        else:
            print(f"     FAILED: {r.status_code}")
    except Exception as e:
        print(f"     ERROR: {e}")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)
