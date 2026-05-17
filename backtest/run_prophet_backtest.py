"""
Prophet Arena Sample-Resolved Dataset Backtest

Runs the agent against the ACTUAL Prophet Arena sample-resolved dataset
using their exact format and scoring formula. This is the most realistic
test possible before the competition.

Usage:
    1. Start server: venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8888
    2. Run: venv\Scripts\python.exe backtest\run_prophet_backtest.py
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Fix Windows console encoding for emoji/unicode
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import httpx

SERVER_URL = "http://127.0.0.1:8888"
PREDICT_ENDPOINT = f"{SERVER_URL}/predict"
TIMEOUT_SECONDS = 180  # 3 min per event (generous)
DELAY_BETWEEN_EVENTS = 2
RESULTS_DIR = Path(__file__).parent / "results"
DATASET_FILE = Path(__file__).parent / "datasets" / "sample-resolved.jsonl"


def load_resolved_tasks() -> List[Dict]:
    """Load tasks from the sample-resolved dataset that have resolved outcomes."""
    if not DATASET_FILE.exists():
        print(f"❌ Dataset not found: {DATASET_FILE}")
        print("   Run: venv\\Scripts\\python.exe fetch_datasets.py")
        sys.exit(1)

    tasks = []
    with open(DATASET_FILE, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                task = json.loads(line)
            except json.JSONDecodeError:
                # Try fixing common encoding issues
                line = line.replace('\x93', '"').replace('\x94', '"').replace('\x92', "'")
                try:
                    task = json.loads(line)
                except json.JSONDecodeError:
                    continue
            # Only include tasks with resolved outcomes
            if task.get("resolved_outcome") and task["resolved_outcome"].get("value"):
                tasks.append(task)

    return tasks


def compute_brier_score(predicted_probs: Dict[str, float], resolved_values: List[str], outcomes: List[str]) -> float:
    """Compute Brier score using Prophet Arena's exact formula.

    Formula: sum((p_i - outcome_i)^2) across all outcome probabilities
    where outcome_i = 1 if outcome resolved, 0 otherwise.

    Note: Prophet Arena normalizes probabilities before scoring,
    so we normalize here too.
    """
    # Normalize predictions to sum to 1
    total = sum(predicted_probs.values())
    if total > 0:
        normalized = {k: v / total for k, v in predicted_probs.items()}
    else:
        n = len(outcomes)
        normalized = {o: 1.0 / n for o in outcomes}

    brier = 0.0
    for outcome in outcomes:
        p = normalized.get(outcome, 0.0)
        actual = 1.0 if outcome in resolved_values else 0.0
        brier += (p - actual) ** 2

    return brier


async def check_server() -> bool:
    """Check if the prediction server is running."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{SERVER_URL}/health")
            return r.status_code == 200
    except Exception:
        return False


async def predict_task(client: httpx.AsyncClient, task: Dict) -> Dict:
    """Send a task to the prediction API in Prophet Arena format."""
    start = time.time()

    # Send the task as-is (our API handles the task_id format)
    try:
        r = await client.post(PREDICT_ENDPOINT, json=task, timeout=TIMEOUT_SECONDS)
        duration = time.time() - start

        if r.status_code == 200:
            data = r.json()
            probs = {p["market"]: p["probability"] for p in data["probabilities"]}
            return {"probs": probs, "duration": duration, "error": None}
        else:
            return {"probs": {}, "duration": duration, "error": f"HTTP {r.status_code}: {r.text[:100]}"}

    except httpx.TimeoutException:
        return {"probs": {}, "duration": time.time() - start, "error": "Timeout"}
    except Exception as e:
        return {"probs": {}, "duration": time.time() - start, "error": str(e)}


async def run_backtest():
    """Run the full backtest against sample-resolved dataset."""
    print("=" * 70)
    print("  PROPHET ARENA — SAMPLE-RESOLVED DATASET BACKTEST")
    print("=" * 70)
    print()

    # Load tasks
    tasks = load_resolved_tasks()
    print(f"📋 Loaded {len(tasks)} resolved tasks from sample-resolved dataset")
    print()

    # Check server
    print("🔍 Checking server...")
    if not await check_server():
        print("❌ Server not available at", SERVER_URL)
        print("   Start with: venv\\Scripts\\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8888")
        sys.exit(1)
    print("✅ Server is running")
    print("-" * 70)

    # Run predictions
    results = []
    start_time = time.time()

    async with httpx.AsyncClient() as client:
        for i, task in enumerate(tasks, 1):
            title = task["title"][:60]
            outcomes = task["outcomes"]
            resolved = task["resolved_outcome"]["value"]

            print(f"\n[{i}/{len(tasks)}] {title}...")
            print(f"   Outcomes: {outcomes[:5]}{'...' if len(outcomes) > 5 else ''}")
            print(f"   Resolved: {resolved}")

            result = await predict_task(client, task)

            if result["error"]:
                print(f"   ❌ ERROR: {result['error'][:80]}")
                results.append({
                    "task_id": task["task_id"],
                    "title": task["title"],
                    "outcomes": outcomes,
                    "resolved": resolved,
                    "predicted_probs": {},
                    "brier_score": None,
                    "duration": result["duration"],
                    "error": result["error"],
                })
            else:
                probs = result["probs"]
                brier = compute_brier_score(probs, resolved, outcomes)

                # Show prediction vs actual
                resolved_prob = sum(probs.get(r, 0) for r in resolved)
                print(f"   ✅ Brier: {brier:.4f} | P(resolved): {resolved_prob:.2%} | {result['duration']:.1f}s")

                results.append({
                    "task_id": task["task_id"],
                    "title": task["title"],
                    "outcomes": outcomes,
                    "resolved": resolved,
                    "predicted_probs": probs,
                    "brier_score": brier,
                    "duration": result["duration"],
                    "error": None,
                })

            if i < len(tasks):
                await asyncio.sleep(DELAY_BETWEEN_EVENTS)

    total_duration = time.time() - start_time

    # Compute summary
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    successful = [r for r in results if r["brier_score"] is not None]
    failed = [r for r in results if r["error"]]

    if successful:
        brier_scores = [r["brier_score"] for r in successful]
        avg_brier = sum(brier_scores) / len(brier_scores)

        # Uniform baseline
        uniform_briers = []
        for r in successful:
            n = len(r["outcomes"])
            uniform_p = 1.0 / n
            ub = sum((uniform_p - (1.0 if o in r["resolved"] else 0.0))**2 for o in r["outcomes"])
            uniform_briers.append(ub)
        avg_uniform = sum(uniform_briers) / len(uniform_briers)

        print(f"\n📊 Overall Brier Score: {avg_brier:.4f}")
        print(f"   Uniform Baseline:    {avg_uniform:.4f}")
        print(f"   Improvement:         {((avg_uniform - avg_brier) / avg_uniform * 100):.1f}%")
        print(f"   Successful:          {len(successful)}/{len(results)}")
        print(f"   Failed:              {len(failed)}/{len(results)}")
        print(f"   Total time:          {total_duration:.0f}s ({total_duration/60:.1f} min)")
        print()

        if avg_brier < 0.15:
            print("   ✅ BEATS market average target (0.15)!")
        elif avg_brier < 0.25:
            print("   ⚠️  Close to market average but needs improvement")
        else:
            print("   ❌ Below market average")

        # Best and worst
        sorted_results = sorted(successful, key=lambda x: x["brier_score"])
        print(f"\n   Best 3:")
        for r in sorted_results[:3]:
            print(f"     {r['brier_score']:.4f} — {r['title'][:50]}")
        print(f"\n   Worst 3:")
        for r in sorted_results[-3:]:
            print(f"     {r['brier_score']:.4f} — {r['title'][:50]}")
    else:
        print("\n❌ No successful predictions!")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "prophet_backtest_results.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "dataset": "sample-resolved",
                "total_tasks": len(tasks),
                "successful": len(successful),
                "failed": len(failed),
                "duration_seconds": total_duration,
                "avg_brier_score": avg_brier if successful else None,
                "uniform_baseline": avg_uniform if successful else None,
            },
            "results": results,
        }, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {output}")
    print("\n✨ Backtest complete!")


if __name__ == "__main__":
    asyncio.run(run_backtest())
