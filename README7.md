# Prophet Forecasting Agent

A multi-model AI forecasting system built for [Prophet Hacks](https://prophethacks.com) (May 16-17, 2026, University of Chicago). The agent receives prediction market events, researches them using web search, reasons about outcomes using a 3-model ensemble with logit-space averaging, and returns calibrated probability predictions anchored to live market data.

**Scored on Brier score** against Kalshi market baselines over a 14-day evaluation window (May 17 – May 31, 2026).

**Live:** http://3.81.84.99:8080 | **GitHub:** https://github.com/Shyamistic/TheProphetOracle

## Key Features

- **3-Model Ensemble + Tiebreaker** — Claude Sonnet 4, Gemini 3.1 Pro, GPT-5 via OpenRouter; Featherless Qwen 72B as 4th tiebreaker when models disagree >15%
- **Logit-Space Averaging (BLF)** — Aggregates in log-odds space instead of simple median, per Murphy 2026
- **Adaptive Shrinkage** — 5% when models agree, 10% moderate disagreement, 15% high disagreement
- **Market-Anchored Predictions** — Kalshi + Polymarket cross-reference as Bayesian prior
- **Time-to-Resolution Anchoring** — 80% anchor weight for 2-day events, 30% for 2-week events
- **Category-Specific Multipliers** — Sports 2.0x, Entertainment 1.5x, Geopolitics 0.7x
- **Confidence Threshold** — Uses market directly when no edge detected (<5% deviation)
- **Structured YES/NO Thesis Reasoning** — With Resolution Analysis step and category-specific base rate priors
- **Counter-Evidence Search** — Triggered for strong predictions (>70%)
- **Iterative Research** — Additional research rounds for moderate confidence (40-70%)
- **Triple Search Fallback** — Tavily → Serper.dev (Google) → DuckDuckGo
- **Non-Mutually-Exclusive Handling** — Top-K events scored independently
- **Supervisor Reconciliation Agent** — Final sanity check against market consensus
- **SQLite Caching** — 6-hour TTL with cost tracking
- **Live Monitoring Dashboard** — Real-time prediction tracking at `/dashboard`
- **100% Completion Rate** — Graceful degradation ensures every event gets a prediction

## Quick Start

```bash
# Clone
git clone https://github.com/Shyamistic/TheProphetOracle.git
cd prophethacks

# Setup
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys

# Run
uvicorn src.api:app --host 0.0.0.0 --port 8080
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Main prediction endpoint (Prophet Arena compatible) |
| GET | `/health` | Health check (API connectivity) |
| GET | `/costs` | Cost tracking summary |
| GET | `/dashboard` | Live monitoring UI |
| GET | `/logs` | Prediction history |

## Input Format (Prophet Arena)

```json
{
  "task_id": "KXNBASERIES-26TEST",
  "title": "Who will win the NBA Eastern Conference Finals?",
  "outcomes": ["New York Knicks", "Cleveland Cavaliers"],
  "predict_by": "2026-06-01T00:00:00Z",
  "source": "KXNBASERIES-26TEST",
  "context": "Predict the winner of the series.",
  "metadata": {"category": "Sports"}
}
```

## Output Format

```json
{
  "probabilities": [
    {"market": "New York Knicks", "probability": 0.743},
    {"market": "Cleveland Cavaliers", "probability": 0.257}
  ]
}
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design, pipeline diagram, and research foundations.

### Pipeline Summary

```
Event → Market Fetch (Kalshi + Polymarket) → Research (Tavily/Serper/DDG)
  → Counter-evidence search (if >70%) / Iterative research (if 40-70%)
  → 3-Model Ensemble (Claude + Gemini + GPT-5) → Logit-Space Averaging (BLF)
  → Qwen 72B Tiebreaker (if disagreement >15%)
  → Adaptive Shrinkage → Supervisor Reconciliation
  → Time-to-Resolution Anchoring → Category Multipliers → Validate → Respond
```

## Performance

| Metric | Value |
|--------|-------|
| Avg prediction time | 60-90 seconds |
| Success rate | 100% |
| Models in ensemble | 3 (+ Qwen 72B tiebreaker) |
| Search sources | 3 (triple fallback) |
| Budget remaining | $47.51 of $50 |
| Estimated full eval cost | $16-20 for 14-day window |

## Research Foundations

| Paper / System | Technique Adopted |
|----------------|-------------------|
| BLF (Kevin Murphy, 2026) | Logit-space averaging, iterative belief updating |
| AIA Forecaster | Multi-agent search, supervisor reconciliation |
| FutureSearch | YES/NO thesis prompting, 6 research agents, median of 3 models |
| Prophet Arena Leaderboard | Market anchoring as the key competitive edge |

## Environment Variables

See [.env.example](.env.example) for all configuration options.

### Required
- `PROPHET_ANTHROPIC_API_KEY` — OpenRouter API key
- `PROPHET_TAVILY_API_KEY` — Tavily search API key

### Optional (Recommended)
- `PROPHET_FEATHERLESS_API_KEY` — Featherless AI (Qwen 72B tiebreaker)
- `PROPHET_SERPER_API_KEY` — Serper.dev Google search fallback
- `PROPHET_ENSEMBLE_MODEL_1/2/3` — Model IDs for the ensemble

## Deployment

**Production:** AWS EC2 t2.small (Ubuntu 26.04) — http://3.81.84.99:8080

```bash
# EC2 setup
sudo apt update && sudo apt install python3.11 python3.11-venv -y
git clone https://github.com/Shyamistic/TheProphetOracle.git && cd prophethacks
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Edit with your keys
uvicorn src.api:app --host 0.0.0.0 --port 8080
```

Runs as a **systemd service** with auto-restart on crash.

### Docker (Alternative)
```bash
docker build -t prophet-agent .
docker run -p 8080:8080 --env-file .env prophet-agent
```

## Testing

```bash
# Unit tests
pytest tests/ -q

# Backtest against Prophet Arena sample dataset
python backtest/run_prophet_backtest.py

# Stress test (20 diverse events)
python backtest/run_stress_test.py
```

## SDK Compatibility

```bash
prophet forecast predict --events events.json --agent-url http://3.81.84.99:8080/predict
```

## License

Built for Prophet Hacks 2026. MIT License.
