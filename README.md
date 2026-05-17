# Prophet Forecasting Agent

A multi-model AI forecasting system built for [Prophet Hacks](https://prophethacks.com) (May 16-17, 2026, University of Chicago). The agent receives prediction market events, researches them using web search, reasons about outcomes using a 3-model ensemble (Claude + Gemini + GPT-5), and returns calibrated probability predictions.

**Scored on Brier score** against Kalshi market baselines over a 14-day evaluation window (May 17 - May 31, 2026).

## Key Features

- **3-Model Ensemble** — Claude Sonnet 4, Gemini 3.1 Pro, GPT-5 run in parallel, median probability taken
- **Market-Anchored Predictions** — Fetches live Kalshi prices as Bayesian prior
- **Structured YES/NO Thesis Reasoning** — Forces consideration of both sides before committing
- **Triple Search Fallback** — Tavily → Serper.dev (Google) → DuckDuckGo
- **Featherless Tiebreaker** — Qwen 72B resolves disagreements when models diverge >15%
- **100% Completion Rate** — Graceful degradation ensures every event gets a prediction

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/prophethacks.git
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
| POST | `/predict` | Predict probabilities for a single event |
| POST | `/predict/batch` | Batch prediction (multiple events) |
| GET | `/health` | Health check (API connectivity) |
| GET | `/costs` | Cost tracking summary |

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

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design, pipeline diagram, and competitive analysis.

### Pipeline Summary

```
Event → Market Price Fetch → Research (Tavily/Serper/DDG)
  → 3-Model Ensemble (Claude + Gemini + GPT-5) → Median
  → Supervisor Reconciliation → Calibration → Validate → Respond
```

## Performance

| Metric | Value |
|--------|-------|
| Avg prediction time | 39 seconds |
| Success rate | 100% |
| Models in ensemble | 3 (+ tiebreaker) |
| Search sources | 3 (triple fallback) |
| Budget used | $2.49 / $47.51 remaining |

## Environment Variables

See [.env.example](.env.example) for all configuration options.

### Required
- `PROPHET_ANTHROPIC_API_KEY` — OpenRouter API key
- `PROPHET_TAVILY_API_KEY` — Tavily search API key

### Optional (Recommended)
- `PROPHET_FEATHERLESS_API_KEY` — Featherless AI (ensemble tiebreaker)
- `PROPHET_SERPER_API_KEY` — Serper.dev Google search fallback
- `PROPHET_ENSEMBLE_MODEL_1/2/3` — Model IDs for the ensemble

## Deployment

### AWS EC2 (Recommended)
```bash
# On EC2 (Ubuntu 22.04, t3.small)
sudo apt update && sudo apt install python3.11 python3.11-venv -y
git clone <repo-url> && cd prophethacks
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Edit with your keys
uvicorn src.api:app --host 0.0.0.0 --port 8080
```

### Docker
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
prophet forecast predict --events events.json --agent-url http://YOUR_HOST:8080/predict
```

## License

Built for Prophet Hacks 2026. MIT License.
