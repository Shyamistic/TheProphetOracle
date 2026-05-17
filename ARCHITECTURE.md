# Architecture Document — Prophet Forecasting Agent

## System Overview

A multi-model AI forecasting agent that predicts real-world event outcomes for the Prophet Arena platform. Built for the Prophet Hacks hackathon (May 16-17, 2026, University of Chicago). Evaluated over a 14-day window (May 17 - May 31, 2026) on Brier score against Kalshi prediction market baselines.

## Design Philosophy

1. **Market-anchored predictions** — Start from Kalshi market prices as a Bayesian prior. Only deviate when research provides strong evidence the market is wrong.
2. **Multi-model consensus** — Run 3 frontier models in parallel, take the median. Reduces individual model biases and hallucinations.
3. **Structured reasoning** — Force YES/NO thesis analysis before committing to probabilities. Counters confirmation bias.
4. **Never fail** — Triple search fallback, graceful model degradation, always return a valid prediction.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        REQUEST HANDLING                               │
│                                                                       │
│  POST /predict ──→ Format Normalization ──→ Request Validation       │
│  (Prophet Arena format: task_id, predict_by, context, outcomes)       │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
┌───────────────────────────────▼───────────────────────────────────────┐
│                        MARKET DATA                                     │
│                                                                       │
│  Fetch Kalshi prices (public API, no auth required)                   │
│  GET https://external-api.kalshi.com/trade-api/v2/markets/{ticker}    │
│  → Provides Bayesian prior for predictions                            │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
┌───────────────────────────────▼───────────────────────────────────────┐
│                        ROUTING & CACHE                                 │
│                                                                       │
│  Category detection (Sports/Economics/Geopolitics/Tech/Science)        │
│  Complexity classification (LOW/MEDIUM/HIGH)                          │
│  Cache check (SQLite, 6-hour TTL)                                     │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
┌───────────────────────────────▼───────────────────────────────────────┐
│                        RESEARCH PIPELINE                               │
│                                                                       │
│  Triple-fallback search:                                              │
│    1. Tavily (advanced, news/finance topics, include_answer)          │
│    2. Serper.dev (Google SERP, 2500 free searches)                    │
│    3. DuckDuckGo (unlimited, rate-limited)                            │
│                                                                       │
│  Entity extraction → Query generation → Search → Evidence filtering   │
│  (90-day recency + corroboration check)                               │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
┌───────────────────────────────▼───────────────────────────────────────┐
│                     3-MODEL ENSEMBLE REASONING                         │
│                                                                       │
│  Structured prompt (FutureSearch-inspired):                           │
│    1. BASE RATES — Historical frequency                               │
│    2. CURRENT STATE — What's happening now                            │
│    3. YES THESIS — Steelman the bull case                             │
│    4. NO THESIS — Steelman the bear case                              │
│    5. KEY FACTORS — What determines the outcome                       │
│    6. SYNTHESIS — Final probability assignment                        │
│                                                                       │
│  Run in PARALLEL via OpenRouter:                                      │
│    ┌──────────────────────┐                                           │
│    │ Claude Sonnet 4      │──┐                                        │
│    ├──────────────────────┤  │                                        │
│    │ Gemini 3.1 Pro       │──┼──→ MEDIAN per outcome                  │
│    ├──────────────────────┤  │                                        │
│    │ GPT-5                │──┘                                        │
│    └──────────────────────┘                                           │
│                                                                       │
│  If models disagree >15% on any outcome:                              │
│    → Qwen 72B (Featherless) as 4th tiebreaker                        │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
┌───────────────────────────────▼───────────────────────────────────────┐
│                     SUPERVISOR & CALIBRATION                           │
│                                                                       │
│  Supervisor Agent (if market prices available):                       │
│    - Reconciles our prediction with market consensus                  │
│    - Single LLM call with 30s timeout                                 │
│    - Fallback: weighted blend (70% ours, 30% market)                  │
│                                                                       │
│  Calibration:                                                         │
│    - With market: skip Platt, light anchor toward market              │
│    - Without market: Platt scaling (coefficient 1.5)                  │
│                                                                       │
│  Confidence Check:                                                    │
│    - If deviation from market < 5%: use market directly               │
│    - "Only predict when confident" (organizer's strategy)             │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
┌───────────────────────────────▼───────────────────────────────────────┐
│                        VALIDATION & RESPONSE                          │
│                                                                       │
│  Validate: outcome count, range [0.01, 0.99], sum to 1.0             │
│  Correct: clamp, normalize, redistribute                              │
│  Fallback: uniform distribution if all else fails                     │
│                                                                       │
│  Response: {"probabilities": [{"market": "X", "probability": 0.7}]}  │
└───────────────────────────────────────────────────────────────────────┘
```

## Component Map

| File | Responsibility | Key Dependencies |
|------|---------------|-----------------|
| `src/api.py` | HTTP endpoint, orchestration | FastAPI, all other modules |
| `src/ensemble_reasoner.py` | 3-model ensemble + structured prompting | OpenAI SDK (OpenRouter) |
| `src/research.py` | Multi-agent web search pipeline | Tavily, AsyncOpenAI |
| `src/supervisor.py` | Market-anchored reconciliation | OpenAI SDK |
| `src/calibrator.py` | Platt scaling + market anchoring | math |
| `src/validator.py` | Response format validation/correction | — |
| `src/router.py` | Category/complexity classification | — |
| `src/search_client.py` | Triple-fallback search (Tavily/Serper/DDG) | httpx, tavily, duckduckgo_search |
| `src/market_data.py` | Kalshi public API price fetcher | httpx |
| `src/cache.py` | SQLite prediction cache (6hr TTL) | aiosqlite |
| `src/cost_tracker.py` | API cost monitoring + budget enforcement | sqlite3 |
| `src/config.py` | Environment variable management | pydantic-settings |
| `src/models.py` | Pydantic models + data classes | pydantic |
| `src/aggregator.py` | Mean/weighted aggregation | — |

## API Keys & Services

| Service | Purpose | Cost Model |
|---------|---------|-----------|
| OpenRouter | Claude + Gemini + GPT-5 ensemble | $47.51 remaining (pay-per-token) |
| Featherless | Qwen 72B tiebreaker | Flat rate (concurrency-limited) |
| Tavily | Primary web search | 823 credits remaining |
| Serper.dev | Google SERP fallback | 2,500 free searches |
| DuckDuckGo | Tertiary fallback | Free (rate-limited) |
| Kalshi | Market price data | Free public API |

## Scoring & Competition Context

- **Metric:** Brier score (lower is better). Prophet Arena reports 1 - Brier (higher is better).
- **Baseline:** Kalshi market prices (1 - Brier ≈ 0.8506)
- **Target:** Beat market by >3.79% (current #1 agent: Gemini 3 at +0.0379)
- **Events:** Binary and multi-outcome, sourced from Kalshi, closing in 2 days to 2 weeks
- **Volume:** Max 200 predictions over 14 days
- **Timeout:** 10 minutes per event (we average 40 seconds)

## Competitive Advantages Over Leaderboard Agents

| Our Feature | Leaderboard Agents | Edge |
|-------------|-------------------|------|
| 3-model ensemble (median) | Single model | Reduces variance |
| Market price anchoring | No market data | Never worse than baseline |
| YES/NO thesis prompting | Standard prompting | Counters confirmation bias |
| Confidence threshold | Always deviate | Avoids adding noise |
| Triple search fallback | Single search | Never fails |
| Disagreement-triggered tiebreaker | N/A | Resolves uncertainty |

## Deployment

- **Platform:** AWS EC2 (t3.small, Ubuntu 22.04)
- **Process manager:** systemd (auto-restart on crash)
- **Port:** 8080 (configurable via PROPHET_PORT)
- **Health check:** GET /health
- **Monitoring:** GET /costs, GET /dashboard (planned)

## Future Refinements

1. **Polymarket cross-reference** — Fetch prices from both Kalshi and Polymarket for stronger market prior
2. **Time-to-resolution weighting** — Anchor more to market for events closing soon
3. **Category-specific confidence** — Trust research more for politics/economics, less for sports/entertainment
4. **Historical calibration** — Track our predictions vs outcomes and adjust calibration parameters
5. **Deliberation round** — Let models review each other's forecasts before final median (per arxiv 2512.22625)
