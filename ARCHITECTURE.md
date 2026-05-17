# Architecture Document — Prophet Forecasting Agent

## System Overview

A multi-model AI forecasting agent that predicts real-world event outcomes for the Prophet Arena platform. Built for Prophet Hacks (May 16-17, 2026, University of Chicago). Evaluated over a 14-day window (May 17 – May 31, 2026) on Brier score against Kalshi prediction market baselines.

**Live endpoint:** http://3.81.84.99:8080  
**Repository:** https://github.com/Shyamistic/TheProphetOracle

## Design Philosophy

1. **Market-anchored predictions** — Start from Kalshi + Polymarket cross-referenced prices as a Bayesian prior. Only deviate when research provides strong evidence the market is wrong.
2. **Logit-space consensus** — Aggregate model outputs in log-odds space (BLF method) rather than simple median. Produces better-calibrated probabilities.
3. **Adaptive confidence** — Shrink toward prior more aggressively when models disagree. Trust the ensemble more when it converges.
4. **Structured reasoning** — Force YES/NO thesis analysis with Resolution Analysis step before committing to probabilities. Counters confirmation bias.
5. **Never fail** — Triple search fallback, graceful model degradation, 100% completion rate with valid predictions.

## Full Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          REQUEST HANDLING                                 │
│                                                                           │
│  POST /predict ──→ Format Normalization ──→ Request Validation           │
│  (Prophet Arena format: task_id, predict_by, context, outcomes)           │
│                                                                           │
│  Additional endpoints:                                                    │
│    GET /health     — API connectivity check                              │
│    GET /costs      — Budget & cost tracking                              │
│    GET /dashboard  — Live monitoring UI                                  │
│    GET /logs       — Prediction history                                  │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼───────────────────────────────────────┐
│                     MARKET DATA (Dual Source)                              │
│                                                                           │
│  ┌─────────────┐    ┌──────────────┐                                     │
│  │   Kalshi    │    │  Polymarket  │                                      │
│  │ Public API  │    │  Cross-ref   │                                      │
│  └──────┬──────┘    └──────┬───────┘                                     │
│         └────────┬─────────┘                                              │
│                  ▼                                                         │
│         Market Consensus Price                                            │
│         (Bayesian prior for predictions)                                  │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼───────────────────────────────────────┐
│                     ROUTING, CACHE & CLASSIFICATION                        │
│                                                                           │
│  Category detection: Sports / Economics / Geopolitics / Tech / Science /  │
│                      Entertainment                                         │
│  Complexity classification: LOW / MEDIUM / HIGH                           │
│  Cache check: SQLite, 6-hour TTL                                         │
│                                                                           │
│  Category-specific base rate priors injected into prompt                  │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼───────────────────────────────────────┐
│                        RESEARCH PIPELINE                                   │
│                                                                           │
│  Triple-fallback search:                                                  │
│    1. Tavily (advanced, news/finance, include_answer)                     │
│    2. Serper.dev (Google SERP, 2500 free searches)                        │
│    3. DuckDuckGo (unlimited, rate-limited)                                │
│                                                                           │
│  Entity extraction → Query generation → Search → Evidence filtering       │
│  (90-day recency + corroboration check)                                   │
│                                                                           │
│  Adaptive research depth:                                                 │
│    • Strong predictions (>70%): Counter-evidence search triggered         │
│    • Moderate confidence (40-70%): Iterative research rounds              │
│    • Low confidence (<40%): Standard single-pass research                 │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼───────────────────────────────────────┐
│                  3-MODEL ENSEMBLE + LOGIT-SPACE AVERAGING                  │
│                                                                           │
│  Structured prompt (FutureSearch + BLF inspired):                         │
│    1. CATEGORY BASE RATES — Historical frequency for this event type     │
│    2. CURRENT STATE — What's happening now (from research)               │
│    3. YES THESIS — Steelman the bull case                                │
│    4. NO THESIS — Steelman the bear case                                 │
│    5. KEY FACTORS — What determines the outcome                          │
│    6. RESOLUTION ANALYSIS — How/when will this resolve                   │
│    7. SYNTHESIS — Final probability assignment                           │
│                                                                           │
│  Run in PARALLEL via OpenRouter:                                          │
│    ┌──────────────────────┐                                               │
│    │ Claude Sonnet 4      │──┐                                            │
│    ├──────────────────────┤  │                                            │
│    │ Gemini 3.1 Pro       │──┼──→ LOGIT-SPACE AVERAGING (BLF)            │
│    ├──────────────────────┤  │    (not simple median)                     │
│    │ GPT-5                │──┘                                            │
│    └──────────────────────┘                                               │
│                                                                           │
│  Disagreement handling (>15% spread):                                     │
│    ┌──────────────────────┐                                               │
│    │ Qwen 72B             │──→ 4th tiebreaker vote                        │
│    │ (Featherless)        │    (included in final aggregation)            │
│    └──────────────────────┘                                               │
│                                                                           │
│  Non-mutually-exclusive outcomes:                                         │
│    → Top-K events scored independently (no forced normalization)          │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼───────────────────────────────────────┐
│                       ADAPTIVE SHRINKAGE                                   │
│                                                                           │
│  Shrinkage toward market prior based on model agreement:                  │
│    • Models agree (spread <5%):       5% shrinkage toward prior          │
│    • Moderate disagreement (5-15%):  10% shrinkage toward prior          │
│    • High disagreement (>15%):       15% shrinkage toward prior          │
│                                                                           │
│  Rationale: When models disagree, we're less certain our ensemble         │
│  has an edge over the market. Shrink more toward the prior.              │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼───────────────────────────────────────┐
│                  SUPERVISOR RECONCILIATION AGENT                           │
│                                                                           │
│  Single LLM call with 30s timeout:                                        │
│    - Reconciles ensemble prediction with market consensus                 │
│    - Reviews research evidence quality                                    │
│    - Checks for logical inconsistencies                                   │
│    - Fallback: weighted blend (70% ours, 30% market)                     │
│                                                                           │
│  Confidence threshold:                                                    │
│    - If deviation from market < 5%: use market directly                  │
│    - "Only predict when confident" strategy                              │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼───────────────────────────────────────┐
│              TIME-TO-RESOLUTION & CATEGORY ANCHORING                       │
│                                                                           │
│  Time-to-resolution adaptive anchoring:                                   │
│    • 2-day events:  80% weight to market anchor                          │
│    • 1-week events: 50% weight to market anchor                          │
│    • 2-week events: 30% weight to market anchor                          │
│                                                                           │
│  Category-specific anchor multipliers:                                    │
│    • Sports:        2.0x (markets very efficient for sports)             │
│    • Entertainment: 1.5x (markets moderately efficient)                  │
│    • Geopolitics:   0.7x (markets often wrong, trust research more)      │
│    • Economics:     1.0x (baseline)                                       │
│    • Tech/Science:  0.8x (markets less informed)                         │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼───────────────────────────────────────┐
│                     VALIDATION & RESPONSE                                  │
│                                                                           │
│  Validate: outcome count, range [0.01, 0.99], sum to 1.0                 │
│  Correct: clamp, normalize, redistribute                                  │
│  Fallback: uniform distribution if all else fails                         │
│                                                                           │
│  Response format:                                                         │
│  {"probabilities": [{"market": "X", "probability": 0.7}, ...]}           │
└───────────────────────────────────────────────────────────────────────────┘
```

## Component Map

| File | Responsibility | Key Dependencies |
|------|---------------|-----------------|
| `src/api.py` | HTTP endpoints, orchestration, dashboard serving | FastAPI, all modules |
| `src/ensemble_reasoner.py` | 3-model ensemble + logit-space averaging + structured prompting | OpenAI SDK (OpenRouter) |
| `src/research.py` | Multi-agent web search pipeline, counter-evidence & iterative research | Tavily, AsyncOpenAI |
| `src/supervisor.py` | Market-anchored reconciliation agent | OpenAI SDK |
| `src/calibrator.py` | Adaptive shrinkage + time-to-resolution anchoring + category multipliers | math |
| `src/validator.py` | Response format validation/correction | — |
| `src/router.py` | Category/complexity classification, base rate priors | — |
| `src/search_client.py` | Triple-fallback search (Tavily/Serper/DDG) | httpx, tavily, duckduckgo_search |
| `src/market_data.py` | Kalshi + Polymarket price fetcher | httpx |
| `src/cache.py` | SQLite prediction cache (6hr TTL) | aiosqlite |
| `src/cost_tracker.py` | API cost monitoring + budget enforcement | sqlite3 |
| `src/dashboard.py` | Live monitoring dashboard UI | FastAPI, Jinja2 |
| `src/config.py` | Environment variable management | pydantic-settings |
| `src/models.py` | Pydantic models + data classes | pydantic |
| `src/aggregator.py` | Logit-space averaging (BLF), adaptive shrinkage | math |

## API Keys & Services

| Service | Purpose | Cost Model |
|---------|---------|-----------|
| OpenRouter | Claude Sonnet 4 + Gemini 3.1 Pro + GPT-5 ensemble | $47.51 remaining (pay-per-token) |
| Featherless | Qwen 72B tiebreaker | Flat rate (concurrency-limited) |
| Tavily | Primary web search | Credits-based |
| Serper.dev | Google SERP fallback | 2,500 free searches |
| DuckDuckGo | Tertiary fallback | Free (rate-limited) |
| Kalshi | Market price data (primary) | Free public API |
| Polymarket | Market price data (cross-reference) | Free public API |

## Research Foundations

### BLF Paper (Kevin Murphy, 2026)
- **Logit-space averaging**: Convert probabilities to log-odds, average, convert back. Produces better-calibrated aggregates than simple mean/median.
- **Iterative belief updating**: Refine predictions through multiple rounds of evidence incorporation.
- **Adaptive shrinkage**: Shrink toward prior proportional to disagreement.

### AIA Forecaster
- **Multi-agent search**: Multiple specialized search agents gather diverse evidence.
- **Supervisor reconciliation**: A meta-agent reviews and reconciles sub-agent outputs.

### FutureSearch
- **YES/NO thesis prompting**: Force structured consideration of both sides.
- **6 research agents**: Parallel evidence gathering from multiple angles.
- **Median of 3 models**: Multi-model consensus reduces individual bias.

### Prophet Arena Leaderboard Analysis
- **Market anchoring is the key edge**: Top agents anchor heavily to market prices.
- **Never be worse than baseline**: If you can't beat the market, match it.
- **Category awareness**: Different event types have different market efficiency levels.

## Scoring & Competition Context

- **Metric:** Brier score (lower is better). Prophet Arena reports 1 - Brier (higher is better).
- **Baseline:** Kalshi market prices (1 - Brier ≈ 0.8506)
- **Target:** Beat market by >3.79% (current #1 agent: Gemini 3 at +0.0379)
- **Events:** Binary and multi-outcome, sourced from Kalshi, closing in 2 days to 2 weeks
- **Volume:** Max 200 predictions over 14 days
- **Timeout:** 10 minutes per event (we average 60-90 seconds)

## Competitive Advantages Over Leaderboard Agents

| Our Feature | Leaderboard Agents | Edge |
|-------------|-------------------|------|
| Logit-space averaging (BLF) | Simple median/mean | Better calibration |
| 3-model ensemble + tiebreaker | Single model | Reduces variance |
| Dual market anchoring (Kalshi + Polymarket) | No/single market data | Stronger prior |
| Time-to-resolution anchoring | Static weights | Adapts to event horizon |
| Category-specific multipliers | One-size-fits-all | Domain-aware confidence |
| Adaptive shrinkage | Fixed blending | Responds to uncertainty |
| YES/NO thesis + Resolution Analysis | Standard prompting | Counters confirmation bias |
| Counter-evidence search (>70%) | No adversarial check | Catches overconfidence |
| Iterative research (40-70%) | Single-pass | Deeper analysis when needed |
| Confidence threshold (<5% edge) | Always deviate | Avoids adding noise |
| Triple search fallback | Single search | Never fails |
| Non-mutually-exclusive handling | Forced normalization | Correct for top-K events |

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Average prediction time | 60-90 seconds |
| Maximum observed time | ~120 seconds |
| Prophet Arena timeout | 10 minutes |
| Success rate | 100% |
| Completion rate | 100% (graceful degradation) |
| Budget remaining | $47.51 of $50 |
| Estimated 14-day cost | $16-20 |

## Deployment

- **Platform:** AWS EC2 t2.small (Ubuntu 26.04)
- **Public IP:** http://3.81.84.99:8080
- **Process manager:** systemd (auto-restart on crash)
- **Port:** 8080 (configurable via PROPHET_PORT)
- **Health check:** GET /health
- **Monitoring:** GET /dashboard (live UI), GET /costs, GET /logs

## Graceful Degradation Strategy

The system is designed to never fail. Each component has fallbacks:

| Component | Primary | Fallback 1 | Fallback 2 | Last Resort |
|-----------|---------|-----------|-----------|-------------|
| Search | Tavily | Serper.dev | DuckDuckGo | Proceed without research |
| Models | 3-model ensemble | 2-model (if one fails) | Single model | Market price only |
| Market data | Kalshi + Polymarket | Kalshi only | Polymarket only | No anchor (uniform prior) |
| Tiebreaker | Qwen 72B | Skip tiebreaker | — | Use 3-model result |
| Supervisor | LLM reconciliation | Weighted blend | — | Use ensemble directly |
| Cache | SQLite read | Recompute | — | — |
