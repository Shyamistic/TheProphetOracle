"""Event routing and complexity classification for the Prophet Forecasting Agent.

Classifies events by category and complexity to determine the appropriate
number of research agents, search queries, and LLM calls to allocate.
"""

from typing import Dict, List

from src.models import ComplexityTier, EventCategory, EventRequest, RoutingConfig


# === Category-Specific Search Strategies ===

CATEGORY_STRATEGIES: Dict[EventCategory, Dict[str, List[str]]] = {
    EventCategory.SPORTS: {
        "source_types": ["team_statistics", "recent_performance", "head_to_head"],
        "query_templates": [
            "{team_a} vs {team_b} recent results {year}",
            "{team_a} season record stats {year}",
            "{sport} {team_a} {team_b} odds prediction",
        ],
        "preferred_sources": ["espn.com", "sports-reference.com", "oddschecker.com"],
    },
    EventCategory.ECONOMICS: {
        "source_types": ["market_indicators", "economic_data", "expert_forecasts"],
        "query_templates": [
            "{indicator} forecast {timeframe}",
            "{entity} economic outlook {year}",
            "Federal Reserve {topic} latest {year}",
        ],
        "preferred_sources": ["fred.stlouisfed.org", "bls.gov", "reuters.com"],
    },
    EventCategory.GEOPOLITICS: {
        "source_types": ["political_analysis", "historical_precedents", "diplomatic_context"],
        "query_templates": [
            "{country} {topic} latest developments {year}",
            "{entity} diplomatic relations {year}",
            "{region} political analysis {topic}",
        ],
        "preferred_sources": ["reuters.com", "apnews.com", "foreignaffairs.com"],
    },
    EventCategory.TECHNOLOGY: {
        "source_types": ["industry_trends", "company_announcements", "technical_feasibility"],
        "query_templates": [
            "{company} {product} announcement {year}",
            "{technology} industry forecast {year}",
            "{company} {topic} latest news",
        ],
        "preferred_sources": ["techcrunch.com", "arstechnica.com", "theverge.com"],
    },
    EventCategory.SCIENCE: {
        "source_types": ["research_publications", "expert_consensus", "experimental_timelines"],
        "query_templates": [
            "{topic} research progress {year}",
            "{field} expert consensus {topic}",
            "{institution} {topic} timeline",
        ],
        "preferred_sources": ["nature.com", "science.org", "arxiv.org"],
    },
    EventCategory.ENTERTAINMENT: {
        "source_types": ["entertainment_news", "charts", "betting_odds"],
        "query_templates": [
            "{title_keywords} latest news {year}",
            "{topic} odds prediction betting",
            "{topic} chart ranking standings",
        ],
        "preferred_sources": ["billboard.com", "variety.com", "spotify.com"],
    },
    EventCategory.GENERAL: {
        "source_types": ["general_news"],
        "query_templates": [
            "{title_keywords} latest news {year}",
        ],
        "preferred_sources": [],
    },
}


# === Complexity Tier Resource Allocation ===

_COMPLEXITY_RESOURCES = {
    ComplexityTier.LOW: {"num_agents": 1, "max_searches": 1, "max_llm_calls": 2},
    ComplexityTier.MEDIUM: {"num_agents": 2, "max_searches": 1, "max_llm_calls": 3},
    ComplexityTier.HIGH: {"num_agents": 3, "max_searches": 2, "max_llm_calls": 5},
}


def detect_category(category_str: str) -> EventCategory:
    """Case-insensitive category matching with keyword-based fallback.

    First tries exact match against known EventCategory values.
    Then tries keyword-based detection from the category string.
    Falls back to GENERAL if nothing matches.

    Args:
        category_str: The category string from the event request.

    Returns:
        The matched EventCategory enum value, or GENERAL if unrecognized.
    """
    normalized = category_str.strip().lower()
    
    # Direct match
    for cat in EventCategory:
        if cat.value == normalized:
            return cat
    
    # Keyword-based fallback for common variations
    keyword_map = {
        EventCategory.SPORTS: ["sport", "nba", "nfl", "mlb", "soccer", "football", "tennis", "cricket", "baseball", "basketball", "hockey", "mma", "ufc", "boxing", "golf", "f1", "racing"],
        EventCategory.ECONOMICS: ["econom", "finance", "market", "stock", "crypto", "bitcoin", "inflation", "gdp", "trade", "fed", "interest rate", "price", "gas price"],
        EventCategory.GEOPOLITICS: ["politic", "geopolitic", "election", "government", "war", "military", "diplomat", "sanction", "trump", "biden", "congress", "senate", "legislation"],
        EventCategory.TECHNOLOGY: ["tech", "ai", "software", "hardware", "startup", "silicon", "computing", "cyber", "digital", "app", "platform"],
        EventCategory.SCIENCE: ["science", "space", "nasa", "spacex", "climate", "medical", "health", "pharma", "research", "biology", "physics", "who"],
        EventCategory.ENTERTAINMENT: ["entertain", "music", "movie", "film", "tv", "show", "celebrity", "spotify", "netflix", "award", "grammy", "oscar", "eurovision", "concert", "album", "song"],
    }
    
    for cat, keywords in keyword_map.items():
        if any(kw in normalized for kw in keywords):
            return cat
    
    return EventCategory.GENERAL


def assess_complexity(event: EventRequest) -> ComplexityTier:
    """Classify event complexity based on outcome count and description length.

    Rules:
        LOW: 2 outcomes AND description < 200 chars
        MEDIUM: 2-3 outcomes AND description < 500 chars
        HIGH: 4+ outcomes OR description >= 500 chars

    The HIGH tier triggers on either condition independently (OR logic),
    while LOW and MEDIUM require both conditions to be met (AND logic).

    Args:
        event: The event request to classify.

    Returns:
        The appropriate ComplexityTier.
    """
    num_outcomes = len(event.outcomes)
    desc_length = len(event.description)

    # HIGH triggers on either condition
    if num_outcomes >= 4 or desc_length >= 500:
        return ComplexityTier.HIGH

    # LOW requires both: exactly 2 outcomes AND short description
    if num_outcomes == 2 and desc_length < 200:
        return ComplexityTier.LOW

    # MEDIUM: 2-3 outcomes AND description < 500 (already excluded >= 500 above)
    return ComplexityTier.MEDIUM


def classify_event(event: EventRequest) -> RoutingConfig:
    """Classify event and produce a full routing configuration.

    Combines category detection and complexity assessment to determine
    the number of research agents, search queries, LLM calls, and
    category-specific search strategies to use.

    Args:
        event: The event request to classify.

    Returns:
        A RoutingConfig with all routing parameters set.
    """
    category = detect_category(event.category)
    complexity = assess_complexity(event)

    resources = _COMPLEXITY_RESOURCES[complexity]
    strategy = CATEGORY_STRATEGIES[category]

    return RoutingConfig(
        category=category,
        complexity=complexity,
        num_agents=resources["num_agents"],
        max_searches=resources["max_searches"],
        max_llm_calls=resources["max_llm_calls"],
        search_strategies=strategy["query_templates"],
    )
