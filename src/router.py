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
    """Case-insensitive category matching with fallback to GENERAL.

    Matches the input string against known EventCategory values. If no match
    is found, returns EventCategory.GENERAL as the default fallback.

    Args:
        category_str: The category string from the event request.

    Returns:
        The matched EventCategory enum value, or GENERAL if unrecognized.
    """
    normalized = category_str.strip().lower()
    for cat in EventCategory:
        if cat.value == normalized:
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
