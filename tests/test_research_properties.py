"""Property-based tests for ResearchAgent.filter_evidence (src/research.py).

Property 4: Evidence Recency Filter
Test that filter returns only items within 90 days of close_time AND corroborated.
Generate random evidence items with various dates and corroboration flags.

Validates: Requirements 2.4
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.research import ResearchAgent


# --- Strategies ---

# Generate domain names for URLs
domain_strategy = st.sampled_from([
    "espn.com", "reuters.com", "bbc.com", "cnn.com", "nytimes.com",
    "apnews.com", "theguardian.com", "washingtonpost.com", "foxnews.com",
    "bloomberg.com", "techcrunch.com", "nature.com", "science.org",
])

# Generate a close_time datetime (UTC, within a reasonable range)
close_time_strategy = st.datetimes(
    min_value=datetime(2024, 1, 1),
    max_value=datetime(2026, 12, 31),
    timezones=st.just(timezone.utc),
)

# Generate a timedelta offset from close_time for publication dates
# Negative means before close_time, positive means after
date_offset_strategy = st.integers(min_value=-365, max_value=30)


# Generate content words that can be used for corroboration overlap
content_words = st.lists(
    st.sampled_from([
        "strong", "performance", "season", "record", "championship",
        "analysis", "forecast", "market", "growth", "decline",
        "victory", "defeat", "election", "policy", "research",
        "technology", "innovation", "breakthrough", "economic", "political",
    ]),
    min_size=5,
    max_size=15,
)


@st.composite
def evidence_item_strategy(draw, close_time):
    """Generate a single raw search result dict with controlled date and domain."""
    domain = draw(domain_strategy)
    url = f"https://{domain}/article-{draw(st.integers(min_value=1, max_value=9999))}"

    # Generate publication date as offset from close_time
    offset_days = draw(date_offset_strategy)
    pub_date = close_time + timedelta(days=offset_days)

    # Generate content
    words = draw(content_words)
    content = " ".join(words)

    score = draw(st.floats(min_value=0.1, max_value=1.0))

    return {
        "url": url,
        "content": content,
        "score": score,
        "published_date": pub_date.isoformat(),
        "_offset_days": offset_days,  # metadata for assertions
        "_domain": domain,  # metadata for assertions
    }


@st.composite
def evidence_list_strategy(draw):
    """Generate a list of evidence items and a close_time."""
    close_time = draw(close_time_strategy)
    num_items = draw(st.integers(min_value=1, max_value=15))

    items = []
    for _ in range(num_items):
        item = draw(evidence_item_strategy(close_time))
        items.append(item)

    return items, close_time


# --- Helper ---

def _create_agent():
    """Create a ResearchAgent with mock clients for testing filter_evidence."""
    mock_search = MagicMock()
    mock_llm = AsyncMock()
    return ResearchAgent(agent_id=0, search_client=mock_search, llm_client=mock_llm)


def _is_within_90_days(offset_days: int) -> bool:
    """Check if an offset (days before close_time is negative) is within 90 days."""
    # offset_days is relative to close_time: negative means before, positive means after
    # The filter keeps items where pub_date >= close_time - 90 days
    # i.e., offset_days >= -90
    return offset_days >= -90


def _extract_domain(url: str) -> str:
    """Extract domain from URL, matching the agent's logic."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _items_are_corroborated(items_within_recency):
    """Determine which items would be corroborated based on the filter logic.

    An item is corroborated if at least 1 other item from a different domain
    has >= 3 meaningful words (len > 3) overlapping in the first 20 words.
    """
    corroborated_indices = set()

    for i, item in enumerate(items_within_recency):
        item_domain = _extract_domain(item["url"])
        item_keywords = set(item["content"].lower().split()[:20])

        for j, other in enumerate(items_within_recency):
            if i == j:
                continue
            other_domain = _extract_domain(other["url"])
            if item_domain == other_domain:
                continue

            other_keywords = set(other["content"].lower().split()[:20])
            overlap = item_keywords & other_keywords
            meaningful_overlap = {w for w in overlap if len(w) > 3}
            if len(meaningful_overlap) >= 3:
                corroborated_indices.add(i)
                break

    return corroborated_indices


# --- Property Tests ---


class TestEvidenceRecencyFilterProperty:
    """Property 4: Evidence Recency Filter.

    **Validates: Requirements 2.4**
    """

    @given(data=st.data())
    @settings(max_examples=200, deadline=None)
    def test_no_items_older_than_90_days_in_output(self, data):
        """All items in filter output have publication dates within 90 days of close_time.

        **Validates: Requirements 2.4**
        """
        items, close_time = data.draw(evidence_list_strategy(), label="evidence_data")

        agent = _create_agent()
        result = asyncio.get_event_loop().run_until_complete(
            agent.filter_evidence(items, close_time)
        )

        cutoff_date = close_time - timedelta(days=90)

        for evidence_item in result:
            if evidence_item.publication_date is not None:
                assert evidence_item.publication_date >= cutoff_date, (
                    f"Evidence item with pub_date {evidence_item.publication_date} "
                    f"is older than cutoff {cutoff_date} (close_time={close_time}).\n"
                    f"Source: {evidence_item.source_url}"
                )

    @given(data=st.data())
    @settings(max_examples=200, deadline=None)
    def test_corroborated_items_returned_when_available(self, data):
        """When corroborated items exist, only corroborated items are returned.

        **Validates: Requirements 2.4**
        """
        items, close_time = data.draw(evidence_list_strategy(), label="evidence_data")

        agent = _create_agent()
        result = asyncio.get_event_loop().run_until_complete(
            agent.filter_evidence(items, close_time)
        )

        # Determine which items pass recency
        recency_passed = [
            item for item in items
            if item["_offset_days"] >= -90
        ]

        # Determine which of those are corroborated
        corroborated_indices = _items_are_corroborated(recency_passed)

        if corroborated_indices:
            # When corroborated items exist, all returned items should be corroborated
            for evidence_item in result:
                assert evidence_item.corroborated, (
                    f"Non-corroborated item returned when corroborated items exist.\n"
                    f"Source: {evidence_item.source_url}\n"
                    f"Summary: {evidence_item.summary[:100]}"
                )

    @given(data=st.data())
    @settings(max_examples=200, deadline=None)
    def test_filter_returns_subset_of_recency_valid_items(self, data):
        """All returned items must have passed the 90-day recency check.

        **Validates: Requirements 2.4**
        """
        items, close_time = data.draw(evidence_list_strategy(), label="evidence_data")

        agent = _create_agent()
        result = asyncio.get_event_loop().run_until_complete(
            agent.filter_evidence(items, close_time)
        )

        # Collect URLs that pass recency
        cutoff_date = close_time - timedelta(days=90)
        recency_valid_urls = set()
        for item in items:
            pub_date = agent._parse_publication_date(item)
            if pub_date is None or pub_date >= cutoff_date:
                recency_valid_urls.add(item["url"])

        # All returned items must be from the recency-valid set
        for evidence_item in result:
            assert evidence_item.source_url in recency_valid_urls, (
                f"Returned item {evidence_item.source_url} did not pass recency filter.\n"
                f"Recency-valid URLs: {recency_valid_urls}"
            )

    @given(data=st.data())
    @settings(max_examples=200, deadline=None)
    def test_empty_input_returns_empty_output(self, data):
        """Empty input always produces empty output.

        **Validates: Requirements 2.4**
        """
        close_time = data.draw(close_time_strategy, label="close_time")

        agent = _create_agent()
        result = asyncio.get_event_loop().run_until_complete(
            agent.filter_evidence([], close_time)
        )

        assert result == [], (
            f"Expected empty output for empty input, got {len(result)} items."
        )
