"""Tests for the multi-agent research pipeline."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models import (
    EvidenceItem,
    EventRequest,
    ResearchResult,
    RoutingConfig,
    SearchQuery,
    ComplexityTier,
    EventCategory,
)
from src.research import ResearchAgent, run_parallel_research


# === Fixtures ===


@pytest.fixture
def sample_event():
    """Create a sample event request for testing."""
    return EventRequest(
        event_ticker="EVT-001",
        market_ticker="MKT-001",
        title="Will the Lakers beat the Celtics in the NBA Finals 2025?",
        description="The Los Angeles Lakers face the Boston Celtics in the 2025 NBA Finals.",
        category="Sports",
        rules="Resolves YES if Lakers win the series.",
        close_time="2025-06-30T23:59:00Z",
        outcomes=["Yes", "No"],
    )


@pytest.fixture
def sports_config():
    """Create a sports routing config."""
    return RoutingConfig(
        category=EventCategory.SPORTS,
        complexity=ComplexityTier.MEDIUM,
        num_agents=2,
        max_searches=2,
        max_llm_calls=3,
        search_strategies=[
            "{team_a} vs {team_b} recent results {year}",
            "{team_a} season record stats {year}",
        ],
    )


@pytest.fixture
def mock_search_client():
    """Create a mock Tavily search client."""
    client = MagicMock()
    client.search = MagicMock(
        return_value={
            "results": [
                {
                    "url": "https://espn.com/nba/lakers-celtics-preview",
                    "content": "The Lakers have won 5 of their last 7 games against the Celtics this season.",
                    "score": 0.85,
                    "published_date": "2025-06-01T12:00:00Z",
                },
                {
                    "url": "https://sports-reference.com/nba/teams/lakers",
                    "content": "Lakers season record shows strong performance with key players healthy.",
                    "score": 0.78,
                    "published_date": "2025-05-28T10:00:00Z",
                },
            ]
        }
    )
    return client


@pytest.fixture
def mock_llm_client():
    """Create a mock Anthropic async client."""
    client = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(
            text=json.dumps(
                {
                    "entities": ["Lakers", "Celtics", "NBA"],
                    "dates": ["2025-06-30"],
                    "context": "NBA Finals matchup between Lakers and Celtics",
                    "keywords": ["Lakers", "Celtics", "NBA Finals", "2025"],
                }
            )
        )
    ]
    client.messages.create = AsyncMock(return_value=mock_response)
    return client


# === Unit Tests ===


class TestResearchAgent:
    """Tests for the ResearchAgent class."""

    @pytest.mark.asyncio
    async def test_extract_entities_success(self, sample_event, mock_search_client, mock_llm_client):
        """Test successful entity extraction via LLM."""
        agent = ResearchAgent(
            agent_id=0, search_client=mock_search_client, llm_client=mock_llm_client
        )
        entities = await agent.extract_entities(sample_event)

        assert "entities" in entities
        assert "dates" in entities
        assert "context" in entities
        assert "keywords" in entities
        assert "Lakers" in entities["entities"]
        assert "Celtics" in entities["entities"]

    @pytest.mark.asyncio
    async def test_extract_entities_fallback_on_llm_failure(
        self, sample_event, mock_search_client
    ):
        """Test fallback entity extraction when LLM fails."""
        llm_client = AsyncMock()
        llm_client.messages.create = AsyncMock(side_effect=Exception("API Error"))

        agent = ResearchAgent(
            agent_id=0, search_client=mock_search_client, llm_client=llm_client
        )
        entities = await agent.extract_entities(sample_event)

        # Should still return a valid structure
        assert "entities" in entities
        assert "keywords" in entities
        assert len(entities["keywords"]) > 0

    @pytest.mark.asyncio
    async def test_generate_queries(
        self, sample_event, sports_config, mock_search_client, mock_llm_client
    ):
        """Test query generation from entities and templates."""
        agent = ResearchAgent(
            agent_id=0, search_client=mock_search_client, llm_client=mock_llm_client
        )
        entities = {
            "entities": ["Lakers", "Celtics"],
            "dates": ["2025-06-30"],
            "context": "NBA Finals",
            "keywords": ["Lakers", "Celtics", "NBA", "Finals"],
        }

        queries = await agent.generate_queries(sample_event, entities, sports_config)

        assert len(queries) <= sports_config.max_searches
        assert all(isinstance(q, SearchQuery) for q in queries)
        # Queries should contain entity names
        query_texts = [q.query_text for q in queries]
        assert any("Lakers" in q for q in query_texts)

    @pytest.mark.asyncio
    async def test_generate_queries_respects_max_searches(
        self, sample_event, mock_search_client, mock_llm_client
    ):
        """Test that query generation respects max_searches limit."""
        config = RoutingConfig(
            category=EventCategory.SPORTS,
            complexity=ComplexityTier.LOW,
            num_agents=1,
            max_searches=1,
            max_llm_calls=2,
            search_strategies=[
                "{team_a} vs {team_b} recent results {year}",
                "{team_a} season record stats {year}",
                "{sport} {team_a} {team_b} odds prediction",
            ],
        )
        agent = ResearchAgent(
            agent_id=0, search_client=mock_search_client, llm_client=mock_llm_client
        )
        entities = {
            "entities": ["Lakers", "Celtics"],
            "dates": [],
            "context": "NBA game",
            "keywords": ["Lakers", "Celtics"],
        }

        queries = await agent.generate_queries(sample_event, entities, config)
        assert len(queries) == 1

    @pytest.mark.asyncio
    async def test_filter_evidence_recency(self, mock_search_client, mock_llm_client):
        """Test that filter_evidence removes results older than 90 days."""
        agent = ResearchAgent(
            agent_id=0, search_client=mock_search_client, llm_client=mock_llm_client
        )
        close_time = datetime(2025, 6, 30, tzinfo=timezone.utc)

        results = [
            {
                "url": "https://espn.com/recent",
                "content": "Recent Lakers game analysis shows strong performance this season",
                "score": 0.9,
                "published_date": "2025-06-01T00:00:00Z",  # Within 90 days
            },
            {
                "url": "https://sports-reference.com/old",
                "content": "Old Lakers game analysis from last year shows different trends",
                "score": 0.7,
                "published_date": "2024-01-01T00:00:00Z",  # Older than 90 days
            },
        ]

        evidence = await agent.filter_evidence(results, close_time)

        # The old result should be filtered out
        urls = [e.source_url for e in evidence]
        assert "https://espn.com/recent" in urls
        assert "https://sports-reference.com/old" not in urls

    @pytest.mark.asyncio
    async def test_filter_evidence_corroboration(self, mock_search_client, mock_llm_client):
        """Test that filter_evidence checks corroboration between sources."""
        agent = ResearchAgent(
            agent_id=0, search_client=mock_search_client, llm_client=mock_llm_client
        )
        close_time = datetime(2025, 6, 30, tzinfo=timezone.utc)

        # Two results from different domains with overlapping content
        results = [
            {
                "url": "https://espn.com/article1",
                "content": "Lakers strong performance season record wins championship contender",
                "score": 0.9,
                "published_date": "2025-06-01T00:00:00Z",
            },
            {
                "url": "https://sportsref.com/article2",
                "content": "Lakers strong performance season record wins playoff contender",
                "score": 0.8,
                "published_date": "2025-06-02T00:00:00Z",
            },
        ]

        evidence = await agent.filter_evidence(results, close_time)

        # Both should be corroborated (different domains, overlapping content)
        assert len(evidence) == 2
        assert all(e.corroborated for e in evidence)

    @pytest.mark.asyncio
    async def test_filter_evidence_empty_results(self, mock_search_client, mock_llm_client):
        """Test filter_evidence with empty results."""
        agent = ResearchAgent(
            agent_id=0, search_client=mock_search_client, llm_client=mock_llm_client
        )
        close_time = datetime(2025, 6, 30, tzinfo=timezone.utc)

        evidence = await agent.filter_evidence([], close_time)
        assert evidence == []

    @pytest.mark.asyncio
    async def test_research_full_pipeline(
        self, sample_event, sports_config, mock_search_client, mock_llm_client
    ):
        """Test the full research pipeline end-to-end."""
        agent = ResearchAgent(
            agent_id=0, search_client=mock_search_client, llm_client=mock_llm_client
        )

        result = await agent.research(sample_event, sports_config)

        assert isinstance(result, ResearchResult)
        assert result.event_ticker == "EVT-001"
        assert len(result.search_queries_used) > 0
        assert result.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_research_handles_search_failure(
        self, sample_event, sports_config, mock_llm_client
    ):
        """Test that research handles search failures gracefully."""
        failing_search = MagicMock()
        failing_search.search = MagicMock(side_effect=Exception("Network error"))

        agent = ResearchAgent(
            agent_id=0, search_client=failing_search, llm_client=mock_llm_client
        )

        result = await agent.research(sample_event, sports_config)

        assert isinstance(result, ResearchResult)
        assert len(result.failed_sources) > 0


class TestRunParallelResearch:
    """Tests for the run_parallel_research function."""

    @pytest.mark.asyncio
    async def test_parallel_research_spawns_correct_agents(
        self, sample_event, sports_config, mock_search_client, mock_llm_client
    ):
        """Test that parallel research spawns the correct number of agents."""
        results = await run_parallel_research(
            sample_event, sports_config, mock_search_client, mock_llm_client
        )

        # Should have results from num_agents agents
        assert len(results) == sports_config.num_agents

    @pytest.mark.asyncio
    async def test_parallel_research_single_agent(
        self, sample_event, mock_search_client, mock_llm_client
    ):
        """Test parallel research with a single agent (LOW complexity)."""
        config = RoutingConfig(
            category=EventCategory.GENERAL,
            complexity=ComplexityTier.LOW,
            num_agents=1,
            max_searches=1,
            max_llm_calls=2,
            search_strategies=["{title_keywords} latest news {year}"],
        )

        results = await run_parallel_research(
            sample_event, config, mock_search_client, mock_llm_client
        )

        assert len(results) == 1
        assert results[0].event_ticker == "EVT-001"

    @pytest.mark.asyncio
    async def test_parallel_research_all_results_have_event_ticker(
        self, sample_event, sports_config, mock_search_client, mock_llm_client
    ):
        """Test that all parallel results reference the correct event."""
        results = await run_parallel_research(
            sample_event, sports_config, mock_search_client, mock_llm_client
        )

        for result in results:
            assert result.event_ticker == sample_event.event_ticker

    @pytest.mark.asyncio
    async def test_parallel_research_handles_agent_failure(
        self, sample_event, sports_config, mock_llm_client
    ):
        """Test that parallel research handles individual agent failures."""
        # Search client that fails intermittently
        failing_search = MagicMock()
        failing_search.search = MagicMock(side_effect=Exception("Intermittent failure"))

        results = await run_parallel_research(
            sample_event, sports_config, failing_search, mock_llm_client
        )

        # Should still return results (even if evidence is empty)
        assert len(results) == sports_config.num_agents
        for result in results:
            assert isinstance(result, ResearchResult)
