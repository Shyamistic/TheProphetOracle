"""Multi-agent research pipeline for the Prophet Forecasting Agent.

Implements parallel research agents that use Tavily web search and OpenAI-compatible LLM
to gather, filter, and summarize evidence for event prediction.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from tavily import TavilyClient

from src.models import (
    EvidenceItem,
    EventRequest,
    ResearchResult,
    RoutingConfig,
    SearchQuery,
)
from src.router import CATEGORY_STRATEGIES
from src.search_client import SearchClient

logger = logging.getLogger(__name__)

# Timeout for individual search queries (seconds)
SEARCH_TIMEOUT_SECONDS = 30

# Timeout for LLM calls (seconds)
LLM_TIMEOUT_SECONDS = 60

# Maximum research duration per agent (seconds)
MAX_RESEARCH_DURATION_SECONDS = 180


class ResearchAgent:
    """A single research agent that searches and gathers evidence for an event.

    Each agent independently extracts entities, generates queries, executes
    web searches, filters evidence, and produces a ResearchResult.
    """

    def __init__(self, agent_id: int, search_client, llm_client: AsyncOpenAI):
        """Initialize a research agent.

        Args:
            agent_id: Unique identifier for this agent instance.
            search_client: Search client (SearchClient with Tavily+DDG fallback, or TavilyClient).
            llm_client: OpenAI-compatible async client for LLM calls.
        """
        self.agent_id = agent_id
        self.search_client = search_client
        self.llm_client = llm_client

    async def extract_entities(self, event: EventRequest) -> dict:
        """Extract named entities, dates, and context from event fields using LLM.

        Uses the Anthropic API to parse the event title, description, and rules
        to identify key entities for search query generation.

        Args:
            event: The event request to extract entities from.

        Returns:
            Dict with keys: entities (list of names), dates (list of date strings),
            context (str summary), keywords (list of key terms).
        """
        prompt = f"""Extract named entities, dates, and context from this event for web search.

Event Title: {event.title}
Event Description: {event.description}
Event Category: {event.category}
Event Rules: {event.rules}
Close Time: {event.close_time}
Outcomes: {', '.join(event.outcomes)}

Return a JSON object with these fields:
- "entities": list of named entities (people, organizations, teams, places)
- "dates": list of relevant dates or time periods mentioned
- "context": a brief summary of what this event is about (1-2 sentences)
- "keywords": list of 3-5 key search terms derived from the event

Return ONLY valid JSON, no other text."""

        try:
            response = await asyncio.wait_for(
                self.llm_client.chat.completions.create(
                    model="anthropic/claude-sonnet-4",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=LLM_TIMEOUT_SECONDS,
            )

            content = response.choices[0].message.content.strip() if response.choices else ""
            # Try to parse JSON from the response
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            result = json.loads(content)
            return {
                "entities": result.get("entities", []),
                "dates": result.get("dates", []),
                "context": result.get("context", ""),
                "keywords": result.get("keywords", []),
            }

        except asyncio.TimeoutError:
            logger.warning(
                f"Agent {self.agent_id}: LLM entity extraction timed out"
            )
        except json.JSONDecodeError:
            logger.warning(
                f"Agent {self.agent_id}: Failed to parse LLM entity extraction response"
            )
        except Exception as e:
            logger.warning(
                f"Agent {self.agent_id}: Entity extraction failed: {e}"
            )

        # Fallback: extract basic entities from title/description
        return self._fallback_extract_entities(event)

    def _fallback_extract_entities(self, event: EventRequest) -> dict:
        """Fallback entity extraction using simple heuristics.

        Extracts capitalized words and key terms from the event fields
        when LLM extraction fails.
        """
        title_words = event.title.split()
        # Extract capitalized words as potential entities
        entities = [
            w.strip(".,!?;:'\"()[]")
            for w in title_words
            if w[0:1].isupper() and len(w) > 2
        ]
        # Use title words as keywords
        keywords = [
            w.lower().strip(".,!?;:'\"()[]")
            for w in title_words
            if len(w) > 3
        ][:5]

        return {
            "entities": entities[:5],
            "dates": [event.close_time[:10]] if event.close_time else [],
            "context": event.title,
            "keywords": keywords,
        }

    async def generate_queries(
        self, event: EventRequest, entities: dict, config: RoutingConfig
    ) -> List[SearchQuery]:
        """Generate category-specific search queries from entities and templates.

        Uses the category-specific query templates from CATEGORY_STRATEGIES,
        filling in extracted entities and keywords.

        Args:
            event: The event request.
            entities: Extracted entities dict from extract_entities.
            config: Routing configuration with category and search strategies.

        Returns:
            List of SearchQuery objects, limited to config.max_searches.
        """
        queries: List[SearchQuery] = []
        strategy = CATEGORY_STRATEGIES.get(config.category, CATEGORY_STRATEGIES[config.category])
        source_types = strategy.get("source_types", ["general_news"])
        templates = config.search_strategies

        # Build template fill values from entities
        entity_list = entities.get("entities", [])
        keywords = entities.get("keywords", [])
        year = datetime.now().year

        # Create fill values for templates
        fill_values = {
            "year": str(year),
            "title_keywords": " ".join(keywords[:3]) if keywords else event.title[:50],
            "topic": entities.get("context", event.title)[:50],
        }

        # Category-specific fill values
        if entity_list:
            fill_values["team_a"] = entity_list[0] if len(entity_list) > 0 else ""
            fill_values["team_b"] = entity_list[1] if len(entity_list) > 1 else ""
            fill_values["entity"] = entity_list[0]
            fill_values["company"] = entity_list[0]
            fill_values["country"] = entity_list[0]
            fill_values["institution"] = entity_list[0]
            fill_values["region"] = entity_list[0] if len(entity_list) > 0 else ""
        else:
            # Use keywords as fallback
            fill_values["team_a"] = keywords[0] if keywords else event.title.split()[0]
            fill_values["team_b"] = keywords[1] if len(keywords) > 1 else ""
            fill_values["entity"] = keywords[0] if keywords else event.title.split()[0]
            fill_values["company"] = keywords[0] if keywords else ""
            fill_values["country"] = keywords[0] if keywords else ""
            fill_values["institution"] = keywords[0] if keywords else ""
            fill_values["region"] = keywords[0] if keywords else ""

        fill_values["sport"] = event.category if event.category.lower() == "sports" else ""
        fill_values["indicator"] = keywords[0] if keywords else ""
        fill_values["timeframe"] = f"{year}"
        fill_values["product"] = keywords[1] if len(keywords) > 1 else ""
        fill_values["technology"] = keywords[0] if keywords else ""
        fill_values["field"] = event.category

        # Generate queries from templates
        for i, template in enumerate(templates):
            if len(queries) >= config.max_searches:
                break
            try:
                query_text = template.format(**fill_values)
                source_type = source_types[i % len(source_types)]
                queries.append(
                    SearchQuery(
                        query_text=query_text,
                        source_type=source_type,
                        max_results=5,
                    )
                )
            except (KeyError, IndexError):
                # If template formatting fails, use a direct keyword query
                fallback_query = f"{' '.join(keywords[:3])} {event.category} {year}"
                queries.append(
                    SearchQuery(
                        query_text=fallback_query,
                        source_type=source_types[0] if source_types else "general_news",
                        max_results=5,
                    )
                )

        # If no queries generated, create a basic one from the event title
        if not queries:
            queries.append(
                SearchQuery(
                    query_text=f"{event.title} {year}",
                    source_type="general_news",
                    max_results=5,
                )
            )

        return queries[: config.max_searches]

    async def _execute_search(self, query: SearchQuery, topic: str = None,
                             time_range: str = None, search_depth: str = "basic",
                             include_answer: bool = False) -> List[dict]:
        """Execute a single web search query via Tavily with advanced features.

        Args:
            query: The search query to execute.
            topic: Tavily topic filter ("news" or "finance").
            time_range: Time range filter ("day", "week", "month", "year").
            search_depth: Search depth ("basic" or "advanced").
            include_answer: Whether to include Tavily's LLM-generated answer.

        Returns:
            List of raw search result dicts from Tavily.
        """
        try:
            # Tavily's search is synchronous, run in executor
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.search_client.search(
                        query=query.query_text,
                        max_results=query.max_results,
                        search_depth=search_depth,
                        topic=topic,
                        time_range=time_range,
                        include_answer=include_answer,
                    ),
                ),
                timeout=SEARCH_TIMEOUT_SECONDS,
            )
            # Store the answer if available for later use
            if include_answer and response.get("answer"):
                self._last_tavily_answer = response.get("answer")
            return response.get("results", [])
        except asyncio.TimeoutError:
            logger.warning(
                f"Agent {self.agent_id}: Search timed out for query: {query.query_text}"
            )
            return []
        except Exception as e:
            logger.warning(
                f"Agent {self.agent_id}: Search failed for query '{query.query_text}': {e}"
            )
            return []

    async def filter_evidence(
        self, results: List[dict], close_time: datetime
    ) -> List[EvidenceItem]:
        """Filter search results by 90-day recency and corroboration.

        Only includes results that:
        1. Were published within 90 days before the event close_time
        2. Are corroborated by at least 1 other source (URL domain differs
           but content overlaps)

        Args:
            results: Raw search results from web searches.
            close_time: The event's close time for recency calculation.

        Returns:
            List of filtered EvidenceItem objects.
        """
        if not results:
            return []

        cutoff_date = close_time - timedelta(days=90)
        evidence_items: List[EvidenceItem] = []

        # First pass: parse results and check recency
        for result in results:
            pub_date = self._parse_publication_date(result)

            # Check recency: must be within 90 days before close_time
            if pub_date and pub_date < cutoff_date:
                continue  # Too old, skip

            # If no publication date, include it (benefit of the doubt)
            source_url = result.get("url", "")
            summary = result.get("content", result.get("snippet", ""))[:500]
            relevance_score = result.get("score", 0.5)

            evidence_items.append(
                EvidenceItem(
                    source_url=source_url,
                    publication_date=pub_date,
                    summary=summary,
                    relevance_score=float(relevance_score),
                    corroborated=False,  # Will be set in corroboration check
                )
            )

        # Second pass: check corroboration
        # An item is corroborated if at least 1 other item from a different
        # domain has overlapping content (shared keywords)
        for i, item in enumerate(evidence_items):
            item_domain = self._extract_domain(item.source_url)
            item_keywords = set(item.summary.lower().split()[:20])

            for j, other in enumerate(evidence_items):
                if i == j:
                    continue
                other_domain = self._extract_domain(other.source_url)
                if item_domain == other_domain:
                    continue  # Same source doesn't count

                other_keywords = set(other.summary.lower().split()[:20])
                overlap = item_keywords & other_keywords
                # If at least 3 meaningful words overlap, consider corroborated
                meaningful_overlap = {
                    w for w in overlap if len(w) > 3
                }
                if len(meaningful_overlap) >= 3:
                    item.corroborated = True
                    break

        # Return only corroborated items (at least 1 other source confirms)
        filtered = [item for item in evidence_items if item.corroborated]

        # If no items pass corroboration, return all recency-filtered items
        # to avoid returning empty evidence sets
        if not filtered and evidence_items:
            # Mark them as not corroborated but still return them
            return evidence_items

        return filtered

    def _parse_publication_date(self, result: dict) -> Optional[datetime]:
        """Parse publication date from a search result.

        Tries multiple date field names and formats.
        """
        date_fields = ["published_date", "publishedDate", "date", "published"]
        for field_name in date_fields:
            date_str = result.get(field_name)
            if date_str:
                try:
                    # Try ISO format
                    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except (ValueError, AttributeError):
                    pass
                try:
                    # Try common date format
                    dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
                    return dt.replace(tzinfo=timezone.utc)
                except (ValueError, AttributeError):
                    pass
        return None

    def _extract_domain(self, url: str) -> str:
        """Extract domain from a URL for corroboration checking."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return url

    async def research(self, event: EventRequest, config: RoutingConfig) -> ResearchResult:
        """Execute the full research pipeline for one event.

        Steps:
        1. Extract entities from event title/description/rules
        2. Generate category-specific search queries
        3. Execute web searches with advanced Tavily features
        4. Filter results by recency (90 days) and corroboration
        5. Return structured research result

        Args:
            event: The event to research.
            config: Routing configuration with category and resource limits.

        Returns:
            ResearchResult with evidence, queries used, and metadata.
        """
        start_time = time.time()
        failed_sources: List[dict] = []
        all_raw_results: List[dict] = []
        queries_used: List[str] = []
        self._last_tavily_answer = None

        # Determine advanced search parameters based on category and complexity
        topic = self._get_topic_for_category(config)
        time_range = self._get_time_range(event, config)
        search_depth = "advanced" if config.complexity == "high" else "basic"
        include_answer = True  # Always request Tavily's answer summary

        try:
            # Step 1: Extract entities
            entities = await self.extract_entities(event)

            # Step 2: Generate queries
            queries = await self.generate_queries(event, entities, config)

            # Step 3: Execute searches with advanced features
            for query in queries:
                queries_used.append(query.query_text)
                results = await self._execute_search(
                    query,
                    topic=topic,
                    time_range=time_range,
                    search_depth=search_depth,
                    include_answer=include_answer,
                )
                if results:
                    all_raw_results.extend(results)
                else:
                    failed_sources.append({
                        "source": query.query_text,
                        "reason": "No results or search failed",
                    })

            # Step 4: Filter evidence
            close_time = self._parse_close_time(event.close_time)
            evidence = await self.filter_evidence(all_raw_results, close_time)

            # Step 5: Add Tavily answer as high-relevance evidence if available
            if self._last_tavily_answer:
                evidence.insert(0, EvidenceItem(
                    source_url="tavily_answer_summary",
                    publication_date=datetime.now(timezone.utc),
                    summary=self._last_tavily_answer[:500],
                    relevance_score=0.95,
                    corroborated=True,
                ))

        except asyncio.TimeoutError:
            logger.error(
                f"Agent {self.agent_id}: Research timed out for event {event.event_ticker}"
            )
            evidence = []
            failed_sources.append({
                "source": "research_pipeline",
                "reason": "Overall research timeout exceeded",
            })
        except Exception as e:
            logger.error(
                f"Agent {self.agent_id}: Research failed for event {event.event_ticker}: {e}"
            )
            evidence = []
            failed_sources.append({
                "source": "research_pipeline",
                "reason": str(e),
            })

        duration = time.time() - start_time

        return ResearchResult(
            event_ticker=event.event_ticker,
            evidence=evidence,
            search_queries_used=queries_used,
            failed_sources=failed_sources,
            duration_seconds=duration,
        )

    def _get_topic_for_category(self, config: RoutingConfig) -> Optional[str]:
        """Determine the Tavily topic based on event category.

        Args:
            config: Routing configuration with category info.

        Returns:
            "finance" for economics, "news" for most others, None for general.
        """
        from src.models import EventCategory
        if config.category == EventCategory.ECONOMICS:
            return "finance"
        elif config.category in (EventCategory.GEOPOLITICS, EventCategory.SPORTS,
                                  EventCategory.TECHNOLOGY, EventCategory.SCIENCE):
            return "news"
        return "news"  # Default to news for better date filtering

    def _get_time_range(self, event: EventRequest, config: RoutingConfig) -> Optional[str]:
        """Determine time range based on event close time proximity.

        Events closing within 7 days use "week", others use "month".

        Args:
            event: The event request.
            config: Routing configuration.

        Returns:
            Time range string for Tavily.
        """
        try:
            close_time = self._parse_close_time(event.close_time)
            now = datetime.now(timezone.utc)
            days_until_close = (close_time - now).days
            if days_until_close <= 7:
                return "week"
            else:
                return "month"
        except Exception:
            return "month"

    def _parse_close_time(self, close_time_str: str) -> datetime:
        """Parse the event close_time string to a datetime object."""
        try:
            dt = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, AttributeError):
            # Default to now + 30 days if parsing fails
            return datetime.now(timezone.utc) + timedelta(days=30)


async def run_parallel_research(
    event: EventRequest,
    config: RoutingConfig,
    search_client,
    llm_client: AsyncOpenAI,
) -> List[ResearchResult]:
    """Run N research agents in parallel and return all evidence sets.

    Spawns config.num_agents research agents concurrently, each independently
    researching the event. Results are collected with a timeout to ensure
    the research phase completes within the allocated time budget.

    Args:
        event: The event to research.
        config: Routing configuration specifying num_agents and resource limits.
        search_client: Shared Tavily API client.
        llm_client: Shared Anthropic async client.

    Returns:
        List of ResearchResult from all agents that completed successfully.
    """
    num_agents = config.num_agents
    agents = [
        ResearchAgent(agent_id=i, search_client=search_client, llm_client=llm_client)
        for i in range(num_agents)
    ]

    # Create tasks for parallel execution
    tasks = [
        asyncio.create_task(agent.research(event, config))
        for agent in agents
    ]

    results: List[ResearchResult] = []

    try:
        # Wait for all agents with overall timeout
        done, pending = await asyncio.wait(
            tasks,
            timeout=MAX_RESEARCH_DURATION_SECONDS,
            return_when=asyncio.ALL_COMPLETED,
        )

        # Cancel any pending tasks
        for task in pending:
            task.cancel()
            logger.warning(
                f"Research agent task cancelled due to timeout for event {event.event_ticker}"
            )

        # Collect results from completed tasks
        for task in done:
            try:
                result = task.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Research agent task failed: {e}")
                # Create a failed result
                results.append(
                    ResearchResult(
                        event_ticker=event.event_ticker,
                        evidence=[],
                        search_queries_used=[],
                        failed_sources=[{"source": "agent", "reason": str(e)}],
                        duration_seconds=0.0,
                    )
                )

    except Exception as e:
        logger.error(f"Parallel research failed for event {event.event_ticker}: {e}")
        # Cancel all tasks on unexpected error
        for task in tasks:
            task.cancel()
        results.append(
            ResearchResult(
                event_ticker=event.event_ticker,
                evidence=[],
                search_queries_used=[],
                failed_sources=[{"source": "parallel_research", "reason": str(e)}],
                duration_seconds=0.0,
            )
        )

    return results
