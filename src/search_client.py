"""Search client with Tavily primary, Serper.dev secondary, DuckDuckGo tertiary.

Fallback chain: Tavily → Serper.dev (Google SERP) → DuckDuckGo
This ensures we NEVER fail to get search results, even if Tavily credits run out.
"""

import logging
from typing import Dict, List, Optional

import httpx
from duckduckgo_search import DDGS
from tavily import TavilyClient

logger = logging.getLogger(__name__)

SERPER_API_URL = "https://google.serper.dev/search"


class SearchClient:
    """Unified search client with triple fallback.

    Fallback chain:
    1. Tavily (advanced features, news/finance topics) — primary
    2. Serper.dev (Google SERP, 2500 free searches) — secondary
    3. DuckDuckGo (free, unlimited, but rate-limited) — tertiary
    """

    def __init__(
        self,
        tavily_api_key: str,
        serper_api_key: Optional[str] = None,
        tavily_failure_threshold: int = 3,
    ):
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        self.serper_api_key = serper_api_key
        self.ddgs = DDGS()
        self._tavily_consecutive_failures = 0
        self._tavily_failure_threshold = tavily_failure_threshold
        self._tavily_disabled = False

    @property
    def is_tavily_available(self) -> bool:
        return not self._tavily_disabled

    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        topic: str = "general",
        time_range: Optional[str] = None,
        include_answer: bool = False,
        include_domains: Optional[List[str]] = None,
    ) -> Dict:
        """Execute a search with triple fallback.

        Args:
            query: Search query string.
            max_results: Max results (1-20).
            search_depth: "basic" (1 credit) or "advanced" (2 credits).
            topic: "general", "news", or "finance".
            time_range: "day", "week", "month", or "year".
            include_answer: Get LLM-generated answer (Tavily only).
            include_domains: Restrict to specific domains.

        Returns:
            Dict with "results" list and optionally "answer" field.
        """
        # Try Tavily first
        if not self._tavily_disabled:
            try:
                kwargs = {
                    "query": query,
                    "max_results": max_results,
                    "search_depth": search_depth,
                    "topic": topic,
                }
                if time_range:
                    kwargs["time_range"] = time_range
                if include_answer:
                    kwargs["include_answer"] = True
                if include_domains:
                    kwargs["include_domains"] = include_domains

                response = self.tavily_client.search(**kwargs)
                self._tavily_consecutive_failures = 0
                return response

            except Exception as e:
                self._tavily_consecutive_failures += 1
                error_str = str(e).lower()

                if "credit" in error_str or "limit" in error_str or "429" in error_str or "402" in error_str:
                    logger.warning(f"Tavily credits exhausted: {e}. Switching to Serper/DDG.")
                    self._tavily_disabled = True
                elif self._tavily_consecutive_failures >= self._tavily_failure_threshold:
                    logger.warning(f"Tavily failed {self._tavily_consecutive_failures}x. Switching to Serper/DDG.")
                    self._tavily_disabled = True
                else:
                    logger.warning(f"Tavily failed ({self._tavily_consecutive_failures}/{self._tavily_failure_threshold}): {e}")

        # Try Serper.dev (Google) second
        if self.serper_api_key:
            serper_result = self._search_serper(query, max_results)
            if serper_result.get("results"):
                return serper_result

        # Fall back to DuckDuckGo
        return self._search_duckduckgo(query, max_results)

    def _search_serper(self, query: str, max_results: int = 5) -> Dict:
        """Search using Serper.dev (Google SERP API, 2500 free searches).

        Args:
            query: Search query.
            max_results: Number of results.

        Returns:
            Dict with "results" in Tavily-compatible format.
        """
        try:
            response = httpx.post(
                SERPER_API_URL,
                headers={
                    "X-API-KEY": self.serper_api_key,
                    "Content-Type": "application/json",
                },
                json={"q": query, "num": max_results},
                timeout=15.0,
            )

            if response.status_code == 200:
                data = response.json()
                formatted = []
                for item in data.get("organic", [])[:max_results]:
                    formatted.append({
                        "url": item.get("link", ""),
                        "content": item.get("snippet", ""),
                        "title": item.get("title", ""),
                        "score": 0.8,  # Serper gives good results
                    })

                result = {"results": formatted}

                # Include answer snippet if available
                if data.get("answerBox"):
                    answer = data["answerBox"].get("answer") or data["answerBox"].get("snippet", "")
                    if answer:
                        result["answer"] = answer

                return result
            else:
                logger.warning(f"Serper returned status {response.status_code}")
                return {"results": []}

        except Exception as e:
            logger.warning(f"Serper search failed: {e}")
            return {"results": []}

    def extract(self, urls: List[str], query: Optional[str] = None) -> Dict:
        """Extract clean content from URLs using Tavily Extract."""
        if self._tavily_disabled:
            return {"results": [], "failed_results": [{"url": u, "error": "Tavily disabled"} for u in urls]}

        try:
            kwargs = {"urls": urls}
            if query:
                kwargs["query"] = query
            return self.tavily_client.extract(**kwargs)
        except Exception as e:
            logger.warning(f"Tavily extract failed: {e}")
            return {"results": [], "failed_results": [{"url": u, "error": str(e)} for u in urls]}

    def _search_duckduckgo(self, query: str, max_results: int = 5) -> Dict:
        """Tertiary fallback using DuckDuckGo (free but rate-limited)."""
        try:
            results = self.ddgs.text(query, max_results=max_results)
            formatted = []
            for r in results:
                formatted.append({
                    "url": r.get("href", r.get("link", "")),
                    "content": r.get("body", r.get("snippet", "")),
                    "title": r.get("title", ""),
                    "score": 0.7,
                })
            return {"results": formatted}
        except Exception as e:
            logger.error(f"DuckDuckGo search also failed: {e}")
            return {"results": []}
