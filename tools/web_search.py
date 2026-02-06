"""Web search tool implementation."""

import json
import requests
from typing import Any, Optional
from .base import Tool, ToolResult


class WebSearchTool(Tool):
    """Tool for searching the web using DuckDuckGo or Google."""

    def __init__(
        self,
        name: str = "web_search",
        description: str = "Search the internet for real-time information",
        provider: str = "duckduckgo",
        api_key: Optional[str] = None,
        search_engine_id: Optional[str] = None,
        num_results: int = 5
    ):
        """Initialize web search tool.

        Args:
            name: Tool name
            description: Tool description
            provider: Search provider (duckduckgo, google)
            api_key: Google API key (for Google search)
            search_engine_id: Google Search Engine ID (for Google search)
            num_results: Number of results to return
        """
        super().__init__(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": f"Number of results (default: {num_results})",
                        "default": num_results
                    }
                },
                "required": ["query"]
            }
        )
        self.provider = provider.lower()
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.num_results = num_results
        self.session = requests.Session()

    def _search_duckduckgo(self, query: str) -> list:
        """Search using DuckDuckGo Instant Answer API (free, no API key needed)."""
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
            "kl": "us-en"
        }

        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        if data.get("Abstract"):
            results.append({
                "title": data.get("Heading", query),
                "source": "DuckDuckGo",
                "snippet": data["Abstract"],
                "url": data.get("AbstractURL", "")
            })

        if data.get("Answer"):
            results.append({
                "title": "Instant Answer",
                "source": "DuckDuckGo",
                "snippet": data["Answer"],
                "url": ""
            })

        for topic in data.get("RelatedTopics", [])[:self.num_results]:
            if topic.get("Text"):
                results.append({
                    "title": topic.get("FirstURL", "").split("/")[-1] if topic.get("FirstURL") else "Related",
                    "source": "DuckDuckGo",
                    "snippet": topic["Text"],
                    "url": topic.get("FirstURL", "")
                })

        return results

    def _search_google(self, query: str) -> list:
        """Search using Google Custom Search JSON API."""
        if not self.api_key or not self.search_engine_id:
            raise ValueError("Google search requires GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID")

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": min(self.num_results, 10)
        }

        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        items = data.get("items", [])
        for item in items:
            results.append({
                "title": item.get("title", ""),
                "source": "Google",
                "snippet": item.get("snippet", ""),
                "url": item.get("link", "")
            })

        return results

    def execute(self, query: str, num_results: int = None) -> ToolResult:
        """Execute web search.

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            ToolResult with search results
        """
        import uuid

        try:
            if self.provider == "google":
                results = self._search_google(query)
            else:
                results = self._search_duckduckgo(query)

            # Limit results
            limit = num_results or self.num_results
            results = results[:limit]

            content = json.dumps(results, ensure_ascii=False, indent=2)

            return ToolResult(
                success=True,
                content=content,
                tool_call_id=f"search_{uuid.uuid4().hex[:8]}"
            )

        except ValueError as e:
            return ToolResult(
                success=False,
                content="",
                error=str(e),
                tool_call_id=f"search_{uuid.uuid4().hex[:8]}"
            )
        except requests.RequestException as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Search request failed: {str(e)}",
                tool_call_id=f"search_{uuid.uuid4().hex[:8]}"
            )
        except json.JSONDecodeError as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Failed to parse search results: {str(e)}",
                tool_call_id=f"search_{uuid.uuid4().hex[:8]}"
            )
