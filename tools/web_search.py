"""Web search tool implementation."""

import json
import requests
from typing import Any, Optional
from .base import Tool, ToolResult


class WebSearchTool(Tool):
    """Tool for searching the web."""

    def __init__(
        self,
        name: str = "web_search",
        description: str = "Search the internet for real-time information",
        api_key: Optional[str] = None,
        num_results: int = 5
    ):
        """Initialize web search tool.

        Args:
            name: Tool name
            description: Tool description
            api_key: Optional API key (for paid services)
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
        self.api_key = api_key
        self.num_results = num_results
        self.session = requests.Session()

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
            # Use a free search API (DuckDuckGo Instant Answer API)
            # This is a simple fallback; for production, use a real search API
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

            # Format results
            results = []
            if data.get("Abstract"):
                results.append({
                    "title": data.get("Heading", query),
                    "source": "DuckDuckGo",
                    "snippet": data["Abstract"],
                    "url": data.get("AbstractURL", "")
                })

            # Add Instant Answer results
            if data.get("Answer"):
                results.append({
                    "title": "Instant Answer",
                    "source": "DuckDuckGo",
                    "snippet": data["Answer"],
                    "url": ""
                })

            # Add Related Topics
            for topic in data.get("RelatedTopics", [])[:self.num_results]:
                if topic.get("Text"):
                    results.append({
                        "title": topic.get("FirstURL", "").split("/")[-1] if topic.get("FirstURL") else "Related",
                        "source": "DuckDuckGo",
                        "snippet": topic["Text"],
                        "url": topic.get("FirstURL", "")
                    })

            # Limit results
            limit = num_results or self.num_results
            results = results[:limit]

            content = json.dumps(results, ensure_ascii=False, indent=2)

            return ToolResult(
                success=True,
                content=content,
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
