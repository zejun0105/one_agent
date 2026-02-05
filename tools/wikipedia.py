"""Wikipedia search tool."""

import json
import uuid
from typing import Optional
import requests
from .base import Tool, ToolResult


class WikipediaTool(Tool):
    """Tool for searching Wikipedia."""

    def __init__(
        self,
        name: str = "wikipedia",
        description: str = "Search Wikipedia for information",
        lang: str = "en",
        num_results: int = 5
    ):
        """Initialize Wikipedia search tool.

        Args:
            name: Tool name
            description: Tool description
            lang: Language code (en, zh, ja, etc.)
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
                        "description": "Search query"
                    },
                    "lang": {
                        "type": "string",
                        "description": "Language code (default: en)"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": f"Number of results (default: {num_results})"
                    }
                },
                "required": ["query"]
            }
        )
        self.lang = lang
        self.num_results = num_results
        self.session = requests.Session()
        self.base_url = f"https://{lang}.wikipedia.org/w/api.php"

    def execute(
        self,
        query: str,
        lang: str = None,
        num_results: int = None
    ) -> ToolResult:
        """Search Wikipedia.

        Args:
            query: Search query
            lang: Optional language override
            num_results: Optional number of results override

        Returns:
            ToolResult with search results
        """
        import uuid

        query_lang = lang or self.lang
        limit = num_results or self.num_results
        url = f"https://{query_lang}.wikipedia.org/w/api.php"
        tool_id = f"wiki_{uuid.uuid4().hex[:8]}"

        try:
            # Search for pages
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "format": "json"
            }

            response = self.session.get(url, params=search_params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("query", {}).get("search", []):
                # Get page summary
                page_id = item["pageid"]
                summary_params = {
                    "action": "query",
                    "prop": "extracts|pageimages",
                    "exintro": True,
                    "explaintext": True,
                    "piprop": "thumbnail",
                    "pithumbsize": 300,
                    "pageids": page_id,
                    "format": "json"
                }

                summary_response = self.session.get(url, params=summary_params, timeout=10)
                summary_data = summary_response.json()
                pages = summary_data.get("query", {}).get("pages", {})
                page_data = pages.get(str(page_id), {})

                results.append({
                    "title": item["title"],
                    "snippet": item["snippet"],
                    "pageid": page_id,
                    "url": f"https://{query_lang}.wikipedia.org/wiki/{item['title'].replace(' ', '_')}",
                    "extract": page_data.get("extract", "")[:500],
                    "thumbnail": page_data.get("thumbnail", {}).get("source")
                })

            content = json.dumps({
                "query": query,
                "language": query_lang,
                "results": results
            }, ensure_ascii=False, indent=2)

            return ToolResult(
                success=True,
                content=content,
                tool_call_id=tool_id
            )

        except requests.RequestException as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Wikipedia search failed: {str(e)}",
                tool_call_id=tool_id
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Wikipedia error: {str(e)}",
                tool_call_id=tool_id
            )

    def get_page(self, title: str, lang: str = None) -> ToolResult:
        """Get a specific Wikipedia page.

        Args:
            title: Page title
            lang: Optional language override

        Returns:
            ToolResult with page content
        """
        import uuid

        query_lang = lang or self.lang
        url = f"https://{query_lang}.wikipedia.org/w/api.php"
        tool_id = f"wiki_page_{uuid.uuid4().hex[:8]}"

        try:
            params = {
                "action": "query",
                "prop": "extracts|revisions|categories",
                "explaintext": True,
                "rvprop": "content",
                "clprop": "timestamp",
                "titles": title,
                "format": "json"
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id == "-1":
                    return ToolResult(
                        success=False,
                        content="",
                        error=f"Page not found: {title}",
                        tool_call_id=tool_id
                    )

                return ToolResult(
                    success=True,
                    content=json.dumps({
                        "title": page_data.get("title"),
                        "extract": page_data.get("extract", ""),
                        "url": f"https://{query_lang}.wikipedia.org/wiki/{title.replace(' ', '_')}",
                        "categories": [c["title"].replace("Category:", "")
                                     for c in page_data.get("categories", [])]
                    }, ensure_ascii=False, indent=2),
                    tool_call_id=tool_id
                )

            return ToolResult(
                success=False,
                content="",
                error="Page not found",
                tool_call_id=tool_id
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Wikipedia error: {str(e)}",
                tool_call_id=tool_id
            )
