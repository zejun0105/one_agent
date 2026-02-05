"""MCP tool wrapper for One-Agent."""

import uuid
import sys
from pathlib import Path
from typing import Any, Optional, Dict
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.base import Tool, ToolResult

# Prefix for MCP tool names
MCP_TOOL_PREFIX = "mcp_"


@dataclass
class MCPToolInfo:
    """Information about an MCP tool."""
    server_name: str
    original_name: str
    description: str
    input_schema: Dict[str, Any]


class MCPTool(Tool):
    """Wrapper for MCP tools to work with One-Agent."""

    def __init__(
        self,
        mcp_client,
        tool_info: MCPToolInfo,
        name: Optional[str] = None,
    ):
        """Initialize MCP tool wrapper.

        Args:
            mcp_client: MCP client instance
            tool_info: MCP tool information
            name: Optional custom name (defaults to mcp_{server}_{original_name})
        """
        self.mcp_client = mcp_client
        self.tool_info = tool_info
        self.server_name = tool_info.server_name

        # Generate tool name
        tool_name = name or f"{MCP_TOOL_PREFIX}{tool_info.server_name}_{tool_info.original_name}"

        # Convert MCP input schema to our format
        parameters = self._convert_schema(tool_info.input_schema)

        super().__init__(
            name=tool_name,
            description=f"[{tool_info.server_name}] {tool_info.description}",
            parameters=parameters,
        )

    def _convert_schema(self, mcp_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert MCP input schema to our tool schema format."""
        return {
            "type": "object",
            "properties": mcp_schema.get("properties", {}),
            "required": mcp_schema.get("required", []),
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute MCP tool.

        Args:
            **kwargs: Tool arguments

        Returns:
            ToolResult with execution result
        """
        import asyncio

        tool_id = f"mcp_{self.server_name}_{uuid.uuid4().hex[:8]}"

        try:
            # Run async MCP call in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                result = loop.run_until_complete(
                    self.mcp_client.call_tool(self.tool_info.original_name, kwargs)
                )
            finally:
                loop.close()

            return ToolResult(
                success=result.success,
                content=result.content,
                error=result.error,
                tool_call_id=tool_id,
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"MCP tool error: {str(e)}",
                tool_call_id=tool_id,
            )


class MCPToolFactory:
    """Factory for creating MCP tool wrappers."""

    def __init__(self):
        """Initialize tool factory."""
        self._clients: dict = {}
        self._tools: list = []

    def add_server(self, name: str, client) -> None:
        """Add an MCP server client.

        Args:
            name: Server name
            client: MCP client instance
        """
        self._clients[name] = client

    def create_tools(self) -> list:
        """Create tool wrappers for all available MCP tools.

        Returns:
            List of MCPTool instances
        """
        tools = []

        for server_name, client in self._clients.items():
            if not client.is_connected:
                continue

            for tool_name, tool_def in client.tools.items():
                tool_info = MCPToolInfo(
                    server_name=server_name,
                    original_name=tool_name,
                    description=tool_def.description,
                    input_schema=tool_def.input_schema,
                )

                tool = MCPTool(mcp_client=client, tool_info=tool_info)
                tools.append(tool)

        self._tools = tools
        return tools

    def get_tool_names(self) -> list:
        """Get names of all available MCP tools."""
        return [t.name for t in self._tools]

    async def connect_all(self) -> dict:
        """Connect to all configured servers.

        Returns:
            Dict of server_name -> connection_success
        """
        results = {}

        for name, client in self._clients.items():
            try:
                success = await client.connect()
                results[name] = success
            except Exception as e:
                print(f"Failed to connect to MCP server '{name}': {e}")
                results[name] = False

        return results

    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        for client in self._clients.values():
            await client.disconnect()


def create_mcp_tools_from_config(mcp_configs: list) -> tuple:
    """Create MCP clients and tools from configuration.

    Args:
        mcp_configs: List of MCP server configurations

    Returns:
        Tuple of (factory, clients_dict)
    """
    from .client import MCPClient, MCPServerConfig

    factory = MCPToolFactory()
    clients = {}

    for config_dict in mcp_configs:
        config = MCPServerConfig.from_dict(config_dict)
        client = MCPClient(config)
        factory.add_server(config.name, client)
        clients[config.name] = client

    return factory, clients
