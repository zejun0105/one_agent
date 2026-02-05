"""One-Agent MCP Integration Module

Provides Model Context Protocol (MCP) server integration:
- MCP client connections
- Tool translation from MCP format
- Server management
"""

from .client import MCPClient, MCPServerConfig
from .tool import MCPTool, MCP_TOOL_PREFIX, create_mcp_tools_from_config
from .registry import MCPToolRegistry

__all__ = [
    "MCPClient",
    "MCPServerConfig",
    "MCPTool",
    "MCPToolRegistry",
    "MCP_TOOL_PREFIX",
    "create_mcp_tools_from_config",
]
