"""MCP Tool Registry for managing multiple MCP servers."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from .client import MCPClient, MCPServerConfig


@dataclass
class RegisteredTool:
    """A registered MCP tool."""
    name: str
    server_name: str
    description: str
    parameters: Dict[str, Any]
    is_available: bool = True


@dataclass
class ServerStatus:
    """Status of an MCP server."""
    name: str
    connected: bool
    tool_count: int
    last_error: Optional[str] = None


class MCPToolRegistry:
    """Registry for managing MCP servers and tools."""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize registry.

        Args:
            config_file: Optional path to MCP config file
        """
        self._clients: Dict[str, MCPClient] = {}
        self._servers: Dict[str, MCPServerConfig] = {}
        self._tools: List[RegisteredTool] = []
        self._config_file = config_file

    @property
    def servers(self) -> Dict[str, MCPServerConfig]:
        """Get configured servers."""
        return self._servers

    @property
    def tools(self) -> List[RegisteredTool]:
        """Get all registered tools."""
        return self._tools

    @property
    def server_names(self) -> List[str]:
        """Get list of server names."""
        return list(self._servers.keys())

    def add_server(self, config: MCPServerConfig) -> None:
        """Add an MCP server configuration.

        Args:
            config: Server configuration
        """
        self._servers[config.name] = config
        client = MCPClient(config)
        self._clients[config.name] = client

    def remove_server(self, name: str) -> bool:
        """Remove an MCP server.

        Args:
            name: Server name

        Returns:
            True if removed, False if not found
        """
        if name in self._servers:
            del self._servers[name]
            if name in self._clients:
                client = self._clients.pop(name)
                # Disconnect if connected
                import asyncio
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(client.disconnect())
                except Exception:
                    pass
            return True
        return False

    def get_server(self, name: str) -> Optional[MCPServerConfig]:
        """Get server configuration.

        Args:
            name: Server name

        Returns:
            Server configuration or None
        """
        return self._servers.get(name)

    def get_client(self, name: str) -> Optional[MCPClient]:
        """Get MCP client for a server.

        Args:
            name: Server name

        Returns:
            MCP client or None
        """
        return self._clients.get(name)

    async def connect(self, name: Optional[str] = None) -> Dict[str, bool]:
        """Connect to MCP server(s).

        Args:
            name: Optional specific server name, or all servers

        Returns:
            Dict of server_name -> connection_success
        """
        results = {}

        if name:
            if name in self._clients:
                client = self._clients[name]
                success = await client.connect()
                results[name] = success
                if success:
                    self._register_tools(name, client)
        else:
            for server_name, client in self._clients.items():
                try:
                    success = await client.connect()
                    results[server_name] = success
                    if success:
                        self._register_tools(server_name, client)
                except Exception as e:
                    print(f"Failed to connect to {server_name}: {e}")
                    results[server_name] = False

        return results

    async def disconnect(self, name: Optional[str] = None) -> None:
        """Disconnect from MCP server(s).

        Args:
            name: Optional specific server name, or all servers
        """
        if name:
            if name in self._clients:
                await self._clients[name].disconnect()
        else:
            for client in self._clients.values():
                await client.disconnect()

    def _register_tools(self, server_name: str, client: MCPClient) -> None:
        """Register tools from a connected server.

        Args:
            server_name: Server name
            client: MCP client instance
        """
        for tool_name, tool_def in client.tools.items():
            tool = RegisteredTool(
                name=f"mcp_{server_name}_{tool_name}",
                server_name=server_name,
                description=tool_def.description,
                parameters=tool_def.input_schema,
            )
            self._tools.append(tool)

    def get_tool(self, name: str) -> Optional[RegisteredTool]:
        """Get a registered tool by name.

        Args:
            name: Tool name

        Returns:
            RegisteredTool or None
        """
        for tool in self._tools:
            if tool.name == name:
                return tool
        return None

    def list_tools(self, server_name: Optional[str] = None) -> List[RegisteredTool]:
        """List registered tools.

        Args:
            server_name: Optional filter by server

        Returns:
            List of registered tools
        """
        if server_name:
            return [t for t in self._tools if t.server_name == server_name]
        return self._tools

    def list_servers(self) -> List[ServerStatus]:
        """List all servers with their status.

        Returns:
            List of ServerStatus
        """
        statuses = []
        for name, config in self._servers.items():
            client = self._clients.get(name)
            connected = client.is_connected if client else False
            tool_count = len([t for t in self._tools if t.server_name == name])

            statuses.append(ServerStatus(
                name=name,
                connected=connected,
                tool_count=tool_count,
            ))

        return statuses

    def save_config(self, path: Optional[str] = None) -> str:
        """Save registry configuration to file.

        Args:
            path: Optional path, uses config_file if not provided

        Returns:
            Path to saved file
        """
        save_path = Path(path or self._config_file or "mcp_servers.json")

        configs = []
        for config in self._servers.values():
            configs.append(config.to_dict())

        # Save in the same format as mcp_config.example.json
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"servers": configs}, f, ensure_ascii=False, indent=2)

        return str(save_path)

    def load_config(self, path: Optional[str] = None) -> bool:
        """Load registry configuration from file.

        Args:
            path: Optional path, uses config_file if not provided

        Returns:
            True if loaded successfully
        """
        load_path = Path(path or self._config_file or "mcp_servers.json")

        if not load_path.exists():
            return False

        try:
            with open(load_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle both formats: {"servers": [...]} or [...]
            if isinstance(data, dict) and "servers" in data:
                configs = data["servers"]
            elif isinstance(data, list):
                configs = data
            else:
                print(f"Invalid MCP config format in {load_path}")
                return False

            for config_dict in configs:
                config = MCPServerConfig.from_dict(config_dict)
                self.add_server(config)

            return True

        except Exception as e:
            print(f"Failed to load MCP config: {e}")
            return False

    @classmethod
    def from_mcp_config(cls, config_file: str = "mcp_servers.json") -> "MCPToolRegistry":
        """Create registry from MCP configuration file.

        Args:
            config_file: Path to MCP servers config

        Returns:
            Initialized registry
        """
        registry = cls(config_file=config_file)
        registry.load_config()
        return registry
