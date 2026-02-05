"""MCP client implementation."""

import json
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path


# MCP message types
MCP_METHOD_INITIALIZE = "initialize"
MCP_METHOD_TOOLS_LIST = "tools/list"
MCP_METHOD_TOOLS_CALL = "tools/call"
MCP_METHOD_RESOURCES_LIST = "resources/list"
MCP_METHOD_RESOURCES_READ = "resources/read"
MCP_METHOD_PROMPTS_LIST = "prompts/list"
MCP_METHOD_PROMPTS_GET = "prompts/get"

MCP_NOTIFICATION_INITIALIZED = "notifications/initialized"


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30  # seconds

    @classmethod
    def from_dict(cls, data: dict) -> "MCPServerConfig":
        """Create from dictionary."""
        return cls(
            name=data.get("name", "unknown"),
            command=data.get("command", ""),
            args=data.get("args", []),
            env=data.get("env", {}),
            timeout=data.get("timeout", 30),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "command": self.command,
            "args": self.args,
            "env": self.env,
            "timeout": self.timeout,
        }


@dataclass
class MCPToolDefinition:
    """Definition of an MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mcp(cls, data: dict) -> "MCPToolDefinition":
        """Create from MCP format."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            input_schema=data.get("inputSchema", {}),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass
class MCPToolResult:
    """Result from calling an MCP tool."""
    success: bool
    content: str
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "content": self.content,
            "error": self.error,
        }


class MCPClient:
    """Client for connecting to MCP servers."""

    def __init__(self, config: MCPServerConfig):
        """Initialize MCP client.

        Args:
            config: Server configuration
        """
        self.config = config
        self._process: Optional[Any] = None
        self._reader: Optional[Any] = None
        self._writer: Optional[Any] = None
        self._request_id = 0
        self._connected = False
        self._tools: Dict[str, MCPToolDefinition] = {}

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._connected

    @property
    def tools(self) -> Dict[str, MCPToolDefinition]:
        """Get available tools."""
        return self._tools

    async def connect(self) -> bool:
        """Connect to MCP server.

        Returns:
            True if connected successfully
        """
        try:
            import subprocess

            # Start the MCP server process
            env = dict(os.environ)
            env.update(self.config.env)
            self._process = subprocess.Popen(
                [self.config.command] + self.config.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=0,
            )

            # Use stdio for communication
            self._reader = self._process.stdout
            self._writer = self._process.stdin

            # Initialize the connection
            await self._send_request(MCP_METHOD_INITIALIZE, {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {},
                },
                "clientInfo": {
                    "name": "one-agent",
                    "version": "1.0.0",
                },
            })

            # Wait for initialization response
            response = await self._read_response()
            if response and response.get("result"):
                self._connected = True

                # List available tools
                await self._list_tools()

                return True

            return False

        except Exception as e:
            print(f"MCP connection error: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
        self._connected = False
        if self._process:
            self._process.terminate()
            self._process = None
        self._reader = None
        self._writer = None

    async def _send_request(self, method: str, params: Dict = None) -> Optional[Dict]:
        """Send a JSON-RPC request.

        Args:
            method: The method name
            params: Request parameters

        Returns:
            Response data or None
        """
        if not self._writer:
            return None

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params or {},
        }

        try:
            # Use lock for thread safety
            import threading
            if not hasattr(self, '_write_lock'):
                self._write_lock = threading.Lock()

            with self._write_lock:
                self._writer.write(json.dumps(request) + "\n")
                self._writer.flush()

            return request

        except Exception as e:
            print(f"MCP send error: {e}")
            return None

    async def _read_response(self) -> Optional[Dict]:
        """Read a JSON-RPC response.

        Returns:
            Response data or None
        """
        if not self._reader:
            return None

        try:
            line = self._reader.readline()
            if line:
                return json.loads(line.strip())
            return None

        except Exception as e:
            print(f"MCP read error: {e}")
            return None

    async def _list_tools(self) -> None:
        """List available tools from server."""
        await self._send_request(MCP_METHOD_TOOLS_LIST)
        response = await self._read_response()

        if response and response.get("result"):
            tools = response["result"].get("tools", [])
            self._tools = {
                tool["name"]: MCPToolDefinition.from_mcp(tool)
                for tool in tools
            }

    async def call_tool(self, name: str, arguments: Dict = None) -> MCPToolResult:
        """Call an MCP tool.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result
        """
        if not self._connected:
            return MCPToolResult(
                success=False,
                content="",
                error="Not connected to MCP server",
            )

        await self._send_request(MCP_METHOD_TOOLS_CALL, {
            "name": name,
            "arguments": arguments or {},
        })

        response = await self._read_response()

        if response and response.get("result"):
            result = response["result"]
            return MCPToolResult(
                success=True,
                content=result.get("content", ""),
            )

        return MCPToolResult(
            success=False,
            content="",
            error=response.get("error", {}).get("message", "Unknown error") if response else "No response",
        )

    async def list_resources(self) -> List[Dict]:
        """List available resources.

        Returns:
            List of resource definitions
        """
        if not self._connected:
            return []

        await self._send_request(MCP_METHOD_RESOURCES_LIST)
        response = await self._read_response()

        if response and response.get("result"):
            return response["result"].get("resources", [])

        return []

    async def read_resource(self, uri: str) -> Optional[str]:
        """Read a resource.

        Args:
            uri: Resource URI

        Returns:
            Resource content or None
        """
        if not self._connected:
            return None

        await self._send_request(MCP_METHOD_RESOURCES_READ, {"uri": uri})
        response = await self._read_response()

        if response and response.get("result"):
            return response["result"].get("contents", [{}])[0].get("text", "")

        return None


# Import os for environment handling
import os
