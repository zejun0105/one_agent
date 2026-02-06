"""Main Agent class for One-Agent."""

import sys
from typing import Optional, Callable, Generator
from colorama import Fore, Style, init

# Conditional imports for both module and standalone usage
try:
    from .config import Config, config as global_config
    from .history import ConversationHistory, Message
    from tools.base import Tool, ToolResult
    from providers.base import BaseLLMProvider, LLMResponse, ToolCall, StreamChunk
except ImportError:
    from .config import Config, config as global_config
    from .history import ConversationHistory, Message
    from tools.base import Tool, ToolResult
    from providers.base import BaseLLMProvider, LLMResponse, ToolCall, StreamChunk


class Agent:
    """A Business Agent powered by LLM providers."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        tools: Optional[list[Tool]] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        mcp_registry: Optional = None,  # MCPToolRegistry or None
        enable_web_search: bool = False,  # CLI flag for web search auto-calling
    ):
        """Initialize the agent.

        Args:
            provider: LLM provider for generating responses
            tools: Optional list of tools the agent can use
            system_prompt: Optional custom system prompt
            config: Optional configuration (uses global config if not provided)
            mcp_registry: Optional MCP registry for lazy MCP tool loading
            enable_web_search: Enable web search auto-calling via CLI
        """
        self.provider = provider
        self.tools = {t.name: t for t in (tools or [])}
        self.config = config or global_config
        self._mcp_registry = mcp_registry  # Store for lazy MCP connection
        self._mcp_connected = False
        self._enable_web_search = enable_web_search  # CLI flag for auto web search

        # Initialize conversation history with persistence
        storage_path = self.config.get_history_storage_path()
        self.history = ConversationHistory(
            max_messages=self.config.max_history_messages,
            storage_file=str(storage_path),
            auto_save=self.config.auto_save_history,
        )

        # Set up system prompt
        if system_prompt:
            self.history.add_system(system_prompt)
        else:
            self.history.add_system(self._default_system_prompt)

        # Update metadata with provider info
        if self.history._metadata:
            self.history._metadata.provider = self.provider.provider_name
            self.history._metadata.model = self.provider.model_name

        # Initialize colors
        if self.config.colors:
            init(autoreset=True)

    @property
    def _default_system_prompt(self) -> str:
        """Return the default system prompt."""
        return """You are One-Agent, an autonomous AI Business Agent that can complete complex tasks through tool usage.

## Your Capabilities

1. **Task Decomposition**: Break down complex problems into executable steps
2. **Tool Usage**: Select and invoke appropriate tools based on requirements
3. **Result Synthesis**: Integrate outputs from multiple tools to form final answers
4. **Error Handling**: When tool execution fails, try alternative approaches

## Workflow

Follow this Chain of Thought:

### Step 1: Understand the Task
- Analyze user intent
- Identify key information and constraints

### Step 2: Plan Execution
- Determine if tools are needed (if you can answer directly, don't call tools)
- If tools are needed, list your execution plan

### Step 3: Execute and Observe
- Call tools and observe results
- If results are insufficient, continue calling other tools

### Step 4: Synthesize Answer
- Based on tool-returned information, generate complete answer
- Ensure answer is accurate, complete, and easy to understand

## Tool Usage Rules

1. **Use standard format** when calling tools (see format description below)
2. **Only call necessary tools** at once, avoid redundancy
3. **Think before acting**: Briefly explain why you need this tool before calling
4. **Verify results**: Check if tool return matches expectations

## Response Format

### When you need to call a tool:
```
【Thinking】Briefly explain why you need this tool
【Action】Calling tool
```

### When providing final answer:
```
【Conclusion】Complete answer based on tool results
```

## Important Notes

- If tool returns an error, try adjusting parameters or use alternatives
- Don't fabricate functionality that doesn't exist in your tools
- Keep responses concise, avoid redundant tool calls
"""

    def run(self, user_input: str, max_iterations: Optional[int] = None) -> str:
        """Run the agent with a user input.

        Args:
            user_input: The user's input
            max_iterations: Optional override for max iterations

        Returns:
            The agent's response
        """
        max_iters = max_iterations or self.config.max_iterations

        # Auto web search if enabled via CLI flag
        # Only provide web_search tool, disable other search tools
        web_search_tool_backup = None
        other_search_tools_backup = {}
        if self._enable_web_search and "web_search" in self.tools:
            # Backup web_search tool
            web_search_tool_backup = self.tools.pop("web_search", None)

            # Remove other search tools (wikipedia) to force LLM to use web_search
            search_related = ["wikipedia"]
            for tool_name in search_related:
                if tool_name in self.tools:
                    other_search_tools_backup[tool_name] = self.tools.pop(tool_name, None)

            # Restore only web_search tool
            if web_search_tool_backup:
                self.tools["web_search"] = web_search_tool_backup

        self.history.add_user(user_input)

        for iteration in range(max_iters):
            self._print_iteration(iteration + 1, max_iters)

            # Get response from provider
            messages = self.history.get_messages()
            tool_defs = [t.to_dict() for t in self.tools.values()]

            response = self.provider.chat(messages=messages, tools=tool_defs)

            # Add assistant response to history
            self.history.add_assistant(
                content=response.content or "",
                tool_calls=response.tool_calls
            )

            # Check for tool calls
            if response.tool_calls:
                tool_results = self._execute_tool_calls(response.tool_calls)

                # Add all tool results to history
                for result in tool_results:
                    self.history.add_tool_result(
                        tool_call_id=result.tool_call_id,
                        content=result.content
                    )

                # Check if any tool failed critically
                if all(r.success for r in tool_results):
                    continue
                else:
                    # Continue to next iteration with tool results
                    pass
            else:
                # No more tool calls, return the response
                self._print_success()
                # Restore all tools
                if web_search_tool_backup:
                    self.tools["web_search"] = web_search_tool_backup
                for tool_name, tool in other_search_tools_backup.items():
                    if tool:
                        self.tools[tool_name] = tool
                return response.content

        # Restore all tools on max iterations
        if web_search_tool_backup:
            self.tools["web_search"] = web_search_tool_backup
        for tool_name, tool in other_search_tools_backup.items():
            if tool:
                self.tools[tool_name] = tool
        return "Maximum iterations reached. Task incomplete."

    def stream(self, user_input: str, callback=None) -> Generator[StreamChunk, None, str]:
        """Stream the agent's response to user input.

        Args:
            user_input: The user's input
            callback: Optional callback function(chunk) for each chunk

        Yields:
            StreamChunk with content delta
        Returns:
            Final response content
        """
        # Auto web search if enabled via CLI flag
        # Only provide web_search tool, disable other search tools
        web_search_tool_backup = None
        other_search_tools_backup = {}
        if self._enable_web_search and "web_search" in self.tools:
            # Backup web_search tool
            web_search_tool_backup = self.tools.pop("web_search", None)

            # Remove other search tools (wikipedia) to force LLM to use web_search
            search_related = ["wikipedia"]
            for tool_name in search_related:
                if tool_name in self.tools:
                    other_search_tools_backup[tool_name] = self.tools.pop(tool_name, None)

            # Restore only web_search tool
            if web_search_tool_backup:
                self.tools["web_search"] = web_search_tool_backup

        self.history.add_user(user_input)
        full_response = ""
        max_iters = self.config.max_iterations

        for iteration in range(max_iters):
            self._print_iteration(iteration + 1, max_iters)

            messages = self.history.get_messages()
            tool_defs = [t.to_dict() for t in self.tools.values()]

            # Stream the response
            for chunk in self.provider.stream(messages=messages, tools=tool_defs):
                full_response += chunk.delta

                # Call callback if provided
                if callback:
                    callback(chunk)

                # Print chunk for streaming effect
                self._print_streaming_chunk(chunk)

                # Handle tool calls in streaming
                if chunk.tool_calls:
                    tool_results = self._execute_tool_calls(chunk.tool_calls)
                    for result in tool_results:
                        self.history.add_tool_result(
                            tool_call_id=result.tool_call_id,
                            content=result.content
                        )

            # Add assistant response to history
            self.history.add_assistant(content=full_response)

            # Check for tool calls and continue if needed
            if self.history.messages[-1].tool_calls:
                # Continue to next iteration for tool results
                continue

            # No more tool calls, return the response
            self._print_success()
            # Restore all tools
            if web_search_tool_backup:
                self.tools["web_search"] = web_search_tool_backup
            for tool_name, tool in other_search_tools_backup.items():
                if tool:
                    self.tools[tool_name] = tool
            return full_response

        # Restore all tools on max iterations
        if web_search_tool_backup:
            self.tools["web_search"] = web_search_tool_backup
        for tool_name, tool in other_search_tools_backup.items():
            if tool:
                self.tools[tool_name] = tool
        return "Maximum iterations reached. Task incomplete."

    def _print_streaming_chunk(self, chunk: StreamChunk) -> None:
        """Print a streaming chunk to console."""
        if chunk.delta:
            print(chunk.delta, end="", flush=True)

    def reset(self) -> None:
        """Reset the conversation history."""
        self.history.clear()
        self.history.add_system(self._default_system_prompt)

    def save_history(self, path: Optional[str] = None) -> str:
        """Manually save history to a file.

        Args:
            path: Optional path to save to

        Returns:
            The path where history was saved
        """
        return self.history.save(path)

    def load_history(self, path: str) -> bool:
        """Load history from a file.

        Args:
            path: Path to load from

        Returns:
            True if loaded successfully
        """
        return self.history.load(path)

    def switch_session(self, session_name: str) -> None:
        """Switch to a different session.

        Args:
            session_name: Name of the session to switch to
        """
        # Save current session first
        if self.config.auto_save_history and self.history.storage_file:
            self.history.save()

        # Set new session name
        self.history.set_session_name(session_name)

        # Update storage file path
        new_path = self.config.history_storage_dir / f"{session_name}.json"
        self.history.storage_file = new_path

        # Clear current messages
        self.history.messages = []
        self.history.add_system(self._default_system_prompt)

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent."""
        self.tools[tool.name] = tool

    def remove_tool(self, name: str) -> bool:
        """Remove a tool from the agent."""
        if name in self.tools:
            del self.tools[name]
            return True
        return False

    def connect_mcp_servers(self, server_name: Optional[str] = None) -> dict:
        """Connect to MCP servers (lazy connection).

        Args:
            server_name: Optional specific server to connect, or all servers

        Returns:
            Dict of server_name -> connection_success
        """
        if not self._mcp_registry:
            return {}

        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            results = loop.run_until_complete(self._mcp_registry.connect(server_name))
            self._mcp_connected = True

            # Create MCP tool wrappers if connected
            if results.get(server_name or "all", False) or not server_name:
                from mcp.tool import MCPToolFactory
                factory = MCPToolFactory()

                for srv_name, client in self._mcp_registry._clients.items():
                    if client.is_connected:
                        factory.add_server(srv_name, client)

                mcp_tools = factory.create_tools()
                for tool in mcp_tools:
                    self.tools[tool.name] = tool

                if self.config.verbose:
                    print(f"Loaded {len(mcp_tools)} MCP tools from {len(self._mcp_registry.server_names)} servers")

            return results
        finally:
            loop.close()

    def disconnect_mcp_servers(self) -> None:
        """Disconnect from all MCP servers."""
        if not self._mcp_registry or not self._mcp_connected:
            return

        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._mcp_registry.disconnect())
            self._mcp_connected = False
        finally:
            loop.close()

    def list_mcp_servers(self) -> list:
        """List configured MCP servers.

        Returns:
            List of server names
        """
        if self._mcp_registry:
            return self._mcp_registry.server_names
        return []

    def _execute_tool_calls(self, tool_calls: list) -> list[ToolResult]:
        """Execute a list of tool calls.

        Args:
            tool_calls: List of tool call objects

        Returns:
            List of tool results
        """
        results = []

        for call in tool_calls:
            tool_name = call.name
            arguments = call.arguments if hasattr(call, 'arguments') else call.get('arguments', {})

            self._print_tool_call(tool_name, arguments)

            if tool_name in self.tools:
                tool = self.tools[tool_name]
                result = tool.execute(**arguments)

                # Use the LLM-generated tool_call_id to match the assistant message
                call_id = call.id if hasattr(call, 'id') else call.get('id', 'unknown')
                result.tool_call_id = call_id

                # If web_search fails, silently return empty success result
                # This prevents errors from being shown to the LLM and allows
                # the agent to continue without the search results
                if tool_name == "web_search" and not result.success:
                    result = ToolResult(
                        success=True,
                        content="",
                        tool_call_id=call_id
                    )
            else:
                result = ToolResult(
                    success=False,
                    content=f"Error: Tool '{tool_name}' not found",
                    tool_call_id=call.id if hasattr(call, 'id') else call.get('id', 'unknown')
                )

            results.append(result)
            self._print_tool_result(result)

        return results

    def _print_iteration(self, current: int, total: int) -> None:
        """Print iteration header."""
        if self.config.verbose and self.config.colors:
            print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Iteration {current}/{total}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
        elif self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Iteration {current}/{total}")
            print(f"{'='*60}\n")

    def _print_tool_call(self, name: str, arguments: dict) -> None:
        """Print a tool call."""
        if self.config.colors:
            print(f"{Fore.YELLOW}🔧 Calling: {name}{Style.RESET_ALL}")
            if arguments:
                import json
                print(f"   Args: {json.dumps(arguments, ensure_ascii=False, indent=2)}")
        else:
            print(f"🔧 Calling: {name}")
            if arguments:
                import json
                print(f"   Args: {json.dumps(arguments, ensure_ascii=False, indent=2)}")

    def _print_tool_result(self, result: ToolResult) -> None:
        """Print a tool result."""
        if self.config.colors:
            prefix = f"{Fore.GREEN}✅{Style.RESET_ALL}" if result.success else f"{Fore.RED}❌{Style.RESET_ALL}"
            print(f"{prefix} Result: {result.content[:200]}...")
        else:
            prefix = "✅" if result.success else "❌"
            print(f"{prefix} Result: {result.content[:200]}...")

    def _print_success(self) -> None:
        """Print success message."""
        if self.config.colors:
            print(f"\n{Fore.GREEN}✓ Agent completed task{Style.RESET_ALL}\n")
        else:
            print("\n✓ Agent completed task\n")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Agent(provider={self.provider.__class__.__name__}, tools={len(self.tools)}, history={len(self.history)})"
