#!/usr/bin/env python3
"""
One-Agent: Multi-Model Business Agent using Claude Agent SDK

A modular Business Agent framework with support for:
- Multiple LLM providers (Anthropic Claude, OpenAI GPT-4, GLM-4, Kimi)
- Tool calling capabilities
- Conversation history with persistence
- Configuration management
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Import core modules
from core import Agent, Config, config as global_config
from core.config import ProviderConfig
from core.history import ConversationHistory

# Import providers
from providers import (
    AnthropicProvider,
    OpenAIProvider,
    CompatibleProvider,
)

# Import tools
from tools import (
    WebSearchTool,
    CalculatorTool,
    PythonCodeTool,
    FileReadTool,
    FileWriteTool,
    SystemCommandTool,
    WikipediaTool,
)

# Import MCP modules
from mcp import (
    MCPClient,
    MCPServerConfig,
    MCPToolRegistry,
    create_mcp_tools_from_config,
    MCP_TOOL_PREFIX,
)


def load_config(env_file: Optional[str] = None) -> Config:
    """Load configuration from environment.

    Args:
        env_file: Optional path to .env file

    Returns:
        Config object
    """
    if env_file:
        env_path = os.path.expanduser(env_file)
        if os.path.exists(env_path):
            load_dotenv(env_path)
    else:
        # Try default .env
        if os.path.exists(".env"):
            load_dotenv(".env")

    return Config.load()


def create_provider(provider_config: ProviderConfig) -> Optional:
    """Create an LLM provider from config.

    Args:
        provider_config: Provider configuration

    Returns:
        Provider instance or None if not configured
    """
    if not provider_config.api_key:
        return None

    provider_type = provider_config.provider.lower()

    if provider_type == "anthropic":
        return AnthropicProvider(
            api_key=provider_config.api_key,
            model=provider_config.model,
            max_tokens=provider_config.max_tokens,
            temperature=provider_config.temperature,
        )
    elif provider_type == "openai":
        return OpenAIProvider(
            api_key=provider_config.api_key,
            model=provider_config.model,
            base_url=provider_config.base_url,
            max_tokens=provider_config.max_tokens,
            temperature=provider_config.temperature,
        )
    elif provider_type in ("glm", "kimi", "compatible"):
        return CompatibleProvider(
            api_key=provider_config.api_key,
            model=provider_config.model,
            base_url=provider_config.base_url or "https://open.bigmodel.cn/api/paas/v4",
            max_tokens=provider_config.max_tokens,
            temperature=provider_config.temperature,
        )

    return None


def create_tools(config: Config, enable_web_search_cli: bool = False) -> tuple:
    """Create tools based on configuration.

    Args:
        config: Configuration object
        enable_web_search_cli: CLI flag to enable web search auto-calling

    Returns:
        Tuple of (list of Tool instances, MCP registry if enabled)
    """
    tools = []

    # Core tools
    # web_search is controlled by CLI flag, not by LLM semantic decision
    if config.enable_web_search or enable_web_search_cli:
        tools.append(WebSearchTool(
            provider=config.web_search_provider,
            api_key=config.google_api_key,
            search_engine_id=config.google_search_engine_id,
        ))

    if config.enable_calculator:
        tools.append(CalculatorTool())

    # Advanced tools
    if config.enable_python_code:
        tools.append(PythonCodeTool())

    if config.enable_file_read:
        tools.append(FileReadTool())

    if config.enable_file_write:
        tools.append(FileWriteTool())

    if config.enable_system:
        tools.append(SystemCommandTool())

    if config.enable_wikipedia:
        tools.append(WikipediaTool())

    # MCP tools
    mcp_registry = None
    if config.enable_mcp:
        mcp_registry = MCPToolRegistry.from_mcp_config(config.mcp_config_file)
        if mcp_registry.server_names:
            # MCP tools will be added after connection
            pass

    return tools, mcp_registry


def create_agent(
    provider_name: Optional[str] = None,
    config: Optional[Config] = None,
    verbose: bool = False,
    enable_web_search: bool = False,
) -> Optional[Agent]:
    """Create an agent instance.

    Args:
        provider_name: Name of provider to use
        config: Optional configuration override
        verbose: Enable verbose output
        enable_web_search: Enable web search auto-calling via CLI

    Returns:
        Agent instance or None if no provider available
    """
    cfg = config or global_config

    if verbose:
        cfg.verbose = True

    # Determine provider
    provider_key = provider_name or cfg.default_provider

    if provider_key not in cfg.providers:
        print(f"Error: Provider '{provider_key}' not configured.")
        print(f"\nAvailable providers: {list(cfg.providers.keys())}")
        print("\nPlease set the appropriate API key in .env:")
        print("  ANTHROPIC_API_KEY=your_key      # For Claude")
        print("  OPENAI_API_KEY=your_key          # For GPT-4")
        print("  GLM_API_KEY=your_key             # For GLM-4")
        print("  KIMI_API_KEY=your_key           # For Kimi")
        return None

    # Create provider
    provider_config = cfg.providers[provider_key]
    provider = create_provider(provider_config)

    if not provider:
        print(f"Error: Could not create provider '{provider_key}'")
        return None

    # Create tools
    tools, mcp_registry = create_tools(cfg, enable_web_search_cli=enable_web_search)

    # NOTE: MCP servers are NOT connected at startup to avoid hanging.
    # Use --mcp-connect to test connections, or use /mcp in interactive mode.
    # MCP tools will be available after connection (lazy loading).
    mcp_tools = []

    # Create agent
    agent = Agent(
        provider=provider,
        tools=tools,
        config=cfg,
        mcp_registry=mcp_registry,  # Pass registry for lazy connection
        enable_web_search=enable_web_search,  # Store CLI flag
    )

    return agent


def parse_api_error(error: Exception) -> str:
    """Parse API error and return user-friendly message.

    Args:
        error: The exception object

    Returns:
        User-friendly error message
    """
    error_str = str(error)

    # Try to extract JSON error details
    try:
        if hasattr(error, 'response') and error.response:
            response_data = error.response.json()
            error_data = response_data.get('error', {})
            code = error_data.get('code', '')
            message = error_data.get('message', '')
            return f"[{code}] {message}" if code else message
    except:
        pass

    # Try to parse from string
    try:
        if 'RateLimitError' in error_str or '速率限制' in error_str:
            return "Rate limit exceeded. Please slow down your requests."
        if 'AuthenticationError' in error_str or 'API key' in error_str.lower():
            return "Authentication failed. Please check your API key."
        if 'BadRequestError' in error_str or 'invalid' in error_str.lower():
            return f"Invalid request: {error_str}"
        if 'NotFoundError' in error_str or 'not found' in error_str.lower():
            return f"Resource not found: {error_str}"
    except:
        pass

    return error_str


def interactive_mode(agent: Agent, stream: bool = False) -> None:
    """Run agent in interactive mode.

    Args:
        agent: Agent instance
        stream: Whether to use streaming
    """
    print("=" * 60)
    print("  One-Agent - Multi-Model Business Agent")
    print("=" * 60)
    print(f"\nAgent initialized: {agent}")
    print(f"Provider: {agent.provider.provider_name} ({agent.provider.model_name})")
    print(f"History: {len(agent.history)} messages (auto-save: {agent.config.auto_save_history})")
    print(f"Storage: {agent.history.storage_file}")
    print(f"Streaming: {'enabled' if stream else 'disabled'}")
    print(f"Available tools: {list(agent.tools.keys())}")

    # Show MCP servers if configured
    mcp_servers = agent.list_mcp_servers()
    if mcp_servers:
        print(f"MCP servers: {mcp_servers} (use /mcp to connect)")

    print("\nCommands:")
    print("  /help       - Show this help")
    print("  /reset      - Reset conversation")
    print("  /save       - Save history manually")
    print("  /sessions   - List all sessions")
    print("  /switch NAME - Switch to a different session")
    print("  /export PATH - Export history to file (json or text)")
    print("  /clear      - Clear current session history")
    print("  /stream     - Toggle streaming on/off")
    print("  /mcp        - Connect and list MCP servers/tools")
    print("  quit        - Exit")
    print("-" * 60)

    # Keywords to trigger web search in interactive mode
    SEARCH_KEYWORDS = ["搜索", "search", "查找", "最新", "新闻", "recent", "latest"]

    try:
        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Command handling
            if user_input.startswith("/"):
                cmd = user_input.lower().split()[0]
                args = user_input.split()[1:] if " " in user_input else []

                if cmd == "/help":
                    print("\nAvailable commands:")
                    print("  /help       - Show this help")
                    print("  /reset      - Reset conversation")
                    print("  /save       - Save history manually")
                    print("  /sessions   - List all sessions")
                    print("  /switch NAME - Switch to a different session")
                    print("  /export PATH - Export history (e.g., /export ./history.txt)")
                    print("  /clear      - Clear current session history")
                    print("  /stream     - Toggle streaming on/off")
                    print("  /mcp        - List MCP servers and tools")
                    print("  quit        - Exit")
                    continue

                elif cmd == "/reset":
                    agent.reset()
                    print("Conversation reset.")
                    continue

                elif cmd == "/save":
                    path = agent.save_history()
                    print(f"History saved to: {path}")
                    continue

                elif cmd == "/sessions":
                    sessions = agent.history.list_sessions(agent.config.history_storage_dir)
                    if not sessions:
                        print("No saved sessions found.")
                    else:
                        print(f"\nSessions in {agent.config.history_storage_dir}:")
                        for i, sess in enumerate(sessions, 1):
                            current = " (current)" if sess["name"] == agent.history.metadata.session_name else ""
                            print(f"  {i}. {sess['name']}{current}")
                            print(f"     Messages: {sess['message_count']}, Updated: {sess['updated_at'][:19].replace('T', ' ')}")
                    continue

                elif cmd == "/switch":
                    if not args:
                        print("Usage: /switch SESSION_NAME")
                    else:
                        session_name = args[0]
                        agent.switch_session(session_name)
                        print(f"Switched to session: {session_name}")
                    continue

                elif cmd == "/export":
                    if not args:
                        print("Usage: /export PATH [--format json|text]")
                    else:
                        path = args[0]
                        fmt = "text" if "--format" in args and args[-1] != "text" else "json"
                        if "--format" in args:
                            idx = args.index("--format")
                            if idx + 1 < len(args):
                                fmt = args[idx + 1]
                        saved_path = agent.history.export(path, fmt)
                        print(f"History exported to: {saved_path}")
                    continue

                elif cmd == "/clear":
                    agent.history.clear()
                    print("History cleared.")
                    continue

                elif cmd == "/stream":
                    stream = not stream
                    print(f"Streaming {'enabled' if stream else 'disabled'}.")
                    continue

                elif cmd == "/mcp":
                    import asyncio

                    servers = agent.list_mcp_servers()
                    if not servers:
                        print("No MCP servers configured.")
                        print(f"Create mcp_servers.json or set MCP_CONFIG_FILE env var.")
                    else:
                        print(f"\nMCP Servers ({len(servers)}):")
                        print("-" * 60)

                        # Connect and show status
                        results = agent.connect_mcp_servers()

                        for server_name in servers:
                            client = agent._mcp_registry._clients.get(server_name)
                            connected = client.is_connected if client else False
                            tools = agent._mcp_registry.list_tools(server_name)
                            conn = "[connected]" if connected else "[disconnected]"
                            print(f"  {server_name} {conn}")
                            print(f"    Tools: {len(tools)} available")
                            for t in tools[:3]:
                                print(f"      - {t.name}")
                            if len(tools) > 3:
                                print(f"      ... and {len(tools) - 3} more")

                        print("\nNote: Use /mcp to reconnect after adding new servers.")
                    continue

                else:
                    print(f"Unknown command: {cmd}")
                    print("Type /help for available commands")
                    continue

            print()

            # Check if user input contains search intent
            should_search = any(kw.lower() in user_input.lower() for kw in SEARCH_KEYWORDS)

            # Save original state and temporarily enable web_search if needed
            original_web_search = agent._enable_web_search
            if should_search and "web_search" in agent.tools:
                agent._enable_web_search = True

            if stream:
                response = agent.stream(user_input)
                print()  # Newline after streaming
            else:
                response = agent.run(user_input)

            # Restore original state
            agent._enable_web_search = original_web_search

            print(f"\nAgent: {response}")

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        error_msg = parse_api_error(e)
        print(f"\nError: {error_msg}")
        sys.exit(1)


def single_query(agent: Agent, query: str, stream: bool = False) -> str:
    """Run a single query and return result.

    Args:
        agent: Agent instance
        query: User query
        stream: Whether to use streaming

    Returns:
        Agent response
    """
    if stream:
        return agent.stream(query)
    return agent.run(query)


def cmd_list_sessions(config: Config) -> None:
    """List all saved sessions."""
    history = ConversationHistory()
    sessions = history.list_sessions(config.history_storage_dir)

    if not sessions:
        print(f"No saved sessions in {config.history_storage_dir}")
    else:
        print(f"\nSessions in {config.history_storage_dir}:")
        print("-" * 60)
        for i, sess in enumerate(sessions, 1):
            print(f"{i}. {sess['name']}")
            print(f"   Messages: {sess['message_count']}")
            print(f"   Updated: {sess['updated_at'][:19].replace('T', ' ')}")
            print(f"   Path: {sess['path']}")
            print()


def cmd_save_history(config: Config, session_name: str = "default") -> None:
    """Save current session history."""
    history = ConversationHistory(
        storage_file=str(Path(config.history_storage_dir) / f"{session_name}.json")
    )
    path = history.save()
    print(f"History saved to: {path}")


def cmd_load_history(config: Config, session_name: str = "default") -> bool:
    """Load a session history."""
    history = ConversationHistory(
        storage_file=str(Path(config.history_storage_dir) / f"{session_name}.json")
    )
    if history.load():
        print(f"Loaded session: {session_name} ({len(history)} messages)")
        return True
    else:
        print(f"Session not found: {session_name}")
        return False


def cmd_clear_history(config: Config, session_name: str = "default") -> None:
    """Clear a session history."""
    history = ConversationHistory(
        storage_file=str(Path(config.history_storage_dir) / f"{session_name}.json")
    )
    history.clear()
    print(f"History cleared for session: {session_name}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="One-Agent: Multi-Model Business Agent with MCP Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MCP Commands:
  --list-mcp-servers           List configured MCP servers
  --mcp-config FILE             Path to MCP config file (default: mcp_servers.json)

Examples:
  # Interactive mode
  PYTHONPATH=. python main.py

  # Single query
  PYTHONPATH=. python main.py --query "What is 25 * 4?"

  # Use specific provider
  PYTHONPATH=. python main.py --provider openai --query "Hello"

  # List MCP servers
  PYTHONPATH=. python main.py --list-mcp-servers

  # List all sessions
  PYTHONPATH=. python main.py --list-sessions
        """
    )

    parser.add_argument(
        "--query", "-q",
        help="Single query to run"
    )

    parser.add_argument(
        "--provider", "-p",
        help="Provider to use (anthropic, openai, glm, kimi)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--env",
        help="Path to .env file",
        default=".env"
    )

    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List configured providers and exit"
    )

    # Streaming option
    parser.add_argument(
        "--stream", "-s",
        action="store_true",
        help="Enable streaming response"
    )

    # Web search option
    parser.add_argument(
        "--web-search", "-w",
        action="store_true",
        help="Enable web search auto-calling (searches for every query)"
    )

    # History commands
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List all saved sessions"
    )

    parser.add_argument(
        "--save-history",
        nargs="?",
        const="default",
        metavar="NAME",
        help="Save current history (default session)"
    )

    parser.add_argument(
        "--load-history",
        metavar="NAME",
        help="Load a saved session"
    )

    parser.add_argument(
        "--clear-history",
        nargs="?",
        const="default",
        metavar="NAME",
        help="Clear a session"
    )

    parser.add_argument(
        "--export-history",
        nargs=2,
        metavar=("NAME", "PATH"),
        help="Export session to file (json or text)"
    )

    # MCP arguments
    parser.add_argument(
        "--list-mcp-servers",
        action="store_true",
        help="List configured MCP servers"
    )

    parser.add_argument(
        "--mcp-config",
        metavar="FILE",
        help="Path to MCP config file (default: mcp_servers.json)"
    )

    parser.add_argument(
        "--mcp-connect",
        metavar="SERVER",
        help="Connect to a specific MCP server"
    )

    args = parser.parse_args()

    # Override MCP config if specified
    if args.mcp_config:
        os.environ["MCP_CONFIG_FILE"] = args.mcp_config

    # Load config
    config = load_config(args.env)

    # List providers if requested
    if args.list_providers:
        print("Configured providers:")
        for name, cfg in config.providers.items():
            print(f"  - {name}: {cfg.model}")
        print(f"\nDefault provider: {config.default_provider}")
        return 0

    # MCP commands (no provider needed)
    if args.list_mcp_servers:
        from mcp import MCPToolRegistry
        mcp_config_file = args.mcp_config or config.mcp_config_file
        registry = MCPToolRegistry.from_mcp_config(mcp_config_file)
        print("\nConfigured MCP servers:")
        print("-" * 60)
        for server_name in registry.server_names:
            print(f"  - {server_name}")
        print(f"\nTotal: {len(registry.server_names)} servers")
        return 0

    if args.mcp_connect:
        from mcp import MCPToolRegistry
        import asyncio
        mcp_config_file = args.mcp_config or config.mcp_config_file
        registry = MCPToolRegistry.from_mcp_config(mcp_config_file)
        server_name = args.mcp_connect

        if server_name not in registry.server_names:
            print(f"Error: MCP server '{server_name}' not found in config")
            print(f"Available servers: {registry.server_names}")
            return 1

        print(f"Connecting to MCP server '{server_name}'...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(registry.connect(server_name))

        if results.get(server_name, False):
            print(f"Successfully connected to '{server_name}'")
            tools = registry.list_tools(server_name)
            print(f"Available tools ({len(tools)}):")
            for tool in tools[:10]:  # Show first 10 tools
                print(f"  - {tool.name}")
            if len(tools) > 10:
                print(f"  ... and {len(tools) - 10} more")
        else:
            print(f"Failed to connect to '{server_name}'")
            return 1

        loop.run_until_complete(registry.disconnect(server_name))
        return 0

    if args.save_history is not None:
        cmd_save_history(config, args.save_history)
        return 0

    if args.load_history:
        cmd_load_history(config, args.load_history)
        return 0

    if args.clear_history is not None:
        cmd_clear_history(config, args.clear_history)
        return 0

    if args.export_history:
        name, path = args.export_history
        history = ConversationHistory(
            storage_file=str(Path(config.history_storage_dir) / f"{name}.json")
        )
        if history.load():
            fmt = "text" if path.endswith(".txt") else "json"
            saved_path = history.export(path, fmt)
            print(f"Exported to: {saved_path}")
        else:
            print(f"Session not found: {name}")
            return 1
        return 0

    # Create agent
    agent = create_agent(
        provider_name=args.provider,
        config=config,
        verbose=args.verbose,
        enable_web_search=args.web_search,
    )

    if not agent:
        print("\nNo provider available. Please configure an API key in .env")
        print("\nExample .env:")
        print("  ANTHROPIC_API_KEY=your_key")
        print("  DEFAULT_PROVIDER=anthropic")
        return 1

    # Run mode
    if args.query:
        try:
            response = single_query(agent, args.query, stream=args.stream)
            print()  # Add newline after streaming output
            if response:
                print(f"Agent: {response}")
        except Exception as e:
            error_msg = parse_api_error(e)
            print(f"\nError: {error_msg}")
            return 1
    else:
        interactive_mode(agent, stream=args.stream)

    return 0


if __name__ == "__main__":
    sys.exit(main())
