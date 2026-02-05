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


def create_tools(config: Config) -> list:
    """Create tools based on configuration.

    Args:
        config: Configuration object

    Returns:
        List of Tool instances
    """
    tools = []

    if config.enable_web_search:
        tools.append(WebSearchTool())

    if config.enable_calculator:
        tools.append(CalculatorTool())

    return tools


def create_agent(
    provider_name: Optional[str] = None,
    config: Optional[Config] = None,
    verbose: bool = False,
) -> Optional[Agent]:
    """Create an agent instance.

    Args:
        provider_name: Name of provider to use
        config: Optional configuration override
        verbose: Enable verbose output

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
        print("  OPENAI_API_KEY=your_key         # For GPT-4")
        print("  GLM_API_KEY=your_key            # For GLM-4")
        print("  KIMI_API_KEY=your_key           # For Kimi")
        return None

    # Create provider
    provider_config = cfg.providers[provider_key]
    provider = create_provider(provider_config)

    if not provider:
        print(f"Error: Could not create provider '{provider_key}'")
        return None

    # Create tools
    tools = create_tools(cfg)

    # Create agent
    agent = Agent(
        provider=provider,
        tools=tools,
        config=cfg,
    )

    return agent


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
    print("\nCommands:")
    print("  /help       - Show this help")
    print("  /reset      - Reset conversation")
    print("  /save       - Save history manually")
    print("  /sessions   - List all sessions")
    print("  /switch NAME - Switch to a different session")
    print("  /export PATH - Export history to file (json or text)")
    print("  /clear      - Clear current session history")
    print("  /stream     - Toggle streaming on/off")
    print("  quit        - Exit")
    print("-" * 60)

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

                else:
                    print(f"Unknown command: {cmd}")
                    print("Type /help for available commands")
                    continue

            print()
            if stream:
                response = agent.stream(user_input)
                print()  # Newline after streaming
            else:
                response = agent.run(user_input)
            print(f"\nAgent: {response}")

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nError: {e}")
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
        description="One-Agent: Multi-Model Business Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
History Commands:
  --list-sessions              List all saved sessions
  --save-history [NAME]        Save current history (default: default)
  --load-history NAME          Load a saved session
  --clear-history [NAME]       Clear a session (default: default)
  --export-history NAME PATH   Export session to file

Examples:
  # Interactive mode
  PYTHONPATH=. python main.py

  # Single query
  PYTHONPATH=. python main.py --query "What is 25 * 4?"

  # Use specific provider
  PYTHONPATH=. python main.py --provider openai --query "Hello"

  # List all sessions
  PYTHONPATH=. python main.py --list-sessions

  # Export session to text
  PYTHONPATH=. python main.py --export-history default ./history.txt
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

    args = parser.parse_args()

    # Load config
    config = load_config(args.env)

    # List providers if requested
    if args.list_providers:
        print("Configured providers:")
        for name, cfg in config.providers.items():
            print(f"  - {name}: {cfg.model}")
        print(f"\nDefault provider: {config.default_provider}")
        return 0

    # History commands (no provider needed)
    if args.list_sessions:
        cmd_list_sessions(config)
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
    )

    if not agent:
        print("\nNo provider available. Please configure an API key in .env")
        print("\nExample .env:")
        print("  ANTHROPIC_API_KEY=your_key")
        print("  DEFAULT_PROVIDER=anthropic")
        return 1

    # Run mode
    if args.query:
        response = single_query(agent, args.query, stream=args.stream)
        print()  # Add newline after streaming output
    else:
        interactive_mode(agent, stream=args.stream)

    return 0


if __name__ == "__main__":
    sys.exit(main())
