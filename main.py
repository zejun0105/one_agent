#!/usr/bin/env python3
"""
One-Agent: Multi-Model Business Agent using Claude Agent SDK

A modular Business Agent framework with support for:
- Multiple LLM providers (Anthropic Claude, OpenAI GPT-4, GLM-4, Kimi)
- Tool calling capabilities
- Conversation history
- Configuration management
"""

import os
import sys
import argparse
from typing import Optional

from dotenv import load_dotenv

# Import core modules
from core import Agent, Config, config as global_config
from core.config import ProviderConfig

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


def interactive_mode(agent: Agent) -> None:
    """Run agent in interactive mode.

    Args:
        agent: Agent instance
    """
    print("=" * 60)
    print("  One-Agent - Multi-Model Business Agent")
    print("=" * 60)
    print(f"\nAgent initialized with: {agent}")
    print(f"Available tools: {[t.name for t in agent.tools]}")
    print("\nType 'quit' or 'exit' to quit.")
    print("-" * 60)

    try:
        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() == "/reset":
                agent.reset()
                print("Conversation reset.")
                continue

            if user_input.lower() == "/help":
                print("\nAvailable commands:")
                print("  /reset - Reset conversation")
                print("  /help  - Show this help")
                print("  quit   - Exit")
                continue

            print()
            response = agent.run(user_input)
            print(f"\nAgent: {response}")

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


def single_query(agent: Agent, query: str) -> str:
    """Run a single query and return result.

    Args:
        agent: Agent instance
        query: User query

    Returns:
        Agent response
    """
    return agent.run(query)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="One-Agent: Multi-Model Business Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python main.py

  # Single query
  python main.py --query "What is 25 * 4?"

  # Use specific provider
  python main.py --provider openai --query "Hello"

  # With verbose output
  python main.py --verbose --query "Search for latest AI news"
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
        response = single_query(agent, args.query)
        print(response)
    else:
        interactive_mode(agent)

    return 0


if __name__ == "__main__":
    sys.exit(main())
