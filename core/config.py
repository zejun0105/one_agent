"""Configuration management for One-Agent."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    provider: str = Field(..., description="Provider name (anthropic, openai, glm, kimi)")
    api_key: Optional[str] = Field(None, description="API key for the provider")
    model: str = Field(default="claude-3-5-sonnet-20241022", description="Model name")
    base_url: Optional[str] = Field(None, description="Base URL for API (for compatible providers)")
    max_tokens: int = Field(default=4096, description="Maximum tokens in response")
    temperature: float = Field(default=0.7, description="Temperature for sampling")


class Config(BaseModel):
    """Main configuration for One-Agent."""

    # Provider settings
    default_provider: str = Field(default="anthropic", description="Default LLM provider")
    providers: dict[str, ProviderConfig] = Field(default_factory=dict, description="Provider configurations")

    # Agent settings
    max_iterations: int = Field(default=10, description="Maximum agent iterations")
    max_history_messages: int = Field(default=50, description="Maximum conversation history messages")

    # Streaming settings
    streaming: bool = Field(default=False, description="Enable streaming response")
    streaming_echo: bool = Field(default=True, description="Print streaming chunks to console")

    # History persistence settings
    history_storage_dir: str = Field(default="~/.one_agent/history", description="Directory for history storage")
    auto_save_history: bool = Field(default=True, description="Auto-save history after each message")
    session_name: str = Field(default="default", description="Session name for history file")

    # Tool settings
    enable_web_search: bool = Field(default=True, description="Enable web search tool")
    enable_calculator: bool = Field(default=True, description="Enable calculator tool")

    # UI settings
    verbose: bool = Field(default=False, description="Verbose output")
    colors: bool = Field(default=True, description="Enable colored output")

    def get_history_storage_path(self) -> Path:
        """Get the resolved history storage path."""
        path = Path(self.history_storage_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{self.session_name}.json"

    @classmethod
    def load(cls, env_file: Optional[str] = None) -> "Config":
        """Load configuration from environment variables."""
        # Load .env file if provided
        if env_file:
            env_path = Path(env_file)
            if env_path.exists():
                load_dotenv(env_path)
        else:
            # Try to load from default .env
            env_path = Path(".env")
            if env_path.exists():
                load_dotenv(env_path)

        # Build provider configs from environment
        providers = {}

        # Anthropic provider
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            providers["anthropic"] = ProviderConfig(
                provider="anthropic",
                api_key=anthropic_key,
                model=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            )

        # OpenAI provider
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            providers["openai"] = ProviderConfig(
                provider="openai",
                api_key=openai_key,
                model=os.environ.get("OPENAI_MODEL", "gpt-4-turbo"),
                base_url=os.environ.get("OPENAI_BASE_URL"),
            )

        # GLM provider (智谱 AI)
        glm_key = os.environ.get("GLM_API_KEY")
        if glm_key:
            providers["glm"] = ProviderConfig(
                provider="glm",
                api_key=glm_key,
                model=os.environ.get("GLM_MODEL", "glm-4-plus"),
                base_url=os.environ.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
            )

        # Kimi provider (月之暗面)
        kimi_key = os.environ.get("KIMI_API_KEY")
        if kimi_key:
            providers["kimi"] = ProviderConfig(
                provider="kimi",
                api_key=kimi_key,
                model=os.environ.get("KIMI_MODEL", "moonshot-v1-8k"),
                base_url=os.environ.get("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
            )

        return cls(
            default_provider=os.environ.get("DEFAULT_PROVIDER", "anthropic"),
            providers=providers,
            max_iterations=int(os.environ.get("MAX_ITERATIONS", 10)),
            max_history_messages=int(os.environ.get("MAX_HISTORY_MESSAGES", 50)),
            streaming=os.environ.get("STREAMING", "false").lower() == "true",
            streaming_echo=os.environ.get("STREAMING_ECHO", "true").lower() == "true",
            history_storage_dir=os.environ.get("HISTORY_STORAGE_DIR", "~/.one_agent/history"),
            auto_save_history=os.environ.get("AUTO_SAVE_HISTORY", "true").lower() == "true",
            session_name=os.environ.get("SESSION_NAME", "default"),
            enable_web_search=os.environ.get("ENABLE_WEB_SEARCH", "true").lower() == "true",
            enable_calculator=os.environ.get("ENABLE_CALCULATOR", "true").lower() == "true",
            verbose=os.environ.get("VERBOSE", "false").lower() == "true",
            colors=os.environ.get("COLORS", "true").lower() == "true",
        )


# Global config instance
config = Config.load()
