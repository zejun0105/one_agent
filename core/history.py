"""Conversation history management for One-Agent."""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # system, user, assistant, tool
    content: str
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        """Create from dictionary."""
        return cls(**data)


class ConversationHistory:
    """Manages conversation history with persistence."""

    def __init__(self, max_messages: int = 50, storage_file: Optional[str] = None):
        """Initialize conversation history.

        Args:
            max_messages: Maximum number of messages to keep
            storage_file: Optional file to persist history
        """
        self.max_messages = max_messages
        self.storage_file = Path(storage_file) if storage_file else None
        self.messages: list[Message] = []

    def add(self, message: Message) -> None:
        """Add a message to the history."""
        self.messages.append(message)
        self._trim()

    def add_system(self, content: str) -> None:
        """Add a system message."""
        self.add(Message(role="system", content=content))

    def add_user(self, content: str) -> None:
        """Add a user message."""
        self.add(Message(role="user", content=content))

    def add_assistant(self, content: str, tool_calls: Optional[list] = None) -> None:
        """Add an assistant message."""
        self.add(Message(role="assistant", content=content, tool_calls=tool_calls))

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        """Add a tool result message."""
        self.add(Message(role="tool", content=content, tool_call_id=tool_call_id))

    def get_messages(self) -> list[dict]:
        """Get all messages as dictionaries for API calls."""
        return [msg.to_dict() for msg in self.messages]

    def clear(self) -> None:
        """Clear the conversation history."""
        self.messages = []

    def save(self) -> None:
        """Save history to file."""
        if self.storage_file:
            self.storage_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump([msg.to_dict() for msg in self.messages], f, ensure_ascii=False)

    def load(self) -> None:
        """Load history from file."""
        if self.storage_file and self.storage_file.exists():
            try:
                with open(self.storage_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.messages = [Message.from_dict(m) for m in data]
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to load history: {e}")
                self.messages = []

    def _trim(self) -> None:
        """Trim history to max_messages, keeping system messages."""
        if len(self.messages) <= self.max_messages:
            return

        # Keep all system messages, trim from the beginning of non-system
        system_messages = [m for m in self.messages if m.role == "system"]
        non_system = [m for m in self.messages if m.role != "system"]

        # Calculate how many to keep
        keep_count = self.max_messages - len(system_messages)
        if keep_count < 0:
            keep_count = 0

        # Trim non-system messages
        self.messages = system_messages + non_system[-keep_count:]

    def __len__(self) -> int:
        """Return the number of messages."""
        return len(self.messages)

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"ConversationHistory(messages={len(self.messages)}, max={self.max_messages})"
