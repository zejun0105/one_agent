"""Conversation history management for One-Agent."""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
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


@dataclass
class SessionMetadata:
    """Metadata for a conversation session."""
    session_name: str
    created_at: str
    updated_at: str
    message_count: int
    provider: Optional[str] = None
    model: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SessionMetadata":
        """Create from dictionary."""
        return cls(**data)


class ConversationHistory:
    """Manages conversation history with persistence."""

    def __init__(
        self,
        max_messages: int = 50,
        storage_file: Optional[str] = None,
        auto_save: bool = True,
    ):
        """Initialize conversation history.

        Args:
            max_messages: Maximum number of messages to keep
            storage_file: Optional file to persist history
            auto_save: Whether to auto-save after each message
        """
        self.max_messages = max_messages
        self.storage_file = Path(storage_file) if storage_file else None
        self.auto_save = auto_save
        self.messages: list[Message] = []
        self._metadata: Optional[SessionMetadata] = None

    @property
    def metadata(self) -> SessionMetadata:
        """Get session metadata, creating if needed."""
        if self._metadata is None:
            self._metadata = SessionMetadata(
                session_name="default",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                message_count=len(self.messages),
            )
        return self._metadata

    def add(self, message: Message) -> None:
        """Add a message to the history."""
        self.messages.append(message)
        self._trim()
        self._update_metadata()
        if self.auto_save and self.storage_file:
            self.save()

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

    def get_last_n_messages(self, n: int) -> list[dict]:
        """Get the last n messages for API calls (preserving order)."""
        messages = self.messages[-n:] if n < len(self.messages) else self.messages
        return [msg.to_dict() for msg in messages]

    def clear(self) -> None:
        """Clear the conversation history."""
        self.messages = []
        self._metadata = None
        if self.auto_save and self.storage_file:
            self.save()

    def save(self, path: Optional[str] = None) -> str:
        """Save history to file.

        Args:
            path: Optional path to save to (overrides storage_file)

        Returns:
            The path where history was saved
        """
        save_path = Path(path) if path else self.storage_file
        if not save_path:
            raise ValueError("No storage file specified for saving history")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": self.metadata.to_dict(),
            "messages": [msg.to_dict() for msg in self.messages]
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return str(save_path)

    def load(self, path: Optional[str] = None) -> bool:
        """Load history from file.

        Args:
            path: Optional path to load from (overrides storage_file)

        Returns:
            True if loaded successfully, False otherwise
        """
        load_path = Path(path) if path else self.storage_file
        if not load_path:
            raise ValueError("No storage file specified for loading history")

        if not load_path.exists():
            return False

        try:
            with open(load_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load metadata
            if "metadata" in data:
                self._metadata = SessionMetadata.from_dict(data["metadata"])

            # Load messages
            self.messages = [Message.from_dict(m) for m in data.get("messages", [])]
            return True

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: Failed to load history from {load_path}: {e}")
            self.messages = []
            self._metadata = None
            return False

    def export(self, path: str, format: str = "json") -> str:
        """Export history to a file.

        Args:
            path: Path to export to
            format: Export format (json, text)

        Returns:
            The path where history was exported
        """
        if format == "json":
            return self.save(path)
        elif format == "text":
            lines = []
            for msg in self.messages:
                role = msg.role.upper()
                timestamp = msg.timestamp.split("T")[1].split(".")[0] if "T" in msg.timestamp else msg.timestamp
                lines.append(f"[{timestamp}] {role}: {msg.content[:200]}")
                if msg.tool_calls:
                    lines.append(f"  Tools: {len(msg.tool_calls)} call(s)")
            content = "\n".join(lines)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return path
        else:
            raise ValueError(f"Unknown export format: {format}")

    def list_sessions(self, directory: str = "~/.one_agent/history") -> List[Dict[str, Any]]:
        """List all available sessions in the storage directory.

        Args:
            directory: Directory to search for sessions

        Returns:
            List of session info dictionaries
        """
        dir_path = Path(directory).expanduser()
        if not dir_path.exists():
            return []

        sessions = []
        for file in dir_path.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                meta = data.get("metadata", {})
                sessions.append({
                    "name": file.stem,
                    "path": str(file),
                    "created_at": meta.get("created_at", "unknown"),
                    "updated_at": meta.get("updated_at", "unknown"),
                    "message_count": meta.get("message_count", 0),
                })
            except (json.JSONDecodeError, KeyError):
                sessions.append({
                    "name": file.stem,
                    "path": str(file),
                    "created_at": "unknown",
                    "updated_at": "unknown",
                    "message_count": 0,
                })

        # Sort by updated_at descending
        sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return sessions

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

    def _update_metadata(self) -> None:
        """Update session metadata."""
        if self._metadata:
            self._metadata.updated_at = datetime.now().isoformat()
            self._metadata.message_count = len(self.messages)

    def set_session_name(self, name: str) -> None:
        """Set the session name."""
        self._metadata = SessionMetadata(
            session_name=name,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            message_count=len(self.messages),
        )

    def __len__(self) -> int:
        """Return the number of messages."""
        return len(self.messages)

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"ConversationHistory(messages={len(self.messages)}, max={self.max_messages}, auto_save={self.auto_save})"
