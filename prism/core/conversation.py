# prism/core/conversation.py
"""Typed conversation log for multi-agent simulations. Extends MAGPIE's flat dict log."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class MessageType(Enum):
    SYSTEM = "system"
    MESSAGE = "message"
    PROPOSAL = "proposal"
    ACCEPT = "accept_proposal"
    REJECT = "reject_proposal"


@dataclass
class LogEntry:
    sender: str
    recipients: list[str]
    msg_type: MessageType
    content: str
    timestamp: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "sender": self.sender,
            "recipients": self.recipients,
            "type": self.msg_type.value,
            "content": self.content,
            "metadata": self.metadata,
        }


class ConversationLog:
    def __init__(self):
        self.entries: list[LogEntry] = []

    def add(self, entry: LogEntry) -> None:
        self.entries.append(entry)

    def get_visible_to(self, agent_name: str) -> list[LogEntry]:
        """Return entries visible to the given agent.

        An agent can see:
        - Messages they sent
        - Messages directed to them
        - System messages (empty recipients = broadcast)
        """
        visible = []
        for e in self.entries:
            if e.sender == agent_name:
                visible.append(e)
            elif agent_name in e.recipients:
                visible.append(e)
            elif e.msg_type == MessageType.SYSTEM:
                visible.append(e)
            elif not e.recipients:  # broadcast
                visible.append(e)
        return visible

    def get_last_message_by(self, sender: str) -> LogEntry | None:
        """Return the most recent MESSAGE entry from the given sender."""
        for e in reversed(self.entries):
            if e.sender == sender and e.msg_type == MessageType.MESSAGE:
                return e
        return None

    def format_readable(self) -> str:
        lines = []
        for e in self.entries:
            to_str = ", ".join(e.recipients) if e.recipients else "all"
            prefix = {
                MessageType.SYSTEM: f"[{e.timestamp}] SYSTEM:",
                MessageType.MESSAGE: f"[{e.timestamp}] {e.sender} -> [{to_str}]:",
                MessageType.PROPOSAL: f"[{e.timestamp}] {e.sender} -> [{to_str}] [PROPOSAL {e.metadata.get('proposal_id', '?')}]:",
                MessageType.ACCEPT: f"[{e.timestamp}] {e.sender} ACCEPTS {e.metadata.get('proposal_id', '?')}:",
                MessageType.REJECT: f"[{e.timestamp}] {e.sender} REJECTS {e.metadata.get('proposal_id', '?')}:",
            }.get(e.msg_type, f"[{e.timestamp}] {e.sender}:")
            lines.append(f"{prefix} {e.content}")
        return "\n".join(lines)

    def to_json(self) -> list[dict]:
        return [e.to_dict() for e in self.entries]
