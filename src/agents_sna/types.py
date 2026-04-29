from __future__ import annotations

from dataclasses import dataclass

MessageContent = str | list[dict[str, object]]
ChatMessage = dict[str, MessageContent]


@dataclass(frozen=True)
class AgentAnswer:
    iteration: int
    agent_name: str
    content: str
