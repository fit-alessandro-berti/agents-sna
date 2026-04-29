from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentAnswer:
    iteration: int
    agent_name: str
    content: str
