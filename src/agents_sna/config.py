from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    model: str | None = None


@dataclass(frozen=True)
class AgenticConfig:
    max_iterations: int
    agents: tuple[AgentSpec, ...]

    def agent_by_name(self) -> dict[str, AgentSpec]:
        return {agent.name: agent for agent in self.agents}


def load_config(path: str | Path) -> AgenticConfig:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{config_path} is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a JSON object.")

    max_iterations = raw.get("max_iterations")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("Configuration field 'max_iterations' must be an integer >= 1.")

    raw_agents = raw.get("agents")
    if not isinstance(raw_agents, list) or not raw_agents:
        raise ValueError("Configuration field 'agents' must be a non-empty list.")

    agents: list[AgentSpec] = []
    seen_names: set[str] = set()
    for index, item in enumerate(raw_agents, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Agent #{index} must be a JSON object.")

        name = item.get("name")
        description = item.get("description")
        model = item.get("model")

        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Agent #{index} field 'name' must be a non-empty string.")
        name = name.strip()
        if name in seen_names:
            raise ValueError(f"Duplicate agent name: {name}")
        seen_names.add(name)

        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                f"Agent '{name}' field 'description' must be a non-empty string."
            )
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ValueError(f"Agent '{name}' field 'model' must be a string when set.")

        agents.append(
            AgentSpec(
                name=name,
                description=description.strip(),
                model=model.strip() if isinstance(model, str) else None,
            )
        )

    return AgenticConfig(
        max_iterations=max_iterations,
        agents=tuple(agents),
    )
