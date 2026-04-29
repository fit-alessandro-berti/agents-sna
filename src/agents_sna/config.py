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
class HandoffExclusion:
    source: str
    target: str


@dataclass(frozen=True)
class AgenticConfig:
    max_iterations: int
    agents: tuple[AgentSpec, ...]
    single_agent_per_iteration: bool = False
    excluded_handoffs: tuple[HandoffExclusion, ...] = ()

    def agent_by_name(self) -> dict[str, AgentSpec]:
        return {agent.name: agent for agent in self.agents}

    def allowed_agent_names_after(self, source: str | None) -> set[str]:
        agent_names = set(self.agent_by_name())
        if not self.single_agent_per_iteration or source is None:
            return agent_names

        excluded_targets = {
            exclusion.target
            for exclusion in self.excluded_handoffs
            if exclusion.source == source
        }
        return agent_names.difference(excluded_targets)


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

    single_agent_per_iteration = raw.get("single_agent_per_iteration", False)
    if not isinstance(single_agent_per_iteration, bool):
        raise ValueError(
            "Configuration field 'single_agent_per_iteration' must be a boolean."
        )

    return AgenticConfig(
        max_iterations=max_iterations,
        agents=tuple(agents),
        single_agent_per_iteration=single_agent_per_iteration,
        excluded_handoffs=parse_excluded_handoffs(raw, seen_names),
    )


def parse_excluded_handoffs(
    raw: dict[str, object],
    known_agents: set[str],
) -> tuple[HandoffExclusion, ...]:
    raw_handoffs = raw.get("excluded_handoffs", [])
    if not isinstance(raw_handoffs, list):
        raise ValueError("Configuration field 'excluded_handoffs' must be a list.")

    handoffs: list[HandoffExclusion] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(raw_handoffs, start=1):
        source, target = _parse_handoff_item(item, index)

        if source not in known_agents:
            raise ValueError(
                f"Excluded handoff #{index} references unknown source agent '{source}'."
            )
        if target not in known_agents:
            raise ValueError(
                f"Excluded handoff #{index} references unknown target agent '{target}'."
            )

        edge = (source, target)
        if edge not in seen:
            handoffs.append(HandoffExclusion(source=source, target=target))
            seen.add(edge)

    return tuple(handoffs)


def _parse_handoff_item(item: object, index: int) -> tuple[str, str]:
    source: object
    target: object

    if isinstance(item, dict):
        source = item.get("from", item.get("source"))
        target = item.get("to", item.get("target"))
    elif isinstance(item, list) and len(item) == 2:
        source, target = item
    else:
        raise ValueError(
            "Excluded handoff "
            f"#{index} must be an object with from/to fields or a two-item list."
        )

    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"Excluded handoff #{index} field 'from' must be a string.")
    if not isinstance(target, str) or not target.strip():
        raise ValueError(f"Excluded handoff #{index} field 'to' must be a string.")

    return source.strip(), target.strip()
