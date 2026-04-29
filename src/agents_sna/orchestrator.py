from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Protocol

from agents_sna.config import AgenticConfig
from agents_sna.prompts import (
    FINAL_SENTINEL,
    build_agent_messages,
    build_final_messages,
    build_selection_messages,
)
from agents_sna.types import AgentAnswer


class ChatClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
    ) -> str:
        ...


@dataclass(frozen=True)
class OrchestrationResult:
    final_answer: str
    agent_answers: tuple[AgentAnswer, ...]


@dataclass(frozen=True)
class AgentSelection:
    agent_names: tuple[str, ...] = ()
    final_requested: bool = False


class AgenticOrchestrator:
    def __init__(self, *, config: AgenticConfig, client: ChatClient):
        self.config = config
        self.client = client

    def run(self, original_prompt: str) -> OrchestrationResult:
        prompt = original_prompt.strip()
        if not prompt:
            raise ValueError("Original prompt must not be empty.")

        previous_answers: list[AgentAnswer] = []
        agents_by_name = self.config.agent_by_name()

        for iteration in range(1, self.config.max_iterations):
            selection_messages = build_selection_messages(
                original_prompt=prompt,
                agents=self.config.agents,
                previous_answers=previous_answers,
                current_iteration=iteration,
                max_iterations=self.config.max_iterations,
            )
            raw_selection = self.client.complete(selection_messages)
            selection = parse_agent_selection(raw_selection, set(agents_by_name))
            if selection.final_requested:
                break

            for agent_name in selection.agent_names:
                agent = agents_by_name[agent_name]
                agent_messages = build_agent_messages(
                    original_prompt=prompt,
                    agent=agent,
                    previous_answers=previous_answers,
                    current_iteration=iteration,
                    max_iterations=self.config.max_iterations,
                )
                answer = self.client.complete(
                    agent_messages,
                    model=agent.model,
                )
                previous_answers.append(
                    AgentAnswer(
                        iteration=iteration,
                        agent_name=agent.name,
                        content=answer.strip(),
                    )
                )

        final_answer = self.client.complete(
            build_final_messages(
                original_prompt=prompt,
                previous_answers=previous_answers,
            )
        )
        return OrchestrationResult(
            final_answer=final_answer.strip(),
            agent_answers=tuple(previous_answers),
        )


def parse_agent_selection(raw_selection: str, known_agents: set[str]) -> AgentSelection:
    try:
        parsed = json.loads(_extract_json_list(raw_selection))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Selector did not return a valid JSON list: {raw_selection}") from exc

    if not isinstance(parsed, list):
        raise ValueError(f"Selector response must be a JSON list: {raw_selection}")

    names: list[str] = []
    for item in parsed:
        candidate = _selection_item_to_name(item)
        if not candidate:
            raise ValueError(f"Invalid selector item: {item!r}")

        normalized = candidate.strip()
        if normalized.upper() == FINAL_SENTINEL:
            return AgentSelection(final_requested=True)

        if normalized not in known_agents:
            raise ValueError(
                f"Selector chose unknown agent '{normalized}'. Known agents: "
                f"{', '.join(sorted(known_agents))}"
            )
        if normalized not in names:
            names.append(normalized)

    if not names:
        raise ValueError("Selector returned an empty agent list.")

    return AgentSelection(agent_names=tuple(names))


def _selection_item_to_name(item: object) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("agent", "name", "id", "next_agent"):
            value = item.get(key)
            if isinstance(value, str):
                return value
    return None


def _extract_json_list(raw: str) -> str:
    text = raw.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.DOTALL)
    if fenced_match:
        return fenced_match.group(1)

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and start < end:
        return text[start : end + 1]
    return text
