from __future__ import annotations

from collections.abc import Callable
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
from agents_sna.types import AgentAnswer, ChatMessage, MessageContent

EventHandler = Callable[[str, dict[str, object]], None]


class ChatClient(Protocol):
    def complete(
        self,
        messages: list[ChatMessage],
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
    def __init__(
        self,
        *,
        config: AgenticConfig,
        client: ChatClient,
        event_handler: EventHandler | None = None,
    ):
        self.config = config
        self.client = client
        self.event_handler = event_handler

    def run(self, original_prompt: MessageContent) -> OrchestrationResult:
        prompt = normalize_original_prompt(original_prompt)
        prompt_text = message_content_to_text(prompt).strip()
        if not prompt_text:
            raise ValueError("Original prompt must not be empty.")

        previous_answers: list[AgentAnswer] = []
        agents_by_name = self.config.agent_by_name()
        previous_agent_name: str | None = None
        self._emit(
            "run_start",
            prompt=prompt_text,
            max_iterations=self.config.max_iterations,
            agent_names=tuple(agents_by_name),
            single_agent_per_iteration=self.config.single_agent_per_iteration,
            excluded_handoffs=tuple(
                {
                    "from": exclusion.source,
                    "to": exclusion.target,
                }
                for exclusion in self.config.excluded_handoffs
            ),
        )

        for iteration in range(1, self.config.max_iterations):
            allowed_agent_names = self.config.allowed_agent_names_after(previous_agent_name)
            selection_messages = build_selection_messages(
                original_prompt=prompt,
                agents=self.config.agents,
                previous_answers=previous_answers,
                current_iteration=iteration,
                max_iterations=self.config.max_iterations,
                single_agent_per_iteration=self.config.single_agent_per_iteration,
                allowed_agent_names=allowed_agent_names,
                previous_agent_name=previous_agent_name,
            )
            self._emit(
                "request",
                kind="selector",
                iteration=iteration,
                messages=selection_messages,
                model=None,
                previous_agent_name=previous_agent_name,
                allowed_agent_names=tuple(sorted(allowed_agent_names)),
            )
            raw_selection = self.client.complete(selection_messages)
            self._emit(
                "response",
                kind="selector",
                iteration=iteration,
                content=raw_selection,
            )
            selection = parse_agent_selection(
                raw_selection,
                set(agents_by_name),
                single_agent_per_iteration=self.config.single_agent_per_iteration,
                allowed_agents=allowed_agent_names,
            )
            self._emit(
                "selection",
                iteration=iteration,
                agent_names=selection.agent_names,
                final_requested=selection.final_requested,
                previous_agent_name=previous_agent_name,
                allowed_agent_names=tuple(sorted(allowed_agent_names)),
            )
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
                self._emit(
                    "request",
                    kind="agent",
                    iteration=iteration,
                    agent_name=agent.name,
                    messages=agent_messages,
                    model=agent.model,
                )
                answer = self.client.complete(
                    agent_messages,
                    model=agent.model,
                )
                self._emit(
                    "response",
                    kind="agent",
                    iteration=iteration,
                    agent_name=agent.name,
                    content=answer,
                )
                previous_answers.append(
                    AgentAnswer(
                        iteration=iteration,
                        agent_name=agent.name,
                        content=answer.strip(),
                    )
                )
                previous_agent_name = agent.name

        final_messages = build_final_messages(
            original_prompt=prompt,
            previous_answers=previous_answers,
        )
        self._emit(
            "request",
            kind="final",
            iteration=self.config.max_iterations,
            messages=final_messages,
            model=None,
        )
        final_answer = self.client.complete(
            final_messages
        )
        self._emit(
            "response",
            kind="final",
            iteration=self.config.max_iterations,
            content=final_answer,
        )
        return OrchestrationResult(
            final_answer=final_answer.strip(),
            agent_answers=tuple(previous_answers),
        )

    def _emit(self, event: str, **payload: object) -> None:
        if self.event_handler:
            self.event_handler(event, payload)


def parse_agent_selection(
    raw_selection: str,
    known_agents: set[str],
    *,
    single_agent_per_iteration: bool = False,
    allowed_agents: set[str] | None = None,
) -> AgentSelection:
    try:
        parsed = json.loads(_extract_json_list(raw_selection))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Selector did not return a valid JSON list: {raw_selection}") from exc

    if not isinstance(parsed, list):
        raise ValueError(f"Selector response must be a JSON list: {raw_selection}")

    allowed = allowed_agents if allowed_agents is not None else known_agents
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
        if normalized not in allowed:
            raise ValueError(
                f"Selector chose disallowed handoff target '{normalized}'. Allowed "
                f"agents now: {', '.join(sorted(allowed)) or '(none)'}"
            )
        if normalized not in names:
            names.append(normalized)
        if single_agent_per_iteration:
            break

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


def normalize_original_prompt(original_prompt: MessageContent) -> MessageContent:
    if isinstance(original_prompt, str):
        return original_prompt.strip()
    return original_prompt


def message_content_to_text(content: MessageContent) -> str:
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for part in content:
        part_type = part.get("type")
        if part_type == "text":
            parts.append(str(part.get("text", "")))
        elif part_type == "image_url":
            parts.append("[image]")
        else:
            parts.append(str(part))
    return "\n".join(parts)
