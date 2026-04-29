from __future__ import annotations

from agents_sna.config import AgentSpec
from agents_sna.types import AgentAnswer


FINAL_SENTINEL = "FINAL"


def build_selection_messages(
    *,
    original_prompt: str,
    agents: tuple[AgentSpec, ...],
    previous_answers: list[AgentAnswer],
    current_iteration: int,
    max_iterations: int,
) -> list[dict[str, str]]:
    system_prompt = (
        "You coordinate a multi-agent LLM discussion before the final answer.\n\n"
        "Available agents:\n"
        f"{format_agent_descriptions(agents)}\n\n"
        "Choose the next agent or agents that should contribute. Pick only agents "
        "whose perspective is useful now. If the discussion is ready for synthesis, "
        f"return [{FINAL_SENTINEL!r}]."
    )
    final_user_prompt = (
        f"Current iteration: {current_iteration}\n"
        f"Maximum iterations, including final synthesis: {max_iterations}\n\n"
        "Return only valid JSON. The JSON must be a list of agent names, for example "
        '["planner", "critic"]. To request final answer generation, return '
        f'["{FINAL_SENTINEL}"]. Do not include explanation or Markdown.'
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": original_prompt},
        *previous_answer_messages(previous_answers),
        {"role": "user", "content": final_user_prompt},
    ]


def build_agent_messages(
    *,
    original_prompt: str,
    agent: AgentSpec,
    previous_answers: list[AgentAnswer],
    current_iteration: int,
    max_iterations: int,
) -> list[dict[str, str]]:
    system_prompt = (
        f"You are the '{agent.name}' agent in a multi-agent LLM discussion.\n\n"
        f"Persona:\n{agent.description}"
    )
    insight_prompt = (
        "Apply your persona to provide insights for the original inquiry.\n"
        f"Current iteration: {current_iteration}\n"
        f"Maximum iterations, including final synthesis: {max_iterations}\n\n"
        "Be concise, avoid repeating earlier agents unless you are correcting or "
        "refining them, and make your contribution useful for the final synthesis."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": original_prompt},
        *previous_answer_messages(previous_answers),
        {"role": "user", "content": insight_prompt},
    ]


def build_final_messages(
    *,
    original_prompt: str,
    previous_answers: list[AgentAnswer],
) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": original_prompt},
        *previous_answer_messages(previous_answers),
        {
            "role": "user",
            "content": (
                "Please provide the final answer to my original inquiry, based on all "
                "the different perspectives."
            ),
        },
    ]


def previous_answer_messages(answers: list[AgentAnswer]) -> list[dict[str, str]]:
    return [
        {
            "role": "assistant",
            "content": (
                f"Iteration {answer.iteration}, agent '{answer.agent_name}':\n"
                f"{answer.content}"
            ),
        }
        for answer in answers
    ]


def format_agent_descriptions(agents: tuple[AgentSpec, ...]) -> str:
    return "\n".join(f"- {agent.name}: {agent.description}" for agent in agents)
