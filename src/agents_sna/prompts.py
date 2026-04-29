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
    single_agent_per_iteration: bool = False,
    allowed_agent_names: set[str] | None = None,
    previous_agent_name: str | None = None,
) -> list[dict[str, str]]:
    if allowed_agent_names is None:
        allowed_names = sorted(agent.name for agent in agents)
    else:
        allowed_names = sorted(allowed_agent_names)
    allowed_prompt = format_allowed_agents(
        allowed_names=allowed_names,
        previous_agent_name=previous_agent_name,
        single_agent_per_iteration=single_agent_per_iteration,
    )
    selection_instruction = (
        "Choose exactly one next agent that should contribute."
        if single_agent_per_iteration
        else "Choose the next agent or agents that should contribute."
    )
    system_prompt = (
        "You coordinate a multi-agent LLM discussion before the final answer.\n\n"
        "Available agents:\n"
        f"{format_agent_descriptions(agents)}\n\n"
        f"{allowed_prompt}\n\n"
        f"{selection_instruction} Pick only agents "
        "whose perspective is useful now. If the discussion is ready for synthesis, "
        f"return [{FINAL_SENTINEL!r}]."
    )
    if single_agent_per_iteration:
        json_instruction = (
            "The JSON must be a list containing exactly one allowed agent name, "
            f'for example ["{allowed_names[0]}"], or ["{FINAL_SENTINEL}"] to '
            "request final answer generation."
            if allowed_names
            else f'There are no allowed next agents; return ["{FINAL_SENTINEL}"].'
        )
    else:
        json_instruction = (
            "The JSON must be a list of agent names, for example "
            '["planner", "critic"]. To request final answer generation, return '
            f'["{FINAL_SENTINEL}"].'
        )
    final_user_prompt = (
        f"Current iteration: {current_iteration}\n"
        f"Maximum iterations, including final synthesis: {max_iterations}\n\n"
        f"Allowed next agents: {allowed_names}\n\n"
        f"Return only valid JSON. {json_instruction} "
        "Do not include explanation or Markdown."
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


def format_allowed_agents(
    *,
    allowed_names: list[str],
    previous_agent_name: str | None,
    single_agent_per_iteration: bool,
) -> str:
    if not single_agent_per_iteration:
        return "No single-agent handoff restriction is active."

    previous = previous_agent_name or "none; this is the first selection"
    if not allowed_names:
        allowed = "none; select FINAL"
    else:
        allowed = ", ".join(allowed_names)
    return (
        "Single-agent-per-iteration mode is active.\n"
        f"Most recent called agent: {previous}.\n"
        f"Allowed next agents after handoff constraints: {allowed}."
    )
