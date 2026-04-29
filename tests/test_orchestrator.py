from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents_sna.config import AgentSpec, AgenticConfig
from agents_sna.orchestrator import AgenticOrchestrator, parse_agent_selection
from agents_sna.prompts import build_selection_messages
from agents_sna.types import AgentAnswer


class FakeClient:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "model": model,
            }
        )
        return self.responses.pop(0)


class OrchestratorTests(unittest.TestCase):
    def test_parse_selection_accepts_list_and_fenced_json(self) -> None:
        selection = parse_agent_selection('```json\n["planner", "critic"]\n```', {
            "planner",
            "critic",
        })

        self.assertEqual(selection.agent_names, ("planner", "critic"))
        self.assertFalse(selection.final_requested)

    def test_parse_selection_accepts_final(self) -> None:
        selection = parse_agent_selection('["FINAL"]', {"planner"})

        self.assertTrue(selection.final_requested)

    def test_prompt_builder_includes_iteration_and_agents(self) -> None:
        messages = build_selection_messages(
            original_prompt="Solve this",
            agents=(AgentSpec(name="planner", description="Plans."),),
            previous_answers=[
                AgentAnswer(iteration=1, agent_name="planner", content="A plan")
            ],
            current_iteration=2,
            max_iterations=4,
        )

        self.assertIn("planner: Plans.", messages[0]["content"])
        self.assertIn("Current iteration: 2", messages[-1]["content"])
        self.assertIn("Maximum iterations", messages[-1]["content"])
        self.assertEqual(messages[2]["role"], "assistant")

    def test_orchestrator_runs_agents_then_final(self) -> None:
        config = AgenticConfig(
            max_iterations=3,
            agents=(
                AgentSpec(name="planner", description="Plans."),
                AgentSpec(name="critic", description="Critiques.", model="critic-model"),
            ),
        )
        client = FakeClient(
            [
                '["planner", "critic"]',
                "planner answer",
                "critic answer",
                '["FINAL"]',
                "final answer",
            ]
        )

        result = AgenticOrchestrator(config=config, client=client).run("Original")

        self.assertEqual(result.final_answer, "final answer")
        self.assertEqual([answer.agent_name for answer in result.agent_answers], [
            "planner",
            "critic",
        ])
        self.assertEqual(client.calls[2]["model"], "critic-model")
        self.assertEqual(len(client.calls), 5)

    def test_orchestrator_emits_prompt_and_response_events(self) -> None:
        config = AgenticConfig(
            max_iterations=2,
            agents=(AgentSpec(name="planner", description="Plans."),),
        )
        client = FakeClient(
            [
                '["planner"]',
                "planner answer",
                "final answer",
            ]
        )
        events: list[tuple[str, dict[str, object]]] = []

        result = AgenticOrchestrator(
            config=config,
            client=client,
            event_handler=lambda event, payload: events.append((event, payload)),
        ).run("Original")

        self.assertEqual(result.final_answer, "final answer")
        self.assertEqual([event for event, _ in events], [
            "run_start",
            "request",
            "response",
            "selection",
            "request",
            "response",
            "request",
        ])
        self.assertEqual(events[1][1]["kind"], "selector")
        self.assertEqual(events[4][1]["kind"], "agent")
        self.assertEqual(events[-1][1]["kind"], "final")

    def test_max_iterations_one_goes_directly_to_final(self) -> None:
        config = AgenticConfig(
            max_iterations=1,
            agents=(AgentSpec(name="planner", description="Plans."),),
        )
        client = FakeClient(["final answer"])

        result = AgenticOrchestrator(config=config, client=client).run("Original")

        self.assertEqual(result.final_answer, "final answer")
        self.assertEqual(result.agent_answers, ())
        self.assertEqual(len(client.calls), 1)


if __name__ == "__main__":
    unittest.main()
