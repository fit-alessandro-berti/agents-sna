from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents_sna.agent_judge import (
    MAX_EXPLANATION_CHARS,
    build_evaluation_candidates,
    build_judge_messages,
    collect_request_files,
    infer_response_path,
    output_file_path,
    parse_judge_response,
)


class AgentJudgeTests(unittest.TestCase):
    def test_collect_request_files_accepts_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            requests_dir = run_dir / "requests"
            requests_dir.mkdir(parents=True)
            (requests_dir / "b.requests.json").write_text("[]", encoding="utf-8")
            (requests_dir / "a.requests.json").write_text("[]", encoding="utf-8")

            files = collect_request_files([str(run_dir)])

        self.assertEqual([path.name for path in files], ["a.requests.json", "b.requests.json"])

    def test_infer_response_path_from_requests_directory(self) -> None:
        request_path = Path("benchmark_runs/run/requests/cat.requests.json")

        response_path = infer_response_path(request_path)

        self.assertEqual(response_path, Path("benchmark_runs/run/responses/cat.responses.json"))

    def test_output_file_path_uses_source_run_and_judge_model(self) -> None:
        request_path = Path("benchmark_runs/source-run/requests/cat.requests.json")

        output_path = output_file_path(
            output_dir=Path("agent_evaluations"),
            request_path=request_path,
            judge_model="openai/gpt-5.4",
        )

        self.assertEqual(
            output_path,
            Path("agent_evaluations/source-run/openaigpt-5.4/cat.agent_evaluations.json"),
        )

    def test_build_candidates_skips_later_selectors_and_includes_complete(self) -> None:
        requests = [
            {
                "kind": "selector",
                "iteration": 1,
                "agent": None,
                "allowed_agents": ["planner"],
                "messages": [{"role": "user", "content": "Original"}],
            },
            {
                "kind": "agent",
                "iteration": 1,
                "agent": "planner",
                "messages": [{"role": "system", "content": "Persona"}],
            },
            {
                "kind": "selector",
                "iteration": 2,
                "agent": None,
                "messages": [],
            },
        ]
        responses = [
            {"kind": "selector", "iteration": 1, "agent": None, "content": '["planner"]'},
            {"kind": "agent", "iteration": 1, "agent": "planner", "content": "Plan"},
            {"kind": "selector", "iteration": 2, "agent": None, "content": '["FINAL"]'},
            {"kind": "final", "iteration": 15, "agent": None, "content": "Final"},
        ]

        candidates = build_evaluation_candidates(requests, responses)

        self.assertEqual(
            [candidate["agent_type"] for candidate in candidates],
            ["START", "planner", "COMPLETE"],
        )

    def test_build_candidates_keeps_complete_last(self) -> None:
        requests = []
        responses = [
            {"kind": "final", "iteration": 15, "agent": None, "content": "Final"},
            {"kind": "agent", "iteration": 1, "agent": "planner", "content": "Plan"},
        ]

        candidates = build_evaluation_candidates(requests, responses)

        self.assertEqual(
            [candidate["agent_type"] for candidate in candidates],
            ["planner", "COMPLETE"],
        )
        self.assertEqual([candidate["node_id"] for candidate in candidates], ["n1", "n2"])

    def test_parse_judge_response_returns_candidate_scores(self) -> None:
        candidates = [
            {"node_id": "n1", "agent_type": "START"},
            {"node_id": "n2", "agent_type": "planner"},
        ]
        raw = (
            '[{"node_id":"n1","evaluation":12,"explanation":"Good first choice"},'
            '{"node_id":"n2","evaluation":8.5,"explanation":"Useful plan"}]'
        )

        parsed = parse_judge_response(raw, candidates)

        self.assertEqual(parsed, [
            {"agent_type": "START", "evaluation": 10.0, "explanation": "Good first choice"},
            {"agent_type": "planner", "evaluation": 8.5, "explanation": "Useful plan"},
        ])

    def test_parse_judge_response_requires_explanations(self) -> None:
        candidates = [{"node_id": "n1", "agent_type": "START"}]
        raw = '[{"node_id":"n1","evaluation":8}]'

        with self.assertRaisesRegex(ValueError, "Missing explanation"):
            parse_judge_response(raw, candidates)

    def test_parse_judge_response_trims_long_explanations(self) -> None:
        candidates = [{"node_id": "n1", "agent_type": "START"}]
        long_explanation = "x" * 80
        raw = (
            '[{"node_id":"n1","evaluation":8,"explanation":"'
            f'{long_explanation}'
            '"}]'
        )

        parsed = parse_judge_response(raw, candidates)

        self.assertEqual(len(parsed[0]["explanation"]), MAX_EXPLANATION_CHARS)

    def test_judge_prompt_requires_harsh_penalties(self) -> None:
        messages = build_judge_messages(
            candidates=[
                {
                    "node_id": "n1",
                    "agent_type": "planner",
                    "kind": "agent",
                    "iteration": 1,
                    "request_context": "Agent: planner",
                    "output": "Plan",
                }
            ],
            original_prompt="Original",
        )

        system_prompt = messages[0]["content"]

        self.assertIn("Apply a harsh grading standard", system_prompt)
        self.assertIn("small defects", system_prompt)
        self.assertIn("large decrease", system_prompt)
        self.assertIn("explanation", messages[1]["content"])
        self.assertIn("at most 50 characters", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
