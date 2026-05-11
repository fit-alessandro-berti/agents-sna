from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import agents_sna.agent_judge as agent_judge
from agents_sna.agent_judge import (
    MAX_EXPLANATION_CHARS,
    build_evaluation_candidates,
    build_parser,
    build_judge_messages,
    collect_request_files,
    infer_response_path,
    output_file_path,
    parse_judge_response,
    run_agent_judge,
)


class ConcurrentJudgeClient:
    active = 0
    max_active = 0
    started = 0
    lock = threading.Lock()
    second_started = threading.Event()

    def __init__(self, *args, **kwargs) -> None:
        pass

    def complete(self, messages, *, model=None):
        with type(self).lock:
            type(self).active += 1
            type(self).started += 1
            type(self).max_active = max(type(self).max_active, type(self).active)
            if type(self).started >= 2:
                type(self).second_started.set()

        try:
            if not type(self).second_started.wait(timeout=2):
                raise RuntimeError("second concurrent judge request did not start")
            return '[{"node_id":"n1","evaluation":8,"explanation":"Solid"}]'
        finally:
            with type(self).lock:
                type(self).active -= 1


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

    def test_parser_accepts_additional_payload_json(self) -> None:
        args = build_parser().parse_args(
            [
                "requests",
                "--additional-payload",
                '{"temperature": 0.2}',
                "--max-threads",
                "100",
            ]
        )

        self.assertEqual(args.additional_payload, {"temperature": 0.2})
        self.assertEqual(args.max_threads, 100)
        alias_args = build_parser().parse_args(
            [
                "requests",
                "--max_threads",
                "25",
            ]
        )
        self.assertEqual(alias_args.max_threads, 25)

    def test_run_agent_judge_can_process_files_concurrently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            requests_dir = run_dir / "requests"
            responses_dir = run_dir / "responses"
            requests_dir.mkdir(parents=True)
            responses_dir.mkdir()
            for name, prompt in (("a", "first"), ("b", "second")):
                (requests_dir / f"{name}.requests.json").write_text(
                    (
                        '[{"kind":"final","iteration":1,"agent":null,'
                        f'"messages":[{{"role":"user","content":"{prompt}"}}]}}]'
                    ),
                    encoding="utf-8",
                )
                (responses_dir / f"{name}.responses.json").write_text(
                    '[{"kind":"final","iteration":1,"agent":null,"content":"answer"}]',
                    encoding="utf-8",
                )
            output_dir = root / "evaluations"

            args = build_parser().parse_args(
                [
                    str(run_dir),
                    "--output-dir",
                    str(output_dir),
                    "--api-key",
                    "key",
                    "--no-color",
                    "--max-threads",
                    "2",
                ]
            )

            ConcurrentJudgeClient.active = 0
            ConcurrentJudgeClient.max_active = 0
            ConcurrentJudgeClient.started = 0
            ConcurrentJudgeClient.second_started = threading.Event()
            with (
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
                patch.object(agent_judge, "OpenRouterClient", ConcurrentJudgeClient),
            ):
                run_agent_judge(args)

            self.assertGreaterEqual(ConcurrentJudgeClient.max_active, 2)
            self.assertTrue(
                (
                    output_dir
                    / "run"
                    / "openaigpt-5.4"
                    / "a.agent_evaluations.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    output_dir
                    / "run"
                    / "openaigpt-5.4"
                    / "b.agent_evaluations.json"
                ).is_file()
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
