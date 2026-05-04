from __future__ import annotations

import base64
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import agents_sna.benchmark_runner as benchmark_runner
from agents_sna.benchmark_runner import (
    IMAGE_PROMPT,
    RetryingChatClient,
    answer_received,
    answer_file_path,
    build_parser,
    clean_model_name,
    list_question_paths,
    load_question_prompt,
    run_benchmark,
    safe_question_slug,
)
from agents_sna.orchestrator import OrchestrationResult


class FlakyClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, *, model=None):
        self.calls += 1
        if self.calls < 3:
            raise RuntimeError("temporary failure")
        return "ok"


class FakeOpenRouterClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def complete(self, messages, *, model=None):
        return "unused"


class FailOnceOrchestrator:
    calls = 0

    def __init__(self, *, config, client, event_handler=None) -> None:
        self.event_handler = event_handler

    def run(self, prompt):
        type(self).calls += 1
        if type(self).calls == 1:
            raise RuntimeError("question failure")
        return OrchestrationResult(final_answer="second answer", agent_answers=())


class ConcurrentOrchestrator:
    active = 0
    max_active = 0
    started = 0
    lock = threading.Lock()
    second_started = threading.Event()

    def __init__(self, *, config, client, event_handler=None) -> None:
        pass

    def run(self, prompt):
        with type(self).lock:
            type(self).active += 1
            type(self).started += 1
            type(self).max_active = max(type(self).max_active, type(self).active)
            if type(self).started >= 2:
                type(self).second_started.set()

        if not type(self).second_started.wait(timeout=2):
            raise RuntimeError("second concurrent question did not start")

        with type(self).lock:
            type(self).active -= 1
        return OrchestrationResult(final_answer=f"answer {prompt}", agent_answers=())


class BenchmarkRunnerTests(unittest.TestCase):
    def test_clean_model_name_matches_benchmark_style(self) -> None:
        self.assertEqual(
            clean_model_name("openai/gpt-5.4-mini:verification heavy"),
            "openaigpt-5.4-miniverification_heavy",
        )

    def test_answer_file_path_uses_txt_for_png_questions(self) -> None:
        answer_path = answer_file_path(
            answers_dir=Path("answers"),
            run_slug="run",
            question_path=Path("cat07_01_ocdfg.png"),
        )

        self.assertEqual(answer_path, Path("answers/run_cat07_01_ocdfg.txt"))

    def test_answer_received_requires_non_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "answer.txt"
            self.assertFalse(answer_received(path))

            path.write_text("   \n", encoding="utf-8")
            self.assertFalse(answer_received(path))

            path.write_text("response\n", encoding="utf-8")
            self.assertTrue(answer_received(path))

    def test_retrying_chat_client_retries_failed_requests(self) -> None:
        sleeps: list[float] = []
        client = RetryingChatClient(
            FlakyClient(),
            retry_delay=15,
            sleep_func=sleeps.append,
            use_color=False,
        )

        with redirect_stderr(io.StringIO()):
            result = client.complete([{"role": "user", "content": "hello"}])

        self.assertEqual(result, "ok")
        self.assertEqual(sleeps, [15, 15])

    def test_list_question_paths_filters_supported_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            questions_dir = Path(directory)
            (questions_dir / "b.txt").write_text("b", encoding="utf-8")
            (questions_dir / "a.png").write_bytes(b"png")
            (questions_dir / "__init__.py").write_text("", encoding="utf-8")

            paths = list_question_paths(questions_dir, include_images=True)

        self.assertEqual([path.name for path in paths], ["a.png", "b.txt"])

    def test_load_text_question_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "question.txt"
            path.write_text("Question text", encoding="utf-8")

            prompt = load_question_prompt(path)

        self.assertEqual(prompt, "Question text")

    def test_load_png_question_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "question.png"
            path.write_bytes(b"image-bytes")

            prompt = load_question_prompt(path)

        expected_image = base64.b64encode(b"image-bytes").decode("ascii")
        self.assertIsInstance(prompt, list)
        self.assertEqual(prompt[0]["text"], IMAGE_PROMPT)
        self.assertEqual(
            prompt[1]["image_url"]["url"],
            f"data:image/png;base64,{expected_image}",
        )

    def test_safe_question_slug(self) -> None:
        self.assertEqual(safe_question_slug(Path("cat 01?/x.txt")), "x")

    def test_parser_accepts_additional_payload_json(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "--config",
                "config.json",
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
                "run",
                "--config",
                "config.json",
                "--max_threads",
                "25",
            ]
        )
        self.assertEqual(alias_args.max_threads, 25)

    def test_run_benchmark_continues_after_failed_question_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_dir = root / "pm-llm-benchmark"
            questions_dir = benchmark_dir / "questions"
            questions_dir.mkdir(parents=True)
            (questions_dir / "a.txt").write_text("first", encoding="utf-8")
            (questions_dir / "b.txt").write_text("second", encoding="utf-8")
            output_dir = root / "runs"

            args = build_parser().parse_args(
                [
                    "run",
                    "--config",
                    "config.json",
                    "--benchmark-dir",
                    str(benchmark_dir),
                    "--output-dir",
                    str(output_dir),
                    "--api-key",
                    "key",
                    "--no-color",
                ]
            )

            FailOnceOrchestrator.calls = 0
            with (
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
                patch.object(benchmark_runner, "OpenRouterClient", FakeOpenRouterClient),
                patch.object(benchmark_runner, "AgenticOrchestrator", FailOnceOrchestrator),
                patch.object(benchmark_runner, "load_config", return_value=object()),
                patch.object(benchmark_runner, "resolve_config_path", return_value=Path("config.json")),
            ):
                run_benchmark(args)

            self.assertEqual(FailOnceOrchestrator.calls, 2)
            self.assertFalse((benchmark_dir / "answers" / "run_a.txt").exists())
            self.assertEqual(
                (benchmark_dir / "answers" / "run_b.txt").read_text(encoding="utf-8"),
                "second answer\n",
            )

    def test_run_benchmark_can_process_questions_concurrently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_dir = root / "pm-llm-benchmark"
            questions_dir = benchmark_dir / "questions"
            questions_dir.mkdir(parents=True)
            (questions_dir / "a.txt").write_text("first", encoding="utf-8")
            (questions_dir / "b.txt").write_text("second", encoding="utf-8")
            output_dir = root / "runs"

            args = build_parser().parse_args(
                [
                    "run",
                    "--config",
                    "config.json",
                    "--benchmark-dir",
                    str(benchmark_dir),
                    "--output-dir",
                    str(output_dir),
                    "--api-key",
                    "key",
                    "--no-color",
                    "--max-threads",
                    "2",
                ]
            )

            ConcurrentOrchestrator.active = 0
            ConcurrentOrchestrator.max_active = 0
            ConcurrentOrchestrator.started = 0
            ConcurrentOrchestrator.second_started = threading.Event()
            with (
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
                patch.object(benchmark_runner, "OpenRouterClient", FakeOpenRouterClient),
                patch.object(benchmark_runner, "AgenticOrchestrator", ConcurrentOrchestrator),
                patch.object(benchmark_runner, "load_config", return_value=object()),
                patch.object(benchmark_runner, "resolve_config_path", return_value=Path("config.json")),
            ):
                run_benchmark(args)

            self.assertGreaterEqual(ConcurrentOrchestrator.max_active, 2)
            self.assertTrue((benchmark_dir / "answers" / "run_a.txt").is_file())
            self.assertTrue((benchmark_dir / "answers" / "run_b.txt").is_file())


if __name__ == "__main__":
    unittest.main()
