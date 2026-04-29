from __future__ import annotations

import argparse
import base64
from collections.abc import Callable
import json
import os
from pathlib import Path
import re
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents_sna.cli import (
    color_text,
    composite_event_handler,
    normalize_messages,
    resolve_config_path,
    write_json_file,
    Colors,
)
from agents_sna.config import load_config
from agents_sna.openrouter import DEFAULT_BASE_URL, DEFAULT_MODEL, OpenRouterClient
from agents_sna.orchestrator import AgenticOrchestrator
from agents_sna.types import ChatMessage, MessageContent


IMAGE_PROMPT = "Can you describe the provided visualization?"


class RetryingChatClient:
    def __init__(
        self,
        client: OpenRouterClient,
        *,
        retry_delay: float = 15.0,
        max_retries: int = 0,
        sleep_func: Callable[[float], None] = time.sleep,
        use_color: bool = True,
    ):
        self.client = client
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.sleep_func = sleep_func
        self.use_color = use_color

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
    ) -> str:
        failed_attempts = 0
        while True:
            try:
                return self.client.complete(messages, model=model)
            except Exception as exc:
                failed_attempts += 1
                if self.max_retries and failed_attempts > self.max_retries:
                    raise

                print(
                    color_text(
                        "request failed "
                        f"(attempt {failed_attempts}); retrying in "
                        f"{self.retry_delay:g}s: {exc}",
                        Colors.YELLOW,
                        enabled=self.use_color,
                    ),
                    file=sys.stderr,
                )
                self.sleep_func(self.retry_delay)


class QuestionRunRecorder:
    def __init__(
        self,
        *,
        run_name: str,
        question_name: str,
        answer_path: Path,
        request_path: Path,
        response_path: Path,
        trace_path: Path,
        metadata_path: Path,
        model: str,
        config_path: Path,
    ):
        self.run_name = run_name
        self.question_name = question_name
        self.answer_path = answer_path
        self.request_path = request_path
        self.response_path = response_path
        self.trace_path = trace_path
        self.metadata_path = metadata_path
        self.model = model
        self.config_path = config_path
        self.requests: list[dict[str, object]] = []
        self.responses: list[dict[str, object]] = []
        self.trace: dict[str, object] = {
            "run_name": run_name,
            "question": question_name,
            "original_prompt": "",
            "events": [],
        }
        self._selector_responses: dict[int, str] = {}

    def handle(self, event: str, payload: dict[str, object]) -> None:
        if event == "run_start":
            self.trace["original_prompt"] = str(payload.get("prompt", ""))
        elif event == "request":
            self._request(payload)
        elif event == "response":
            self._response(payload)
        elif event == "selection":
            self._selection(payload)

    def write(self, *, final_answer: str, status: str, error: str | None = None) -> None:
        metadata = {
            "run_name": self.run_name,
            "question": self.question_name,
            "status": status,
            "error": error,
            "model": self.model,
            "config": str(self.config_path),
            "answer_path": str(self.answer_path),
            "request_path": str(self.request_path),
            "response_path": str(self.response_path),
            "trace_path": str(self.trace_path),
            "final_answer_length": len(final_answer),
        }
        write_json_file(self.request_path, self.requests)
        write_json_file(self.response_path, self.responses)
        write_json_file(self.trace_path, self.trace)
        write_json_file(self.metadata_path, metadata)

    def _request(self, payload: dict[str, object]) -> None:
        messages = payload.get("messages")
        self.requests.append(
            {
                "kind": str(payload.get("kind", "")),
                "iteration": payload.get("iteration"),
                "agent": payload.get("agent_name"),
                "model": payload.get("model") or self.model,
                "previous_agent": payload.get("previous_agent_name"),
                "allowed_agents": list(payload.get("allowed_agent_names", ()) or ()),
                "messages": normalize_messages(messages if isinstance(messages, list) else []),
            }
        )

    def _response(self, payload: dict[str, object]) -> None:
        kind = str(payload.get("kind", ""))
        iteration = int(payload.get("iteration", 0) or 0)
        content = str(payload.get("content", ""))
        record = {
            "kind": kind,
            "iteration": iteration,
            "agent": payload.get("agent_name"),
            "content": content,
        }
        self.responses.append(record)

        if kind == "selector":
            self._selector_responses[iteration] = content
        elif kind == "agent":
            self._append_trace(
                {
                    "type": "agent_response",
                    "iteration": iteration,
                    "agent": str(payload.get("agent_name", "")),
                    "response": content,
                }
            )
        elif kind == "final":
            self._append_trace(
                {
                    "type": "final_answer",
                    "iteration": iteration,
                    "response": content,
                }
            )

    def _selection(self, payload: dict[str, object]) -> None:
        iteration = int(payload.get("iteration", 0) or 0)
        agent_names = payload.get("agent_names", ())
        if isinstance(agent_names, tuple):
            agents = list(agent_names)
        elif isinstance(agent_names, list):
            agents = agent_names
        else:
            agents = []

        self._append_trace(
            {
                "type": "choice",
                "iteration": iteration,
                "raw_response": self._selector_responses.pop(iteration, ""),
                "agents": [str(agent) for agent in agents],
                "final_requested": bool(payload.get("final_requested")),
                "allowed_agents": list(payload.get("allowed_agent_names", ()) or ()),
            }
        )

    def _append_trace(self, item: dict[str, object]) -> None:
        events = self.trace.get("events")
        if isinstance(events, list):
            events.append(item)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        run_benchmark(args)
    except Exception as exc:
        print(
            color_text(
                f"error: {exc}",
                Colors.RED,
                enabled=not getattr(args, "no_color", False),
            ),
            file=sys.stderr,
        )
        return 1
    return 0


def run_benchmark(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OpenRouter API key missing. Set OPENROUTER_API_KEY or pass --api-key.")

    benchmark_dir = args.benchmark_dir.resolve()
    questions_dir = benchmark_dir / "questions"
    answers_dir = benchmark_dir / "answers"
    if not questions_dir.is_dir():
        raise ValueError(f"Questions directory not found: {questions_dir}")
    answers_dir.mkdir(parents=True, exist_ok=True)

    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    run_slug = clean_model_name(args.run_name)
    if not run_slug:
        raise ValueError("Run name must contain at least one filename-safe character.")

    run_dir = args.output_dir.resolve() / run_slug
    request_dir = run_dir / "requests"
    response_dir = run_dir / "responses"
    trace_dir = run_dir / "traces"
    metadata_dir = run_dir / "metadata"
    for directory in (request_dir, response_dir, trace_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    question_paths = list_question_paths(questions_dir, include_images=not args.skip_images)
    if args.limit is not None:
        question_paths = question_paths[: args.limit]
    if not question_paths:
        raise ValueError(f"No questions found in {questions_dir}")

    client = RetryingChatClient(
        OpenRouterClient(
            api_key=api_key,
            model=args.model,
            base_url=args.base_url,
            timeout=args.timeout,
            app_url=args.app_url,
            app_name=args.app_name,
        ),
        retry_delay=args.retry_delay,
        max_retries=args.max_retries,
        use_color=not args.no_color,
    )

    summary: list[dict[str, object]] = []
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(
        color_text(
            f"Running {len(question_paths)} questions as {run_slug} using {config_path}",
            Colors.GREEN,
            enabled=not args.no_color,
        )
    )

    for index, question_path in enumerate(question_paths, start=1):
        answer_path = answer_file_path(
            answers_dir=answers_dir,
            run_slug=run_slug,
            question_path=question_path,
        )
        question_slug = safe_question_slug(question_path)
        request_path = request_dir / f"{question_slug}.requests.json"
        response_path = response_dir / f"{question_slug}.responses.json"
        trace_path = trace_dir / f"{question_slug}.trace.json"
        metadata_path = metadata_dir / f"{question_slug}.metadata.json"

        if answer_received(answer_path) and not args.overwrite:
            print(
                f"[{index}/{len(question_paths)}] skipping answered "
                f"{answer_path.name}"
            )
            summary.append(
                {
                    "question": question_path.name,
                    "status": "skipped_existing_response",
                    "answer_path": str(answer_path),
                }
            )
            continue

        print(f"[{index}/{len(question_paths)}] answering {question_path.name}")
        recorder = QuestionRunRecorder(
            run_name=run_slug,
            question_name=question_path.name,
            answer_path=answer_path,
            request_path=request_path,
            response_path=response_path,
            trace_path=trace_path,
            metadata_path=metadata_path,
            model=args.model,
            config_path=config_path,
        )

        try:
            prompt = load_question_prompt(question_path)
            result = AgenticOrchestrator(
                config=config,
                client=client,
                event_handler=recorder.handle,
            ).run(prompt)
            answer_path.write_text(result.final_answer + "\n", encoding="utf-8")
            recorder.write(final_answer=result.final_answer, status="completed")
            summary.append(
                {
                    "question": question_path.name,
                    "status": "completed",
                    "answer_path": str(answer_path),
                    "request_path": str(request_path),
                    "response_path": str(response_path),
                    "trace_path": str(trace_path),
                }
            )
        except Exception as exc:
            recorder.write(final_answer="", status="failed", error=str(exc))
            summary.append(
                {
                    "question": question_path.name,
                    "status": "failed",
                    "error": str(exc),
                    "request_path": str(request_path),
                    "response_path": str(response_path),
                    "trace_path": str(trace_path),
                }
            )
            print(color_text(f"failed {question_path.name}: {exc}", Colors.RED, enabled=not args.no_color))
            if not args.continue_on_error:
                break

    write_json_file(
        run_dir / "summary.json",
        {
            "run_name": run_slug,
            "requested_run_name": args.run_name,
            "model": args.model,
            "config": str(config_path),
            "benchmark_dir": str(benchmark_dir),
            "started_at": started_at,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "questions": summary,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an agents-sna config over PM-LLM-Benchmark questions."
    )
    parser.add_argument(
        "run_name",
        help=(
            "Benchmark run/model alias used in answer filenames, e.g. "
            "gpt-5.4-mini-verification-heavy."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the agents-sna JSON config to execute.",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("../pm-llm-benchmark"),
        help="Path to the PM-LLM-Benchmark checkout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_runs"),
        help="Local directory where request/response/trace artifacts are stored.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model to call. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--api-key",
        help="OpenRouter API key. Defaults to OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"OpenRouter-compatible API base URL. Defaults to {DEFAULT_BASE_URL}.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--app-url",
        help="Optional HTTP-Referer header value for OpenRouter rankings.",
    )
    parser.add_argument(
        "--app-name",
        default="agents-sna-benchmark",
        help="X-Title header value sent to OpenRouter.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute answers even if the answer file already exists.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later questions after a failed question.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=15.0,
        help="Seconds to wait before retrying a failed request. Defaults to 15.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Maximum retries per failed request. Use 0 for unlimited retries.",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip PNG questions. By default PNG questions are sent as image_url content.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process the first N discovered questions.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output.",
    )
    return parser


def list_question_paths(questions_dir: Path, *, include_images: bool) -> list[Path]:
    allowed_suffixes = {".txt"}
    if include_images:
        allowed_suffixes.add(".png")
    return sorted(
        path
        for path in questions_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in allowed_suffixes
        and not path.name.startswith("__")
    )


def load_question_prompt(question_path: Path) -> MessageContent:
    if question_path.suffix.lower() == ".txt":
        return question_path.read_text(encoding="utf-8")
    if question_path.suffix.lower() == ".png":
        encoded = base64.b64encode(question_path.read_bytes()).decode("ascii")
        return [
            {"type": "text", "text": IMAGE_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            },
        ]
    raise ValueError(f"Unsupported question type: {question_path}")


def answer_file_path(*, answers_dir: Path, run_slug: str, question_path: Path) -> Path:
    answer_name = f"{run_slug}_{question_path.name}"
    if question_path.suffix.lower() == ".png":
        answer_name = answer_name[: -len(".png")] + ".txt"
    return answers_dir / answer_name


def answer_received(answer_path: Path) -> bool:
    if not answer_path.exists() or not answer_path.is_file():
        return False
    return bool(answer_path.read_text(encoding="utf-8", errors="ignore").strip())


def safe_question_slug(question_path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", question_path.stem).strip("._-")


def clean_model_name(name: str) -> str:
    cleaned = name.replace("/", "").replace(":", "")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", cleaned)
    return cleaned.strip("._-")


if __name__ == "__main__":
    raise SystemExit(main())
