from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import textwrap
from typing import TextIO

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents_sna.config import load_config
from agents_sna.openrouter import DEFAULT_BASE_URL, DEFAULT_MODEL, OpenRouterClient
from agents_sna.orchestrator import AgenticOrchestrator


JsonObject = dict[str, object]


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GRAY = "\033[90m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[97m"


class CliLogger:
    def __init__(
        self,
        *,
        enabled: bool,
        use_color: bool,
        stream: TextIO,
        config_path: Path,
        default_model: str,
        base_url: str,
    ):
        self.enabled = enabled
        self.use_color = use_color
        self.stream = stream
        self.config_path = config_path
        self.default_model = default_model
        self.base_url = base_url

    def handle(self, event: str, payload: dict[str, object]) -> None:
        if not self.enabled:
            return

        if event == "run_start":
            self._run_start(payload)
        elif event == "request":
            self._request(payload)
        elif event == "response":
            self._response(payload)
        elif event == "selection":
            self._selection(payload)

    def _run_start(self, payload: dict[str, object]) -> None:
        prompt = str(payload.get("prompt", ""))
        max_iterations = payload.get("max_iterations", "?")
        agent_names = payload.get("agent_names", ())
        agents = ", ".join(str(name) for name in agent_names)
        single_agent = bool(payload.get("single_agent_per_iteration"))
        excluded_handoffs = payload.get("excluded_handoffs", ())

        self._line("agents-sna starting", Colors.GREEN, bold=True)
        self._line(f"config: {self.config_path}", Colors.GRAY)
        self._line(f"openrouter base url: {self.base_url}", Colors.GRAY)
        self._line(f"default model: {self.default_model}", Colors.GRAY)
        self._line(f"max iterations: {max_iterations}", Colors.GRAY)
        self._line(f"agents: {agents}", Colors.GRAY)
        self._line(f"single agent per iteration: {single_agent}", Colors.GRAY)
        if isinstance(excluded_handoffs, tuple) and excluded_handoffs:
            handoffs = ", ".join(
                f"{item.get('from')} -> {item.get('to')}"
                for item in excluded_handoffs
                if isinstance(item, dict)
            )
            self._line(f"excluded handoffs: {handoffs}", Colors.GRAY)
        self._block("original prompt sent by the user", prompt, Colors.YELLOW)

    def _request(self, payload: dict[str, object]) -> None:
        kind = str(payload.get("kind", "request"))
        iteration = payload.get("iteration", "?")
        agent_name = payload.get("agent_name")
        model = payload.get("model") or self.default_model
        messages = payload.get("messages")

        if kind == "selector":
            title = f"selector prompt sent | iteration {iteration} | model {model}"
            color = Colors.MAGENTA
        elif kind == "agent":
            title = (
                f"agent prompt sent | iteration {iteration} | "
                f"agent {agent_name} | model {model}"
            )
            color = Colors.CYAN
        elif kind == "final":
            title = f"final synthesis prompt sent | model {model}"
            color = Colors.BLUE
        else:
            title = f"{kind} prompt sent | iteration {iteration} | model {model}"
            color = Colors.BLUE

        self._line("")
        self._line(title, color, bold=True)
        if isinstance(messages, list):
            self._messages(messages)

    def _response(self, payload: dict[str, object]) -> None:
        kind = str(payload.get("kind", "response"))
        iteration = payload.get("iteration", "?")
        content = str(payload.get("content", ""))

        if kind == "selector":
            title = f"selector response received | iteration {iteration}"
            color = Colors.MAGENTA
        elif kind == "agent":
            agent_name = payload.get("agent_name", "?")
            title = f"agent response received | iteration {iteration} | agent {agent_name}"
            color = Colors.GREEN
        elif kind == "final":
            self._line("")
            self._line(
                "final response received; printing final answer on stdout",
                Colors.BLUE,
                bold=True,
            )
            return
        else:
            title = f"{kind} response received | iteration {iteration}"
            color = Colors.CYAN

        self._block(title, content, color)

    def _selection(self, payload: dict[str, object]) -> None:
        iteration = payload.get("iteration", "?")
        if payload.get("final_requested"):
            text = f"selector requested final answer generation at iteration {iteration}"
        else:
            names = payload.get("agent_names", ())
            selected = ", ".join(str(name) for name in names)
            text = f"selector chose next agent(s) at iteration {iteration}: {selected}"
        self._line(text, Colors.BLUE)

    def _messages(self, messages: list[object]) -> None:
        for index, raw_message in enumerate(messages, start=1):
            if not isinstance(raw_message, dict):
                continue
            role = str(raw_message.get("role", "?"))
            content = str(raw_message.get("content", ""))
            color = {
                "system": Colors.BLUE,
                "user": Colors.YELLOW,
                "assistant": Colors.GREEN,
            }.get(role, Colors.CYAN)

            self._line(f"  message {index}: {role}", color)
            self._line(textwrap.indent(content.rstrip() or "<empty>", "    "), Colors.GRAY)

    def _block(self, title: str, content: str, color: str) -> None:
        self._line("")
        self._line(title, color, bold=True)
        self._line(textwrap.indent(content.rstrip() or "<empty>", "  "), color)

    def _line(self, text: str, color: str = Colors.GRAY, *, bold: bool = False) -> None:
        print(self._paint(text, color, bold=bold), file=self.stream)

    def _paint(self, text: str, color: str, *, bold: bool = False) -> str:
        if not self.use_color:
            return text
        prefix = f"{Colors.BOLD if bold else ''}{color}"
        return f"{prefix}{text}{Colors.RESET}"


class ReportRecorder:
    def __init__(
        self,
        *,
        request_inputs_path: Path | None,
        conversation_trace_path: Path | None,
    ):
        self.request_inputs_path = request_inputs_path
        self.conversation_trace_path = conversation_trace_path
        self.request_inputs: list[list[dict[str, str]]] = []
        self.trace: JsonObject = {
            "original_prompt": "",
            "events": [],
        }
        self._selector_responses: dict[int, str] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.request_inputs_path or self.conversation_trace_path)

    def handle(self, event: str, payload: dict[str, object]) -> None:
        if not self.enabled:
            return

        if event == "run_start":
            self.trace["original_prompt"] = str(payload.get("prompt", ""))
        elif event == "request":
            self._request(payload)
        elif event == "response":
            self._response(payload)
        elif event == "selection":
            self._selection(payload)

    def write(self) -> None:
        if self.request_inputs_path:
            write_json_file(self.request_inputs_path, self.request_inputs)
        if self.conversation_trace_path:
            write_json_file(self.conversation_trace_path, self.trace)

    def _request(self, payload: dict[str, object]) -> None:
        if not self.request_inputs_path:
            return

        messages = payload.get("messages")
        if not isinstance(messages, list):
            return

        self.request_inputs.append(normalize_messages(messages))

    def _response(self, payload: dict[str, object]) -> None:
        kind = str(payload.get("kind", "response"))
        iteration = int(payload.get("iteration", 0) or 0)
        content = str(payload.get("content", ""))

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
            }
        )

    def _append_trace(self, item: JsonObject) -> None:
        events = self.trace.get("events")
        if isinstance(events, list):
            events.append(item)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        prompt = read_prompt(args)
        config_path = resolve_config_path(args.config)
        config = load_config(config_path)
        api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key missing. Set OPENROUTER_API_KEY or pass --api-key."
            )

        logger = CliLogger(
            enabled=not args.quiet,
            use_color=not args.no_color and os.getenv("NO_COLOR") is None,
            stream=sys.stderr,
            config_path=config_path,
            default_model=args.model,
            base_url=args.base_url,
        )
        report_recorder = ReportRecorder(
            request_inputs_path=args.request_inputs_file,
            conversation_trace_path=args.conversation_trace_file,
        )
        client = OpenRouterClient(
            api_key=api_key,
            model=args.model,
            base_url=args.base_url,
            timeout=args.timeout,
            app_url=args.app_url,
            app_name=args.app_name,
        )
        result = AgenticOrchestrator(
            config=config,
            client=client,
            event_handler=composite_event_handler(logger.handle, report_recorder.handle),
        ).run(prompt)
        report_recorder.write()
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

    if args.json:
        print(
            json.dumps(
                {
                    "final_answer": result.final_answer,
                    "agent_answers": [
                        {
                            "iteration": answer.iteration,
                            "agent": answer.agent_name,
                            "content": answer.content,
                        }
                        for answer in result.agent_answers
                    ],
                },
                indent=2,
            )
        )
    else:
        print(color_text(result.final_answer, Colors.WHITE, enabled=not args.no_color))

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an OpenRouter-backed multi-agent LLM discussion."
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Original prompt. If omitted, stdin is used.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Read the original prompt from a UTF-8 text file.",
    )
    parser.add_argument(
        "--config",
        default="configs/agents.example.json",
        help="Path to the JSON config containing agents and max_iterations.",
    )
    parser.add_argument(
        "--api-key",
        help="OpenRouter API key. Defaults to OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Default OpenRouter model. Defaults to {DEFAULT_MODEL}.",
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
        default="agents-sna",
        help="X-Title header value sent to OpenRouter.",
    )
    parser.add_argument(
        "--show-transcript",
        action="store_true",
        help="Deprecated: progress output now shows the transcript during the run.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print final answer and transcript as JSON.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output. The final answer is still printed.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output.",
    )
    parser.add_argument(
        "--request-inputs-file",
        type=Path,
        help=(
            "Write a JSON report of every LLM request input message list "
            "with system/user/assistant roles."
        ),
    )
    parser.add_argument(
        "--conversation-trace-file",
        type=Path,
        help=(
            "Write a compact JSON trace with the original prompt, selector "
            "choices, agent responses, and final answer."
        ),
    )
    return parser


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file and args.prompt:
        raise ValueError("Use either positional prompt text or --prompt-file, not both.")
    if args.prompt_file:
        return args.prompt_file.read_text(encoding="utf-8")
    if args.prompt:
        return " ".join(args.prompt)
    if sys.stdin.isatty():
        raise ValueError("Provide a prompt argument, --prompt-file, or pipe text on stdin.")
    return sys.stdin.read()


def resolve_config_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path

    repo_relative = Path(__file__).resolve().parents[2] / path
    if repo_relative.exists():
        return repo_relative
    return path


def color_text(text: str, color: str, *, enabled: bool = True) -> str:
    if not enabled or os.getenv("NO_COLOR") is not None:
        return text
    return f"{color}{text}{Colors.RESET}"


def composite_event_handler(*handlers: object):
    def handle(event: str, payload: dict[str, object]) -> None:
        for handler in handlers:
            if callable(handler):
                handler(event, payload)

    return handle


def normalize_messages(messages: list[object]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        normalized.append(
            {
                "role": str(message.get("role", "")),
                "content": str(message.get("content", "")),
            }
        )
    return normalized


def write_json_file(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
