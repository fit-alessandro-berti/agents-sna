from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from agents_sna.config import load_config
from agents_sna.openrouter import DEFAULT_BASE_URL, DEFAULT_MODEL, OpenRouterClient
from agents_sna.orchestrator import AgenticOrchestrator


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        prompt = read_prompt(args)
        config = load_config(args.config)
        api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key missing. Set OPENROUTER_API_KEY or pass --api-key."
            )

        client = OpenRouterClient(
            api_key=api_key,
            model=args.model,
            base_url=args.base_url,
            timeout=args.timeout,
            app_url=args.app_url,
            app_name=args.app_name,
        )
        result = AgenticOrchestrator(config=config, client=client).run(prompt)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
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
        if args.show_transcript:
            for answer in result.agent_answers:
                print(f"\n[{answer.iteration}] {answer.agent_name}\n{answer.content}")
            print("\nFinal answer\n")
        print(result.final_answer)

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
        help="Print intermediate agent answers before the final answer.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print final answer and transcript as JSON.",
    )
    return parser


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file and args.prompt:
        raise ValueError("Use either positional prompt text or --prompt-file, not both.")
    if args.prompt_file:
        return args.prompt_file.read_text(encoding="utf-8")
    if args.prompt:
        return " ".join(args.prompt)
    return sys.stdin.read()


if __name__ == "__main__":
    raise SystemExit(main())
