from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import re
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents_sna.benchmark_runner import RetryingChatClient, clean_model_name
from agents_sna.cli import color_text, write_json_file, Colors
from agents_sna.openrouter import DEFAULT_BASE_URL, OpenRouterClient


DEFAULT_JUDGE_MODEL = "openai/gpt-5.4"
MAX_EXPLANATION_CHARS = 50


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        run_agent_judge(args)
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


def run_agent_judge(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OpenRouter API key missing. Set OPENROUTER_API_KEY or pass --api-key.")

    request_files = collect_request_files(args.request_inputs)
    if args.limit is not None:
        request_files = request_files[: args.limit]
    if not request_files:
        raise ValueError("No request files found.")

    client = RetryingChatClient(
        OpenRouterClient(
            api_key=api_key,
            model=args.judge_model,
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
    for index, request_path in enumerate(request_files, start=1):
        response_path = infer_response_path(request_path, args.responses_dir)
        output_path = output_file_path(
            output_dir=args.output_dir,
            request_path=request_path,
            judge_model=args.judge_model,
        )

        if output_path.exists() and output_path.read_text(encoding="utf-8").strip() and not args.overwrite:
            print(f"[{index}/{len(request_files)}] skipping judged {output_path.name}")
            summary.append(
                {
                    "request_path": str(request_path),
                    "response_path": str(response_path),
                    "output_path": str(output_path),
                    "status": "skipped_existing",
                }
            )
            continue

        print(f"[{index}/{len(request_files)}] judging {request_path.name}")
        try:
            requests = read_json_list(request_path)
            responses = read_json_list(response_path)
            candidates = build_evaluation_candidates(requests, responses)
            evaluations = judge_candidates(
                client=client,
                candidates=candidates,
                original_prompt=extract_original_prompt(requests),
                judge_model=args.judge_model,
            )
            write_json_file(output_path, evaluations)
            summary.append(
                {
                    "request_path": str(request_path),
                    "response_path": str(response_path),
                    "output_path": str(output_path),
                    "status": "completed",
                }
            )
        except Exception as exc:
            summary.append(
                {
                    "request_path": str(request_path),
                    "response_path": str(response_path),
                    "output_path": str(output_path),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(
                color_text(
                    f"failed {request_path.name}: {exc}",
                    Colors.RED,
                    enabled=not args.no_color,
                ),
                file=sys.stderr,
            )
            if not args.continue_on_error:
                break

    write_json_file(
        args.output_dir / "summary.json",
        {
            "judge_model": args.judge_model,
            "evaluated_files": summary,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate saved benchmark agent work with an OpenRouter LLM judge."
    )
    parser.add_argument(
        "request_inputs",
        nargs="+",
        help=(
            "Request artifact files, request directories, benchmark run directories, "
            "or glob patterns."
        ),
    )
    parser.add_argument(
        "--responses-dir",
        type=Path,
        help="Optional directory containing matching *.responses.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("agent_evaluations"),
        help="Directory where agent evaluation JSON files are written.",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"OpenRouter judge model. Defaults to {DEFAULT_JUDGE_MODEL}.",
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
        default="agents-sna-agent-judge",
        help="X-Title header value sent to OpenRouter.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=15.0,
        help="Seconds to wait before retrying a failed judge request. Defaults to 15.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Maximum retries per failed judge request. Use 0 for unlimited retries.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute existing non-empty evaluation files.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later files after a failed evaluation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process the first N discovered request files.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output.",
    )
    return parser


def collect_request_files(inputs: list[str]) -> list[Path]:
    discovered: list[Path] = []
    for raw_input in inputs:
        matches = [Path(match) for match in glob.glob(raw_input)]
        paths = matches or [Path(raw_input)]
        for path in paths:
            if path.is_dir():
                if (path / "requests").is_dir():
                    discovered.extend(sorted((path / "requests").glob("*.requests.json")))
                else:
                    discovered.extend(sorted(path.glob("*.requests.json")))
            elif path.is_file():
                discovered.append(path)

    unique: dict[Path, None] = {}
    for path in discovered:
        unique[path.resolve()] = None
    return sorted(unique)


def infer_response_path(request_path: Path, responses_dir: Path | None = None) -> Path:
    response_name = request_path.name.replace(".requests.json", ".responses.json")
    if response_name == request_path.name:
        response_name = f"{request_path.stem}.responses.json"

    if responses_dir is not None:
        return responses_dir / response_name
    if request_path.parent.name == "requests":
        return request_path.parent.parent / "responses" / response_name
    return request_path.parent / response_name


def output_file_path(*, output_dir: Path, request_path: Path, judge_model: str) -> Path:
    source_run = source_run_name(request_path)
    judge_slug = clean_model_name(judge_model)
    question_slug = request_path.name.replace(".requests.json", "")
    return output_dir / source_run / judge_slug / f"{question_slug}.agent_evaluations.json"


def source_run_name(request_path: Path) -> str:
    if request_path.parent.name == "requests":
        return clean_model_name(request_path.parent.parent.name)
    return clean_model_name(request_path.parent.name)


def read_json_list(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise ValueError(f"JSON artifact not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Expected JSON list of objects in {path}")
    return data


def build_evaluation_candidates(
    requests: list[dict[str, object]],
    responses: list[dict[str, object]],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    first_selector_seen = False
    used_request_indexes: set[int] = set()
    final_candidate: dict[str, object] | None = None

    for response in responses:
        kind = str(response.get("kind", ""))
        if kind == "selector":
            if first_selector_seen:
                continue
            first_selector_seen = True
            agent_type = "START"
        elif kind == "agent":
            agent_type = str(response.get("agent", "") or "agent")
        elif kind == "final":
            agent_type = "COMPLETE"
        else:
            continue

        request = find_matching_request(response, requests, used_request_indexes)
        candidate = {
            "node_id": "",
            "agent_type": agent_type,
            "kind": kind,
            "iteration": response.get("iteration"),
            "request_context": summarize_request_context(request),
            "output": str(response.get("content", "")),
        }
        if kind == "final":
            final_candidate = candidate
        else:
            candidates.append(candidate)

    if final_candidate is None:
        final_candidate = {
            "node_id": "",
            "agent_type": "COMPLETE",
            "kind": "final",
            "iteration": None,
            "request_context": "Missing final response.",
            "output": "",
        }

    candidates.append(final_candidate)
    for index, candidate in enumerate(candidates, start=1):
        candidate["node_id"] = f"n{index}"

    return candidates


def find_matching_request(
    response: dict[str, object],
    requests: list[dict[str, object]],
    used_request_indexes: set[int],
) -> dict[str, object] | None:
    response_kind = response.get("kind")
    response_iteration = response.get("iteration")
    response_agent = response.get("agent")

    for index, request in enumerate(requests):
        if index in used_request_indexes:
            continue
        if (
            request.get("kind") == response_kind
            and request.get("iteration") == response_iteration
            and request.get("agent") == response_agent
        ):
            used_request_indexes.add(index)
            return request
    return None


def summarize_request_context(request: dict[str, object] | None) -> str:
    if not request:
        return ""

    parts: list[str] = []
    if request.get("kind") == "selector":
        parts.append(f"Allowed agents: {request.get('allowed_agents', [])}")
    agent = request.get("agent")
    if agent:
        parts.append(f"Agent: {agent}")

    messages = request.get("messages", [])
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "system":
                parts.append(f"System/persona: {message_content_to_text(message.get('content', ''))}")
                break
    return "\n".join(parts)


def extract_original_prompt(requests: list[dict[str, object]]) -> str:
    for request in requests:
        messages = request.get("messages", [])
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "user":
                return message_content_to_text(message.get("content", ""))
    return ""


def message_content_to_text(content: object) -> str:
    if isinstance(content, str):
        return redact_large_data_urls(content)
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                parts.append(redact_large_data_urls(str(part.get("text", ""))))
            elif part_type == "image_url":
                parts.append("[image_url omitted]")
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)


def redact_large_data_urls(text: str) -> str:
    return re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "[image_url omitted]", text)


def judge_candidates(
    *,
    client: RetryingChatClient,
    candidates: list[dict[str, object]],
    original_prompt: str,
    judge_model: str,
) -> list[dict[str, float | str]]:
    judge_messages = build_judge_messages(
        candidates=candidates,
        original_prompt=original_prompt,
    )
    raw_response = client.complete(judge_messages, model=judge_model)
    return parse_judge_response(raw_response, candidates)


def build_judge_messages(
    *,
    candidates: list[dict[str, object]],
    original_prompt: str,
) -> list[dict[str, str]]:
    candidate_payload = [
        {
            "node_id": candidate["node_id"],
            "agent_type": candidate["agent_type"],
            "kind": candidate["kind"],
            "iteration": candidate["iteration"],
            "request_context": candidate["request_context"],
            "output": candidate["output"],
        }
        for candidate in candidates
    ]
    expected_ids = [candidate["node_id"] for candidate in candidates]

    system_prompt = (
        "You are a strict LLM-as-a-judge for a process-mining multi-agent "
        "benchmark. Score each listed node's contribution from 1.0 "
        "(minimum) to 10.0 (maximum). Judge usefulness, correctness, grounding "
        "in the original prompt, process-mining rigor, and whether the node "
        "advanced the final answer. Apply a harsh grading standard: small "
        "defects, unsupported assumptions, vague claims, missing edge cases, "
        "or minor process-mining inaccuracies must cause a large decrease in "
        "the score. Do not reward fluent but weak answers. Reserve 9.0-10.0 "
        "for nearly flawless, specific, well-grounded contributions; use "
        "7.0-8.9 only for useful answers with no meaningful defects; use "
        "4.0-6.9 for partially useful answers with any clear defect; use "
        "1.0-3.9 for misleading, shallow, irrelevant, or harmful work. Return "
        "only valid JSON."
    )
    user_prompt = (
        "Original benchmark prompt:\n"
        f"{original_prompt}\n\n"
        "Nodes to evaluate, in order:\n"
        f"{json.dumps(candidate_payload, indent=2, ensure_ascii=False)}\n\n"
        "Return a JSON list with exactly one object per node_id, in the same "
        "order. Each object must contain: node_id, agent_type, evaluation, "
        "explanation. The explanation must be a terse grading reason, ideally "
        f"at most {MAX_EXPLANATION_CHARS} characters. "
        f"Expected node_id order: {expected_ids}. "
        "The evaluation must be a float between 1.0 and 10.0. Do not include "
        "Markdown or any text outside the JSON."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_judge_response(
    raw_response: str,
    candidates: list[dict[str, object]],
) -> list[dict[str, float | str]]:
    parsed = json.loads(extract_json_list(raw_response))
    if not isinstance(parsed, list):
        raise ValueError(f"Judge response must be a JSON list: {raw_response}")
    if len(parsed) != len(candidates):
        raise ValueError(
            "Judge response must contain exactly "
            f"{len(candidates)} objects; got {len(parsed)}"
        )

    by_node_id: dict[str, dict[str, object]] = {}
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"Judge item {index + 1} must be an object")
        raw_node_id = item.get("node_id")
        if raw_node_id is None:
            raise ValueError(f"Judge item {index + 1} is missing node_id")
        node_id = str(raw_node_id)
        if node_id in by_node_id:
            raise ValueError(f"Duplicate judge score for node {node_id}")
        by_node_id[node_id] = item

    scored: list[dict[str, float | str]] = []
    for candidate in candidates:
        candidate_id = str(candidate["node_id"])
        item = by_node_id.get(candidate_id)
        if item is None:
            raise ValueError(f"Missing judge score for node {candidate_id}")

        raw_score = item.get("evaluation", item.get("score"))
        if raw_score is None:
            raise ValueError(f"Missing evaluation for node {candidate_id}")

        scored.append(
            {
                "agent_type": str(candidate["agent_type"]),
                "evaluation": clamp_score(parse_score(raw_score, candidate_id)),
                "explanation": parse_explanation(item, candidate_id),
            }
        )
    return scored


def parse_score(raw_score: object, candidate_id: str) -> float:
    if isinstance(raw_score, bool):
        raise ValueError(f"Evaluation for node {candidate_id} must be numeric")
    try:
        return float(raw_score)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Evaluation for node {candidate_id} must be numeric") from exc


def parse_explanation(item: dict[str, object], candidate_id: str) -> str:
    raw_explanation = item.get("explanation")
    if not isinstance(raw_explanation, str) or not raw_explanation.strip():
        raise ValueError(f"Missing explanation for node {candidate_id}")
    return raw_explanation.strip()[:MAX_EXPLANATION_CHARS]


def extract_json_list(raw_response: str) -> str:
    text = raw_response.strip()
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and start < end:
        return text[start : end + 1]
    return text


def clamp_score(score: float) -> float:
    return max(1.0, min(10.0, score))


if __name__ == "__main__":
    raise SystemExit(main())
