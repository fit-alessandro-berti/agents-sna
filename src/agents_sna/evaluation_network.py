from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_OUTPUT_NAME = "social_network_analysis.json"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        analysis = analyze_evaluation_folder(
            args.evaluations_dir,
            edge_score=args.edge_score,
            continue_on_error=args.continue_on_error,
        )
        output_path = args.output or args.evaluations_dir / DEFAULT_OUTPUT_NAME
        write_json_file(output_path, analysis)
        print(f"wrote {output_path}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate agent evaluation JSON files into a social-network analysis."
    )
    parser.add_argument(
        "evaluations_dir",
        type=Path,
        help="Folder containing *.agent_evaluations.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output JSON path. Defaults to social_network_analysis.json inside "
            "the evaluations folder."
        ),
    )
    parser.add_argument(
        "--edge-score",
        choices=("target", "mean"),
        default="target",
        help=(
            "How to score each observed edge. 'target' uses the target node's "
            "evaluation; 'mean' averages source and target evaluations."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Skip malformed evaluation files instead of aborting.",
    )
    return parser


def analyze_evaluation_folder(
    evaluations_dir: Path,
    *,
    edge_score: str = "target",
    continue_on_error: bool = False,
) -> dict[str, object]:
    if edge_score not in {"target", "mean"}:
        raise ValueError("edge_score must be either 'target' or 'mean'")
    if not evaluations_dir.is_dir():
        raise ValueError(f"Evaluation folder not found: {evaluations_dir}")

    files = sorted(evaluations_dir.glob("*.agent_evaluations.json"))
    if not files:
        raise ValueError(f"No *.agent_evaluations.json files found in {evaluations_dir}")

    node_scores: dict[str, list[float]] = {}
    edge_scores: dict[tuple[str, str], list[float]] = {}
    skipped_files: list[dict[str, str]] = []
    processed_files: list[str] = []

    for path in files:
        try:
            records = read_evaluation_file(path)
        except Exception as exc:
            if not continue_on_error:
                raise
            skipped_files.append({"path": str(path), "error": str(exc)})
            continue

        processed_files.append(str(path))
        for record in records:
            node_scores.setdefault(record["agent_type"], []).append(record["evaluation"])

        for source, target in zip(records, records[1:]):
            edge_key = (source["agent_type"], target["agent_type"])
            edge_scores.setdefault(edge_key, []).append(edge_evaluation(source, target, edge_score))

    return {
        "source_dir": str(evaluations_dir),
        "files_processed": len(processed_files),
        "files_skipped": skipped_files,
        "edge_score": edge_score,
        "edge_score_description": edge_score_description(edge_score),
        "nodes": [
            {"agent_type": agent_type, **score_summary(scores)}
            for agent_type, scores in sorted(node_scores.items())
        ],
        "edges": [
            {"source": source, "target": target, **score_summary(scores)}
            for (source, target), scores in sorted(edge_scores.items())
        ],
    }


def read_evaluation_file(path: Path) -> list[dict[str, float | str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")

    records: list[dict[str, float | str]] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Item {index} in {path} must be an object")

        agent_type = item.get("agent_type")
        if not isinstance(agent_type, str) or not agent_type.strip():
            raise ValueError(f"Item {index} in {path} has invalid agent_type")

        raw_score = item.get("evaluation")
        if isinstance(raw_score, bool):
            raise ValueError(f"Item {index} in {path} has invalid evaluation")
        try:
            evaluation = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Item {index} in {path} has invalid evaluation") from exc

        records.append({"agent_type": agent_type.strip(), "evaluation": evaluation})

    if not records:
        raise ValueError(f"Evaluation file is empty: {path}")
    return records


def edge_evaluation(
    source: dict[str, float | str],
    target: dict[str, float | str],
    edge_score: str,
) -> float:
    source_score = float(source["evaluation"])
    target_score = float(target["evaluation"])
    if edge_score == "target":
        return target_score
    if edge_score == "mean":
        return (source_score + target_score) / 2.0
    raise ValueError("edge_score must be either 'target' or 'mean'")


def edge_score_description(edge_score: str) -> str:
    if edge_score == "target":
        return "Each edge occurrence is scored with the target node evaluation."
    if edge_score == "mean":
        return "Each edge occurrence is scored with the mean of source and target evaluations."
    raise ValueError("edge_score must be either 'target' or 'mean'")


def score_summary(scores: list[float]) -> dict[str, float | int]:
    if not scores:
        raise ValueError("Cannot summarize an empty score list")
    average = sum(scores) / len(scores)
    variance = sum((score - average) ** 2 for score in scores) / len(scores)
    return {
        "count": len(scores),
        "average": average,
        "stddev": sqrt(variance),
    }


def write_json_file(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
