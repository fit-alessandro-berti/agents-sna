from __future__ import annotations

import argparse
import html
import json
from math import sqrt
from pathlib import Path
import re
import subprocess
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_OUTPUT_NAME = "social_network_analysis.json"
DEFAULT_LATEX_OUTPUT_NAME = "agent_usage_by_category.tex"
DEFAULT_GRAPHVIZ_OUTPUT_NAME = "social_network_analysis.gv"
DEFAULT_SVG_OUTPUT_NAME = "social_network_analysis.svg"
EXCLUDED_USAGE_AGENT_TYPES = frozenset({"START", "COMPLETE"})
LATEX_TABLE_CAPTION = "Agent usage by benchmark question category."
LATEX_TABLE_LABEL = "tab:agent-usage-by-category"


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
        latex_output_path = (
            args.latex_output or args.evaluations_dir / DEFAULT_LATEX_OUTPUT_NAME
        )
        graphviz_output_path = (
            args.graphviz_output or args.evaluations_dir / DEFAULT_GRAPHVIZ_OUTPUT_NAME
        )
        svg_output_path = (
            args.svg_output or args.evaluations_dir / DEFAULT_SVG_OUTPUT_NAME
        )
        write_json_file(output_path, analysis)
        write_agent_usage_latex_table(
            args.evaluations_dir,
            latex_output_path,
            continue_on_error=args.continue_on_error,
        )
        write_graphviz_network_files(
            analysis,
            graphviz_output_path,
            svg_output_path,
            dot_command=args.dot_command,
        )
        print(f"wrote {output_path}")
        print(f"wrote {latex_output_path}")
        print(f"wrote {graphviz_output_path}")
        print(f"wrote {svg_output_path}")
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
        "--latex-output",
        type=Path,
        help=(
            "Output LaTeX table path. Defaults to agent_usage_by_category.tex "
            "inside the evaluations folder."
        ),
    )
    parser.add_argument(
        "--graphviz-output",
        type=Path,
        help=(
            "Output Graphviz DOT path. Defaults to social_network_analysis.gv "
            "inside the evaluations folder."
        ),
    )
    parser.add_argument(
        "--svg-output",
        type=Path,
        help=(
            "Output SVG graph path. Defaults to social_network_analysis.svg "
            "inside the evaluations folder."
        ),
    )
    parser.add_argument(
        "--dot-command",
        default="dot",
        help="Graphviz DOT executable used to compile the SVG. Defaults to dot.",
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
    edge_counts_per_question: list[int] = []
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
        edge_counts_per_question.append(max(len(records) - 1, 0))
        for record in records:
            node_scores.setdefault(record["agent_type"], []).append(record["evaluation"])

        for source, target in zip(records, records[1:]):
            edge_key = (source["agent_type"], target["agent_type"])
            edge_scores.setdefault(edge_key, []).append(
                edge_evaluation(source, target, edge_score)
            )

    return {
        "source_dir": str(evaluations_dir),
        "files_processed": len(processed_files),
        "files_skipped": skipped_files,
        "edge_score": edge_score,
        "edge_score_description": edge_score_description(edge_score),
        "edge_instantiations_per_question": count_summary(edge_counts_per_question),
        "nodes": [
            {"agent_type": agent_type, **score_summary(scores)}
            for agent_type, scores in sorted(node_scores.items())
        ],
        "edges": [
            {"source": source, "target": target, **score_summary(scores)}
            for (source, target), scores in sorted(edge_scores.items())
        ],
    }


def build_agent_usage_by_category_table(
    evaluations_dir: Path,
    *,
    continue_on_error: bool = False,
) -> pd.DataFrame:
    if not evaluations_dir.is_dir():
        raise ValueError(f"Evaluation folder not found: {evaluations_dir}")

    files = sorted(evaluations_dir.glob("*.agent_evaluations.json"))
    if not files:
        raise ValueError(f"No *.agent_evaluations.json files found in {evaluations_dir}")

    usage: dict[str, dict[str, int]] = {}
    categories: set[str] = set()
    for path in files:
        try:
            records = read_evaluation_file(path)
        except Exception:
            if not continue_on_error:
                raise
            continue

        category = question_category_from_path(path)
        categories.add(category)
        for record in records:
            agent_type = str(record["agent_type"])
            if agent_type in EXCLUDED_USAGE_AGENT_TYPES:
                continue
            usage.setdefault(agent_type, {})
            usage[agent_type][category] = usage[agent_type].get(category, 0) + 1

    table = pd.DataFrame.from_dict(usage, orient="index")
    if categories:
        table = table.reindex(columns=sorted(categories), fill_value=0)
    if not table.empty:
        table = table.fillna(0).astype(int)
    table.index.name = "Agent"
    return table.sort_index()


def write_agent_usage_latex_table(
    evaluations_dir: Path,
    output_path: Path,
    *,
    continue_on_error: bool = False,
) -> None:
    table = build_agent_usage_by_category_table(
        evaluations_dir,
        continue_on_error=continue_on_error,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(agent_usage_table_to_latex(table), encoding="utf-8")


def agent_usage_table_to_latex(table: pd.DataFrame) -> str:
    try:
        return table.to_latex(
            caption=LATEX_TABLE_CAPTION,
            label=LATEX_TABLE_LABEL,
            escape=True,
        )
    except ImportError:
        return render_latex_table(
            table,
            caption=LATEX_TABLE_CAPTION,
            label=LATEX_TABLE_LABEL,
        )


def render_latex_table(table: pd.DataFrame, *, caption: str, label: str) -> str:
    column_format = "l" + ("r" * len(table.columns))
    header = [table.index.name or "Agent", *(str(column) for column in table.columns)]
    lines = [
        "\\begin{table}",
        f"\\caption{{{escape_latex(caption)}}}",
        f"\\label{{{escape_latex(label)}}}",
        f"\\begin{{tabular}}{{{column_format}}}",
        "\\hline",
        " & ".join(escape_latex(cell) for cell in header) + r" \\",
        "\\hline",
    ]
    for index_value, row in table.iterrows():
        cells = [str(index_value), *(str(value) for value in row)]
        lines.append(" & ".join(escape_latex(cell) for cell in cells) + r" \\")
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def write_graphviz_network_files(
    analysis: dict[str, object],
    graphviz_output_path: Path,
    svg_output_path: Path,
    *,
    dot_command: str = "dot",
) -> None:
    graphviz_output_path.parent.mkdir(parents=True, exist_ok=True)
    graphviz_output_path.write_text(build_graphviz_dot(analysis), encoding="utf-8")
    compile_graphviz_to_svg(
        graphviz_output_path,
        svg_output_path,
        dot_command=dot_command,
    )


def compile_graphviz_to_svg(
    graphviz_path: Path,
    svg_output_path: Path,
    *,
    dot_command: str = "dot",
) -> None:
    svg_output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                dot_command,
                "-Tsvg",
                str(graphviz_path),
                "-o",
                str(svg_output_path),
            ],
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Graphviz DOT executable not found: {dot_command}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Graphviz DOT failed with exit code {exc.returncode}: {dot_command}"
        ) from exc


def build_graphviz_dot(analysis: dict[str, object]) -> str:
    nodes = sorted_graph_items(extract_graph_nodes(analysis), name_keys=("agent_type",))
    edges = sorted_graph_items(
        extract_graph_edges(analysis),
        name_keys=("source", "target"),
    )
    max_edge_count = max((int(edge["count"]) for edge in edges), default=1)

    lines = [
        "digraph social_network_analysis {",
        "  graph [",
        '    rankdir="TB",',
        '    bgcolor="transparent",',
        '    margin="0.03",',
        '    pad="0.02",',
        '    nodesep="0.22",',
        '    ranksep="0.32",',
        '    splines="polyline",',
        '    concentrate="true",',
        '    outputorder="edgesfirst",',
        '    ordering="out"',
        "  ];",
        "  node [",
        '    shape="box",',
        '    style="rounded,filled",',
        '    fontname="Helvetica-Bold",',
        '    fontsize="9",',
        '    margin="0.045,0.035",',
        '    penwidth="0.8",',
        '    color="#666666"',
        "  ];",
        "  edge [",
        '    fontname="Helvetica-Bold",',
        '    fontsize="6.8",',
        '    arrowsize="0.55",',
        '    fontcolor="#333333"',
        "  ];",
        "",
    ]

    for node in nodes:
        agent_type = str(node["agent_type"])
        average = float(node["average"])
        lines.append(
            "  "
            f"{dot_quote(agent_type)} "
            "["
            f"label=<{node_html_label(agent_type, node)}>, "
            f'fillcolor="{score_to_color(average, pastel=True)}"'
            "];"
        )

    if nodes and edges:
        lines.append("")

    for edge in edges:
        source = str(edge["source"])
        target = str(edge["target"])
        average = float(edge["average"])
        color = score_to_color(average)
        lines.append(
            "  "
            f"{dot_quote(source)} -> {dot_quote(target)} "
            "["
            f"label=<{edge_html_label(edge)}>, "
            f'color="{color}", '
            f'fontcolor="{color}", '
            f'penwidth="{edge_penwidth(int(edge["count"]), max_edge_count):.2f}"'
            "];"
        )

    lines.append("}")
    return "\n".join(lines) + "\n"


def extract_graph_nodes(analysis: dict[str, object]) -> list[dict[str, object]]:
    raw_nodes = analysis.get("nodes")
    if not isinstance(raw_nodes, list):
        raise ValueError("Analysis field 'nodes' must be a list")
    return [dict(item) for item in raw_nodes if isinstance(item, dict)]


def extract_graph_edges(analysis: dict[str, object]) -> list[dict[str, object]]:
    raw_edges = analysis.get("edges")
    if not isinstance(raw_edges, list):
        raise ValueError("Analysis field 'edges' must be a list")
    return [dict(item) for item in raw_edges if isinstance(item, dict)]


def sorted_graph_items(
    items: list[dict[str, object]],
    *,
    name_keys: tuple[str, ...],
) -> list[dict[str, object]]:
    return sorted(
        items,
        key=lambda item: (
            -int(item.get("count", 0)),
            -float(item.get("average", 0.0)),
            float(item.get("stddev", 0.0)),
            *(str(item.get(key, "")) for key in name_keys),
        ),
    )


def node_html_label(agent_type: str, summary: dict[str, object]) -> str:
    name = "<BR/>".join(html.escape(part) for part in agent_type.split("_"))
    return (
        '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="1">'
        f"<TR><TD><B>{name}</B></TD></TR>"
        '<TR><TD><FONT POINT-SIZE="7"><B>'
        f'f={int(summary["count"])}'
        "</B></FONT></TD></TR>"
        '<TR><TD><FONT POINT-SIZE="7"><B>'
        f'mu={format_decimal(float(summary["average"]))} '
        f'std={format_decimal(float(summary["stddev"]))}'
        "</B></FONT></TD></TR>"
        "</TABLE>"
    )


def edge_html_label(summary: dict[str, object]) -> str:
    return (
        '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
        '<TR><TD><FONT POINT-SIZE="6.8"><B>'
        f'f={int(summary["count"])}'
        "</B></FONT></TD></TR>"
        '<TR><TD><FONT POINT-SIZE="6.8"><B>'
        f'mu={format_decimal(float(summary["average"]))} '
        f'std={format_decimal(float(summary["stddev"]))}'
        "</B></FONT></TD></TR>"
        "</TABLE>"
    )


def format_decimal(value: float) -> str:
    return f"{value:.1f}".replace(".", ",")


def score_to_color(score: float, *, pastel: bool = False) -> str:
    ratio = min(max(score / 10.0, 0.0), 1.0)
    if ratio < 0.5:
        local = ratio * 2.0
        start = (215, 48, 39)
        end = (254, 224, 139)
    else:
        local = (ratio - 0.5) * 2.0
        start = (254, 224, 139)
        end = (26, 152, 80)

    rgb = tuple(
        round(start[index] + (end[index] - start[index]) * local)
        for index in range(3)
    )
    if pastel:
        rgb = blend_with_white(rgb, 0.5)
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def blend_with_white(rgb: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return tuple(round(channel + (255 - channel) * amount) for channel in rgb)


def edge_penwidth(count: int, max_count: int) -> float:
    if max_count <= 0:
        return 1.0
    return 0.9 + 3.6 * (count / max_count)


def dot_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', r"\"") + '"'


def question_category_from_path(path: Path) -> str:
    match = re.match(r"^(cat\d+)_", path.name)
    if match:
        return match.group(1)
    return "unknown"


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


def count_summary(counts: list[int]) -> dict[str, float | int]:
    if not counts:
        return {
            "count": 0,
            "average": 0.0,
            "stddev": 0.0,
        }
    return score_summary([float(count) for count in counts])


def write_json_file(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
