from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents_sna.evaluation_network import (
    analyze_evaluation_folder,
    build_agent_usage_by_category_table,
    count_summary,
    question_category_from_path,
    read_evaluation_file,
    score_summary,
    write_agent_usage_latex_table,
)


class EvaluationNetworkTests(unittest.TestCase):
    def test_score_summary_uses_population_stddev(self) -> None:
        summary = score_summary([2.0, 4.0])

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["average"], 3.0)
        self.assertEqual(summary["stddev"], 1.0)

    def test_count_summary_allows_empty_counts(self) -> None:
        summary = count_summary([])

        self.assertEqual(summary["count"], 0)
        self.assertEqual(summary["average"], 0.0)
        self.assertEqual(summary["stddev"], 0.0)

    def test_analyze_evaluation_folder_aggregates_nodes_and_edges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_evaluation(
                root / "a.agent_evaluations.json",
                [
                    ("START", 8.0),
                    ("parser", 6.0),
                    ("COMPLETE", 7.0),
                ],
            )
            write_evaluation(
                root / "b.agent_evaluations.json",
                [
                    ("START", 10.0),
                    ("parser", 8.0),
                    ("COMPLETE", 9.0),
                ],
            )

            analysis = analyze_evaluation_folder(root)

        nodes = {node["agent_type"]: node for node in analysis["nodes"]}
        edges = {(edge["source"], edge["target"]): edge for edge in analysis["edges"]}

        self.assertEqual(analysis["files_processed"], 2)
        self.assertEqual(analysis["edge_instantiations_per_question"]["count"], 2)
        self.assertEqual(analysis["edge_instantiations_per_question"]["average"], 2.0)
        self.assertEqual(analysis["edge_instantiations_per_question"]["stddev"], 0.0)
        self.assertEqual(nodes["START"]["average"], 9.0)
        self.assertEqual(nodes["START"]["stddev"], 1.0)
        self.assertEqual(nodes["parser"]["average"], 7.0)
        self.assertEqual(edges[("START", "parser")]["average"], 7.0)
        self.assertEqual(edges[("START", "parser")]["stddev"], 1.0)
        self.assertEqual(edges[("parser", "COMPLETE")]["average"], 8.0)
        self.assertEqual(edges[("parser", "COMPLETE")]["stddev"], 1.0)

    def test_analyze_evaluation_folder_summarizes_edge_counts_per_question(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_evaluation(
                root / "a.agent_evaluations.json",
                [
                    ("START", 8.0),
                    ("parser", 6.0),
                    ("COMPLETE", 7.0),
                ],
            )
            write_evaluation(
                root / "b.agent_evaluations.json",
                [
                    ("START", 10.0),
                    ("COMPLETE", 9.0),
                ],
            )

            analysis = analyze_evaluation_folder(root)

        summary = analysis["edge_instantiations_per_question"]
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["average"], 1.5)
        self.assertEqual(summary["stddev"], 0.5)

    def test_analyze_evaluation_folder_supports_mean_edge_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_evaluation(
                root / "a.agent_evaluations.json",
                [
                    ("START", 8.0),
                    ("parser", 6.0),
                ],
            )

            analysis = analyze_evaluation_folder(root, edge_score="mean")

        edge = analysis["edges"][0]
        self.assertEqual(edge["source"], "START")
        self.assertEqual(edge["target"], "parser")
        self.assertEqual(edge["average"], 7.0)

    def test_question_category_from_path_uses_benchmark_prefix(self) -> None:
        self.assertEqual(
            question_category_from_path(
                Path("cat03_08_powl_discovery.agent_evaluations.json")
            ),
            "cat03",
        )

    def test_build_agent_usage_by_category_table_counts_agents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_evaluation(
                root / "cat01_01_case_id_inference.agent_evaluations.json",
                [
                    ("START", 8.0),
                    ("parser", 6.0),
                    ("parser", 7.0),
                    ("COMPLETE", 8.0),
                ],
            )
            write_evaluation(
                root / "cat02_01_conformance_textual.agent_evaluations.json",
                [
                    ("START", 8.0),
                    ("critic", 6.0),
                    ("parser", 7.0),
                    ("COMPLETE", 8.0),
                ],
            )

            table = build_agent_usage_by_category_table(root)

        self.assertEqual(list(table.columns), ["cat01", "cat02"])
        self.assertEqual(list(table.index), ["critic", "parser"])
        self.assertEqual(table.loc["parser", "cat01"], 2)
        self.assertEqual(table.loc["parser", "cat02"], 1)
        self.assertEqual(table.loc["critic", "cat01"], 0)
        self.assertEqual(table.loc["critic", "cat02"], 1)

    def test_write_agent_usage_latex_table_writes_tex_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_evaluation(
                root / "cat01_01_case_id_inference.agent_evaluations.json",
                [
                    ("START", 8.0),
                    ("parser", 6.0),
                    ("COMPLETE", 8.0),
                ],
            )
            output_path = root / "usage.tex"

            write_agent_usage_latex_table(root, output_path)

            content = output_path.read_text(encoding="utf-8")

        self.assertIn("\\begin{tabular}", content)
        self.assertIn("cat01", content)
        self.assertIn("parser", content)

    def test_read_evaluation_file_rejects_invalid_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.agent_evaluations.json"
            path.write_text(
                json.dumps([{"agent_type": "START", "evaluation": True}]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "invalid evaluation"):
                read_evaluation_file(path)

    def test_analyze_evaluation_folder_can_skip_malformed_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_evaluation(root / "good.agent_evaluations.json", [("START", 8.0)])
            (root / "bad.agent_evaluations.json").write_text("{}", encoding="utf-8")

            analysis = analyze_evaluation_folder(root, continue_on_error=True)

        self.assertEqual(analysis["files_processed"], 1)
        self.assertEqual(len(analysis["files_skipped"]), 1)


def write_evaluation(path: Path, rows: list[tuple[str, float]]) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "agent_type": agent_type,
                    "evaluation": evaluation,
                    "explanation": "test",
                }
                for agent_type, evaluation in rows
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
