from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents_sna.evaluation_network import (
    analyze_evaluation_folder,
    read_evaluation_file,
    score_summary,
)


class EvaluationNetworkTests(unittest.TestCase):
    def test_score_summary_uses_population_stddev(self) -> None:
        summary = score_summary([2.0, 4.0])

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["average"], 3.0)
        self.assertEqual(summary["stddev"], 1.0)

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
        self.assertEqual(nodes["START"]["average"], 9.0)
        self.assertEqual(nodes["START"]["stddev"], 1.0)
        self.assertEqual(nodes["parser"]["average"], 7.0)
        self.assertEqual(edges[("START", "parser")]["average"], 7.0)
        self.assertEqual(edges[("START", "parser")]["stddev"], 1.0)
        self.assertEqual(edges[("parser", "COMPLETE")]["average"], 8.0)
        self.assertEqual(edges[("parser", "COMPLETE")]["stddev"], 1.0)

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
