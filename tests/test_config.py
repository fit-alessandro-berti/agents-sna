from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents_sna.config import HandoffExclusion, load_config


class ConfigTests(unittest.TestCase):
    def test_load_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agents.json"
            path.write_text(
                json.dumps(
                    {
                        "max_iterations": 3,
                        "single_agent_per_iteration": True,
                        "excluded_handoffs": [
                            {"from": "planner", "to": "critic"},
                            ["critic", "planner"],
                        ],
                        "agents": [
                            {
                                "name": "planner",
                                "description": "Plans the work.",
                                "model": "provider/model",
                            },
                            {
                                "name": "critic",
                                "description": "Critiques the work.",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(path)

        self.assertEqual(config.max_iterations, 3)
        self.assertTrue(config.single_agent_per_iteration)
        self.assertEqual(config.agents[0].name, "planner")
        self.assertEqual(config.agents[0].model, "provider/model")
        self.assertEqual(config.excluded_handoffs[0], HandoffExclusion("planner", "critic"))
        self.assertEqual(config.allowed_agent_names_after("planner"), {"planner"})

    def test_load_config_allows_empty_agents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agents.json"
            path.write_text(
                json.dumps(
                    {
                        "max_iterations": 3,
                        "agents": [],
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(path)

        self.assertEqual(config.max_iterations, 3)
        self.assertEqual(config.agents, ())
        self.assertEqual(config.allowed_agent_names_after(None), set())

    def test_rejects_duplicate_agents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agents.json"
            path.write_text(
                json.dumps(
                    {
                        "max_iterations": 3,
                        "agents": [
                            {"name": "planner", "description": "A"},
                            {"name": "planner", "description": "B"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Duplicate agent"):
                load_config(path)

    def test_rejects_unknown_excluded_handoff_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agents.json"
            path.write_text(
                json.dumps(
                    {
                        "max_iterations": 3,
                        "agents": [
                            {"name": "planner", "description": "A"},
                        ],
                        "excluded_handoffs": [
                            {"from": "planner", "to": "critic"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unknown target agent"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
