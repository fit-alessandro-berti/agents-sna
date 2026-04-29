from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents_sna.config import load_config


class ConfigTests(unittest.TestCase):
    def test_load_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agents.json"
            path.write_text(
                json.dumps(
                    {
                        "max_iterations": 3,
                        "agents": [
                            {
                                "name": "planner",
                                "description": "Plans the work.",
                                "model": "provider/model",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(path)

        self.assertEqual(config.max_iterations, 3)
        self.assertEqual(config.agents[0].name, "planner")
        self.assertEqual(config.agents[0].model, "provider/model")

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


if __name__ == "__main__":
    unittest.main()
