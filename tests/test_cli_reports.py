from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents_sna.cli import ReportRecorder


class ReportRecorderTests(unittest.TestCase):
    def test_writes_request_inputs_and_compact_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request_inputs_path = Path(directory) / "requests.json"
            conversation_trace_path = Path(directory) / "trace.json"
            recorder = ReportRecorder(
                request_inputs_path=request_inputs_path,
                conversation_trace_path=conversation_trace_path,
            )

            recorder.handle(
                "run_start",
                {
                    "prompt": "Original prompt",
                    "max_iterations": 2,
                    "agent_names": ("planner",),
                },
            )
            recorder.handle(
                "request",
                {
                    "kind": "selector",
                    "iteration": 1,
                    "messages": [
                        {"role": "system", "content": "Agents"},
                        {"role": "user", "content": "Original prompt"},
                    ],
                    "model": None,
                },
            )
            recorder.handle(
                "response",
                {
                    "kind": "selector",
                    "iteration": 1,
                    "content": '["planner"]',
                },
            )
            recorder.handle(
                "selection",
                {
                    "iteration": 1,
                    "agent_names": ("planner",),
                    "final_requested": False,
                },
            )
            recorder.handle(
                "response",
                {
                    "kind": "agent",
                    "iteration": 1,
                    "agent_name": "planner",
                    "content": "Planner response",
                },
            )
            recorder.handle(
                "response",
                {
                    "kind": "final",
                    "iteration": 2,
                    "content": "Final answer",
                },
            )
            recorder.write()

            requests_report = json.loads(request_inputs_path.read_text(encoding="utf-8"))
            trace_report = json.loads(conversation_trace_path.read_text(encoding="utf-8"))

        self.assertEqual(requests_report[0][0]["role"], "system")
        self.assertEqual(requests_report[0][1]["role"], "user")
        self.assertEqual(trace_report["original_prompt"], "Original prompt")
        self.assertEqual(trace_report["events"][0]["type"], "choice")
        self.assertEqual(trace_report["events"][0]["agents"], ["planner"])
        self.assertEqual(trace_report["events"][1]["type"], "agent_response")
        self.assertEqual(trace_report["events"][-1]["type"], "final_answer")


if __name__ == "__main__":
    unittest.main()
