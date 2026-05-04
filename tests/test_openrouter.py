from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents_sna.openrouter import OpenRouterClient


class FakeResponse:
    status_code = 200
    text = '{"choices":[{"message":{"content":"ok"}}]}'

    def json(self) -> dict[str, object]:
        return {"choices": [{"message": {"content": "ok"}}]}


class OpenRouterClientTests(unittest.TestCase):
    def test_complete_posts_chat_completion_payload_without_temperature(self) -> None:
        client = OpenRouterClient(
            api_key="secret",
            model="default-model",
            base_url="https://example.test/api/v1",
            timeout=5,
        )
        messages = [{"role": "user", "content": "hello"}]

        with patch("agents_sna.openrouter.requests.post", return_value=FakeResponse()) as post:
            result = client.complete(messages, model="agent-model")

        self.assertEqual(result, "ok")
        post.assert_called_once()
        _, kwargs = post.call_args
        self.assertEqual(
            kwargs["json"],
            {
                "model": "agent-model",
                "messages": messages,
            },
        )
        self.assertNotIn("temperature", kwargs["json"])

    def test_complete_merges_additional_payload(self) -> None:
        client = OpenRouterClient(
            api_key="secret",
            model="default-model",
            base_url="https://example.test/api/v1",
            additional_payload={
                "temperature": 0.2,
                "reasoning": {"effort": "high"},
            },
        )
        messages = [{"role": "user", "content": "hello"}]

        with patch("agents_sna.openrouter.requests.post", return_value=FakeResponse()) as post:
            client.complete(messages)

        _, kwargs = post.call_args
        self.assertEqual(kwargs["json"]["model"], "default-model")
        self.assertEqual(kwargs["json"]["messages"], messages)
        self.assertEqual(kwargs["json"]["temperature"], 0.2)
        self.assertEqual(kwargs["json"]["reasoning"], {"effort": "high"})


if __name__ == "__main__":
    unittest.main()
