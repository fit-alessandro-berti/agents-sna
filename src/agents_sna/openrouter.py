from __future__ import annotations

from dataclasses import dataclass

import requests


DEFAULT_MODEL = "openai/gpt-5.4-mini"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenRouterClient:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 120.0
    app_url: str | None = None
    app_name: str = "agents-sna"

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
    ) -> str:
        payload: dict[str, object] = {
            "model": model or self.model,
            "messages": messages,
        }

        try:
            response = requests.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc

        if response.status_code >= 400:
            raise OpenRouterError(
                "OpenRouter request failed with "
                f"HTTP {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise OpenRouterError(f"Unexpected OpenRouter response: {response.text}") from exc

        if not isinstance(content, str):
            raise OpenRouterError(f"Unexpected OpenRouter message content: {content!r}")
        return content

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": self.app_name,
        }
        if self.app_url:
            headers["HTTP-Referer"] = self.app_url
        return headers
