from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx


@dataclass
class OpenAICompatClient:
    """Minimal OpenAI-compatible chat client.

    Works with local servers that implement the OpenAI Chat Completions API,
    e.g. vLLM, TGI, llama.cpp server (when configured for OpenAI routes), etc.

    Endpoint: POST {base_url}/v1/chat/completions
    """

    base_url: str
    model: str
    api_key: str = ""
    timeout_s: float = 30.0
    temperature: float = 0.2
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client

        timeout = httpx.Timeout(self.timeout_s, connect=min(5.0, self.timeout_s))
        limits = httpx.Limits(
            max_connections=10,
            max_keepalive_connections=5,
            keepalive_expiry=30.0,
        )
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self._client = httpx.AsyncClient(
            base_url=self.base_url.rstrip("/"),
            timeout=timeout,
            limits=limits,
            headers=headers,
        )
        return self._client

    async def aclose(self) -> None:
        if self._client is None:
            return
        await self._client.aclose()
        self._client = None

    async def chat_json(self, system: str, user: str) -> str:
        """Return the model's raw text (expected JSON)."""

        client = self._get_client()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
        }

        # Some servers support forcing JSON.
        payload["response_format"] = {"type": "json_object"}

        resp = await client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            raise RuntimeError(
                f"Unexpected OpenAI-compatible response: {json.dumps(data)[:400]}"
            )

        if not isinstance(content, str):
            raise RuntimeError(
                f"Unexpected OpenAI-compatible content: {json.dumps(data)[:400]}"
            )
        return content.strip()
