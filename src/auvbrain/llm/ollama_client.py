from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx


@dataclass
class OllamaClient:
    base_url: str
    model: str
    timeout_s: float = 30.0
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
        self._client = httpx.AsyncClient(
            base_url=self.base_url.rstrip("/"),
            timeout=timeout,
            limits=limits,
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
            "stream": False,
            # Encourage deterministic structured output
            "options": {"temperature": 0.2},
        }

        resp = await client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()

        content = data.get("message", {}).get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected Ollama response: {json.dumps(data)[:400]}")
        return content.strip()
