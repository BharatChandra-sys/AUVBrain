from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field


@dataclass
class LlamaCppClient:
    """In-process local chat client backed by llama-cpp-python.

    This is the most "offline" option: no HTTP server required.

    Requirements:
    - Install optional deps: `pip install -e .[local-llm]`
    - Provide a GGUF model file via `AUV_LLAMA_CPP_MODEL_PATH`.
    """

    model_path: str
    n_ctx: int = 2048
    n_threads: int | None = None
    n_gpu_layers: int = 0
    temperature: float = 0.2

    _llama: object | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _get_llama(self):
        if self._llama is not None:
            return self._llama

        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "llama-cpp-python not available. Install with: pip install -e .[local-llm]"
            ) from e

        kwargs: dict[str, object] = {
            "model_path": self.model_path,
            "n_ctx": int(self.n_ctx),
            "n_gpu_layers": int(self.n_gpu_layers),
        }
        if self.n_threads is not None:
            kwargs["n_threads"] = int(self.n_threads)

        self._llama = Llama(**kwargs)
        return self._llama

    async def aclose(self) -> None:
        # llama-cpp-python doesn't expose a universal close; drop reference for GC.
        self._llama = None

    async def chat_json(self, system: str, user: str) -> str:
        return await asyncio.to_thread(self._chat_json_blocking, system, user)

    def _chat_json_blocking(self, system: str, user: str) -> str:
        llama = self._get_llama()

        # llama-cpp-python isn't guaranteed thread-safe; serialize access.
        with self._lock:
            try:
                result = llama.create_chat_completion(  # type: ignore[attr-defined]
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=float(self.temperature),
                )
            except Exception as e:
                raise RuntimeError(f"llama.cpp inference failed: {e}") from e

        try:
            content = result["choices"][0]["message"]["content"]
        except Exception:
            raise RuntimeError(f"Unexpected llama.cpp response: {json.dumps(result)[:400]}")

        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected llama.cpp content: {json.dumps(result)[:400]}")
        return content.strip()
