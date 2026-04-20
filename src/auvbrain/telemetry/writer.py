from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any


class TelemetryWriter:
    def __init__(
        self,
        root: Path,
        *,
        flush_interval_s: float = 0.5,
        max_queue: int = 10_000,
    ):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "telemetry.jsonl"

        self._flush_interval_s = float(flush_interval_s)
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=int(max_queue))
        self._stop = threading.Event()
        self._closed = False
        self.dropped_events = 0

        self._thread = threading.Thread(
            target=self._worker,
            name="TelemetryWriter",
            daemon=True,
        )
        self._thread.start()

    def write(self, event: dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            # Enqueue raw event and let the background writer serialize.
            self._queue.put_nowait(event)
        except queue.Full:
            # Drop rather than blocking the control loop.
            self.dropped_events += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _worker(self) -> None:
        # Dedicated writer thread to avoid sync disk I/O in the event loop.
        last_flush = time.monotonic()
        with self.path.open("a", encoding="utf-8") as f:
            while True:
                if self._stop.is_set() and self._queue.empty():
                    break
                try:
                    event = self._queue.get(timeout=0.1)
                except queue.Empty:
                    # Periodic flush even if idle.
                    now = time.monotonic()
                    if now - last_flush >= self._flush_interval_s:
                        f.flush()
                        last_flush = now
                    continue

                line = json.dumps(
                    event,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=_json_default,
                )
                f.write(line + "\n")
                now = time.monotonic()
                if now - last_flush >= self._flush_interval_s:
                    f.flush()
                    last_flush = now

            # Final flush on shutdown.
            f.flush()


def _json_default(obj: object) -> object:
    # Support Pydantic models without paying serialization cost on the event loop.
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except TypeError:
            return model_dump()
    return str(obj)
