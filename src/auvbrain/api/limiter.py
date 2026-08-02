"""In-process token-bucket rate limiter for API endpoints.

One bucket per client IP.  When a bucket is exhausted the request is
rejected with HTTP 429.  The limiter is intentionally simple (no Redis)
because AUVBrain targets single-node embedded deployments.

For multi-node / load-balanced deployments, replace the ``_buckets`` dict
with a Redis-backed store.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict

from fastapi import HTTPException, Request, status


class TokenBucketLimiter:
    """Per-IP token-bucket rate limiter.

    Args:
        rate:       Tokens refilled per second.
        burst:      Maximum token accumulation (bucket capacity).
    """

    def __init__(self, rate: float, burst: int) -> None:
        self._rate = max(0.0, float(rate))
        self._burst = max(1, int(burst))
        # Map[ip, (tokens, last_refill_time)]
        self._buckets: dict[str, list[float]] = defaultdict(
            lambda: [float(self._burst), time.monotonic()]
        )
        self._lock = threading.Lock()

    def check(self, key: str) -> bool:
        """Return True if the request is allowed, False if rate-limited."""
        if self._rate <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets[key]
            elapsed = now - bucket[1]
            bucket[0] = min(self._burst, bucket[0] + elapsed * self._rate)
            bucket[1] = now
            if bucket[0] >= 1.0:
                bucket[0] -= 1.0
                return True
            return False


# Module-level singleton (shared across all request handlers)
_limiter: TokenBucketLimiter | None = None


def configure_limiter(rate: float, burst: int) -> None:
    global _limiter
    _limiter = TokenBucketLimiter(rate=rate, burst=burst)


def get_limiter() -> TokenBucketLimiter:
    global _limiter
    if _limiter is None:
        _limiter = TokenBucketLimiter(rate=20.0, burst=40)
    return _limiter


def enforce_rate_limit(request: Request) -> None:
    """FastAPI dependency — raises 429 when the client IP is over its quota."""
    from ..metrics.registry import METRICS

    client_ip = (request.client.host if request.client else "unknown")
    limiter = get_limiter()
    if not limiter.check(client_ip):
        METRICS.inc("rate_limit_hits_total")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Slow down.",
            headers={"Retry-After": "1"},
        )
