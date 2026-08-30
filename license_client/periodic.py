from __future__ import annotations

import threading
import time

from license_client.manager import LicenseManager
from license_client.runtime import (
    RuntimeMode,
    RuntimeSnapshot,
    get_license_runtime,
)

DEFAULT_REFRESH_SECONDS = 60

_lock = threading.Lock()
_last_attempt_monotonic = 0.0


def refresh_if_due(
    *,
    app_version: str,
    refresh_seconds: int = DEFAULT_REFRESH_SECONDS,
) -> RuntimeSnapshot:
    """
    Periodic online authorization refresh.

    Fast path is local-only. A License Server request is made
    only when the refresh interval has elapsed.
    """
    global _last_attempt_monotonic

    runtime = get_license_runtime()
    snapshot = runtime.snapshot()

    now = time.monotonic()

    # Cold state is handled by E1/E2.
    if snapshot.mode == RuntimeMode.COLD:
        return snapshot

    if (
        _last_attempt_monotonic > 0
        and now - _last_attempt_monotonic < refresh_seconds
    ):
        return snapshot

    # Prevent concurrent requests from causing multiple
    # simultaneous License Server checks.
    if not _lock.acquire(blocking=False):
        return runtime.snapshot()

    try:
        now = time.monotonic()

        if (
            _last_attempt_monotonic > 0
            and now - _last_attempt_monotonic < refresh_seconds
        ):
            return runtime.snapshot()

        _last_attempt_monotonic = now

        return runtime.refresh(
            manager=LicenseManager(
                app_version=app_version,
            )
        )

    finally:
        _lock.release()
