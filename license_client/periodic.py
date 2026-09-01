from __future__ import annotations

import logging
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


def _background_refresh(
    *,
    app_version: str,
) -> None:
    try:
        runtime = get_license_runtime()

        started = time.perf_counter()

        result = runtime.refresh(
            manager=LicenseManager(
                app_version=app_version,
            )
        )

        elapsed = time.perf_counter() - started

        logging.info(
            "LICENSE_REFRESH_PERF elapsed=%.3fs | "
            "mode=%s license=%s generation=%s",
            elapsed,
            result.mode.value,
            result.license_number,
            result.generation,
        )

    except Exception:
        logging.exception(
            "Background license refresh failed"
        )

    finally:
        _lock.release()


def refresh_if_due(
    *,
    app_version: str,
    refresh_seconds: int = DEFAULT_REFRESH_SECONDS,
) -> RuntimeSnapshot:
    """
    Periodic online authorization refresh.

    HTTP requests do not wait for the License Server.
    Only one background refresh may run at a time.
    """
    global _last_attempt_monotonic

    runtime = get_license_runtime()
    snapshot = runtime.snapshot()

    # Initial authorization stays synchronous via E1/E2.
    if snapshot.mode == RuntimeMode.COLD:
        return snapshot

    now = time.monotonic()

    if (
        _last_attempt_monotonic > 0
        and now - _last_attempt_monotonic < refresh_seconds
    ):
        return snapshot

    if not _lock.acquire(blocking=False):
        return snapshot

    now = time.monotonic()

    if (
        _last_attempt_monotonic > 0
        and now - _last_attempt_monotonic < refresh_seconds
    ):
        _lock.release()
        return snapshot

    _last_attempt_monotonic = now

    try:
        thread = threading.Thread(
            target=_background_refresh,
            kwargs={
                "app_version": app_version,
            },
            name="wccr-license-refresh",
            daemon=True,
        )
        thread.start()

    except Exception:
        _lock.release()
        raise

    return snapshot