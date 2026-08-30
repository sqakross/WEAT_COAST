from __future__ import annotations

import logging

from werkzeug.exceptions import ServiceUnavailable

from license_client.manager import LicenseManager
from license_client.runtime import (
    RuntimeMode,
    RuntimeSnapshot,
    get_license_runtime,
)


def prime_application_state(
    *,
    app_version: str | None = None,
) -> RuntimeSnapshot:
    """
    Establish trusted process-local authorization state.

    This is one enforcement point only. Other enforcement
    points must not depend on this call having executed.
    """
    runtime = get_license_runtime()

    snapshot = runtime.refresh(
        manager=LicenseManager(
            app_version=app_version,
        )
    )

    if not snapshot.valid:
        logging.critical(
            "Application authorization unavailable at startup | "
            "mode=%s reason=%s",
            snapshot.mode.value,
            snapshot.reason,
        )

        raise SystemExit(73)

    logging.info(
        "Application authorization initialized | "
        "mode=%s license=%s generation=%s",
        snapshot.mode.value,
        snapshot.license_number,
        snapshot.generation,
    )

    return snapshot


def require_application_state(
    *,
    app_version: str | None = None,
) -> RuntimeSnapshot:
    """
    Independent request-level enforcement.

    If startup initialization was removed, skipped, or never
    executed, this guard can initialize runtime state itself.
    """
    runtime = get_license_runtime()
    snapshot = runtime.snapshot()

    if snapshot.mode == RuntimeMode.COLD:
        snapshot = runtime.refresh(
            manager=LicenseManager(
                app_version=app_version,
            )
        )

    if not snapshot.valid:
        logging.error(
            "Application request rejected | "
            "mode=%s reason=%s",
            snapshot.mode.value,
            snapshot.reason,
        )

        raise ServiceUnavailable(
            description="Application authorization is unavailable."
        )

    return snapshot
