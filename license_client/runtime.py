from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from license_client.manager import (
    LicenseManager,
    LicenseState,
    LicenseStatus,
)


class RuntimeMode(str, Enum):
    COLD = "cold"
    ONLINE = "online"
    OFFLINE = "offline"
    BLOCKED = "blocked"
    ERROR = "error"


class RuntimeAccessError(RuntimeError):
    """Raised when the current runtime state does not permit access."""


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    mode: RuntimeMode
    valid: bool
    generation: int
    checked_at: str | None
    license_number: str | None
    plan: str | None
    offline_until: str | None
    reason: str | None
    error_code: str | None


class LicenseRuntime:
    """
    Process-local licensing state.

    There is intentionally no writable global LICENSE_VALID flag.
    State changes only through refresh().
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

        self._snapshot = RuntimeSnapshot(
            mode=RuntimeMode.COLD,
            valid=False,
            generation=0,
            checked_at=None,
            license_number=None,
            plan=None,
            offline_until=None,
            reason="License state has not been initialized.",
            error_code=None,
        )

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _mode_from_status(
        status: LicenseStatus,
    ) -> RuntimeMode:
        if (
            status.valid
            and status.state == LicenseState.VALID
        ):
            return RuntimeMode.ONLINE

        if (
            status.valid
            and status.state == LicenseState.OFFLINE_GRACE
        ):
            return RuntimeMode.OFFLINE

        if status.state in {
            LicenseState.REJECTED,
            LicenseState.NOT_ACTIVATED,
        }:
            return RuntimeMode.BLOCKED

        if status.state == LicenseState.SERVER_UNAVAILABLE:
            return RuntimeMode.BLOCKED

        return RuntimeMode.ERROR

    def refresh(
        self,
        *,
        manager: LicenseManager,
    ) -> RuntimeSnapshot:
        status = manager.check_license()

        with self._lock:
            previous = self._snapshot

            snapshot = RuntimeSnapshot(
                mode=self._mode_from_status(status),
                valid=status.valid,
                generation=previous.generation + 1,
                checked_at=self._utc_now(),
                license_number=status.license_number,
                plan=status.plan,
                offline_until=status.offline_until,
                reason=status.message,
                error_code=status.error_code,
            )

            self._snapshot = snapshot
            return snapshot

    def snapshot(self) -> RuntimeSnapshot:
        with self._lock:
            return self._snapshot

    def require_access(self) -> RuntimeSnapshot:
        with self._lock:
            snapshot = self._snapshot

        if not snapshot.valid:
            raise RuntimeAccessError(
                "Application authorization is unavailable."
            )

        return snapshot

    def permits_access(self) -> bool:
        with self._lock:
            return self._snapshot.valid


_runtime = LicenseRuntime()


def get_license_runtime() -> LicenseRuntime:
    return _runtime
