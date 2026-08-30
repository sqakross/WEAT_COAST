from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum

from license_client.cache import (
    LicenseCacheError,
    load_successful_check,
    save_successful_check,
)
from license_client.client import (
    LicenseClient,
    LicenseClientError,
    LicenseServerRejected,
    LicenseServerUnavailable,
)
from license_client.machine import get_machine_identity
from license_client.storage import (
    LicenseStorageError,
    load_activation,
)


OFFLINE_GRACE_DAYS = 5


class LicenseState(str, Enum):
    VALID = "valid"
    OFFLINE_GRACE = "offline_grace"
    NOT_ACTIVATED = "not_activated"
    SERVER_UNAVAILABLE = "server_unavailable"
    REJECTED = "rejected"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class LicenseStatus:
    state: LicenseState
    valid: bool
    license_number: str | None = None
    plan: str | None = None
    expires_at: str | None = None
    message: str | None = None
    error_code: str | None = None
    offline_until: str | None = None


class LicenseManager:
    def __init__(
        self,
        *,
        client: LicenseClient | None = None,
        app_version: str | None = None,
    ):
        self.client = client or LicenseClient(
            app_version=app_version,
        )

    def _offline_grace_status(
        self,
        *,
        stored: dict[str, str],
    ) -> LicenseStatus:
        try:
            cached = load_successful_check()
        except LicenseCacheError:
            return LicenseStatus(
                state=LicenseState.SERVER_UNAVAILABLE,
                valid=False,
                license_number=stored.get("license_number"),
                message=(
                    "License Server is unavailable and "
                    "no valid offline cache exists."
                ),
            )

        if (
            cached["license_number"]
            != stored["license_number"]
            or cached["activation_id"]
            != stored["activation_id"]
            or cached["machine_id"]
            != stored["machine_id"]
        ):
            return LicenseStatus(
                state=LicenseState.SERVER_UNAVAILABLE,
                valid=False,
                license_number=stored.get("license_number"),
                message=(
                    "Offline license cache does not match "
                    "the current activation."
                ),
            )

        try:
            last_success = datetime.fromisoformat(
                cached["last_successful_check_at"]
            )
        except ValueError:
            return LicenseStatus(
                state=LicenseState.SERVER_UNAVAILABLE,
                valid=False,
                license_number=stored.get("license_number"),
                message="Offline license cache timestamp is invalid.",
            )

        if last_success.tzinfo is None:
            last_success = last_success.replace(
                tzinfo=timezone.utc,
            )

        now = datetime.now(timezone.utc)

        offline_until = (
            last_success
            + timedelta(days=OFFLINE_GRACE_DAYS)
        )

        if now > offline_until:
            return LicenseStatus(
                state=LicenseState.SERVER_UNAVAILABLE,
                valid=False,
                license_number=stored.get("license_number"),
                message=(
                    "Offline license grace period has expired."
                ),
                offline_until=offline_until.isoformat(),
            )

        return LicenseStatus(
            state=LicenseState.OFFLINE_GRACE,
            valid=True,
            license_number=stored.get("license_number"),
            message=(
                "License Server is unavailable. "
                "Running within offline grace period."
            ),
            offline_until=offline_until.isoformat(),
        )

    def check_license(self) -> LicenseStatus:
        try:
            stored = load_activation()
        except LicenseStorageError as exc:
            return LicenseStatus(
                state=LicenseState.NOT_ACTIVATED,
                valid=False,
                message=str(exc),
            )

        try:
            machine = get_machine_identity()

            result = self.client.check(
                activation_token=stored["activation_token"],
                machine=machine,
            )

        except LicenseServerUnavailable:
            return self._offline_grace_status(
                stored=stored,
            )

        except LicenseServerRejected as exc:
            return LicenseStatus(
                state=LicenseState.REJECTED,
                valid=False,
                license_number=stored.get("license_number"),
                message=str(exc),
                error_code=exc.error_code,
            )

        except LicenseClientError as exc:
            return LicenseStatus(
                state=LicenseState.ERROR,
                valid=False,
                license_number=stored.get("license_number"),
                message=str(exc),
            )

        except Exception as exc:
            return LicenseStatus(
                state=LicenseState.ERROR,
                valid=False,
                license_number=stored.get("license_number"),
                message=f"Unexpected license error: {exc}",
            )

        if not result.valid:
            return LicenseStatus(
                state=LicenseState.REJECTED,
                valid=False,
                license_number=result.license_number,
                plan=result.plan,
                expires_at=result.expires_at,
                message="License is not valid.",
            )

        try:
            save_successful_check(
                license_number=result.license_number,
                activation_id=result.activation_id,
                machine_id=result.machine_id,
            )
        except LicenseCacheError as exc:
            return LicenseStatus(
                state=LicenseState.ERROR,
                valid=False,
                license_number=result.license_number,
                plan=result.plan,
                expires_at=result.expires_at,
                message=(
                    "License is valid, but secure cache "
                    f"could not be updated: {exc}"
                ),
            )

        return LicenseStatus(
            state=LicenseState.VALID,
            valid=True,
            license_number=result.license_number,
            plan=result.plan,
            expires_at=result.expires_at,
            message="License is valid.",
        )
