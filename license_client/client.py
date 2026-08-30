from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from license_client.machine import MachineIdentity, get_machine_identity


DEFAULT_TIMEOUT_SECONDS = 10


class LicenseClientError(Exception):
    """Base error raised by the WCCR license client."""


class LicenseServerUnavailable(LicenseClientError):
    """The License Server could not be reached."""


class LicenseServerRejected(LicenseClientError):
    """The License Server rejected the request."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


@dataclass(frozen=True, slots=True)
class ActivationResult:
    activation_token: str
    activation_id: str
    license_id: str
    license_number: str
    machine_id: str
    status: str
    activated_at: str
    last_check_at: str | None


@dataclass(frozen=True, slots=True)
class CheckResult:
    valid: bool
    activation_id: str
    license_id: str
    license_number: str
    machine_id: str
    status: str
    plan: str
    expires_at: str | None
    last_check_at: str


@dataclass(frozen=True, slots=True)
class DeactivateResult:
    deactivated: bool
    activation_id: str
    license_id: str
    machine_id: str
    status: str
    deactivated_at: str | None


class LicenseClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        app_version: str | None = None,
    ):
        configured_url = (
            base_url
            or os.environ.get("WCCR_LICENSE_SERVER_URL")
            or "https://134.209.9.239"
        )

        self.base_url = configured_url.rstrip("/")
        self.timeout = timeout
        self.app_version = app_version

    def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"

        body = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                raw = response.read().decode("utf-8")

                if not raw:
                    return {}

                return json.loads(raw)

        except urllib.error.HTTPError as exc:
            # Gateway/backend failures mean that the License Server is
            # temporarily unavailable. They are NOT an explicit license
            # rejection, so offline grace may be used.
            if exc.code in (502, 503, 504):
                raise LicenseServerUnavailable(
                    "License Server is temporarily unavailable."
                ) from exc

            raw = exc.read().decode("utf-8", errors="replace")

            try:
                error_data = json.loads(raw)
            except (TypeError, ValueError):
                error_data = {}

            message = (
                error_data.get("message")
                or error_data.get("detail")
                or f"License Server rejected request with HTTP {exc.code}."
            )

            error_code = error_data.get("error")

            raise LicenseServerRejected(
                str(message),
                status_code=exc.code,
                error_code=(
                    str(error_code)
                    if error_code is not None
                    else None
                ),
            ) from exc

        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
        ) as exc:
            raise LicenseServerUnavailable(
                "License Server is unavailable."
            ) from exc

        except json.JSONDecodeError as exc:
            raise LicenseClientError(
                "License Server returned invalid JSON."
            ) from exc

    def activate(
        self,
        *,
        license_key: str,
        machine: MachineIdentity | None = None,
    ) -> ActivationResult:
        identity = machine or get_machine_identity()

        payload = {
            "license_key": license_key,
            "machine_hash": identity.machine_hash,
            "hostname": identity.hostname,
            "operating_system": identity.operating_system,
            "cpu_model": identity.cpu_model,
            "motherboard_serial": identity.motherboard_serial,
            "bios_serial": identity.bios_serial,
            "app_version": self.app_version,
        }

        data = self._post_json(
            "/api/v1/licenses/activate",
            payload,
        )

        token = data.get("activation_token")

        if not isinstance(token, str) or not token.strip():
            raise LicenseClientError(
                "License Server did not return an activation token."
            )

        return ActivationResult(
            activation_token=token,
            activation_id=str(data["activation_id"]),
            license_id=str(data["license_id"]),
            license_number=str(data["license_number"]),
            machine_id=str(data["machine_id"]),
            status=str(data["status"]),
            activated_at=str(data["activated_at"]),
            last_check_at=(
                str(data["last_check_at"])
                if data.get("last_check_at") is not None
                else None
            ),
        )

    def check(
        self,
        *,
        activation_token: str,
        machine: MachineIdentity | None = None,
    ) -> CheckResult:
        identity = machine or get_machine_identity()

        payload = {
            "activation_token": activation_token,
            "machine_hash": identity.machine_hash,
            "app_version": self.app_version,
        }

        data = self._post_json(
            "/api/v1/licenses/check",
            payload,
        )

        return CheckResult(
            valid=bool(data["valid"]),
            activation_id=str(data["activation_id"]),
            license_id=str(data["license_id"]),
            license_number=str(data["license_number"]),
            machine_id=str(data["machine_id"]),
            status=str(data["status"]),
            plan=str(data["plan"]),
            expires_at=(
                str(data["expires_at"])
                if data.get("expires_at") is not None
                else None
            ),
            last_check_at=str(data["last_check_at"]),
        )

    def deactivate(
        self,
        *,
        activation_token: str,
        machine: MachineIdentity | None = None,
    ) -> DeactivateResult:
        identity = machine or get_machine_identity()

        payload = {
            "activation_token": activation_token,
            "machine_hash": identity.machine_hash,
        }

        data = self._post_json(
            "/api/v1/licenses/deactivate",
            payload,
        )

        return DeactivateResult(
            deactivated=bool(data["deactivated"]),
            activation_id=str(data["activation_id"]),
            license_id=str(data["license_id"]),
            machine_id=str(data["machine_id"]),
            status=str(data["status"]),
            deactivated_at=(
                str(data["deactivated_at"])
                if data.get("deactivated_at") is not None
                else None
            ),
        )
