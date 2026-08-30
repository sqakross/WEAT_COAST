from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from license_client.storage import (
    LicenseStorageError,
    protect_bytes,
    unprotect_bytes,
)


DEFAULT_CACHE_PATH = (
    Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData"))
    / "WCCR"
    / "license_cache.dat"
)


class LicenseCacheError(Exception):
    """Protected license cache error."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_successful_check(
    *,
    license_number: str,
    activation_id: str,
    machine_id: str,
    path: Path = DEFAULT_CACHE_PATH,
) -> None:
    payload = {
        "version": 1,
        "last_successful_check_at": _utc_now_iso(),
        "license_number": license_number.strip(),
        "activation_id": activation_id.strip(),
        "machine_id": machine_id.strip(),
    }

    plaintext = json.dumps(
        payload,
        separators=(",", ":"),
    ).encode("utf-8")

    try:
        protected = protect_bytes(plaintext)
    except LicenseStorageError as exc:
        raise LicenseCacheError(str(exc)) from exc

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(".tmp")
    temp_path.write_bytes(protected)
    temp_path.replace(path)


def load_successful_check(
    *,
    path: Path = DEFAULT_CACHE_PATH,
) -> dict[str, str]:
    if not path.exists():
        raise LicenseCacheError(
            "License cache does not exist."
        )

    protected = path.read_bytes()

    if not protected:
        raise LicenseCacheError(
            "License cache is empty."
        )

    try:
        plaintext = unprotect_bytes(protected)
    except LicenseStorageError as exc:
        raise LicenseCacheError(str(exc)) from exc

    try:
        payload = json.loads(
            plaintext.decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LicenseCacheError(
            "License cache is invalid."
        ) from exc

    required = (
        "last_successful_check_at",
        "license_number",
        "activation_id",
        "machine_id",
    )

    for key in required:
        value = payload.get(key)

        if not isinstance(value, str) or not value.strip():
            raise LicenseCacheError(
                f"License cache is missing {key}."
            )

    return {
        key: payload[key].strip()
        for key in required
    }
