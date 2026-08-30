from __future__ import annotations

import ctypes
import json
import os
from ctypes import wintypes
from pathlib import Path


CRYPTPROTECT_LOCAL_MACHINE = 0x4

DEFAULT_STORAGE_PATH = (
    Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData"))
    / "WCCR"
    / "license.dat"
)


class LicenseStorageError(Exception):
    """Base storage error."""


class DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_byte)),
    ]


crypt32 = ctypes.windll.crypt32
kernel32 = ctypes.windll.kernel32


def _bytes_to_blob(data: bytes) -> tuple[DATA_BLOB, ctypes.Array]:
    buffer = ctypes.create_string_buffer(data)

    blob = DATA_BLOB()
    blob.cbData = len(data)
    blob.pbData = ctypes.cast(
        buffer,
        ctypes.POINTER(ctypes.c_byte),
    )

    return blob, buffer


def _blob_to_bytes(blob: DATA_BLOB) -> bytes:
    if not blob.pbData or blob.cbData == 0:
        return b""

    return ctypes.string_at(
        blob.pbData,
        blob.cbData,
    )


def protect_bytes(data: bytes) -> bytes:
    if os.name != "nt":
        raise LicenseStorageError(
            "Windows DPAPI storage is only available on Windows."
        )

    input_blob, input_buffer = _bytes_to_blob(data)
    output_blob = DATA_BLOB()

    result = crypt32.CryptProtectData(
        ctypes.byref(input_blob),
        "WCCR License",
        None,
        None,
        None,
        CRYPTPROTECT_LOCAL_MACHINE,
        ctypes.byref(output_blob),
    )

    _ = input_buffer

    if not result:
        error_code = ctypes.get_last_error()
        raise LicenseStorageError(
            f"CryptProtectData failed: {error_code}"
        )

    try:
        return _blob_to_bytes(output_blob)
    finally:
        if output_blob.pbData:
            kernel32.LocalFree(output_blob.pbData)


def unprotect_bytes(data: bytes) -> bytes:
    if os.name != "nt":
        raise LicenseStorageError(
            "Windows DPAPI storage is only available on Windows."
        )

    input_blob, input_buffer = _bytes_to_blob(data)
    output_blob = DATA_BLOB()

    result = crypt32.CryptUnprotectData(
        ctypes.byref(input_blob),
        None,
        None,
        None,
        None,
        CRYPTPROTECT_LOCAL_MACHINE,
        ctypes.byref(output_blob),
    )

    _ = input_buffer

    if not result:
        error_code = ctypes.get_last_error()
        raise LicenseStorageError(
            f"CryptUnprotectData failed: {error_code}"
        )

    try:
        return _blob_to_bytes(output_blob)
    finally:
        if output_blob.pbData:
            kernel32.LocalFree(output_blob.pbData)


def save_activation(
    *,
    activation_token: str,
    license_number: str,
    activation_id: str,
    machine_id: str,
    path: Path = DEFAULT_STORAGE_PATH,
) -> None:
    if not activation_token.strip():
        raise LicenseStorageError(
            "Activation token is required."
        )

    payload = {
        "version": 1,
        "activation_token": activation_token.strip(),
        "license_number": license_number.strip(),
        "activation_id": activation_id.strip(),
        "machine_id": machine_id.strip(),
    }

    plaintext = json.dumps(
        payload,
        separators=(",", ":"),
    ).encode("utf-8")

    protected = protect_bytes(plaintext)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(".tmp")

    temp_path.write_bytes(protected)
    temp_path.replace(path)


def load_activation(
    *,
    path: Path = DEFAULT_STORAGE_PATH,
) -> dict[str, str]:
    if not path.exists():
        raise LicenseStorageError(
            "License activation data does not exist."
        )

    protected = path.read_bytes()

    if not protected:
        raise LicenseStorageError(
            "License activation data is empty."
        )

    plaintext = unprotect_bytes(protected)

    try:
        payload = json.loads(
            plaintext.decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LicenseStorageError(
            "License activation data is invalid."
        ) from exc

    required = (
        "activation_token",
        "license_number",
        "activation_id",
        "machine_id",
    )

    for key in required:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise LicenseStorageError(
                f"License activation data is missing {key}."
            )

    return {
        key: payload[key].strip()
        for key in required
    }


def delete_activation(
    *,
    path: Path = DEFAULT_STORAGE_PATH,
) -> None:
    if path.exists():
        path.unlink()
