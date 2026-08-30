from __future__ import annotations

import hashlib
import os
import platform
import socket
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MachineIdentity:
    machine_hash: str
    hostname: str
    operating_system: str
    cpu_model: str | None
    motherboard_serial: str | None
    bios_serial: str | None


def _run_powershell(command: str) -> str | None:
    """
    Execute a small PowerShell query and return a normalized single value.

    Failure to read one hardware property must not prevent the application
    from starting. The final machine fingerprint uses several independent
    identifiers.
    """
    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None

    value = result.stdout.strip()

    if not value:
        return None

    # PowerShell can occasionally return multiple lines.
    value = value.splitlines()[0].strip()

    if not value:
        return None

    return value


def _normalize(value: str | None) -> str:
    if value is None:
        return ""

    return " ".join(value.strip().lower().split())


def _get_cpu_model() -> str | None:
    return _run_powershell(
        "(Get-CimInstance Win32_Processor | "
        "Select-Object -First 1 -ExpandProperty Name)"
    )


def _get_motherboard_serial() -> str | None:
    return _run_powershell(
        "(Get-CimInstance Win32_BaseBoard | "
        "Select-Object -First 1 -ExpandProperty SerialNumber)"
    )


def _get_bios_serial() -> str | None:
    return _run_powershell(
        "(Get-CimInstance Win32_BIOS | "
        "Select-Object -First 1 -ExpandProperty SerialNumber)"
    )


def _get_machine_guid() -> str | None:
    return _run_powershell(
        "(Get-ItemProperty "
        "'HKLM:\\SOFTWARE\\Microsoft\\Cryptography' "
        "-Name MachineGuid).MachineGuid"
    )


def get_machine_identity() -> MachineIdentity:
    """
    Build the identity of the WCCR application SERVER.

    This intentionally identifies the Windows host running Flask.
    Browser workstations are not part of license activation.
    """
    hostname = socket.gethostname().strip()

    operating_system = (
        f"{platform.system()} "
        f"{platform.release()} "
        f"{platform.version()}"
    ).strip()

    cpu_model = _get_cpu_model()
    motherboard_serial = _get_motherboard_serial()
    bios_serial = _get_bios_serial()
    machine_guid = _get_machine_guid()

    fingerprint_components = [
        "wccr-machine-v1",
        _normalize(hostname),
        _normalize(machine_guid),
        _normalize(motherboard_serial),
        _normalize(bios_serial),
        _normalize(cpu_model),
        _normalize(platform.machine()),
        _normalize(os.environ.get("PROCESSOR_ARCHITECTURE")),
    ]

    fingerprint_source = "|".join(fingerprint_components)

    machine_hash = hashlib.sha256(
        fingerprint_source.encode("utf-8")
    ).hexdigest()

    return MachineIdentity(
        machine_hash=machine_hash,
        hostname=hostname,
        operating_system=operating_system,
        cpu_model=cpu_model,
        motherboard_serial=motherboard_serial,
        bios_serial=bios_serial,
    )