"""WCCR employee time tracking policy settings.

Settings are stored separately from the operational database.
Only a later Superadmin-protected route may change them.
"""

import json
import os
import tempfile
from pathlib import Path
from threading import RLock

from flask import current_app


DEFAULT_POLICY = {
    "paid_break_minutes": 15,
    "idle_minutes": 15,
    "auto_close_time": "18:00",
    "missing_lunch_after_hours": 6,
    "expected_lunch_minutes": 30,
    "smart_shift_recovery": True,
}

_POLICY_LOCK = RLock()


class TimePolicyError(ValueError):
    pass


def _path():
    return Path(current_app.instance_path) / "employee_time_policy.json"


def validate_policy(values):
    if not isinstance(values, dict):
        raise TimePolicyError("Policy must be an object")

    if set(values) != set(DEFAULT_POLICY):
        raise TimePolicyError("Invalid policy keys")

    result = {}

    limits = {
        "paid_break_minutes": (0, 120),
        "idle_minutes": (1, 240),
        "missing_lunch_after_hours": (1, 24),
        "expected_lunch_minutes": (1, 120),
    }

    for key, (minimum, maximum) in limits.items():
        value = values[key]

        if type(value) is not int:
            raise TimePolicyError(
                key + " must be an integer"
            )

        if not minimum <= value <= maximum:
            raise TimePolicyError(
                key + " is outside allowed range"
            )

        result[key] = value

    clock = values["auto_close_time"]

    if not isinstance(clock, str):
        raise TimePolicyError("Invalid auto-close time")

    import re
    if not re.fullmatch(
        r"(?:[01][0-9]|2[0-3]):[0-5][0-9]",
        clock
    ):
        raise TimePolicyError(
            "Auto-close time must be HH:MM"
        )

    result["auto_close_time"] = clock

    recovery = values["smart_shift_recovery"]

    if type(recovery) is not bool:
        raise TimePolicyError(
            "Smart Shift Recovery must be boolean"
        )

    result["smart_shift_recovery"] = recovery

    return result


def load_policy():
    path = _path()

    with _POLICY_LOCK:
        if not path.exists():
            return dict(DEFAULT_POLICY)

        try:
            data = json.loads(
                path.read_text(encoding="utf-8")
            )
            return validate_policy(data)

        except (OSError, ValueError, TypeError) as exc:
            current_app.logger.error(
                "Employee time policy read failed: %s",
                exc
            )
            raise TimePolicyError(
                "Stored time tracking policy is invalid"
            ) from exc


def save_policy(values):
    policy = validate_policy(values)
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)

    with _POLICY_LOCK:
        # Never silently overwrite a malformed existing policy.
        if path.exists():
            load_policy()

        temp_name = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(path.parent),
                prefix=".employee_time_policy_",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_name = handle.name
                json.dump(
                    policy,
                    handle,
                    indent=2,
                    sort_keys=True
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temp_name, path)

        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)

    return policy
