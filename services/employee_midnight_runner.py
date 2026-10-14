"""
WCCR Employee Midnight Runner V2.3N.6

Production runner. Midnight Engine activation is controlled
exclusively by services.employee_midnight_engine.ACTIVATION_UTC.
No scheduler is installed by this module.
"""

from __future__ import annotations

import ast
import logging
import msvcrt
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE_FILE = ROOT / "services" / "employee_midnight_engine.py"
DATABASE_FILE = ROOT / "instance" / "inventory.db"
LOCK_FILE = ROOT / "instance" / "employee_midnight_runner.lock"


def activation_disabled():
    from services.employee_midnight_engine import activation_cutoff
    cutoff = activation_cutoff()
    if cutoff is None:
        return True
    if cutoff > datetime.now(timezone.utc):
        return True
    return False


def process_midnight():
    from flask import Flask
    from config import Config
    from extensions import db
    from sqlalchemy.engine import make_url
    from services.employee_midnight_engine import (
        close_expired_shifts
    )

    if not DATABASE_FILE.is_file():
        raise RuntimeError("Production database missing")

    app = Flask(
        "app",
        instance_path=str(ROOT / "instance"),
        instance_relative_config=True
    )

    app.config.from_object(Config)
    app.config.from_pyfile("config.py", silent=True)

    uri = app.config.get("SQLALCHEMY_DATABASE_URI")

    if not isinstance(uri, str):
        raise RuntimeError("Missing database URI")

    parsed = make_url(uri)

    if parsed.drivername not in ("sqlite", "sqlite+pysqlite"):
        raise RuntimeError("Unexpected database driver")

    configured = Path(parsed.database).resolve()

    if configured != DATABASE_FILE.resolve():
        raise RuntimeError("Database path mismatch")

    if not configured.is_file():
        raise RuntimeError("Database file missing")

    db.init_app(app)

    with app.app_context():
        try:
            return close_expired_shifts(
                now=datetime.now(timezone.utc)
            )
        finally:
            db.session.remove()
            db.engine.dispose()


def locked_execution():
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(LOCK_FILE, "a+b", buffering=0) as handle:
        if os.path.getsize(LOCK_FILE) == 0:
            handle.write(b"\0")
            handle.flush()

        handle.seek(0)

        try:
            msvcrt.locking(
                handle.fileno(),
                msvcrt.LK_NBLCK,
                1
            )
        except OSError:
            print("MIDNIGHT_RUNNER_BUSY")
            return 0

        try:
            result = process_midnight()
            print("MIDNIGHT_RUNNER_COMPLETED", result)
            return 0
        finally:
            handle.seek(0)
            msvcrt.locking(
                handle.fileno(),
                msvcrt.LK_UNLCK,
                1
            )


def main():
    try:
        if activation_disabled():
            print("MIDNIGHT_RUNNER_DISABLED")
            return 0

        return locked_execution()

    except Exception:
        logging.exception("MIDNIGHT_RUNNER_STOP")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
