import multiprocessing as mp
import sqlite3
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from flask import Flask
from sqlalchemy import select, func, text

from extensions import db
from services.employee_time_tracking_models import (
    EmployeeShift,
    EmployeeTimeEvent,
    EmployeeTimeAudit,
)
from services.employee_time_tracking_service import (
    EmployeeTimeTrackingService as Service,
)

INDEX_NAME = "uq_employee_time_one_open_shift"


def make_app(database):
    app = Flask("employee_time_concurrency_test")
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = (
        "sqlite:///" + Path(database).as_posix()
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "connect_args": {"timeout": 10}
    }
    db.init_app(app)
    return app


def worker(database, ready, start, result):
    app = make_app(database)

    with app.app_context():
        user = SimpleNamespace(id=910001, role="admin")

        ready.put("READY")

        if not start.wait(timeout=15):
            result.put(("ERROR", "Start timeout"))
            return

        try:
            shift = Service.login(user)
            result.put(("OK", shift.id))
        except Exception as exc:
            result.put(("ERROR", type(exc).__name__))


def count_open(database):
    connection = sqlite3.connect(database)
    try:
        return connection.execute("""
            SELECT COUNT(*)
            FROM employee_time_shifts
            WHERE user_id = 910001
              AND ended_at IS NULL
        """).fetchone()[0]
    finally:
        connection.close()


def concurrent_round(database):
    context = mp.get_context("spawn")
    ready = context.Queue()
    result = context.Queue()
    start = context.Event()

    processes = [
        context.Process(
            target=worker,
            args=(database, ready, start, result),
        )
        for _ in range(2)
    ]

    for process in processes:
        process.start()

    try:
        for _ in processes:
            status = ready.get(timeout=20)
            assert status == "READY"

        start.set()

        outcomes = [
            result.get(timeout=25)
            for _ in processes
        ]

    finally:
        start.set()

        for process in processes:
            process.join(timeout=10)

            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    return outcomes


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("PASS:", message)


def main():
    print("=" * 65)
    print("WCCR PATCH #4D.3 - CONCURRENT LOGIN TEST")
    print("=" * 65)

    with tempfile.TemporaryDirectory(
        prefix="wccr_time_test_"
    ) as temporary:
        database = str(Path(temporary) / "test.sqlite3")
        app = make_app(database)

        with app.app_context():
            db.Model.metadata.create_all(
                bind=db.engine,
                tables=[
                    EmployeeShift.__table__,
                    EmployeeTimeEvent.__table__,
                    EmployeeTimeAudit.__table__,
                ],
            )

            with db.engine.begin() as connection:
                connection.exec_driver_sql(f"""
                    CREATE UNIQUE INDEX {INDEX_NAME}
                    ON employee_time_shifts (user_id)
                    WHERE ended_at IS NULL
                """)

        print("OK: Temporary SQLite initialized")
        print("OK: Unique index installed in test database")

        # First concurrent round: both processes attempt Login.
        first = concurrent_round(database)
        print("ROUND 1:", first)

        check(
            all(item[0] == "OK" for item in first),
            "Both concurrent Login requests succeeded"
        )

        check(
            first[0][1] == first[1][1],
            "Both Login requests use same shift"
        )

        check(
            count_open(database) == 1,
            "Exactly one open shift exists"
        )

        # Second round: existing shift must be reused.
        second = concurrent_round(database)
        print("ROUND 2:", second)

        check(
            all(item[0] == "OK" for item in second),
            "Repeated concurrent Login requests succeeded"
        )

        check(
            second[0][1] == first[0][1]
            and second[1][1] == first[0][1],
            "Existing shift reused"
        )

        check(
            count_open(database) == 1,
            "No duplicate shifts after repeated Login"
        )

        # Pending Review must survive subsequent Login.
        with app.app_context():
            user = SimpleNamespace(id=910001, role="admin")

            Service.logout(user)

            shift = Service.get_open_shift(user.id)

            check(
                shift.requires_review is True,
                "Logout requires review"
            )

        third = concurrent_round(database)
        print("ROUND 3:", third)

        check(
            all(item[0] == "OK" for item in third),
            "Concurrent Login after Logout succeeded"
        )

        with app.app_context():
            shift = Service.get_open_shift(910001)

            check(
                shift.requires_review is True
                and shift.status == "pending_review",
                "Pending Review preserved"
            )

        check(
            count_open(database) == 1,
            "Final open-shift count is one"
        )

    print("")
    print("PATCH #4D.3 TESTS PASSED")
    print("PRODUCTION DATABASE MODIFIED: NO")


if __name__ == "__main__":
    mp.freeze_support()
    main()
