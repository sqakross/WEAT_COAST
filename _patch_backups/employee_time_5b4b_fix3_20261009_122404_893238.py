from pathlib import Path
from datetime import datetime, timezone
from tempfile import TemporaryDirectory
from unittest.mock import patch
import traceback
import sys

from flask import Flask
from sqlalchemy import select, event

ROOT = Path.cwd()
REPORTS = ROOT / "_patch_reports"
REPORTS.mkdir(exist_ok=True)

REPORT = REPORTS / (
    "employee_time_5b4b_"
    + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    + ".txt"
)

log = []
passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        log.append("PASS: " + name)
    else:
        failed += 1
        log.append("FAIL: " + name)

try:
    from extensions import db
    from models import User
    from services.employee_time_tracking_models import (
        EmployeeShift,
        EmployeeTimeEvent,
        EmployeeTimeSettingsAudit,
    )
    from services.employee_time_settings import (
        set_tracking_enabled,
        EmployeeTimeSettingsError,
    )
    from services.employee_time_tracking_service import (
        EmployeeTimeTrackingService,
    )

    with TemporaryDirectory(prefix="wccr_time_test_") as temp:
        database = Path(temp) / "test.sqlite"

        app = Flask(__name__)
        app.config.update(
            TESTING=True,
            SECRET_KEY="temporary-integration-test-only",
            SQLALCHEMY_DATABASE_URI="sqlite:///" + database.as_posix(),
            SQLALCHEMY_TRACK_MODIFICATIONS=False,
        )

        db.init_app(app)

        @app.teardown_appcontext
        def _close_test_db(exception):
            db.session.remove()
            db.engine.dispose()

        with app.app_context():
            tables = [
                User.__table__,
                EmployeeShift.__table__,
                EmployeeTimeEvent.__table__,
                EmployeeTimeSettingsAudit.__table__,
            ]

            db.metadata.create_all(bind=db.engine)

            admin = User(
                username="time_test_superadmin",
                password_hash="test-only",
                role="superadmin",
            )
            employee = User(
                username="time_test_employee",
                password_hash="test-only",
                role="admin",
            )
            technician = User(
                username="time_test_technician",
                password_hash="test-only",
                role="technician",
            )

            db.session.add_all([admin, employee, technician])
            db.session.commit()

            check(
                "Default OFF",
                not employee.employee_time_tracking_enabled,
            )

            check(
                "Disabled user not eligible",
                not EmployeeTimeTrackingService.eligible(employee),
            )

            check(
                "Enable returns True",
                set_tracking_enabled(admin, employee.id, True) is True,
            )

            db.session.expire_all()

            check(
                "Enable persisted",
                db.session.get(User, employee.id)
                .employee_time_tracking_enabled is True,
            )

            check(
                "Enabled user eligible",
                EmployeeTimeTrackingService.eligible(employee),
            )

            check(
                "Technician excluded",
                not EmployeeTimeTrackingService.eligible(technician),
            )

            try:
                set_tracking_enabled(admin, technician.id, True)
                check("Technician enable rejected", False)
            except EmployeeTimeSettingsError:
                check("Technician enable rejected", True)

            shift = EmployeeShift(
                user_id=employee.id,
                started_at=datetime.now(timezone.utc),
                status="open",
                requires_review=False,
            )
            db.session.add(shift)
            db.session.commit()

            check(
                "Disable returns True",
                set_tracking_enabled(admin, employee.id, False) is True,
            )

            db.session.expire_all()

            saved_shift = db.session.get(EmployeeShift, shift.id)

            check(
                "Shift closed",
                saved_shift.ended_at is not None,
            )
            check(
                "Shift pending review",
                saved_shift.status == "pending_review"
                and saved_shift.requires_review is True,
            )

            disable_event = db.session.execute(
                select(EmployeeTimeEvent).where(
                    EmployeeTimeEvent.shift_id == shift.id,
                    EmployeeTimeEvent.event_type == "TRACKING_DISABLED",
                )
            ).scalar_one_or_none()

            check(
                "Disable event recorded",
                disable_event is not None,
            )

            audits = db.session.execute(
                select(EmployeeTimeSettingsAudit).where(
                    EmployeeTimeSettingsAudit.user_id == employee.id
                )
            ).scalars().all()

            check(
                "Enable and disable audited",
                len(audits) == 2,
            )

            check(
                "Final state OFF",
                db.session.get(User, employee.id)
                .employee_time_tracking_enabled is False,
            )

            # Test rollback by making audit insertion fail.
            def reject_audit(mapper, connection, target):
                raise RuntimeError("Simulated audit failure")

            event.listen(
                EmployeeTimeSettingsAudit,
                "before_insert",
                reject_audit,
            )

            try:
                try:
                    set_tracking_enabled(admin, employee.id, True)
                    check("Audit failure propagated", False)
                except RuntimeError as exc:
                    check(
                        "Audit failure propagated",
                        str(exc) == "Simulated audit failure",
                    )
            finally:
                event.remove(
                    EmployeeTimeSettingsAudit,
                    "before_insert",
                    reject_audit,
                )

            db.session.expire_all()

            check(
                "Rollback preserved OFF",
                db.session.get(User, employee.id)
                .employee_time_tracking_enabled is False,
            )

            audits_after = db.session.execute(
                select(EmployeeTimeSettingsAudit).where(
                    EmployeeTimeSettingsAudit.user_id == employee.id
                )
            ).scalars().all()

            check(
                "Rollback preserved audit count",
                len(audits_after) == 2,
            )

except Exception:
    failed += 1
    log.append("FAIL: Unexpected exception")
    log.append(traceback.format_exc())
finally:
    try:
        db.session.remove()
        with app.app_context():
            db.engine.dispose()
    except Exception:
        log.append("CLEANUP WARNING:")
        log.append(traceback.format_exc())

log.append("")
log.append(f"PASSED: {passed}")
log.append(f"FAILED: {failed}")

REPORT.write_text("\n".join(log), encoding="utf-8")

print("INTEGRATION RESULT:", "PASS" if failed == 0 else "FAIL")
print("Passed checks:", passed)
print("Failed checks:", failed)
print("Report:", REPORT.relative_to(ROOT))
print("Production database modified by test: NO")

sys.exit(0 if failed == 0 else 1)
