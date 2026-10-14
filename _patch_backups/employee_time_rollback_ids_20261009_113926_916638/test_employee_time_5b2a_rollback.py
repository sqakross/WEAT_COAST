import tempfile
from pathlib import Path
from unittest.mock import patch

from flask import Flask
from sqlalchemy import select

from extensions import db
from models import User
from services.employee_time_tracking_models import (
    EmployeeShift,
    EmployeeTimeEvent,
    EmployeeTimeSettingsAudit,
)
from services.employee_time_settings import set_tracking_enabled


with tempfile.TemporaryDirectory() as directory:
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY="isolated-rollback-test",
        SQLALCHEMY_DATABASE_URI=(
            "sqlite:///" +
            str(Path(directory) / "rollback.db").replace("\\", "/")
        ),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )

    db.init_app(app)
    db.session.session_factory.configure(expire_on_commit=False)

    with app.app_context():
        try:
            db.metadata.create_all(
                bind=db.engine,
                tables=[
                    User.__table__,
                    EmployeeShift.__table__,
                    EmployeeTimeEvent.__table__,
                    EmployeeTimeSettingsAudit.__table__,
                ],
            )

            actor = User(
                username="rollback_superadmin",
                password_hash="isolated-test-only",
                role="superadmin",
                employee_time_tracking_enabled=False,
            )
            employee = User(
                username="rollback_employee",
                password_hash="isolated-test-only",
                role="user",
                employee_time_tracking_enabled=True,
            )

            db.session.add_all([actor, employee])
            db.session.commit()

            shift = EmployeeShift(
                user_id=employee.id,
                started_at=db.func.current_timestamp(),
                status="open",
                requires_review=False,
            )
            db.session.add(shift)
            db.session.commit()

            original_add = db.session.add

            def fail_on_audit(obj):
                if isinstance(obj, EmployeeTimeSettingsAudit):
                    raise RuntimeError("Simulated audit failure")
                return original_add(obj)

            try:
                with patch.object(db.session, "add", side_effect=fail_on_audit):
                    set_tracking_enabled(actor, employee.id, False)
            except RuntimeError as exc:
                assert str(exc) == "Simulated audit failure"
                print("PASS: Simulated audit failure detected")
            else:
                raise AssertionError(
                    "Expected audit failure was not raised"
                )

            # Query persisted scalar columns directly.
            # Avoid loading unrelated WCCR ORM relationships.
            with db.engine.connect() as connection:
                enabled = connection.execute(
                    select(User.__table__.c.employee_time_tracking_enabled)
                    .where(User.__table__.c.id == employee.id)
                ).scalar_one()

                shift_state = connection.execute(
                    select(
                        EmployeeShift.__table__.c.ended_at,
                        EmployeeShift.__table__.c.requires_review,
                    ).where(EmployeeShift.__table__.c.id == shift.id)
                ).one()

                audit_count = connection.execute(
                    select(db.func.count())
                    .select_from(EmployeeTimeSettingsAudit.__table__)
                ).scalar_one()

                event_count = connection.execute(
                    select(db.func.count())
                    .select_from(EmployeeTimeEvent.__table__)
                ).scalar_one()

            assert bool(enabled) is True
            print("PASS: Tracking remains ON after rollback")

            assert shift_state.ended_at is None
            print("PASS: Open shift remains open after rollback")

            assert bool(shift_state.requires_review) is False
            print("PASS: Review flag unchanged after rollback")

            assert audit_count == 0
            print("PASS: No partial audit record")

            assert event_count == 0
            print("PASS: No partial time event")

            print()
            print("TRANSACTION ROLLBACK TEST PASSED")
            print("PRODUCTION DATABASE MODIFIED: NO")

        finally:
            db.session.remove()
            db.engine.dispose()
