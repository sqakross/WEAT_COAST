
"""WCCR V3.0.1 - transactional pay persistence.

Caller supplies an already-authorized actor and an SQLAlchemy session.
This module does not import the Flask application or connect to a DB.
"""
from datetime import date
from decimal import Decimal

from services.employee_pay_validation import (
    PayProfileValidationError,
    periods_overlap,
    validate_amount,
    validate_effective_dates,
    validate_exemption,
    validate_pay_type,
)


class EmployeePayPersistence:

    @staticmethod
    def validate_record(data):
        pay_type = validate_pay_type(data["pay_type"])
        exemption = validate_exemption(
            data["exemption_status"]
        )
        amount = validate_amount(data["amount"])
        start, end = validate_effective_dates(
            data["effective_from"],
            data.get("effective_to"),
        )
        state = str(data["work_state"]).strip().upper()
        if len(state) != 2 or not state.isascii() or not state.isalpha():
            raise PayProfileValidationError("Invalid work state")
        return {
            "pay_type": pay_type,
            "exemption_status": exemption,
            "work_state": state,
            "amount": amount,
            "effective_from": start,
            "effective_to": end,
        }

    @staticmethod
    def create_rate(
        session,
        actor,
        employee_id,
        values,
        models,
        access_service,
    ):
        # Authorization must happen on every write, not only in the UI.
        from services.employee_pay_service import EmployeePayService

        EmployeePayService.require_manage(actor, access_service)

        if type(employee_id) is not int or employee_id <= 0:
            raise PayProfileValidationError("Invalid employee ID")
        actor_id = getattr(actor, "id", None)
        if type(actor_id) is not int or actor_id <= 0:
            raise PayProfileValidationError("Invalid actor ID")

        normalized = EmployeePayPersistence.validate_record(values)

        Profile, Rate, Audit = models

        # Caller owns the transaction boundary. This function deliberately
        # does not commit, so it can be composed with other atomic writes.
        profile = (
            session.query(Profile)
            .filter(Profile.user_id == employee_id)
            .first()
        )

        if profile is None:
            profile = Profile(
                user_id=employee_id,
                work_state=normalized["work_state"],
                pay_type=normalized["pay_type"],
                exemption_status=normalized["exemption_status"],
            )
            session.add(profile)
            session.flush()

        existing = (
            session.query(Rate)
            .filter(Rate.profile_id == profile.id)
            .all()
        )

        for rate in existing:
            if periods_overlap(
                rate.effective_from,
                rate.effective_to,
                normalized["effective_from"],
                normalized["effective_to"],
            ):
                raise PayProfileValidationError(
                    "Overlapping pay periods"
                )

        rate = Rate(
            profile_id=profile.id,
            effective_from=normalized["effective_from"],
            effective_to=normalized["effective_to"],
            amount=normalized["amount"],
            pay_type=normalized["pay_type"],
            created_by_id=actor_id,
        )
        rate.exemption_status = normalized["exemption_status"]
        rate.work_state = normalized["work_state"]

        session.add(rate)
        session.flush()

        import json
        audit = Audit(
            user_id=employee_id,
            actor_id=actor_id,
            action="PAY_RATE_CREATED",
            old_values=None,
            new_values=json.dumps(
                {
                    "rate_id": rate.id,
                    "pay_type": normalized["pay_type"],
                    "exemption_status": normalized["exemption_status"],
                    "work_state": normalized["work_state"],
                    "amount": str(normalized["amount"]),
                    "effective_from": str(normalized["effective_from"]),
                    "effective_to": (
                        str(normalized["effective_to"])
                        if normalized["effective_to"] else None
                    ),
                },
                sort_keys=True,
            ),
        )
        session.add(audit)
        session.flush()

        return rate
