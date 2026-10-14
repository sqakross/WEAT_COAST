
"""WCCR V3.0.1 - Employee Pay Service.

No production database access on import.
All persistence must be performed through an authorized caller.
"""

from datetime import date, timedelta
from decimal import Decimal
from services.employee_pay_validation import (
    validate_pay_type,
    validate_exemption,
    validate_amount,
    validate_effective_dates,
    periods_overlap,
    PayProfileValidationError,
)


class EmployeePayService:

    PERMISSION = "erp.payroll.manage_rates"

    @staticmethod
    def can_manage(actor, access_service):
        if actor is None:
            return False

        return bool(
            access_service.is_allowed(
                actor,
                EmployeePayService.PERMISSION,
                default_allowed=False,
            )
        )

    @staticmethod
    def require_manage(actor, access_service):
        if not EmployeePayService.can_manage(
            actor, access_service
        ):
            raise PermissionError(
                "Employee pay management access denied"
            )

    @staticmethod
    def normalize_settings(
        pay_type,
        exemption_status,
        work_state,
        amount,
        effective_from,
        effective_to=None,
    ):
        pay_type = validate_pay_type(pay_type)
        exemption_status = validate_exemption(
            exemption_status
        )
        amount = validate_amount(amount)

        start, end = validate_effective_dates(
            effective_from,
            effective_to,
        )

        state = str(work_state or "").strip().upper()

        if len(state) != 2 or not state.isalpha():
            raise PayProfileValidationError(
                "Invalid work state"
            )

        return {
            "pay_type": pay_type,
            "exemption_status": exemption_status,
            "work_state": state,
            "amount": amount,
            "effective_from": start,
            "effective_to": end,
        }

    @staticmethod
    def assert_no_overlap(existing_rates, start, end):
        validate_effective_dates(start, end)

        for rate in existing_rates:
            if periods_overlap(
                rate.effective_from,
                rate.effective_to,
                start,
                end,
            ):
                raise PayProfileValidationError(
                    "Pay rate effective periods overlap"
                )

    @staticmethod
    def audit_payload(
        employee_id,
        actor_id,
        old_values,
        new_values,
    ):
        if not isinstance(employee_id, int) or employee_id <= 0:
            raise PayProfileValidationError(
                "Invalid employee ID"
            )

        if not isinstance(actor_id, int) or actor_id <= 0:
            raise PayProfileValidationError(
                "Invalid actor ID"
            )

        def convert(value):
            if isinstance(value, (Decimal, date)):
                return str(value)
            raise TypeError(
                "Unsupported audit value type"
            )

        import json

        return {
            "user_id": employee_id,
            "actor_id": actor_id,
            "action": "PAY_SETTINGS_CHANGED",
            "old_values": (
                json.dumps(
                    old_values,
                    default=convert,
                    sort_keys=True,
                )
                if old_values is not None
                else None
            ),
            "new_values": json.dumps(
                new_values,
                default=convert,
                sort_keys=True,
            ),
        }

    @staticmethod
    def prepare_change(
        actor,
        access_service,
        employee_id,
        existing_rates,
        pay_type,
        exemption_status,
        work_state,
        amount,
        effective_from,
        effective_to=None,
    ):
        EmployeePayService.require_manage(
            actor, access_service
        )

        values = EmployeePayService.normalize_settings(
            pay_type,
            exemption_status,
            work_state,
            amount,
            effective_from,
            effective_to,
        )

        EmployeePayService.assert_no_overlap(
            existing_rates,
            values["effective_from"],
            values["effective_to"],
        )

        if not isinstance(employee_id, int) or employee_id <= 0:
            raise PayProfileValidationError(
                "Invalid employee ID"
            )

        actor_id = getattr(actor, "id", None)

        if not isinstance(actor_id, int) or actor_id <= 0:
            raise PayProfileValidationError(
                "Invalid actor ID"
            )

        return {
            "employee_id": employee_id,
            "actor_id": actor_id,
            "values": values,
        }
