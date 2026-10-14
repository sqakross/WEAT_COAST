
"""Pure validation helpers for WCCR employee pay settings."""
from datetime import date
from decimal import Decimal, InvalidOperation

PAY_TYPES = {"hourly", "salary"}
EXEMPTION_STATUSES = {"exempt", "non_exempt"}


class PayProfileValidationError(ValueError):
    pass


def validate_pay_type(value):
    if value not in PAY_TYPES:
        raise PayProfileValidationError("Invalid pay type")
    return value


def validate_exemption(value):
    if value not in EXEMPTION_STATUSES:
        raise PayProfileValidationError(
            "Invalid exemption classification"
        )
    return value


def validate_amount(value):
    if isinstance(value, (float, bool)):
        raise PayProfileValidationError(
            "Use a decimal string, not float or boolean"
        )
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise PayProfileValidationError("Invalid amount")
    if not amount.is_finite() or amount < 0:
        raise PayProfileValidationError("Invalid amount")
    if amount.as_tuple().exponent < -2:
        raise PayProfileValidationError(
            "Maximum two decimal places"
        )
    if amount >= Decimal("1000000000000"):
        raise PayProfileValidationError("Amount too large")
    return amount.quantize(Decimal("0.01"))


def validate_effective_dates(start, end=None):
    if not isinstance(start, date) or isinstance(
        start, __import__("datetime").datetime
    ):
        raise PayProfileValidationError(
            "Invalid effective start date"
        )
    if end is not None:
        if not isinstance(end, date) or isinstance(
            end, __import__("datetime").datetime
        ) or end < start:
            raise PayProfileValidationError(
                "Invalid effective end date"
            )
    return start, end


def periods_overlap(a_start, a_end, b_start, b_end):
    validate_effective_dates(a_start, a_end)
    validate_effective_dates(b_start, b_end)
    return (
        (a_end is None or b_start <= a_end)
        and (b_end is None or a_start <= b_end)
    )
