"""WCCR California Payroll Rules V1A.

Pure validation foundation.
No Flask, database, payroll calculation or payments.

A saved waiver status never establishes legal validity.
No overtime, meal or rest entitlement is waived here.
"""

from datetime import date, datetime
from decimal import Decimal

VERSION = "WCCR_CA_RULES_V1A"

ALLOWED_WAGE_ORDERS = frozenset(
    str(n) for n in range(1, 18)
)

ALLOWED_WAIVER_STATUSES = frozenset({
    "none",
    "documented_pending_review",
    "verified_by_authorized_reviewer",
})

REQUIRED_KEYS = frozenset({
    "effective_date",
    "wage_order",
    "first_meal_waiver",
    "second_meal_waiver",
    "notes",
})


class CaliforniaRulesError(ValueError):
    pass


def parse_date(value):
    if isinstance(value, datetime):
        raise CaliforniaRulesError(
            "Effective Date must not contain time"
        )
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise CaliforniaRulesError(
            "Invalid Effective Date"
        )
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise CaliforniaRulesError(
            "Invalid Effective Date"
        ) from exc


def validate_rules(values):
    if not isinstance(values, dict):
        raise CaliforniaRulesError(
            "California Rules must be an object"
        )

    if set(values) != REQUIRED_KEYS:
        raise CaliforniaRulesError(
            "Invalid California Rules fields"
        )

    effective = parse_date(
        values["effective_date"]
    )

    order = values["wage_order"]
    if type(order) is not str:
        raise CaliforniaRulesError(
            "Invalid Wage Order"
        )
    order = order.strip()

    if order not in ALLOWED_WAGE_ORDERS:
        raise CaliforniaRulesError(
            "Wage Order requires a valid number 1-17"
        )

    result = {
        "effective_date": effective.isoformat(),
        "wage_order": order,
    }

    for field in (
        "first_meal_waiver",
        "second_meal_waiver",
    ):
        status = values[field]

        if (
            type(status) is not str
            or status not in ALLOWED_WAIVER_STATUSES
        ):
            raise CaliforniaRulesError(
                "Invalid meal waiver status: " + field
            )

        result[field] = status

    notes = values["notes"]

    if (
        type(notes) is not str
        or len(notes.strip()) > 1000
    ):
        raise CaliforniaRulesError(
            "Notes must be text, maximum 1000 characters"
        )

    result["notes"] = notes.strip()
    return result


def validate_history(rows):
    if not isinstance(rows, (list, tuple)):
        raise CaliforniaRulesError(
            "Invalid California Rules history"
        )

    output = []
    seen = set()

    for record in rows:
        normalized = validate_rules(record)
        effective = normalized["effective_date"]

        if effective in seen:
            raise CaliforniaRulesError(
                "Duplicate California Rules Effective Date"
            )

        seen.add(effective)
        output.append(normalized)

    output.sort(key=lambda row: row["effective_date"])
    return output


def rules_on_date(rows, work_date):
    target = parse_date(work_date)
    history = validate_history(rows)

    selected = None
    for record in history:
        if parse_date(record["effective_date"]) <= target:
            selected = record
        else:
            break

    return selected


def compliance_status(values):
    rules = validate_rules(values)

    pending = any(
        rules[field] != "none"
        for field in (
            "first_meal_waiver",
            "second_meal_waiver",
        )
    )

    return {
        "configuration_valid": True,
        "waiver_review_required": pending,
        "legal_eligibility_verified": False,
        "meal_rest_compliance_verified": False,
        "payroll_approval_allowed": False,
        "payment_allowed": False,
    }
