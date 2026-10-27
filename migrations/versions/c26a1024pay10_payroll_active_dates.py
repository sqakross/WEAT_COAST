"""Payroll active effective-date uniqueness.

Revision ID: c26a1024pay10
Revises: c26a1024pay9
"""
from alembic import op

revision = "c26a1024pay10"
down_revision = "c26a1024pay9"
branch_labels = None
depends_on = None


def upgrade():
    for table, constraint in (
        (
            "employee_pay_rate",
            "uq_employee_pay_rate_user_effective",
        ),
        (
            "employee_payroll_status",
            "uq_employee_payroll_status_user_effective",
        ),
    ):
        with op.batch_alter_table(table) as batch:
            batch.drop_constraint(
                constraint, type_="unique"
            )

        op.create_index(
            "ux_" + table + "_active_effective",
            table,
            ["user_id", "effective_date"],
            unique=True,
            sqlite_where=__import__(
                "sqlalchemy"
            ).text("cancelled_at IS NULL"),
        )


def downgrade():
    raise RuntimeError(
        "Downgrade requires manual reconciliation "
        "of historical corrections."
    )
