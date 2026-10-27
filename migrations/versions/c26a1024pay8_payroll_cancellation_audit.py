"""Payroll scheduled cancellation audit.

Revision ID: c26a1024pay8
Revises: c26a1024pay2
"""

from alembic import op
import sqlalchemy as sa

revision = "c26a1024pay8"
down_revision = "c26a1024pay2"
branch_labels = None
depends_on = None


def upgrade():
    for table in (
        "employee_pay_rate",
        "employee_payroll_status",
    ):
        with op.batch_alter_table(table) as batch:
            batch.add_column(
                sa.Column(
                    "cancelled_at",
                    sa.DateTime(),
                    nullable=True,
                )
            )
            batch.add_column(
                sa.Column(
                    "cancelled_by_id",
                    sa.Integer(),
                    nullable=True,
                )
            )
            batch.add_column(
                sa.Column(
                    "cancellation_reason",
                    sa.Text(),
                    nullable=True,
                )
            )
            batch.create_foreign_key(
                "fk_" + table + "_cancelled_by",
                "user",
                ["cancelled_by_id"],
                ["id"],
                ondelete="RESTRICT",
            )


def downgrade():
    for table in (
        "employee_payroll_status",
        "employee_pay_rate",
    ):
        with op.batch_alter_table(table) as batch:
            batch.drop_constraint(
                "fk_" + table + "_cancelled_by",
                type_="foreignkey",
            )
            batch.drop_column("cancellation_reason")
            batch.drop_column("cancelled_by_id")
            batch.drop_column("cancelled_at")
