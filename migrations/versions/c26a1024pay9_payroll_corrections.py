"""Payroll correction audit links.

Revision ID: c26a1024pay9
Revises: c26a1024pay8
"""

from alembic import op
import sqlalchemy as sa

revision = "c26a1024pay9"
down_revision = "c26a1024pay8"
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
                    "corrects_id",
                    sa.Integer(),
                    nullable=True,
                )
            )
            batch.add_column(
                sa.Column(
                    "correction_reason",
                    sa.Text(),
                    nullable=True,
                )
            )
            batch.create_foreign_key(
                "fk_" + table + "_corrects",
                table,
                ["corrects_id"],
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
                "fk_" + table + "_corrects",
                type_="foreignkey",
            )
            batch.drop_column("correction_reason")
            batch.drop_column("corrects_id")
