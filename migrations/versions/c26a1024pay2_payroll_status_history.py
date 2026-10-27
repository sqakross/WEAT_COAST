"""WCCR payroll enabled status history.

Revision ID: c26a1024pay2
Revises: c26a1024pay1
"""
from alembic import op
import sqlalchemy as sa

revision = "c26a1024pay2"
down_revision = "c26a1024pay1"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "employee_payroll_status",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id", sa.Integer(), nullable=False,
        ),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column(
            "effective_date", sa.Date(), nullable=False,
        ),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False,
        ),
        sa.Column(
            "created_by_id", sa.Integer(), nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["user.id"], ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_id"], ["user.id"],
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "user_id", "effective_date",
            name="uq_employee_payroll_status_user_effective",
        ),
    )
    op.create_index(
        "ix_employee_payroll_status_user_id",
        "employee_payroll_status", ["user_id"],
    )
    op.create_index(
        "ix_employee_payroll_status_effective_date",
        "employee_payroll_status", ["effective_date"],
    )


def downgrade():
    op.drop_index(
        "ix_employee_payroll_status_effective_date",
        table_name="employee_payroll_status",
    )
    op.drop_index(
        "ix_employee_payroll_status_user_id",
        table_name="employee_payroll_status",
    )
    op.drop_table("employee_payroll_status")
