"""WCCR employee pay rate history foundation.

Revision ID: c26a1024pay1
Revises: b7c4e2a91f30
"""

from alembic import op
import sqlalchemy as sa

revision = "c26a1024pay1"
down_revision = "b7c4e2a91f30"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "employee_pay_rate",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("user.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("pay_type", sa.String(12), nullable=False),
        sa.Column("classification", sa.String(16), nullable=False),
        sa.Column("amount", sa.Numeric(12, 2), nullable=False),
        sa.Column("salary_frequency", sa.String(20), nullable=True),
        sa.Column("effective_date", sa.Date(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column(
            "created_by_id",
            sa.Integer(),
            sa.ForeignKey("user.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "user_id", "effective_date",
            name="uq_employee_pay_rate_user_effective",
        ),
        sa.CheckConstraint(
            "pay_type IN ('hourly', 'salary')",
            name="ck_employee_pay_rate_type",
        ),
        sa.CheckConstraint(
            "classification IN ('non_exempt', 'exempt')",
            name="ck_employee_pay_rate_classification",
        ),
        sa.CheckConstraint(
            "amount > 0",
            name="ck_employee_pay_rate_positive",
        ),
        sa.CheckConstraint(
            "(pay_type = 'hourly' AND salary_frequency IS NULL) OR "
            "(pay_type = 'salary' AND salary_frequency IS NOT NULL)",
            name="ck_employee_pay_rate_frequency",
        ),
    )

    op.create_index(
        "ix_employee_pay_rate_user_id",
        "employee_pay_rate",
        ["user_id"],
    )
    op.create_index(
        "ix_employee_pay_rate_effective_date",
        "employee_pay_rate",
        ["effective_date"],
    )


def downgrade():
    raise RuntimeError(
        "Payroll rate history must not be dropped automatically."
    )