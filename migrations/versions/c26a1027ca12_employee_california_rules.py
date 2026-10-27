"""Employee California payroll rules V1B.2.

Revision ID: c26a1027ca12
Revises: c26a1024pay10
"""

from alembic import op
import sqlalchemy as sa

revision = "c26a1027ca12"
down_revision = "c26a1024pay10"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "employee_california_payroll_rule",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id", sa.Integer(), nullable=False,
        ),
        sa.Column(
            "effective_date", sa.Date(), nullable=False,
        ),
        sa.Column(
            "wage_order", sa.String(2), nullable=False,
        ),
        sa.Column(
            "first_meal_waiver", sa.String(40),
            nullable=False, server_default="none",
        ),
        sa.Column(
            "second_meal_waiver", sa.String(40),
            nullable=False, server_default="none",
        ),
        sa.Column(
            "notes", sa.Text(),
            nullable=False, server_default="",
        ),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False,
        ),
        sa.Column(
            "created_by_id", sa.Integer(), nullable=False,
        ),
        sa.Column("cancelled_at", sa.DateTime()),
        sa.Column("cancelled_by_id", sa.Integer()),
        sa.Column("cancellation_reason", sa.Text()),
        sa.Column("corrects_id", sa.Integer()),
        sa.Column("correction_reason", sa.Text()),
        sa.CheckConstraint(
            "wage_order IN (" +
            ", ".join("'" + str(n) + "'" for n in range(1, 18)) +
            ")",
            name="ck_employee_ca_rule_wage_order",
        ),
        sa.CheckConstraint(
            "first_meal_waiver IN "
            "('none', 'documented_pending_review', "
            "'verified_by_authorized_reviewer')",
            name="ck_employee_ca_rule_first_waiver",
        ),
        sa.CheckConstraint(
            "second_meal_waiver IN "
            "('none', 'documented_pending_review', "
            "'verified_by_authorized_reviewer')",
            name="ck_employee_ca_rule_second_waiver",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["user.id"], ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_id"], ["user.id"], ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["cancelled_by_id"], ["user.id"], ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["corrects_id"],
            ["employee_california_payroll_rule.id"],
            ondelete="RESTRICT",
        ),
    )

    op.create_index(
        "ix_employee_california_payroll_rule_user_id",
        "employee_california_payroll_rule",
        ["user_id"],
    )
    op.create_index(
        "ix_employee_california_payroll_rule_effective_date",
        "employee_california_payroll_rule",
        ["effective_date"],
    )
    op.create_index(
        "uq_employee_ca_rule_active_date",
        "employee_california_payroll_rule",
        ["user_id", "effective_date"],
        unique=True,
        sqlite_where=sa.text("cancelled_at IS NULL"),
    )


def downgrade():
    raise RuntimeError(
        "Destructive downgrade is disabled. "
        "Restore from a verified database backup."
    )
