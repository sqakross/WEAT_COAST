"""appliance manual work order reference

Revision ID: c31f4a2e9d70
Revises: 87dc3aa9b0c4
Create Date: 2026-08-26

"""

from alembic import op
import sqlalchemy as sa


revision = "c31f4a2e9d70"
down_revision = "87dc3aa9b0c4"
branch_labels = None
depends_on = None


def upgrade():

    # --------------------------------------------------------
    # Appliance Issue
    # --------------------------------------------------------

    op.add_column(
        "appliance_issue",
        sa.Column(
            "work_order_number",
            sa.String(length=120),
            nullable=True,
        ),
    )

    # Preserve W/O values from already created test/real issues.
    op.execute(
        """
        UPDATE appliance_issue
        SET work_order_number = (
            SELECT work_orders.job_numbers
            FROM work_orders
            WHERE work_orders.id = appliance_issue.work_order_id
        )
        WHERE work_order_number IS NULL
          AND work_order_id IS NOT NULL
        """
    )

    # Existing rows without a matching main WO remain valid.
    op.execute(
        """
        UPDATE appliance_issue
        SET work_order_number = 'UNKNOWN'
        WHERE work_order_number IS NULL
        """
    )

    with op.batch_alter_table(
        "appliance_issue"
    ) as batch_op:

        batch_op.alter_column(
            "work_order_number",
            existing_type=sa.String(length=120),
            nullable=False,
        )

        batch_op.create_index(
            "ix_appliance_issue_work_order_number",
            ["work_order_number"],
            unique=False,
        )


    # --------------------------------------------------------
    # Appliance Issue Line
    # --------------------------------------------------------

    op.add_column(
        "appliance_issue_line",
        sa.Column(
            "current_work_order_number",
            sa.String(length=120),
            nullable=True,
        ),
    )

    op.execute(
        """
        UPDATE appliance_issue_line
        SET current_work_order_number = (
            SELECT appliance_issue.work_order_number
            FROM appliance_issue
            WHERE appliance_issue.id =
                  appliance_issue_line.issue_id
        )
        WHERE current_work_order_number IS NULL
        """
    )

    op.create_index(
        "ix_appliance_issue_line_current_work_order_number",
        "appliance_issue_line",
        ["current_work_order_number"],
        unique=False,
    )


    # --------------------------------------------------------
    # Current ApplianceUnit state
    # --------------------------------------------------------

    op.add_column(
        "appliance_unit",
        sa.Column(
            "current_work_order_number",
            sa.String(length=120),
            nullable=True,
        ),
    )

    op.execute(
        """
        UPDATE appliance_unit
        SET current_work_order_number = (
            SELECT work_orders.job_numbers
            FROM work_orders
            WHERE work_orders.id =
                  appliance_unit.current_work_order_id
        )
        WHERE current_work_order_id IS NOT NULL
        """
    )

    op.create_index(
        "ix_appliance_unit_current_work_order_number",
        "appliance_unit",
        ["current_work_order_number"],
        unique=False,
    )


    # --------------------------------------------------------
    # Immutable Movement history
    # --------------------------------------------------------

    op.add_column(
        "appliance_movement",
        sa.Column(
            "from_work_order_number",
            sa.String(length=120),
            nullable=True,
        ),
    )

    op.add_column(
        "appliance_movement",
        sa.Column(
            "to_work_order_number",
            sa.String(length=120),
            nullable=True,
        ),
    )

    op.execute(
        """
        UPDATE appliance_movement
        SET from_work_order_number = (
            SELECT work_orders.job_numbers
            FROM work_orders
            WHERE work_orders.id =
                  appliance_movement.from_work_order_id
        )
        WHERE from_work_order_id IS NOT NULL
        """
    )

    op.execute(
        """
        UPDATE appliance_movement
        SET to_work_order_number = (
            SELECT appliance_issue.work_order_number
            FROM appliance_issue
            WHERE appliance_issue.id =
                  appliance_movement.issue_id
        )
        WHERE to_work_order_number IS NULL
          AND issue_id IS NOT NULL
        """
    )

    op.create_index(
        "ix_appliance_movement_from_work_order_number",
        "appliance_movement",
        ["from_work_order_number"],
        unique=False,
    )

    op.create_index(
        "ix_appliance_movement_to_work_order_number",
        "appliance_movement",
        ["to_work_order_number"],
        unique=False,
    )


def downgrade():

    op.drop_index(
        "ix_appliance_movement_to_work_order_number",
        table_name="appliance_movement",
    )

    op.drop_index(
        "ix_appliance_movement_from_work_order_number",
        table_name="appliance_movement",
    )

    op.drop_column(
        "appliance_movement",
        "to_work_order_number",
    )

    op.drop_column(
        "appliance_movement",
        "from_work_order_number",
    )

    op.drop_index(
        "ix_appliance_unit_current_work_order_number",
        table_name="appliance_unit",
    )

    op.drop_column(
        "appliance_unit",
        "current_work_order_number",
    )

    op.drop_index(
        "ix_appliance_issue_line_current_work_order_number",
        table_name="appliance_issue_line",
    )

    op.drop_column(
        "appliance_issue_line",
        "current_work_order_number",
    )

    with op.batch_alter_table(
        "appliance_issue"
    ) as batch_op:

        batch_op.drop_index(
            "ix_appliance_issue_work_order_number"
        )

        batch_op.drop_column(
            "work_order_number"
        )
