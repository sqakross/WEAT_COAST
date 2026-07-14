"""add asset assignments and assignment receipts

Revision ID: fa00164a1011
Revises: 097c8034d48e
Create Date: 2026-07-11 08:26:22.370610
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "fa00164a1011"
down_revision = "097c8034d48e"
branch_labels = None
depends_on = None


def upgrade():
    # ---------------------------------------------------------
    # 1. Universal asset fields
    # ---------------------------------------------------------
    with op.batch_alter_table("tool_assets") as batch_op:
        batch_op.add_column(
            sa.Column(
                "asset_type",
                sa.String(length=40),
                nullable=False,
                server_default="TOOL",
            )
        )

        batch_op.add_column(
            sa.Column(
                "current_holder_id",
                sa.Integer(),
                nullable=True,
            )
        )

        batch_op.add_column(
            sa.Column(
                "current_holder_name",
                sa.String(length=80),
                nullable=True,
            )
        )

        batch_op.create_foreign_key(
            "fk_tool_assets_current_holder_id_user",
            "user",
            ["current_holder_id"],
            ["id"],
            ondelete="SET NULL",
        )

        batch_op.create_index(
            "ix_tool_assets_asset_type",
            ["asset_type"],
            unique=False,
        )

        batch_op.create_index(
            "ix_tool_assets_current_holder_id",
            ["current_holder_id"],
            unique=False,
        )

        batch_op.create_index(
            "ix_tool_assets_current_holder_name",
            ["current_holder_name"],
            unique=False,
        )

        batch_op.create_index(
            "ix_tool_assets_status_holder",
            ["status", "current_holder_id"],
            unique=False,
        )

    # Existing rows receive TOOL explicitly.
    op.execute(
        """
        UPDATE tool_assets
        SET asset_type = 'TOOL'
        WHERE asset_type IS NULL OR TRIM(asset_type) = ''
        """
    )

    # Copy existing technician custody into the new universal holder fields.
    op.execute(
        """
        UPDATE tool_assets
        SET current_holder_id = current_technician_id
        WHERE current_holder_id IS NULL
          AND current_technician_id IS NOT NULL
        """
    )

    op.execute(
        """
        UPDATE tool_assets
        SET current_holder_name = current_technician_name
        WHERE (
            current_holder_name IS NULL
            OR TRIM(current_holder_name) = ''
        )
        AND current_technician_name IS NOT NULL
        """
    )

    # Remove temporary database default.
    # Python model default remains TOOL.
    with op.batch_alter_table("tool_assets") as batch_op:
        batch_op.alter_column(
            "asset_type",
            existing_type=sa.String(length=40),
            nullable=False,
            server_default=None,
        )

    # ---------------------------------------------------------
    # 2. Asset Assignment Receipt header
    # ---------------------------------------------------------
    op.create_table(
        "asset_assignment_receipts",

        sa.Column(
            "id",
            sa.Integer(),
            primary_key=True,
            nullable=False,
        ),

        sa.Column(
            "receipt_number",
            sa.String(length=30),
            nullable=False,
        ),

        sa.Column(
            "assigned_to_id",
            sa.Integer(),
            nullable=False,
        ),

        sa.Column(
            "assigned_to_name",
            sa.String(length=120),
            nullable=False,
        ),

        sa.Column(
            "issued_by_id",
            sa.Integer(),
            nullable=True,
        ),

        sa.Column(
            "issued_by_name",
            sa.String(length=120),
            nullable=False,
        ),

        sa.Column(
            "work_order_id",
            sa.Integer(),
            nullable=True,
        ),

        sa.Column(
            "assigned_at",
            sa.DateTime(),
            nullable=False,
        ),

        sa.Column(
            "status",
            sa.String(length=30),
            nullable=False,
            server_default="open",
        ),

        sa.Column(
            "note",
            sa.Text(),
            nullable=True,
        ),

        sa.Column(
            "acknowledged_at",
            sa.DateTime(),
            nullable=True,
        ),

        sa.Column(
            "acknowledged_by_id",
            sa.Integer(),
            nullable=True,
        ),

        sa.Column(
            "acknowledged_by_name",
            sa.String(length=120),
            nullable=True,
        ),

        sa.Column(
            "signature_name",
            sa.String(length=200),
            nullable=True,
        ),

        sa.Column(
            "signature_data",
            sa.Text(),
            nullable=True,
        ),

        sa.Column(
            "voided_at",
            sa.DateTime(),
            nullable=True,
        ),

        sa.Column(
            "voided_by_id",
            sa.Integer(),
            nullable=True,
        ),

        sa.Column(
            "voided_by_name",
            sa.String(length=120),
            nullable=True,
        ),

        sa.Column(
            "void_reason",
            sa.String(length=500),
            nullable=True,
        ),

        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
        ),

        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
        ),

        sa.ForeignKeyConstraint(
            ["assigned_to_id"],
            ["user.id"],
            name="fk_asset_receipt_assigned_to_user",
            ondelete="RESTRICT",
        ),

        sa.ForeignKeyConstraint(
            ["issued_by_id"],
            ["user.id"],
            name="fk_asset_receipt_issued_by_user",
            ondelete="SET NULL",
        ),

        sa.ForeignKeyConstraint(
            ["acknowledged_by_id"],
            ["user.id"],
            name="fk_asset_receipt_acknowledged_by_user",
            ondelete="SET NULL",
        ),

        sa.ForeignKeyConstraint(
            ["voided_by_id"],
            ["user.id"],
            name="fk_asset_receipt_voided_by_user",
            ondelete="SET NULL",
        ),

        sa.ForeignKeyConstraint(
            ["work_order_id"],
            ["work_orders.id"],
            name="fk_asset_receipt_work_order",
            ondelete="SET NULL",
        ),

        sa.UniqueConstraint(
            "receipt_number",
            name="uq_asset_assignment_receipt_number",
        ),
    )

    op.create_index(
        "ix_asset_assignment_receipts_receipt_number",
        "asset_assignment_receipts",
        ["receipt_number"],
        unique=True,
    )

    op.create_index(
        "ix_asset_assignment_receipts_assigned_to_id",
        "asset_assignment_receipts",
        ["assigned_to_id"],
        unique=False,
    )

    op.create_index(
        "ix_asset_assignment_receipts_work_order_id",
        "asset_assignment_receipts",
        ["work_order_id"],
        unique=False,
    )

    op.create_index(
        "ix_asset_assignment_receipts_status",
        "asset_assignment_receipts",
        ["status"],
        unique=False,
    )

    op.create_index(
        "ix_asset_assignment_receipts_assigned_at",
        "asset_assignment_receipts",
        ["assigned_at"],
        unique=False,
    )

    op.create_index(
        "ix_asset_receipt_holder_date",
        "asset_assignment_receipts",
        ["assigned_to_id", "assigned_at"],
        unique=False,
    )

    op.create_index(
        "ix_asset_receipt_wo_status",
        "asset_assignment_receipts",
        ["work_order_id", "status"],
        unique=False,
    )

    # ---------------------------------------------------------
    # 3. Asset Assignment rows
    # ---------------------------------------------------------
    op.create_table(
        "asset_assignments",

        sa.Column(
            "id",
            sa.Integer(),
            primary_key=True,
            nullable=False,
        ),

        sa.Column(
            "receipt_id",
            sa.Integer(),
            nullable=False,
        ),

        sa.Column(
            "asset_id",
            sa.Integer(),
            nullable=False,
        ),

        sa.Column(
            "assigned_to_id",
            sa.Integer(),
            nullable=False,
        ),

        sa.Column(
            "assigned_to_name",
            sa.String(length=120),
            nullable=False,
        ),

        sa.Column(
            "assigned_by_id",
            sa.Integer(),
            nullable=True,
        ),

        sa.Column(
            "assigned_by_name",
            sa.String(length=120),
            nullable=False,
        ),

        sa.Column(
            "work_order_id",
            sa.Integer(),
            nullable=True,
        ),

        sa.Column(
            "work_unit_id",
            sa.Integer(),
            nullable=True,
        ),

        sa.Column(
            "quantity",
            sa.Integer(),
            nullable=False,
            server_default="1",
        ),

        sa.Column(
            "assigned_at",
            sa.DateTime(),
            nullable=False,
        ),

        sa.Column(
            "status",
            sa.String(length=30),
            nullable=False,
            server_default="assigned",
        ),

        sa.Column(
            "assignment_note",
            sa.Text(),
            nullable=True,
        ),

        sa.Column(
            "returned_at",
            sa.DateTime(),
            nullable=True,
        ),

        sa.Column(
            "returned_by_id",
            sa.Integer(),
            nullable=True,
        ),

        sa.Column(
            "returned_by_name",
            sa.String(length=120),
            nullable=True,
        ),

        sa.Column(
            "return_condition",
            sa.String(length=40),
            nullable=True,
        ),

        sa.Column(
            "return_note",
            sa.Text(),
            nullable=True,
        ),

        sa.Column(
            "voided_at",
            sa.DateTime(),
            nullable=True,
        ),

        sa.Column(
            "voided_by_id",
            sa.Integer(),
            nullable=True,
        ),

        sa.Column(
            "voided_by_name",
            sa.String(length=120),
            nullable=True,
        ),

        sa.Column(
            "void_reason",
            sa.String(length=500),
            nullable=True,
        ),

        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
        ),

        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
        ),

        sa.ForeignKeyConstraint(
            ["receipt_id"],
            ["asset_assignment_receipts.id"],
            name="fk_asset_assignment_receipt",
            ondelete="CASCADE",
        ),

        sa.ForeignKeyConstraint(
            ["asset_id"],
            ["tool_assets.id"],
            name="fk_asset_assignment_asset",
            ondelete="RESTRICT",
        ),

        sa.ForeignKeyConstraint(
            ["assigned_to_id"],
            ["user.id"],
            name="fk_asset_assignment_assigned_to_user",
            ondelete="RESTRICT",
        ),

        sa.ForeignKeyConstraint(
            ["assigned_by_id"],
            ["user.id"],
            name="fk_asset_assignment_assigned_by_user",
            ondelete="SET NULL",
        ),

        sa.ForeignKeyConstraint(
            ["returned_by_id"],
            ["user.id"],
            name="fk_asset_assignment_returned_by_user",
            ondelete="SET NULL",
        ),

        sa.ForeignKeyConstraint(
            ["voided_by_id"],
            ["user.id"],
            name="fk_asset_assignment_voided_by_user",
            ondelete="SET NULL",
        ),

        sa.ForeignKeyConstraint(
            ["work_order_id"],
            ["work_orders.id"],
            name="fk_asset_assignment_work_order",
            ondelete="SET NULL",
        ),

        sa.ForeignKeyConstraint(
            ["work_unit_id"],
            ["work_units.id"],
            name="fk_asset_assignment_work_unit",
            ondelete="SET NULL",
        ),

        sa.CheckConstraint(
            "quantity > 0",
            name="ck_asset_assignment_quantity_positive",
        ),
    )

    op.create_index(
        "ix_asset_assignments_receipt_id",
        "asset_assignments",
        ["receipt_id"],
        unique=False,
    )

    op.create_index(
        "ix_asset_assignments_asset_id",
        "asset_assignments",
        ["asset_id"],
        unique=False,
    )

    op.create_index(
        "ix_asset_assignments_assigned_to_id",
        "asset_assignments",
        ["assigned_to_id"],
        unique=False,
    )

    op.create_index(
        "ix_asset_assignments_work_order_id",
        "asset_assignments",
        ["work_order_id"],
        unique=False,
    )

    op.create_index(
        "ix_asset_assignments_work_unit_id",
        "asset_assignments",
        ["work_unit_id"],
        unique=False,
    )

    op.create_index(
        "ix_asset_assignments_status",
        "asset_assignments",
        ["status"],
        unique=False,
    )

    op.create_index(
        "ix_asset_assignments_assigned_at",
        "asset_assignments",
        ["assigned_at"],
        unique=False,
    )

    op.create_index(
        "ix_asset_assignments_returned_at",
        "asset_assignments",
        ["returned_at"],
        unique=False,
    )

    op.create_index(
        "ix_asset_assignment_asset_status",
        "asset_assignments",
        ["asset_id", "status"],
        unique=False,
    )

    op.create_index(
        "ix_asset_assignment_holder_status",
        "asset_assignments",
        ["assigned_to_id", "status"],
        unique=False,
    )

    op.create_index(
        "ix_asset_assignment_wo_status",
        "asset_assignments",
        ["work_order_id", "status"],
        unique=False,
    )

    # ---------------------------------------------------------
    # 4. Connect movement audit with assignment and receipt
    # ---------------------------------------------------------
    with op.batch_alter_table("tool_movements") as batch_op:
        batch_op.add_column(
            sa.Column(
                "assignment_id",
                sa.Integer(),
                nullable=True,
            )
        )

        batch_op.add_column(
            sa.Column(
                "receipt_id",
                sa.Integer(),
                nullable=True,
            )
        )

        batch_op.create_foreign_key(
            "fk_tool_movement_assignment",
            "asset_assignments",
            ["assignment_id"],
            ["id"],
            ondelete="SET NULL",
        )

        batch_op.create_foreign_key(
            "fk_tool_movement_receipt",
            "asset_assignment_receipts",
            ["receipt_id"],
            ["id"],
            ondelete="SET NULL",
        )

        batch_op.create_index(
            "ix_tool_movements_assignment_id",
            ["assignment_id"],
            unique=False,
        )

        batch_op.create_index(
            "ix_tool_movements_receipt_id",
            ["receipt_id"],
            unique=False,
        )


def downgrade():
    # ---------------------------------------------------------
    # 1. Remove movement links
    # ---------------------------------------------------------
    with op.batch_alter_table("tool_movements") as batch_op:
        batch_op.drop_index(
            "ix_tool_movements_receipt_id"
        )

        batch_op.drop_index(
            "ix_tool_movements_assignment_id"
        )

        batch_op.drop_constraint(
            "fk_tool_movement_receipt",
            type_="foreignkey",
        )

        batch_op.drop_constraint(
            "fk_tool_movement_assignment",
            type_="foreignkey",
        )

        batch_op.drop_column("receipt_id")
        batch_op.drop_column("assignment_id")

    # ---------------------------------------------------------
    # 2. Remove assignment table
    # ---------------------------------------------------------
    op.drop_index(
        "ix_asset_assignment_wo_status",
        table_name="asset_assignments",
    )

    op.drop_index(
        "ix_asset_assignment_holder_status",
        table_name="asset_assignments",
    )

    op.drop_index(
        "ix_asset_assignment_asset_status",
        table_name="asset_assignments",
    )

    op.drop_index(
        "ix_asset_assignments_returned_at",
        table_name="asset_assignments",
    )

    op.drop_index(
        "ix_asset_assignments_assigned_at",
        table_name="asset_assignments",
    )

    op.drop_index(
        "ix_asset_assignments_status",
        table_name="asset_assignments",
    )

    op.drop_index(
        "ix_asset_assignments_work_unit_id",
        table_name="asset_assignments",
    )

    op.drop_index(
        "ix_asset_assignments_work_order_id",
        table_name="asset_assignments",
    )

    op.drop_index(
        "ix_asset_assignments_assigned_to_id",
        table_name="asset_assignments",
    )

    op.drop_index(
        "ix_asset_assignments_asset_id",
        table_name="asset_assignments",
    )

    op.drop_index(
        "ix_asset_assignments_receipt_id",
        table_name="asset_assignments",
    )

    op.drop_table("asset_assignments")

    # ---------------------------------------------------------
    # 3. Remove receipt table
    # ---------------------------------------------------------
    op.drop_index(
        "ix_asset_receipt_wo_status",
        table_name="asset_assignment_receipts",
    )

    op.drop_index(
        "ix_asset_receipt_holder_date",
        table_name="asset_assignment_receipts",
    )

    op.drop_index(
        "ix_asset_assignment_receipts_assigned_at",
        table_name="asset_assignment_receipts",
    )

    op.drop_index(
        "ix_asset_assignment_receipts_status",
        table_name="asset_assignment_receipts",
    )

    op.drop_index(
        "ix_asset_assignment_receipts_work_order_id",
        table_name="asset_assignment_receipts",
    )

    op.drop_index(
        "ix_asset_assignment_receipts_assigned_to_id",
        table_name="asset_assignment_receipts",
    )

    op.drop_index(
        "ix_asset_assignment_receipts_receipt_number",
        table_name="asset_assignment_receipts",
    )

    op.drop_table("asset_assignment_receipts")

    # ---------------------------------------------------------
    # 4. Remove universal asset fields
    # ---------------------------------------------------------
    with op.batch_alter_table("tool_assets") as batch_op:
        batch_op.drop_index(
            "ix_tool_assets_status_holder"
        )

        batch_op.drop_index(
            "ix_tool_assets_current_holder_name"
        )

        batch_op.drop_index(
            "ix_tool_assets_current_holder_id"
        )

        batch_op.drop_index(
            "ix_tool_assets_asset_type"
        )

        batch_op.drop_constraint(
            "fk_tool_assets_current_holder_id_user",
            type_="foreignkey",
        )

        batch_op.drop_column("current_holder_name")
        batch_op.drop_column("current_holder_id")
        batch_op.drop_column("asset_type")