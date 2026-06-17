"""add tools system

Revision ID: 50ce3abeb7be
Revises: 20260414_link_supplier_return_to_reports
Create Date: 2026-06-16 13:07:48.094818

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "50ce3abeb7be"
down_revision = "20260414_link_supplier_return_to_reports"
branch_labels = None
depends_on = None


def upgrade():
    # --- Tools master table ---
    op.create_table(
        "tool_assets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tool_number", sa.String(length=80), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("serial_number", sa.String(length=120), nullable=True),
        sa.Column("status", sa.String(length=30), nullable=False, server_default="available"),
        sa.Column("condition", sa.String(length=40), nullable=False, server_default="good"),
        sa.Column("location", sa.String(length=80), nullable=False, server_default="TOOLS"),
        sa.Column("current_technician_id", sa.Integer(), nullable=True),
        sa.Column("current_technician_name", sa.String(length=80), nullable=True),
        sa.Column("current_work_order_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["current_technician_id"], ["user.id"]),
        sa.ForeignKeyConstraint(["current_work_order_id"], ["work_orders.id"]),
    )
    op.create_index("ix_tool_assets_tool_number", "tool_assets", ["tool_number"], unique=True)
    op.create_index("ix_tool_assets_serial_number", "tool_assets", ["serial_number"], unique=False)
    op.create_index("ix_tool_assets_status", "tool_assets", ["status"], unique=False)
    op.create_index("ix_tool_assets_current_technician_id", "tool_assets", ["current_technician_id"], unique=False)
    op.create_index("ix_tool_assets_current_technician_name", "tool_assets", ["current_technician_name"], unique=False)
    op.create_index("ix_tool_assets_current_work_order_id", "tool_assets", ["current_work_order_id"], unique=False)

    # --- Tools history table ---
    op.create_table(
        "tool_movements",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tool_id", sa.Integer(), nullable=False),
        sa.Column("work_order_id", sa.Integer(), nullable=True),
        sa.Column("technician_id", sa.Integer(), nullable=True),
        sa.Column("technician_name", sa.String(length=80), nullable=True),
        sa.Column("action", sa.String(length=30), nullable=False),
        sa.Column("from_status", sa.String(length=30), nullable=True),
        sa.Column("to_status", sa.String(length=30), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("actor_user_id", sa.Integer(), nullable=True),
        sa.Column("actor_username", sa.String(length=80), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["tool_id"], ["tool_assets.id"]),
        sa.ForeignKeyConstraint(["work_order_id"], ["work_orders.id"]),
        sa.ForeignKeyConstraint(["technician_id"], ["user.id"]),
        sa.ForeignKeyConstraint(["actor_user_id"], ["user.id"]),
    )
    op.create_index("ix_tool_movements_tool_id", "tool_movements", ["tool_id"], unique=False)
    op.create_index("ix_tool_movements_work_order_id", "tool_movements", ["work_order_id"], unique=False)
    op.create_index("ix_tool_movements_technician_id", "tool_movements", ["technician_id"], unique=False)
    op.create_index("ix_tool_movements_technician_name", "tool_movements", ["technician_name"], unique=False)
    op.create_index("ix_tool_movements_action", "tool_movements", ["action"], unique=False)

    # --- Link WO rows to tools / mark row type ---
    with op.batch_alter_table("work_order_parts") as batch_op:
        batch_op.add_column(sa.Column("item_type", sa.String(length=20), nullable=False, server_default="part"))
        batch_op.add_column(sa.Column("tool_asset_id", sa.Integer(), nullable=True))
        batch_op.create_index("ix_work_order_parts_item_type", ["item_type"], unique=False)
        batch_op.create_index("ix_work_order_parts_tool_asset_id", ["tool_asset_id"], unique=False)


def downgrade():
    with op.batch_alter_table("work_order_parts") as batch_op:
        batch_op.drop_index("ix_work_order_parts_tool_asset_id")
        batch_op.drop_index("ix_work_order_parts_item_type")
        batch_op.drop_column("tool_asset_id")
        batch_op.drop_column("item_type")

    op.drop_index("ix_tool_movements_action", table_name="tool_movements")
    op.drop_index("ix_tool_movements_technician_name", table_name="tool_movements")
    op.drop_index("ix_tool_movements_technician_id", table_name="tool_movements")
    op.drop_index("ix_tool_movements_work_order_id", table_name="tool_movements")
    op.drop_index("ix_tool_movements_tool_id", table_name="tool_movements")
    op.drop_table("tool_movements")

    op.drop_index("ix_tool_assets_current_work_order_id", table_name="tool_assets")
    op.drop_index("ix_tool_assets_current_technician_name", table_name="tool_assets")
    op.drop_index("ix_tool_assets_current_technician_id", table_name="tool_assets")
    op.drop_index("ix_tool_assets_status", table_name="tool_assets")
    op.drop_index("ix_tool_assets_serial_number", table_name="tool_assets")
    op.drop_index("ix_tool_assets_tool_number", table_name="tool_assets")
    op.drop_table("tool_assets")
