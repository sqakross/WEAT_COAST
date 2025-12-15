from alembic import op
import sqlalchemy as sa

revision = "20251214_add_po_and_invoice_fields"
down_revision = "20251208_add_insurance_supplied_flags"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    # ---------- work_orders: customer_po ----------
    if "work_orders" in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns("work_orders")]
        if "customer_po" not in cols:
            op.add_column(
                "work_orders",
                sa.Column("customer_po", sa.String(length=64), nullable=True),
            )
            op.create_index(
                "ix_work_orders_customer_po",
                "work_orders",
                ["customer_po"],
                unique=False,
            )

    # ---------- work_order_parts: invoice_number ----------
    if "work_order_parts" in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns("work_order_parts")]
        if "invoice_number" not in cols:
            op.add_column(
                "work_order_parts",
                sa.Column("invoice_number", sa.String(length=32), nullable=True),
            )
            op.create_index(
                "ix_work_order_parts_invoice_number",
                "work_order_parts",
                ["invoice_number"],
                unique=False,
            )


def downgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    # ---------- work_order_parts ----------
    if "work_order_parts" in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns("work_order_parts")]
        if "invoice_number" in cols:
            op.drop_index(
                "ix_work_order_parts_invoice_number",
                table_name="work_order_parts",
            )
            op.drop_column("work_order_parts", "invoice_number")

    # ---------- work_orders ----------
    if "work_orders" in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns("work_orders")]
        if "customer_po" in cols:
            op.drop_index(
                "ix_work_orders_customer_po",
                table_name="work_orders",
            )
            op.drop_column("work_orders", "customer_po")
