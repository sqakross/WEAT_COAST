from alembic import op
import sqlalchemy as sa

revision = "20251124_add_tech_note_to_supplier_return_item"
down_revision = "20251124_supplier_return_tech_note"  # ← твоя предыдущая ревизия
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    tables = insp.get_table_names()
    if "supplier_return_item" in tables:
        cols = [c["name"] for c in insp.get_columns("supplier_return_item")]
        if "tech_note" not in cols:
            op.add_column(
                "supplier_return_item",
                sa.Column("tech_note", sa.String(length=255), nullable=True),
            )


def downgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    tables = insp.get_table_names()
    if "supplier_return_item" in tables:
        cols = [c["name"] for c in insp.get_columns("supplier_return_item")]
        if "tech_note" in cols:
            op.drop_column("supplier_return_item", "tech_note")
