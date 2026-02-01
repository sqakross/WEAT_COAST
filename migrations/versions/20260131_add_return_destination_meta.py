from alembic import op
import sqlalchemy as sa

revision = "20260131_add_return_destination_meta"
down_revision = "20260103_add_wo_audit_created_updated_by"
branch_labels = None
depends_on = None


def _table_exists(insp, name: str) -> bool:
    try:
        return name in insp.get_table_names()
    except Exception:
        return False


def _col_exists(insp, table: str, col: str) -> bool:
    try:
        return col in [c["name"] for c in insp.get_columns(table)]
    except Exception:
        return False


def _index_exists(insp, table: str, index_name: str) -> bool:
    try:
        idxs = insp.get_indexes(table) or []
        return any((i.get("name") == index_name) for i in idxs)
    except Exception:
        return False


def upgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    # 1) create return_destination table
    if not _table_exists(insp, "return_destination"):
        op.create_table(
            "return_destination",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("name", sa.String(length=120), nullable=False),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("1")),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        )
        op.create_index("ix_return_destination_name", "return_destination", ["name"], unique=True)
        op.create_index("ix_return_destination_is_active", "return_destination", ["is_active"], unique=False)

        # seed defaults
        op.execute(sa.text("""
            INSERT OR IGNORE INTO return_destination(name, is_active)
            VALUES
              ('Marcone', 1),
              ('ReliableParts', 1),
              ('Amazon', 1),
              ('Coast', 1),
              ('Encompass', 1),
              ('Other', 1)
        """))

    # 2) add columns to issued_part_record
    if _table_exists(insp, "issued_part_record"):
        if not _col_exists(insp, "issued_part_record", "return_to"):
            op.add_column("issued_part_record", sa.Column("return_to", sa.String(length=16), nullable=True))
        if not _index_exists(insp, "issued_part_record", "ix_issued_part_record_return_to"):
            op.create_index("ix_issued_part_record_return_to", "issued_part_record", ["return_to"], unique=False)

        if not _col_exists(insp, "issued_part_record", "return_destination_id"):
            op.add_column("issued_part_record", sa.Column("return_destination_id", sa.Integer(), nullable=True))
        if not _index_exists(insp, "issued_part_record", "ix_issued_part_record_return_destination_id"):
            op.create_index("ix_issued_part_record_return_destination_id", "issued_part_record", ["return_destination_id"], unique=False)


def downgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    if _table_exists(insp, "issued_part_record"):
        if _index_exists(insp, "issued_part_record", "ix_issued_part_record_return_destination_id"):
            op.drop_index("ix_issued_part_record_return_destination_id", table_name="issued_part_record")
        if _col_exists(insp, "issued_part_record", "return_destination_id"):
            op.drop_column("issued_part_record", "return_destination_id")

        if _index_exists(insp, "issued_part_record", "ix_issued_part_record_return_to"):
            op.drop_index("ix_issued_part_record_return_to", table_name="issued_part_record")
        if _col_exists(insp, "issued_part_record", "return_to"):
            op.drop_column("issued_part_record", "return_to")

    if _table_exists(insp, "return_destination"):
        # drop indexes first
        try:
            op.drop_index("ix_return_destination_is_active", table_name="return_destination")
        except Exception:
            pass
        try:
            op.drop_index("ix_return_destination_name", table_name="return_destination")
        except Exception:
            pass
        op.drop_table("return_destination")
