"""add job_reservation table

Revision ID: 20260201_add_job_reservation
Revises: 20260131_add_return_destination_meta
Create Date: 2026-02-01
"""

from alembic import op
import sqlalchemy as sa

revision = "20260201_add_job_reservation"
down_revision = "20260131_add_return_destination_meta"
branch_labels = None
depends_on = None


def _table_exists(insp, name: str) -> bool:
    try:
        return name in insp.get_table_names()
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

    # 1) create job_reservation table
    if not _table_exists(insp, "job_reservation"):
        op.create_table(
            "job_reservation",
            sa.Column("id", sa.Integer(), primary_key=True),

            sa.Column("job_token", sa.String(length=64), nullable=False),

            sa.Column("holder_user_id", sa.Integer(), nullable=True),
            sa.Column("holder_username", sa.String(length=64), nullable=True),

            sa.Column("expires_at", sa.DateTime(), nullable=False),

            # SQLite-friendly timestamps
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),

            sa.Column("work_order_id", sa.Integer(), nullable=True),

            sa.ForeignKeyConstraint(["holder_user_id"], ["user.id"]),
            sa.ForeignKeyConstraint(["work_order_id"], ["work_orders.id"]),
        )

    # 2) indexes (idempotent)
    if _table_exists(insp, "job_reservation"):
        if not _index_exists(insp, "job_reservation", "ix_job_reservation_job_token"):
            op.create_index("ix_job_reservation_job_token", "job_reservation", ["job_token"], unique=False)

        if not _index_exists(insp, "job_reservation", "ix_job_reservation_expires_at"):
            op.create_index("ix_job_reservation_expires_at", "job_reservation", ["expires_at"], unique=False)

        # SQLite-safe UNIQUE: unique index instead of constraint
        if not _index_exists(insp, "job_reservation", "uq_job_reservation_token"):
            op.create_index("uq_job_reservation_token", "job_reservation", ["job_token"], unique=True)


def downgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    if _table_exists(insp, "job_reservation"):
        # drop unique index first (if exists)
        try:
            if _index_exists(insp, "job_reservation", "uq_job_reservation_token"):
                op.drop_index("uq_job_reservation_token", table_name="job_reservation")
        except Exception:
            pass

        try:
            if _index_exists(insp, "job_reservation", "ix_job_reservation_expires_at"):
                op.drop_index("ix_job_reservation_expires_at", table_name="job_reservation")
        except Exception:
            pass

        try:
            if _index_exists(insp, "job_reservation", "ix_job_reservation_job_token"):
                op.drop_index("ix_job_reservation_job_token", table_name="job_reservation")
        except Exception:
            pass

        op.drop_table("job_reservation")
