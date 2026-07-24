"""Add supplier statement line components

Revision ID: 9f2c1a7b4d10
Revises: 43d42458ba81
Create Date: 2026-07-22

"""

from alembic import op
import sqlalchemy as sa


revision = "9f2c1a7b4d10"
down_revision = "739b7f1d176a"
depends_on = None


def upgrade():
    op.create_table(
        "supplier_statement_line_component",

        sa.Column(
            "id",
            sa.Integer(),
            primary_key=True,
        ),

        sa.Column(
            "statement_line_id",
            sa.Integer(),
            nullable=False,
        ),

        sa.Column(
            "amount",
            sa.Float(),
            nullable=False,
            server_default="0",
        ),

        sa.Column(
            "matched_issued_part_record_id",
            sa.Integer(),
            nullable=True,
        ),

        sa.Column(
            "note",
            sa.String(length=255),
            nullable=True,
        ),

        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
        ),

        sa.Column(
            "created_by",
            sa.Integer(),
            nullable=True,
        ),

        sa.ForeignKeyConstraint(
            ["statement_line_id"],
            ["supplier_statement_line.id"],
            ondelete="CASCADE",
        ),

        sa.ForeignKeyConstraint(
            ["matched_issued_part_record_id"],
            ["issued_part_record.id"],
            ondelete="SET NULL",
        ),
    )

    op.create_index(
        "ix_sslc_statement_line_id",
        "supplier_statement_line_component",
        ["statement_line_id"],
        unique=False,
    )

    op.create_index(
        "ix_sslc_matched_record_id",
        "supplier_statement_line_component",
        ["matched_issued_part_record_id"],
        unique=False,
    )

    op.create_index(
        "ix_supplier_statement_line_component_created_by",
        "supplier_statement_line_component",
        ["created_by"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        "ix_supplier_statement_line_component_created_by",
        table_name="supplier_statement_line_component",
    )

    op.drop_index(
        "ix_sslc_matched_record_id",
        table_name="supplier_statement_line_component",
    )

    op.drop_index(
        "ix_sslc_statement_line_id",
        table_name="supplier_statement_line_component",
    )

    op.drop_table(
        "supplier_statement_line_component"
    )