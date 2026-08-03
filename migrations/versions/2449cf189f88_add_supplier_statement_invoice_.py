"""add supplier statement invoice components

Revision ID: 6c4e2f8a1d90
Revises: PASTE_CURRENT_HEAD_HERE
Create Date: 2026-08-03
"""

from alembic import op
import sqlalchemy as sa


revision = "6c4e2f8a1d90"
down_revision = "8e50ed0a9440"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "supplier_statement_invoice_component",

        sa.Column(
            "id",
            sa.Integer(),
            nullable=False,
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
            "matched_goods_receipt_line_id",
            sa.Integer(),
            nullable=True,
        ),

        sa.Column(
            "note",
            sa.String(length=500),
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
            ["matched_goods_receipt_line_id"],
            ["goods_receipt_lines.id"],
            ondelete="SET NULL",
        ),

        sa.ForeignKeyConstraint(
            ["created_by"],
            ["user.id"],
            ondelete="SET NULL",
        ),

        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_ssic_statement_line_id",
        "supplier_statement_invoice_component",
        ["statement_line_id"],
        unique=False,
    )

    op.create_index(
        "ix_ssic_receipt_line_id",
        "supplier_statement_invoice_component",
        ["matched_goods_receipt_line_id"],
        unique=False,
    )

    op.create_index(
        op.f(
            "ix_supplier_statement_invoice_component_statement_line_id"
        ),
        "supplier_statement_invoice_component",
        ["statement_line_id"],
        unique=False,
    )

    op.create_index(
        op.f(
            "ix_supplier_statement_invoice_component_"
            "matched_goods_receipt_line_id"
        ),
        "supplier_statement_invoice_component",
        ["matched_goods_receipt_line_id"],
        unique=False,
    )

    op.create_index(
        op.f(
            "ix_supplier_statement_invoice_component_created_by"
        ),
        "supplier_statement_invoice_component",
        ["created_by"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        op.f(
            "ix_supplier_statement_invoice_component_created_by"
        ),
        table_name="supplier_statement_invoice_component",
    )

    op.drop_index(
        op.f(
            "ix_supplier_statement_invoice_component_"
            "matched_goods_receipt_line_id"
        ),
        table_name="supplier_statement_invoice_component",
    )

    op.drop_index(
        op.f(
            "ix_supplier_statement_invoice_component_statement_line_id"
        ),
        table_name="supplier_statement_invoice_component",
    )

    op.drop_index(
        "ix_ssic_receipt_line_id",
        table_name="supplier_statement_invoice_component",
    )

    op.drop_index(
        "ix_ssic_statement_line_id",
        table_name="supplier_statement_invoice_component",
    )

    op.drop_table(
        "supplier_statement_invoice_component"
    )