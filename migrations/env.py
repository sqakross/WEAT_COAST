from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from flask import current_app
from sqlalchemy import engine_from_config, pool


config = context.config


if (
    config.config_file_name
    and os.path.exists(config.config_file_name)
):
    fileConfig(config.config_file_name)


target_metadata = (
    current_app.extensions["migrate"].db.metadata
)


# ============================================================
# Alembic-managed schema
#
# Historical ERP tables existed before schema management became
# strict and currently contain intentional/live schema drift.
#
# Do NOT let Alembic autogenerate destructive "catch-up"
# migrations against those legacy tables.
#
# Tables listed here are under strict Alembic ownership.
#
# As legacy tables are reconciled in the future, they may be
# added here individually.
# ============================================================

ALEMBIC_MANAGED_TABLES = {
    # Access Control Foundation
    "warehouse",
    "user_warehouse_access",
    "permission",
    "user_permission",
    "user_permission_audit",

    # Appliance Inventory
    "appliance_category",
    "appliance_receiving",
    "appliance_receiving_line",
    "appliance_unit",
}


def include_object(
    object_,
    name,
    type_,
    reflected,
    compare_to,
):
    """
    Restrict autogenerate to schema explicitly owned by Alembic.

    Migration execution itself is NOT restricted.
    Existing migration files continue to execute normally.

    This callback affects autogenerate comparison only.
    """

    if type_ == "table":
        return name in ALEMBIC_MANAGED_TABLES

    table = getattr(object_, "table", None)

    if table is not None:
        return table.name in ALEMBIC_MANAGED_TABLES

    compare_table = getattr(compare_to, "table", None)

    if compare_table is not None:
        return compare_table.name in ALEMBIC_MANAGED_TABLES

    return True


def configure_context(**kwargs):
    context.configure(
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=False,
        include_object=include_object,
        **kwargs,
    )


def run_migrations_offline() -> None:
    """Run migrations without a live database connection."""

    url = current_app.config.get(
        "SQLALCHEMY_DATABASE_URI"
    )

    configure_context(
        url=url,
        literal_binds=True,
        dialect_opts={
            "paramstyle": "named",
        },
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations using a live database connection."""

    configuration = (
        config.get_section(
            config.config_ini_section
        )
        or {}
    )

    configuration["sqlalchemy.url"] = (
        current_app.config.get(
            "SQLALCHEMY_DATABASE_URI"
        )
    )

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        configure_context(
            connection=connection,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
