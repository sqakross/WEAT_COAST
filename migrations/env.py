# migrations/env.py
from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool
from flask import current_app

# Alembic config (Flask-Migrate прокинет его сам)
config = context.config

# Логи Alembic — только если ini реально есть
if config.config_file_name and os.path.exists(config.config_file_name):
    fileConfig(config.config_file_name)

# ВАЖНО: берём metadata из Flask-Migrate
target_metadata = current_app.extensions["migrate"].db.metadata

def run_migrations_offline() -> None:
    """Запуск миграций без подключения к БД."""
    url = current_app.config.get("SQLALCHEMY_DATABASE_URI")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Запуск миграций с подключением к БД."""
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = current_app.config.get("SQLALCHEMY_DATABASE_URI")

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
