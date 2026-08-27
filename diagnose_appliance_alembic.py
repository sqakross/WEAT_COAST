from app import app
from extensions import db
from sqlalchemy import inspect

with app.app_context():

    migrate_ext = app.extensions.get("migrate")

    print("=" * 70)
    print("ALEMBIC / DATABASE DIAGNOSTIC")
    print("=" * 70)

    print("APP:", app.name)
    print("DB URL:", db.engine.url)
    print(
        "CONFIG URI:",
        app.config.get("SQLALCHEMY_DATABASE_URI"),
    )

    print()
    print("DB METADATA APPLIANCE TABLES:")

    for name in sorted(
        x
        for x in db.metadata.tables.keys()
        if x.startswith("appliance_")
    ):
        print(" ", name)

    print()
    print("MIGRATE EXTENSION:")

    if migrate_ext is None:
        print("  ERROR: migrate extension not found")
    else:
        migrate_db = migrate_ext.db

        print(
            "  same db object:",
            migrate_db is db,
        )

        print(
            "  migrate metadata same object:",
            migrate_db.metadata is db.metadata,
        )

        print("  MIGRATE METADATA APPLIANCE TABLES:")

        for name in sorted(
            x
            for x in migrate_db.metadata.tables.keys()
            if x.startswith("appliance_")
        ):
            print("   ", name)

    print()
    print("PHYSICAL DATABASE TABLES:")

    inspector = inspect(db.engine)

    for name in sorted(
        x
        for x in inspector.get_table_names()
        if x.startswith("appliance_")
    ):
        print(" ", name)

    print()
    print("EXPECTED NEW TABLES:")

    for name in [
        "appliance_issue",
        "appliance_issue_line",
        "appliance_movement",
    ]:
        print(
            f"  {name}:",
            inspector.has_table(name),
        )

    print("=" * 70)
