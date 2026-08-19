from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext

from app import app
from extensions import db


with app.app_context():
    with db.engine.connect() as connection:
        context = MigrationContext.configure(
            connection,
            opts={
                "compare_type": True,
                "compare_server_default": False,
            },
        )

        diffs = compare_metadata(
            context,
            db.metadata,
        )

        print()
        print("=" * 80)
        print("ALEMBIC SCHEMA DIFF")
        print("=" * 80)
        print("Total differences:", len(diffs))
        print()

        if not diffs:
            print("DATABASE AND MODELS ARE IN SYNC")
        else:
            for i, diff in enumerate(diffs, 1):
                print(f"[{i}] {diff}")
