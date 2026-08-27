from app import app
from extensions import db

from alembic.migration import MigrationContext
from alembic.autogenerate import compare_metadata


with app.app_context():

    print("=" * 70)
    print("ALEMBIC AUTOGENERATE DIRECT CHECK")
    print("=" * 70)

    with db.engine.connect() as connection:

        context = MigrationContext.configure(
            connection
        )

        diffs = compare_metadata(
            context,
            db.metadata,
        )

    print("DIFF COUNT:", len(diffs))
    print()

    for index, diff in enumerate(diffs, 1):
        print(f"{index}. {diff}")

    print()
    print("=" * 70)

    expected = {
        "appliance_issue",
        "appliance_issue_line",
        "appliance_movement",
    }

    found = set()

    for diff in diffs:
        if not diff:
            continue

        if diff[0] == "add_table":
            table = diff[1]
            found.add(table.name)

    print("EXPECTED NEW TABLES FOUND:")

    for name in sorted(expected):
        print(
            f"  {name}:",
            name in found,
        )

    print("=" * 70)
