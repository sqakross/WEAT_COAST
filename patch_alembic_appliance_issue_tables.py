from pathlib import Path
from datetime import datetime

path = Path("migrations/env.py")

if not path.exists():
    raise SystemExit("ERROR: migrations/env.py not found")

text = path.read_text(encoding="utf-8")

backup = Path(
    f"migrations/env.py.before_appliance_issue_tables_"
    f"{datetime.now():%Y%m%d_%H%M%S}.bak"
)

backup.write_text(
    text,
    encoding="utf-8",
)

old = '''    # Appliance Inventory
    "appliance_category",
    "appliance_receiving",
    "appliance_receiving_line",
    "appliance_unit",
}'''

new = '''    # Appliance Inventory
    "appliance_category",
    "appliance_receiving",
    "appliance_receiving_line",
    "appliance_unit",

    # Appliance Issue / Movement
    "appliance_issue",
    "appliance_issue_line",
    "appliance_movement",
}'''

if old not in text:
    raise SystemExit(
        "ERROR: Appliance Inventory managed-table block not found"
    )

text = text.replace(
    old,
    new,
    1,
)

path.write_text(
    text,
    encoding="utf-8",
)

print("OK: migrations/env.py updated")
print("Added Alembic-managed tables:")
print("  appliance_issue")
print("  appliance_issue_line")
print("  appliance_movement")
print("Backup:", backup)
