from pathlib import Path

path = Path("models.py")
text = path.read_text(encoding="utf-8")

replacements = [
    (
'''        backref=db.backref(
            "warehouse_accesses",
            lazy="selectin",
            cascade="all, delete-orphan",
        ),''',
'''        backref=db.backref(
            "warehouse_accesses",
            lazy="select",
            cascade="all, delete-orphan",
        ),'''
    ),
    (
'''        backref=db.backref(
            "user_accesses",
            lazy="selectin",
        ),''',
'''        backref=db.backref(
            "user_accesses",
            lazy="select",
        ),'''
    ),
    (
'''        backref=db.backref(
            "permission_grants",
            lazy="selectin",
            cascade="all, delete-orphan",
        ),''',
'''        backref=db.backref(
            "permission_grants",
            lazy="select",
            cascade="all, delete-orphan",
        ),'''
    ),
    (
'''        backref=db.backref(
            "user_grants",
            lazy="selectin",
        ),''',
'''        backref=db.backref(
            "user_grants",
            lazy="select",
        ),'''
    ),
]

for old, new in replacements:
    if old not in text:
        raise SystemExit(
            "ERROR: expected relationship block not found:\n" + old
        )

    text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")

print("OK: access relationship loading fixed")
print("Changed:")
print("  User.warehouse_accesses: selectin -> select")
print("  Warehouse.user_accesses: selectin -> select")
print("  User.permission_grants: selectin -> select")
print("  Permission.user_grants: selectin -> select")
