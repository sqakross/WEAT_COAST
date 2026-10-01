from pathlib import Path
from datetime import datetime
import shutil
import py_compile
import sys

P = Path(r"inventory\routes.py")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_dir = Path(r"_patch_backups") / f"delete_zero_return_{stamp}"
backup_dir.mkdir(parents=True, exist_ok=True)
backup = backup_dir / "routes.py"

text = P.read_text(encoding="utf-8")
shutil.copy2(P, backup)

print("=" * 78)
print("PATCH: DELETE ZERO-QTY RETURN")
print("=" * 78)
print("Target :", P)
print("Backup :", backup)
print()

OLD = '''    def _is_return_row(r):
        """A row is a 'return' when its quantity is negative."""
        return (getattr(r, 'quantity', 0) or 0) < 0
'''

NEW = '''    def _is_return_row(r):
        """
        Persistent RETURN detection.

        A RETURN remains a RETURN even when its quantity was edited
        from a negative value to 0.  Quantity alone therefore cannot
        be used to identify the record.
        """
        qty = int(getattr(r, "quantity", 0) or 0)

        ref = str(
            getattr(r, "reference_job", "") or ""
        ).strip().upper()

        cost_source = str(
            getattr(r, "cost_source", "") or ""
        ).strip().upper()

        return (
            qty < 0
            or ref.startswith("RETURN")
            or cost_source == "BASE_RETURN"
        )
'''

count = text.count(OLD)

print("Exact helper blocks found:", count)

if count != 1:
    print()
    print("STOP: expected exactly 1 old _is_return_row helper.")
    print("NO CHANGES WRITTEN.")
    sys.exit(1)

new_text = text.replace(OLD, NEW, 1)

# ------------------------------------------------------------
# Static verification before writing
# ------------------------------------------------------------

checks = {
    "persistent RETURN helper": 'ref.startswith("RETURN")' in new_text,
    "BASE_RETURN detection": 'cost_source == "BASE_RETURN"' in new_text,
    "negative RETURN preserved": "qty < 0" in new_text,
    "DELETE RETURN block preserved": "if do_delete_ret:" in new_text,
    "zero RETURN stock safe": "if r.part and (r.quantity or 0) < 0:" in new_text,
    "delete record preserved": "db.session.delete(r)" in new_text,
}

print()
print("STATIC VERIFICATION")
print("-" * 78)

for name, ok in checks.items():
    print(("OK   " if ok else "FAIL "), name)

if not all(checks.values()):
    print()
    print("STOP: verification failed.")
    print("NO CHANGES WRITTEN.")
    sys.exit(1)

P.write_text(new_text, encoding="utf-8")

# ------------------------------------------------------------
# Compile
# ------------------------------------------------------------

try:
    py_compile.compile(str(P), doraise=True)
except Exception as e:
    print()
    print("PY_COMPILE FAILED:", e)
    shutil.copy2(backup, P)
    print("Original routes.py restored.")
    sys.exit(1)

print()
print("PY_COMPILE OK")

# ------------------------------------------------------------
# Readback
# ------------------------------------------------------------

saved = P.read_text(encoding="utf-8")

readback = {
    "RETURN reference detection persisted":
        'ref.startswith("RETURN")' in saved,

    "BASE_RETURN detection persisted":
        'cost_source == "BASE_RETURN"' in saved,

    "stock guard still persisted":
        "if r.part and (r.quantity or 0) < 0:" in saved,
}

print()
print("READBACK VERIFICATION")
print("-" * 78)

for name, ok in readback.items():
    print(("OK   " if ok else "FAIL "), name)

if not all(readback.values()):
    print()
    print("READBACK FAILED - restoring backup.")
    shutil.copy2(backup, P)
    sys.exit(1)

print()
print("=" * 78)
print("PATCH INSTALLED AND VERIFIED")
print("=" * 78)

print("""
Expected behavior:

RETURN -1:
  Delete Return -> record deleted
  stock reduced according to existing delete-return behavior

RETURN 0:
  Delete Return -> record deleted
  stock DOES NOT CHANGE
  empty return invoice/batch is removed

Normal ISSUE:
  not treated as RETURN

RETURN identification:
  quantity < 0
  OR reference_job starts with RETURN
  OR cost_source == BASE_RETURN
""")

print("Backup:", backup)
