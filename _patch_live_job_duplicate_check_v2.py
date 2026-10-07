from pathlib import Path
from datetime import datetime
import shutil
import sys
import re

FILE = Path(r"templates\wo_form_units.html")

if not FILE.exists():
    print(f"STOP: file not found: {FILE}")
    sys.exit(1)

text = FILE.read_text(encoding="utf-8")

print("=" * 90)
print("PATCH: LIVE JOB DUPLICATE CHECK V2")
print("=" * 90)

# ----------------------------------------------------------------------
# Locate handleJobInput only
# ----------------------------------------------------------------------

func_match = re.search(
    r'function\s+handleJobInput\s*\(\s*\)\s*\{',
    text
)

if not func_match:
    print("STOP: handleJobInput() not found.")
    print("Original file NOT changed.")
    sys.exit(1)

# Find next jobInput.addEventListener after the function.
listener_pos = text.find(
    "jobInput.addEventListener",
    func_match.end()
)

if listener_pos == -1:
    print("STOP: jobInput.addEventListener not found after handleJobInput().")
    print("Original file NOT changed.")
    sys.exit(1)

block = text[func_match.start():listener_pos]

print(f"handleJobInput block length: {len(block)} chars")

# ----------------------------------------------------------------------
# Safety checks
# ----------------------------------------------------------------------

if "LIVE JOB CHECK" in block:
    print("STOP: LIVE JOB CHECK already appears installed.")
    print("No changes made.")
    sys.exit(0)

target = "setButtonsEnabled(true);"

target_count = block.count(target)

print(f"setButtonsEnabled(true) inside handleJobInput: {target_count}")

if target_count != 1:
    print("STOP: expected exactly 1 insertion point.")
    print("Original file NOT changed.")
    sys.exit(1)

# ----------------------------------------------------------------------
# Backup BEFORE modification
# ----------------------------------------------------------------------

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

backup_dir = (
    Path("_patch_backups")
    / f"live_job_duplicate_check_v2_{stamp}"
)

backup_dir.mkdir(parents=True, exist_ok=True)

backup_file = backup_dir / FILE.name

shutil.copy2(FILE, backup_file)

print(f"Backup: {backup_file}")

# ----------------------------------------------------------------------
# Insert debounce
# ----------------------------------------------------------------------

INSERT = """setButtonsEnabled(true);

  /*
   * LIVE JOB CHECK
   *
   * Do not request on every keystroke.
   * Wait 800 ms after the last input change,
   * then use the existing reserveNow().
   */
  const raw = (jobInput.value || "").trim();

  if (!hasAnyToken(raw)) {
    setIdle();
    return;
  }

  reserveTimer = setTimeout(() => {
    reserveTimer = null;
    reserveNow();
  }, 800);"""

new_block = block.replace(
    target,
    INSERT,
    1
)

patched = (
    text[:func_match.start()]
    + new_block
    + text[listener_pos:]
)

# ----------------------------------------------------------------------
# Pre-write verification
# ----------------------------------------------------------------------

checks = {
    "LIVE JOB CHECK inserted":
        "LIVE JOB CHECK" in new_block,

    "debounce timer inserted":
        "reserveTimer = setTimeout" in new_block,

    "800 ms debounce":
        "}, 800);" in new_block,

    "reserveNow called":
        "reserveNow();" in new_block,

    "empty input protected":
        "if (!hasAnyToken(raw))" in new_block,

    "existing-WO handling preserved":
        'if (data.status === "exists")' in patched,

    "locked handling preserved":
        'if (data.status === "locked")' in patched,

    "reserved handling preserved":
        'data.status === "reserved"' in patched,

    "blur listener preserved":
        "jobInput.addEventListener('blur'" in patched,

    "final exists protection preserved":
        'lastStatus === "exists"' in patched,
}

print()
print("PRE-WRITE VERIFICATION")
print("-" * 90)

failed = []

for name, ok in checks.items():
    print(("OK   " if ok else "FAIL ") + name)
    if not ok:
        failed.append(name)

if failed:
    print()
    print("PATCH FAILED BEFORE WRITE.")
    print("Original file NOT changed.")
    sys.exit(1)

# ----------------------------------------------------------------------
# Write
# ----------------------------------------------------------------------

FILE.write_text(patched, encoding="utf-8")

# ----------------------------------------------------------------------
# Read-back verification
# ----------------------------------------------------------------------

saved = FILE.read_text(encoding="utf-8")

post_checks = {
    "exactly one LIVE JOB CHECK":
        saved.count("LIVE JOB CHECK") == 1,

    "exactly one new debounce":
        saved.count("reserveTimer = setTimeout") == 1,

    "800 ms present":
        "}, 800);" in saved,

    "exists handling present":
        'if (data.status === "exists")' in saved,

    "locked handling present":
        'if (data.status === "locked")' in saved,

    "reserved handling present":
        'data.status === "reserved"' in saved,

    "blur listener present":
        "jobInput.addEventListener('blur'" in saved,
}

print()
print("READ-BACK VERIFICATION")
print("-" * 90)

post_failed = []

for name, ok in post_checks.items():
    print(("OK   " if ok else "FAIL ") + name)

    if not ok:
        post_failed.append(name)

if post_failed:
    shutil.copy2(backup_file, FILE)

    print()
    print("PATCH FAILED - ORIGINAL RESTORED")
    print("Failed checks:")

    for name in post_failed:
        print(f"  - {name}")

    sys.exit(1)

print()
print("=" * 90)
print("PATCH OK")
print("=" * 90)
print(f"Changed: {FILE}")
print(f"Backup : {backup_file}")
print()
print("Backend/routes.py : NOT CHANGED")
print("Database          : NOT CHANGED")
print("Reservation API   : NOT CHANGED")
print("Appliance History : NOT CHANGED")
print("Debounce           : 800 ms")
print("=" * 90)
