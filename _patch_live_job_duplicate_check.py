from pathlib import Path
from datetime import datetime
import shutil
import sys

FILE = Path(r"templates\wo_form_units.html")

if not FILE.exists():
    print(f"STOP: file not found: {FILE}")
    sys.exit(1)

text = FILE.read_text(encoding="utf-8")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_dir = Path("_patch_backups") / f"live_job_duplicate_check_{stamp}"
backup_dir.mkdir(parents=True, exist_ok=True)

backup_file = backup_dir / "wo_form_units.html"
shutil.copy2(FILE, backup_file)

print("=" * 90)
print("PATCH: LIVE JOB DUPLICATE CHECK")
print("=" * 90)
print(f"Backup: {backup_file}")
print()

OLD = """  function handleJobInput(){
  if (IS_RO) return;

  /*
    * Во время набора НЕ обращаемся к серверу.
    * Только сбрасываем старый статус.
   */
  cancelPendingReserve();

  lastStatus = "typing";
  lastSent = "";

  jobInput.classList.remove('is-invalid');

  statusLine.className = "form-text text-muted";
  statusLine.textContent = "Finish entering job number(s) to check.";

  setButtonsEnabled(true);
}

jobInput.addEventListener(
  'input',
  handleJobInput
);
"""

NEW = """  function handleJobInput(){
  if (IS_RO) return;

  /*
   * LIVE JOB CHECK
   *
   * Не отправляем request на каждую клавишу.
   * После последнего изменения ждём 800 ms,
   * затем используем существующий reserveNow().
   *
   * reserveNow() уже различает:
   *   exists   -> saved Work Order
   *   locked   -> temporary reservation by another user
   *   reserved -> successfully reserved by current user
   */
  cancelPendingReserve();

  lastStatus = "typing";
  lastSent = "";

  jobInput.classList.remove('is-invalid');

  statusLine.className = "form-text text-muted";
  statusLine.textContent = "Checking after you finish typing...";

  setButtonsEnabled(true);

  const raw = (jobInput.value || "").trim();

  if (!hasAnyToken(raw)) {
    setIdle();
    return;
  }

  reserveTimer = setTimeout(() => {
    reserveTimer = null;
    reserveNow();
  }, 800);
}

jobInput.addEventListener(
  'input',
  handleJobInput
);
"""

count = text.count(OLD)

print(f"Target blocks found: {count}")

if count != 1:
    print()
    print("PATCH FAILED")
    print(f"Expected exactly 1 handleJobInput block, found {count}.")
    print("Original file was NOT changed.")
    sys.exit(1)

patched = text.replace(OLD, NEW, 1)

# ----------------------------------------------------------------------
# STATIC SAFETY VERIFICATION BEFORE WRITE
# ----------------------------------------------------------------------

checks = {
    "debounce 800ms":
        "}, 800);" in patched,

    "reserveNow called by debounce":
        "reserveNow();" in patched,

    "timer assigned":
        "reserveTimer = setTimeout" in patched,

    "timer cleared":
        "cancelPendingReserve();" in patched,

    "empty input handled":
        "if (!hasAnyToken(raw))" in patched,

    "existing WO handling preserved":
        'if (data.status === "exists")' in patched,

    "locked handling preserved":
        'if (data.status === "locked")' in patched,

    "reserved handling preserved":
        'data.status === "reserved"' in patched,

    "blur check preserved":
        "jobInput.addEventListener('blur'" in patched,

    "final Save protection preserved":
        'lastStatus === "exists"' in patched,
}

failed = []

print()
print("STATIC VERIFICATION")
print("-" * 90)

for name, ok in checks.items():
    print(("OK   " if ok else "FAIL ") + name)
    if not ok:
        failed.append(name)

if failed:
    print()
    print("PATCH FAILED BEFORE WRITE")
    print("Failed checks:")
    for name in failed:
        print(f"  - {name}")
    print()
    print("Original file was NOT changed.")
    sys.exit(1)

# ----------------------------------------------------------------------
# WRITE
# ----------------------------------------------------------------------

FILE.write_text(patched, encoding="utf-8")

# ----------------------------------------------------------------------
# READ-BACK VERIFICATION
# ----------------------------------------------------------------------

saved = FILE.read_text(encoding="utf-8")

post_checks = {
    "patch marker":
        "LIVE JOB CHECK" in saved,

    "exactly one debounce":
        saved.count("reserveTimer = setTimeout") == 1,

    "800 ms delay":
        "}, 800);" in saved,

    "exists logic still present":
        'if (data.status === "exists")' in saved,

    "locked logic still present":
        'if (data.status === "locked")' in saved,

    "reserved logic still present":
        'data.status === "reserved"' in saved,

    "blur listener still present":
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
print("Changed only:")
print(f"  {FILE}")
print()
print("Backend/routes.py: NOT CHANGED")
print("Database:          NOT CHANGED")
print("Reservation API:   NOT CHANGED")
print("Appliance History: NOT CHANGED")
print()
print("Live duplicate check delay: 800 ms")
print("=" * 90)
