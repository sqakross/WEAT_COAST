from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta, timezone

BASE_DIR = Path(__file__).resolve().parents[1]
UPLOADS = BASE_DIR / "uploads"
LOGS = BASE_DIR / "logs"
INSTANCE = BASE_DIR / "instance"

# ЛОГИ — по возрасту
KEEP_LOGS_DAYS = 14
LOG_PATTERNS = ["*.log", "*.log.*", "*-out.log", "*-err.log"]

# INSTANCE files — по возрасту
INSTANCE_LOG_PATTERNS = [
    "email_queue.log",
    "email_queue.log.*",
    "wo_save_debug.txt",
    "wo_save_debug*.txt",
]

def _older_than(path: Path, days: int) -> bool:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return mtime < cutoff

def _delete_matching(dir_path: Path, patterns: list[str]) -> tuple[int, int, int]:
    deleted = 0
    scanned = 0
    failed = 0
    if not dir_path.exists():
        return (0, 0, 0)

    for pat in patterns:
        for p in dir_path.glob(pat):
            if not p.is_file():
                continue
            scanned += 1
            try:
                p.unlink()
                deleted += 1
            except Exception as e:
                failed += 1
                print(f"[FAIL] cannot delete: {p} -> {e}")
    return deleted, scanned, failed

def _delete_older(dir_path: Path, patterns: list[str], keep_days: int) -> tuple[int, int, int]:
    deleted = 0
    scanned = 0
    failed = 0
    if not dir_path.exists():
        return (0, 0, 0)

    for pat in patterns:
        for p in dir_path.glob(pat):
            if not p.is_file():
                continue
            scanned += 1
            try:
                if _older_than(p, keep_days):
                    p.unlink()
                    deleted += 1
            except Exception as e:
                failed += 1
                print(f"[FAIL] {p} -> {e}")
    return deleted, scanned, failed

def main():
    print("BASE_DIR =", BASE_DIR)
    print("UPLOADS  =", UPLOADS.resolve())
    print("LOGS     =", LOGS.resolve())
    print("INSTANCE =", INSTANCE.resolve())

    deleted = 0
    failed = 0
    skipped = 0

    # -------------------------
    # UPLOADS: delete invoice*.pdf
    # -------------------------
    if not UPLOADS.exists():
        print("Uploads folder not found")
    else:
        for p in UPLOADS.iterdir():
            if not p.is_file():
                continue

            name = p.name.lower()

            # УДАЛЯЕМ ВСЕ invoice*.pdf
            if name.startswith("invoice") and name.endswith(".pdf"):
                try:
                    p.unlink()
                    deleted += 1
                except Exception as e:
                    failed += 1
                    print(f"[FAIL] {p.name} -> {type(e).__name__}: {e}")
            else:
                skipped += 1

    print(f"UPLOAD cleanup: deleted={deleted}, failed={failed}, skipped={skipped}")

    # -------------------------
    # LOGS: delete old log files
    # -------------------------
    d1, s1, f1 = _delete_older(LOGS, LOG_PATTERNS, KEEP_LOGS_DAYS)
    print(f"LOGS cleanup: deleted={d1}, scanned={s1}, failed={f1}")

    # -------------------------
    # INSTANCE: delete old queue/debug logs
    # -------------------------
    d2, s2, f2 = _delete_older(INSTANCE, INSTANCE_LOG_PATTERNS, KEEP_LOGS_DAYS)
    print(f"INSTANCE cleanup: deleted={d2}, scanned={s2}, failed={f2}")

if __name__ == "__main__":
    main()