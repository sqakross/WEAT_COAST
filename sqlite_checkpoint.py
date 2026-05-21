import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(r"C:\Users\administrator.WEST\PycharmProjects\WEAT_COAST\instance\inventory.db")
LOG_PATH = Path(r"C:\Users\administrator.WEST\PycharmProjects\WEAT_COAST\logs\sqlite_checkpoint.log")

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

if not DB_PATH.exists():
    log(f"ERROR: DB not found: {DB_PATH}")
    raise SystemExit(1)

try:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cur = conn.cursor()

    cur.execute("PRAGMA journal_mode;")
    mode = cur.fetchone()
    log(f"Current journal_mode: {mode}")

    cur.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    result = cur.fetchall()
    log(f"Checkpoint result: {result}")

    conn.close()
    log("SQLite WAL checkpoint completed successfully.")

except Exception as e:
    log(f"ERROR: {e}")
    raise SystemExit(1)