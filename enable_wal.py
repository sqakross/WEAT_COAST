import sqlite3
from pathlib import Path

db_path = Path("instance") / "inventory.db"  # если имя DB другое — поменяй

if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("PRAGMA journal_mode=WAL;")
print("journal_mode:", cur.fetchone())

cur.execute("PRAGMA synchronous=NORMAL;")
cur.execute("PRAGMA busy_timeout=30000;")

conn.commit()
conn.close()

print("WAL mode enabled successfully.")