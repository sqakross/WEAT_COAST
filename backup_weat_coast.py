import sqlite3
from pathlib import Path
from datetime import datetime

source_db = Path(r"C:\Users\administrator.WEST\PycharmProjects\WEAT_COAST\instance\inventory.db")
backup_folder = Path(r"F:\Backup")
backup_folder.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
backup_file = backup_folder / f"inventory_{timestamp}.db"

if not source_db.exists():
    raise FileNotFoundError(f"Source DB not found: {source_db}")

src = sqlite3.connect(source_db)
dst = sqlite3.connect(backup_file)

with dst:
    src.backup(dst)

src.close()
dst.close()

print(f"Backup created: {backup_file}")

# Delete backups older than 30 days
for f in backup_folder.glob("inventory_*.db"):
    if f.stat().st_mtime < (datetime.now().timestamp() - 30 * 24 * 60 * 60):
        f.unlink()
        print(f"Deleted old backup: {f}")