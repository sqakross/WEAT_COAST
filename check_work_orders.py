import sqlite3

conn = sqlite3.connect("instance/inventory.db")
cur = conn.cursor()

cur.execute("PRAGMA table_info(work_orders);")
rows = cur.fetchall()

print("Columns in work_orders:")
for r in rows:
    print(r)

conn.close()
