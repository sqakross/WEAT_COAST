import sqlite3

conn = sqlite3.connect('instance/inventory.db')
cursor = conn.cursor()

cursor.execute("ALTER TABLE issued_part_record ADD COLUMN unit_cost_at_issue FLOAT NOT NULL DEFAULT 0;")

conn.commit()
conn.close()
