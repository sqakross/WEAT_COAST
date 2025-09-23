 scripts/ensure_wo_parts_columns.py
from sqlalc#hemy import text
from app import app, db

COLUMNS_TO_ADD = [
    ("work_order_parts", "alt_numbers",       "VARCHAR(200)"),
    ("work_order_parts", "line_status",       "VARCHAR(32)"),
    ("work_order_parts", "unit_cost",         "FLOAT"),
    ("work_order_parts", "unit_id",           "INTEGER"),
    ("work_order_parts", "unit_label",        "VARCHAR(120)"),
    ("work_order_parts", "unit_price_base",   "FLOAT"),
    ("work_order_parts", "unit_price_final",  "FLOAT"),
]

def table_columns(conn, table):
    # PRAGMA table_info: (cid, name, type, notnull, dflt_value, pk)
    rows = conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
    return {r[1] for r in rows}

if __name__ == "__main__":
    with app.app_context():
        # открываем транзакцию (для SQLite alter это ок)
        with db.engine.begin() as conn:
            for table, col, ddl in COLUMNS_TO_ADD:
                try:
                    cols = table_columns(conn, table)
                    if col not in cols:
                        sql = f"ALTER TABLE {table} ADD COLUMN {col} {ddl}"
                        conn.exec_driver_sql(sql)
                        print(f"Added: {table}.{col} {ddl}")
                    else:
                        print(f"OK: {table}.{col} already exists")
                except Exception as e:
                    print(f"ERROR adding {table}.{col}: {e}")

