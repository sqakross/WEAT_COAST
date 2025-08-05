from app import app, db  # Импортируем Flask app и db
from sqlalchemy import text

with app.app_context():
    with db.engine.connect() as conn:
        result = conn.execute(text("PRAGMA table_info(issued_part_record)"))
        columns = result.fetchall()
        for col in columns:
            print(col)


