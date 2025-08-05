import sqlite3

def migrate_add_invoice_number(db_path='instance/inventory.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Проверяем, есть ли колонка invoice_number
    cursor.execute("PRAGMA table_info(issued_part_record)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'invoice_number' not in columns:
        print("Добавляем колонку 'invoice_number' в таблицу 'issued_part_record'...")
        cursor.execute("ALTER TABLE issued_part_record ADD COLUMN invoice_number TEXT")
        print("Колонка добавлена.")
    else:
        print("Колонка 'invoice_number' уже существует. Миграция не требуется.")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    migrate_add_invoice_number()
