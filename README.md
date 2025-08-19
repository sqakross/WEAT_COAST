
# WEST COAST CHIEF REPAIR — Inventory System

## 🛠 Requirements
- Python 3.8+
- pip

## 📁 Installation Steps

1. **Extract ZIP archive**
2. **Navigate to the project folder**:
   ```bash
   cd wccr_inventory_systempip
   ```
3. **Create virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate     # for Mac/Linux
   .venv\Scripts\activate.bat  # for Windows
   source .venv/Scripts/activate #for bash windows
   
   pip freeze > requirements.txt


      ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Start the app**:
   ```bash
   python app.py
   
   power Shell & "C:\Users\administrator.WEST\PycharmProjects\WEAT_COAST\start_server.bat"

   ```
6. Open your browser and go to:  
   `http://127.0.0.1:5000`

## 👤 Default Users
- Super Admin: `admin / admin123`

## 🧾 Features
- Inventory Management
- Barcode Scanning
- PDF Invoice Generation
- Role-Based User Permissions (Super Admin / Admin / User / Viewer)
- Client Issuance Reports
- Import/Export (Excel, CSV)
- Search by part name, part number, shelf, and job reference
Option 1: Create Users via /register Page (Recommended for Admins)

    This is available only to superadmin (which you already created in the shell).

Steps:

    Log in with your superadmin credentials (e.g., admin / admin123)

    Visit this URL manually in your browser:

    http://127.0.0.1:5000/register

    Fill out the form:

        Username

        Password

        Select Role: admin, user, or viewer

    Click Create User

    You should see a success message.

    Now the new user can log in from /login with their credentials.

✅ Option 2: Create Users in Python Shell (Manual)

Use this if /register is not working or if you prefer command line.
Steps:

    Activate your virtual environment, if not already:

.venv\Scripts\activate

Open Flask shell:

python

Вариант 1: Проверить, есть ли уже пользователь

Открой flask shell и проверь:

from app import db
from models import User

User.query.all()

Если видишь пользователя с username='admin' — он уже создан.
🗑 Вариант 2: Удалить старого admin и создать заново

admin = User.query.filter_by(username="andrew").first()
db.session.delete(admin)
db.session.commit()

Теперь снова создай:

from werkzeug.security import generate_password_hash

new_admin = User(
    username="andrew",
    password=generate_password_hash("lion9911"),
    role="superadmin"
)
db.session.add(new_admin)
db.session.commit()

✅ Альтернатива: Создать другого пользователя

Можно создать другого суперпользователя с другим именем:

new_admin = User(
    username="chief",
    password=generate_password_hash("chief123"),
    role="superadmin"
)
db.session.add(new_admin)
db.session.commit()

FOR CLEAN REPORT
Открой этот URL в браузере (например, http://127.0.0.1:5000/clear_issued_records) — и все записи очистятся.

FOR CLEAN ALL PARTS
Открой этот URL в браузере (например, http://127.0.0.1:5000/clear_parts ) — и все записи очистятся.

@echo off
cd /d C:\Users\andrii\PycharmProjects\WEAT_COAST
call .venv\Scripts\activate.bat
python app.py

    cd /d ... — переходит в папку проекта.

    call .venv\Scripts\activate.bat — активирует виртуальное окружение Python (в котором установлены все зависимости).

    python app.py — запускает сам Flask сервер.

Нажми Win+R, введи shell:startup, нажми Enter.

Положи туда ярлык на твой .bat.






