from app import app, db
from models import User

with app.app_context():
    # Проверяем, существует ли суперадмин с таким username, чтобы не создавать дубликат
    existing = User.query.filter_by(username='Andrew').first()
    if existing:
        print("Superadmin already exists.")
    else:
        superadmin = User(
            username='Andrew1',
            role='superadmin'
        )
        superadmin.password = 'lion9911'  # вызываем сеттер, который сам хэширует пароль
        db.session.add(superadmin)
        db.session.commit()
        print("Superadmin user created")
