from app import app, db
import os

print("Current working directory:", os.getcwd())
print("Database URI:", app.config['SQLALCHEMY_DATABASE_URI'])

from models import User

with app.app_context():
    user = User.query.filter_by(username='Andrew').first()
    if user:
        print(f"Found user: {user.username} (ID: {user.id}), deleting...")
        db.session.delete(user)
        db.session.commit()
        print("User deleted.")
    else:
        print("User not found.")
