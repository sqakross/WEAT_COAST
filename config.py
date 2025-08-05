# Placeholder for config.py
import os


class Config:
    # Secret key for session and CSRF protection
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-very-secure-key'

    # SQLite instatce path
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(BASE_DIR, 'instance', 'inventory.db')

    # Disable tracking modifications (improves performance)
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Upload folder for Excel/Word imports
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')

    # Allowed file extensions for import
    ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv', 'docx'}
