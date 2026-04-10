# Placeholder for confi
import os

class Config:
    # Secrets

    SECRET_KEY = os.environ.get("SECRET_KEY", "your-very-secure-key")

    # Paths
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

    # DB
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(BASE_DIR, "instance", "inventory.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Import: allowed file extensions (now includes PDF)
    ALLOWED_EXTENSIONS = {"xlsx", "xls", "csv", "pdf"}

    # OCR tools (Windows defaults; can be overridden by environment variables)
    POPPLER_BIN   = os.environ.get("POPPLER_BIN",  r"C:\Program Files\poppler\bin")
    TESSERACT_EXE = os.environ.get("TESSERACT_EXE", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

    # Import feature flags (keep your defaults)
    WCCR_IMPORT_ENABLED = int(os.environ.get("WCCR_IMPORT_ENABLED", "1"))  # default 1 = применять
    WCCR_IMPORT_DRY_RUN = int(os.environ.get("WCCR_IMPORT_DRY_RUN", "0"))  # default 0 = не dry

    SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
    SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
    SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
    SMTP_FROM = os.environ.get("SMTP_FROM", SMTP_USERNAME)
    EMAIL_ORDERS_TO = os.environ.get("EMAIL_ORDERS_TO", "lion7500000@gmail.com")
