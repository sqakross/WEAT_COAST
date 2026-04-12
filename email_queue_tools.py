from app import app
from models import EmailOutbox
from extensions import db


def reset_failed_emails(limit: int | None = None):
    with app.app_context():
        q = EmailOutbox.query.filter_by(status="error").order_by(EmailOutbox.id.asc())
        rows = q.limit(limit).all() if limit else q.all()

        count = 0
        for row in rows:
            row.status = "pending"
            row.error = None
            count += 1

        db.session.commit()
        print({"reset": count})
        return count


if __name__ == "__main__":
    reset_failed_emails()