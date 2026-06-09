from dotenv import load_dotenv
load_dotenv(override=True)
from models import EmailOutbox
from extensions import db
from app import app
from inventory.routes import process_email_queue
from models import EmailOutbox
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import traceback
import re


PACIFIC_TZ = ZoneInfo("America/Los_Angeles")

REAL_JOB_RE = re.compile(r"\b[12]\d{6}\b")
AS_JOB_RE = re.compile(r"\b(?:AS)?00\d{4,}\b|\b00\d{4,}AS\b", re.I)


def has_allowed_job(text):
    if not text:
        return False

    text = str(text).upper()

    return bool(
        REAL_JOB_RE.search(text)
        or AS_JOB_RE.search(text)
    )


def skip_bad_pending_pickup_emails():
    rows = (
        EmailOutbox.query
        .filter_by(status="pending")
        .all()
    )

    skipped = 0

    for row in rows:
        subject = row.subject or ""
        body = row.body or ""

        is_pickup_email = (
            "Tech has picked up all the parts" in subject
            or "Pending Technician Confirmations Report" in subject
        )

        if is_pickup_email and not has_allowed_job(subject + " " + body):
            row.status = "skipped"
            row.error = "Skipped: no allowed job number found"
            skipped += 1

    if skipped:
        db.session.commit()
        write_log(f"SKIPPED BAD PENDING EMAILS | count={skipped}")

now = datetime.now(ZoneInfo("America/Los_Angeles"))
# Mon–Fri, 8am–6pm
if not (0 <= now.weekday() <= 4 and 8 <= now.hour < 18):
    print("Outside working hours, skipping")
    exit()


def write_log(message: str):
    try:
        log_dir = app.instance_path
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "email_queue.log")
        now_pst = datetime.now(PACIFIC_TZ)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{now_pst.isoformat()} | {message}\n")

    except Exception as e:
        print("LOG ERROR:", e)


def classify_error(err_text: str) -> str:
    s = (err_text or "").lower()

    if "535" in s or "badcredentials" in s or "username and password not accepted" in s:
        return "SMTP auth failed: check SMTP_USERNAME / SMTP_PASSWORD / App Password"
    if "timed out" in s or "timeout" in s:
        return "SMTP timeout: network/firewall/server delay"
    if "recipient address rejected" in s or "550" in s:
        return "Recipient rejected: check destination email"
    if "connection refused" in s or "could not connect" in s:
        return "SMTP connection failed: host/port/firewall"
    if "network is unreachable" in s:
        return "Network unreachable"
    if "authentication required" in s:
        return "SMTP authentication required"
    if not s.strip():
        return "Unknown error"
    return "See raw error"


def log_queue_snapshot():
    pending = EmailOutbox.query.filter_by(status="pending").count()
    error = EmailOutbox.query.filter_by(status="error").count()
    sent = EmailOutbox.query.filter_by(status="sent").count()
    write_log(f"QUEUE SNAPSHOT | pending={pending} error={error} sent={sent}")


def log_pending_rows(limit: int = 10):
    rows = (
        EmailOutbox.query
        .filter_by(status="pending")
        .order_by(EmailOutbox.id.desc())
        .limit(limit)
        .all()
    )

    if not rows:
        write_log("PENDING ROWS | none")
        return

    for row in rows:
        to_email = (row.to_email or "").strip()

        warning = ""
        if not to_email or "@" not in to_email or " " in to_email:
            warning = " | WARNING=invalid_to_email"

        write_log(
            f"PENDING ROW | "
            f"id={row.id} "
            f"attempts={getattr(row, 'attempt_count', None)} "
            f"to={to_email!r} "
            f"subject={row.subject!r}"
            f"{warning}"
        )


def log_failed_rows(limit: int = 10):
    rows = (
        EmailOutbox.query
        .filter_by(status="error")
        .order_by(EmailOutbox.id.desc())
        .limit(limit)
        .all()
    )

    if not rows:
        write_log("FAILED ROWS | none")
        return

    for row in rows:
        raw_error = row.error or ""
        friendly = classify_error(raw_error)

        write_log(
            f"FAILED ROW | "
            f"id={row.id} "
            f"attempts={getattr(row, 'attempt_count', None)} "
            f"to={row.to_email} "
            f"subject={row.subject!r} "
            f"reason={friendly} "
            f"raw_error={raw_error!r}"
        )


if __name__ == "__main__":
    try:
        with app.app_context():
            skip_bad_pending_pickup_emails()

            pending = EmailOutbox.query.filter_by(status="pending").count()

            if pending == 0:
                write_log("SKIP | queue empty")
            else:
                write_log(f"QUEUE SNAPSHOT | pending={pending}")
                log_pending_rows(limit=10)

                result = process_email_queue(limit=20) or {}
                msg = (
                    f"RESULT | sent={result.get('sent', 0)} "
                    f"errors={result.get('errors', 0)} "
                    f"processed={result.get('processed', 0)}"
                )
                print(msg)
                write_log(msg)

                if result.get("errors", 0):
                    log_failed_rows(limit=10)

    except Exception as e:
        err = f"SCRIPT ERROR | {str(e)}"
        print(err)
        write_log(err)
        write_log(traceback.format_exc())