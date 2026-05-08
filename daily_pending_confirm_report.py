import os
from pathlib import Path
from datetime import datetime, date, timedelta, timezone
from zoneinfo import ZoneInfo
from sqlalchemy import func, case

from app import app
from extensions import db
from models import WorkOrder, IssuedBatch, IssuedPartRecord, EmailOutbox


PACIFIC_TZ = ZoneInfo("America/Los_Angeles")
LOG_PATH = Path("instance") / "email_queue.log"


def log_queue(message: str):
    ts = datetime.now(PACIFIC_TZ).strftime("%Y-%m-%d %H:%M:%S")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[{ts} PT] DAILY_PENDING_CONFIRM_REPORT | {message}\n")

def to_pacific(dt):
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(PACIFIC_TZ)


def build_report():
    pending_days = int(os.getenv("PENDING_CONFIRM_DAYS", "3"))
    to_email = os.getenv("PENDING_CONFIRM_REPORT_TO", "vita@chiefappliance.com")

    today_pt = datetime.now(PACIFIC_TZ).date()
    cutoff = today_pt - timedelta(days=pending_days)

    rows = (
        db.session.query(
            IssuedBatch,
            WorkOrder,
            func.count(IssuedPartRecord.id).label("items_count"),
            func.sum(
                case(
                    (IssuedPartRecord.confirmed_by_tech == True, 1),
                    else_=0
                )
            ).label("confirmed_count")
        )
        .join(WorkOrder, WorkOrder.id == IssuedBatch.work_order_id)
        .join(IssuedPartRecord, IssuedPartRecord.batch_id == IssuedBatch.id)
        .filter(
            WorkOrder.status != "cancel_job",
            func.date(IssuedBatch.issue_date) <= cutoff,
        )
        .group_by(IssuedBatch.id, WorkOrder.id)
        .having(
            func.sum(
                case(
                    (IssuedPartRecord.confirmed_by_tech == False, 1),
                    else_=0
                )
            ) > 0
        )
        .order_by(IssuedBatch.issue_date.asc())
        .all()
    )

    if not rows:
        msg = "No pending confirmations found."
        print(msg)
        log_queue(msg)
        return None

    lines = []
    lines.append("Technician Confirmations Pending Report")
    lines.append(f"Date: {today_pt.isoformat()} PT")
    lines.append(f"Showing pending confirmations older than {pending_days} day(s)")
    lines.append("")
    lines.append("Job | Technician | Issued Date (PT) | Invoice # | Confirmed")
    lines.append("-" * 85)

    for batch, wo, items_count, confirmed_count in rows:
        total = int(items_count or 0)
        confirmed = int(confirmed_count or 0)

        job = wo.job_numbers or batch.reference_job or "—"
        tech = batch.issued_to or wo.technician_name or "—"

        issued_dt_pt = to_pacific(batch.issue_date)
        issued_date = issued_dt_pt.strftime("%Y-%m-%d %I:%M %p") if issued_dt_pt else "—"

        invoice = f"{batch.invoice_number:06d}" if batch.invoice_number else "—"

        lines.append(
            f"{job} | {tech} | {issued_date} | {invoice} | {confirmed}/{total}"
        )

    body = "\n".join(lines)
    subject = f"Pending Technician Confirmations Report - {today_pt.isoformat()}"

    unique_key = f"pending_confirm_report_{today_pt.isoformat()}"

    exists = EmailOutbox.query.filter_by(unique_key=unique_key).first()
    if exists:
        msg = f"Report already queued today. email_id={exists.id} status={exists.status}"
        print(msg)
        log_queue(msg)
        return exists

    email = EmailOutbox(
        kind="pending_confirm_report",
        unique_key=unique_key,
        to_email=to_email,
        subject=subject,
        body=body,
        status="pending",
        created_at=datetime.utcnow(),  # keep same style as your existing models
    )



    db.session.add(email)
    db.session.commit()

    msg = f"Queued report email to {to_email} email_id={email.id} rows={len(rows)}"
    print(msg)
    log_queue(msg)
    return email


if __name__ == "__main__":
    with app.app_context():
        build_report()