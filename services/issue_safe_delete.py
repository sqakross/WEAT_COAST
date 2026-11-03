
"""WCCR Issue Safety V2.2 - conservative cancellation foundation.

This module intentionally does not enable deletion routes.
"""

from datetime import datetime
from sqlalchemy import text
from extensions import db


AUDIT_TABLE = "issue_cancellation_audit"


def ensure_audit_table():
    """Create audit storage only during an authorized installation step."""
    db.session.execute(text("""
        CREATE TABLE IF NOT EXISTS issue_cancellation_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_record_id INTEGER NOT NULL,
            invoice_number INTEGER,
            part_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            original_record_json TEXT NOT NULL,
            canceled_by TEXT NOT NULL,
            canceled_at TEXT NOT NULL,
            stock_before INTEGER,
            stock_after INTEGER,
            reason TEXT,
            UNIQUE(original_record_id)
        )
    """))


def cancellation_block_reason(records, supplier_return_model=None):
    """Fail closed on consumption, return rows, and ambiguous references."""
    if not records:
        return "Invoice has no active issue records."

    for record in records:
        quantity = int(record.quantity or 0)

        if quantity <= 0:
            return "Invoice contains return or non-positive quantity rows."

        if int(getattr(record, "consumed_qty", 0) or 0) > 0:
            return "Invoice contains consumed parts."

        if getattr(record, "consumed_flag", False):
            return "Invoice contains consumed parts."

        if getattr(record, "consumption_logs", None):
            if len(record.consumption_logs) > 0:
                return "Invoice has consumption history."

        if getattr(record, "confirmed_by_tech", False):
            return "Invoice contains technician-confirmed parts."

        if getattr(record, "return_to", None):
            return "Invoice has return-related metadata."

        if not getattr(record, "part", None):
            return "Invoice contains a missing inventory part."

    return None

def safe_delete_invoice(invoice_number, actor):
    """V2.5: transactional cancellation of unconfirmed issue invoices."""
    import json
    from collections import defaultdict
    from datetime import datetime, timezone
    from sqlalchemy import text
    from extensions import db
    from models import (
        IssuedPartRecord, IssuedBatch, SupplierReturnBatch,
        WorkOrder, WorkOrderPart
    )

    def norm(value):
        return str(value or "").strip().upper().lstrip("0") or "0"

    def stop(message):
        db.session.rollback()
        return False, message

    try:
        invoice_number = int(invoice_number)
        if invoice_number <= 0:
            return False, "Invalid invoice number."

        records = (
            IssuedPartRecord.query
            .filter_by(invoice_number=invoice_number)
            .order_by(IssuedPartRecord.id)
            .all()
        )
        if not records:
            return False, "Invoice not found or already deleted."

        reason = cancellation_block_reason(records)
        if reason:
            return stop(reason)

        batches = {}
        for record in records:
            if record.batch_id is None or record.batch is None:
                return stop("Missing issue batch. Manual review required.")
            batches[record.batch_id] = record.batch

        if len(batches) != 1:
            return stop("Invoice spans multiple batches. Manual review required.")

        batch = next(iter(batches.values()))
        if batch.invoice_number != invoice_number:
            return stop("Invoice/batch mismatch.")

        if (
            batch.consumed_flag or batch.consumed_at
            or batch.consumed_by or batch.consumed_note
        ):
            return stop("Batch has consumption history.")

        if batch.first_confirm_email_sent_at:
            return stop("Technician confirmation email history requires review.")

        all_batch_rows = (
            IssuedPartRecord.query
            .filter_by(batch_id=batch.id)
            .all()
        )
        if {r.id for r in all_batch_rows} != {r.id for r in records}:
            return stop("Batch contains other invoice records.")

        for record in records:
            if record.confirmed_at or record.confirmed_by:
                return stop("Invoice has technician confirmation history.")
            if record.consumed_at or record.consumed_by or record.consumed_note:
                return stop("Invoice has consumption metadata.")
            if record.return_destination_id:
                return stop("Invoice has return destination history.")
            if SupplierReturnBatch.query.filter_by(
                source_kind="reports_return",
                source_return_record_id=record.id
            ).first():
                return stop("Supplier return history requires review.")

            ref = (record.reference_job or "").strip()
            returns = IssuedPartRecord.query.filter(
                IssuedPartRecord.part_id == record.part_id,
                IssuedPartRecord.issued_to == record.issued_to,
                IssuedPartRecord.quantity < 0
            )
            if ref:
                returns = returns.filter(
                    IssuedPartRecord.reference_job == "RETURN " + ref
                )
            else:
                returns = returns.filter(
                    IssuedPartRecord.reference_job.ilike("RETURN%")
                )
            if returns.first():
                return stop("Matching return history requires review.")

        wo_id = batch.work_order_id
        changes = defaultdict(int)
        matched_lines = {}

        if wo_id:
            wo = db.session.get(WorkOrder, wo_id)
            if wo is None:
                return stop("Linked Work Order not found.")

            jobs = [
                x.strip().upper()
                for x in __import__("re").split(
                    r"[,;\s]+", wo.job_numbers or ""
                )
                if x.strip()
            ]

            for record in records:
                ref = (record.reference_job or "").strip().upper()
                if ref and ref not in jobs:
                    return stop(
                        "Job reference does not match linked Work Order."
                    )

                if not record.part:
                    return stop("Inventory part missing.")

                pn = record.part.part_number
                candidates = (
                    WorkOrderPart.query
                    .filter_by(work_order_id=wo_id, part_number=pn)
                    .all()
                )

                # WCCR WO LINE MATCH V3.5
                # Prefer an exact receiving invoice match.
                # For STOCK issues, use the WO line's
                # creation/last-issue interval instead.
                # WCCR ISSUE DELETE V3.6
                # Exact invoice match first.
                # For STOCK lines, use the issue timestamp
                # and require exactly one matching WO line.
                inv = norm(record.inv_ref)

                if inv != "0":
                    matches = [
                        line for line in candidates
                        if norm(line.invoice_number) == inv
                    ]
                elif len(candidates) == 1:
                    matches = candidates
                else:
                    from datetime import timedelta

                    matches = []
                    issued = record.issue_date

                    if issued is not None:
                        for line in candidates:
                            created = line.created_at
                            last = line.last_issued_at

                            if created is None or last is None:
                                continue

                            if (
                                created - timedelta(seconds=2)
                                <= issued
                                <= last + timedelta(seconds=2)
                            ):
                                matches.append(line)

                if len(matches) != 1:
                    return stop(
                        "Cannot uniquely match WO line for part "
                        + str(pn) + ". No changes made."
                    )

                line = matches[0]
                changes[line.id] += int(record.quantity)
                matched_lines[line.id] = line

            for line_id, qty in changes.items():
                line = matched_lines[line_id]
                if int(line.issued_qty or 0) < qty:
                    return stop(
                        "WO issued quantity mismatch for line "
                        + str(line_id)
                    )
        else:
            for record in records:
                ref = (record.reference_job or "").strip().upper()
                if ref not in ("", "STOCK", "INVENTORY"):
                    return stop(
                        "Unlinked job invoice requires manual review."
                    )

        audit_exists = db.session.execute(text("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='issue_cancellation_audit'
        """)).first()
        if not audit_exists:
            return stop("Cancellation audit table missing.")

        timestamp = datetime.now(timezone.utc).isoformat()

        for record in records:
            part = record.part
            qty = int(record.quantity)
            before = int(part.quantity or 0)
            after = before + qty

            snapshot = {}
            for column in record.__table__.columns:
                value = getattr(record, column.name)
                if isinstance(value, datetime):
                    value = value.isoformat()
                snapshot[column.name] = value

            # V2.6: prevent duplicate cancellation of the same
            # issuance, while allowing SQLite ID reuse.
            existing_audits = db.session.execute(text("""
                SELECT original_record_json
                FROM issue_cancellation_audit
                WHERE original_record_id = :rid
            """), {"rid": record.id}).scalars().all()

            for old_json in existing_audits:
                try:
                    old_snapshot = json.loads(old_json)
                except (ValueError, TypeError):
                    return stop("Invalid historical audit snapshot.")

                if (
                    str(old_snapshot.get("issue_date"))
                    == str(snapshot.get("issue_date"))
                    and old_snapshot.get("batch_id")
                    == snapshot.get("batch_id")
                    and old_snapshot.get("part_id")
                    == snapshot.get("part_id")
                    and old_snapshot.get("invoice_number")
                    == snapshot.get("invoice_number")
                ):
                    return stop(
                        "This exact issuance was already canceled."
                    )

            db.session.execute(text("""
                INSERT INTO issue_cancellation_audit (
                    original_record_id, invoice_number, part_id,
                    quantity, original_record_json, canceled_by,
                    canceled_at, stock_before, stock_after, reason
                ) VALUES (
                    :rid, :invoice, :pid, :qty, :snapshot, :actor,
                    :at, :before, :after, :reason
                )
            """), {
                "rid": record.id,
                "invoice": invoice_number,
                "pid": record.part_id,
                "qty": qty,
                "snapshot": json.dumps(
                    snapshot, ensure_ascii=False, default=str
                ),
                "actor": str(actor),
                "at": timestamp,
                "before": before,
                "after": after,
                "reason": "Issue Safety V2.5 - Cancel Invoice",
            })

            part.quantity = after
            db.session.delete(record)

        for line_id, qty in changes.items():
            line = matched_lines[line_id]
            line.issued_qty = int(line.issued_qty or 0) - qty
            if line.issued_qty == 0:
                line.last_issued_at = None
                line.stock_hint = None

        db.session.flush()
        db.session.delete(batch)
        db.session.commit()

        return True, (
            "Invoice deleted. Stock restored, "
            "Work Order updated, audit saved."
        )

    except Exception as exc:
        import logging
        import traceback

        db.session.rollback()

        logging.getLogger("wccr.issue_safe_delete").error(
            "ISSUE DELETE V2.5 FAILED invoice=%s error=%s",
            invoice_number,
            str(exc),
            exc_info=True,
        )

        return False, (
            "Safe Delete V2.5 failed: "
            + type(exc).__name__
            + ": "
            + str(exc)[:250]
            + ". Transaction rolled back."
        )
