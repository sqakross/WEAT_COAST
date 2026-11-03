"""WCCR Safe Return Delete V1.

Safely cancel a single, posted, unreconciled VENDOR return.
No physical inventory adjustment: the return creation and supplier
posting have already offset each other.
"""

import json
from datetime import datetime, timezone

from sqlalchemy import text

from extensions import db
from models import (
    IssuedPartRecord,
    IssuedBatch,
    SupplierReturnBatch,
    SupplierReturnItem,
)


def safe_delete_vendor_return(invoice_number, actor):
    def reject(message):
        db.session.rollback()
        return False, message

    try:
        invoice_number = int(invoice_number)

        if invoice_number <= 0:
            return reject("Invalid invoice number.")

        records = (
            IssuedPartRecord.query
            .filter_by(invoice_number=invoice_number)
            .all()
        )

        if len(records) != 1:
            return reject(
                "Only single-line VENDOR return invoices are supported."
            )

        record = records[0]

        if (
            int(record.quantity or 0) >= 0
            or (record.return_to or "").upper() != "VENDOR"
            or not (record.reference_job or "").upper().startswith("RETURN")
        ):
            return reject("This is not an eligible VENDOR return.")

        if (
            record.consumed_flag
            or record.consumed_at
            or record.consumed_by
            or record.consumed_note
            or int(record.consumed_qty or 0) != 0
        ):
            return reject("Return has consumption history.")

        if record.part is None:
            return reject("Inventory part is missing.")

        if record.batch_id is None:
            return reject("Issue batch is missing.")

        batch = db.session.get(IssuedBatch, record.batch_id)

        if batch is None or batch.invoice_number != invoice_number:
            return reject("Issue batch mismatch.")

        if batch.work_order_id is not None:
            return reject("Linked Work Order requires manual review.")

        if (
            batch.consumed_flag
            or batch.consumed_at
            or batch.consumed_by
            or batch.consumed_note
            or batch.first_confirm_email_sent_at
        ):
            return reject("Issue batch has protected history.")

        batch_rows = IssuedPartRecord.query.filter_by(
            batch_id=batch.id
        ).all()

        if len(batch_rows) != 1 or batch_rows[0].id != record.id:
            return reject("Issue batch contains other records.")

        supplier_batches = SupplierReturnBatch.query.filter_by(
            source_kind="reports_return",
            source_return_record_id=record.id,
        ).all()

        if len(supplier_batches) != 1:
            return reject("Supplier Return link is not unique.")

        supplier = supplier_batches[0]

        if (supplier.status or "").lower() != "posted":
            return reject("Supplier Return must be posted.")

        if (
            supplier.supplier_credit_document
            or supplier.supplier_credit_date
            or supplier.reconciled_at
            or supplier.reconciled_by
        ):
            return reject(
                "Supplier credit or reconciliation exists. "
                "Deletion is blocked."
            )

        items = SupplierReturnItem.query.filter_by(
            batch_id=supplier.id
        ).all()

        if len(items) != 1:
            return reject("Supplier Return must contain exactly one item.")

        item = items[0]

        if int(item.qty_returned or 0) != abs(int(record.quantity)):
            return reject("Supplier Return quantity mismatch.")

        if (
            (item.part_number or "").strip().upper()
            != (record.part.part_number or "").strip().upper()
        ):
            return reject("Supplier Return part mismatch.")

        if (
            item.issued_part_record_id is not None
            and item.issued_part_record_id != record.id
        ):
            return reject("Supplier Return item references another issue.")

        audit_table = db.session.execute(text("""
            SELECT name FROM sqlite_master
            WHERE type='table'
              AND name='issue_cancellation_audit'
        """)).first()

        if audit_table is None:
            return reject("Cancellation audit table is missing.")

        snapshot = {}
        for column in record.__table__.columns:
            value = getattr(record, column.name)
            if isinstance(value, datetime):
                value = value.isoformat()
            snapshot[column.name] = value

        existing = db.session.execute(text("""
            SELECT original_record_json
            FROM issue_cancellation_audit
            WHERE original_record_id = :rid
        """), {"rid": record.id}).scalars().all()

        for old_json in existing:
            try:
                previous = json.loads(old_json)
            except (ValueError, TypeError):
                return reject("Invalid existing audit history.")

            if (
                previous.get("invoice_number") == invoice_number
                and previous.get("part_id") == record.part_id
                and previous.get("batch_id") == record.batch_id
                and str(previous.get("issue_date"))
                == str(snapshot.get("issue_date"))
            ):
                return reject("This return was already canceled.")

        stock_before = int(record.part.quantity or 0)

        supplier_snapshot = {
            column.name: getattr(supplier, column.name)
            for column in supplier.__table__.columns
        }
        item_snapshot = {
            column.name: getattr(item, column.name)
            for column in item.__table__.columns
        }

        snapshot["_wccr_supplier_return_batch"] = supplier_snapshot
        snapshot["_wccr_supplier_return_item"] = item_snapshot

        db.session.execute(text("""
            INSERT INTO issue_cancellation_audit (
                original_record_id,
                invoice_number,
                part_id,
                quantity,
                original_record_json,
                canceled_by,
                canceled_at,
                stock_before,
                stock_after,
                reason
            ) VALUES (
                :rid, :invoice, :pid, :qty, :snapshot,
                :actor, :at, :before, :after, :reason
            )
        """), {
            "rid": record.id,
            "invoice": invoice_number,
            "pid": record.part_id,
            "qty": int(record.quantity),
            "snapshot": json.dumps(
                snapshot, ensure_ascii=False, default=str
            ),
            "actor": str(actor),
            "at": datetime.now(timezone.utc).isoformat(),
            "before": stock_before,
            "after": stock_before,
            "reason": (
                "WCCR Safe Return Delete V1 - "
                "Cancel posted VENDOR return"
            ),
        })

        # Return creation adds stock; supplier posting subtracts
        # the same quantity. Net physical inventory impact is zero.
        # Therefore cancellation must not adjust Part.quantity.
        db.session.delete(item)
        db.session.flush()

        db.session.delete(supplier)
        db.session.flush()

        db.session.delete(record)
        db.session.flush()

        db.session.delete(batch)
        db.session.flush()

        stock_after = int(record.part.quantity or 0)

        if stock_after != stock_before:
            raise RuntimeError(
                "Inventory quantity changed unexpectedly."
            )

        db.session.commit()

        return True, (
            f"VENDOR return invoice #{invoice_number} deleted. "
            "Supplier Return removed. Inventory unchanged. "
            "Audit saved."
        )

    except Exception as exc:
        db.session.rollback()
        return False, (
            "Safe Return Delete failed: "
            + type(exc).__name__
            + ": "
            + str(exc)[:250]
        )
