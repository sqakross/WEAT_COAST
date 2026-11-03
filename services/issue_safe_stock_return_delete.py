"""WCCR Return Delete V2.5: safe STOCK return cancellation."""

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


def safe_delete_stock_return(invoice_number, actor):
    def reject(message):
        db.session.rollback()
        return False, message

    try:
        invoice_number = int(invoice_number)
        if invoice_number <= 0:
            return reject("Invalid return invoice.")

        records = IssuedPartRecord.query.filter_by(
            invoice_number=invoice_number
        ).all()

        if len(records) != 1:
            return reject(
                "STOCK deletion requires exactly one return line."
            )

        record = records[0]
        qty = int(record.quantity or 0)

        if (
            qty >= 0
            or (record.return_to or "").strip().upper() != "STOCK"
            or not (record.reference_job or "")
                .strip().upper().startswith("RETURN ")
        ):
            return reject("Not an eligible STOCK return.")

        if (
            record.consumed_flag
            or record.consumed_at
            or record.consumed_by
            or record.consumed_note
            or int(record.consumed_qty or 0) != 0
            or record.consumed_job_ref
        ):
            return reject("Return has protected usage history.")

        part = record.part
        if part is None:
            return reject("Inventory part is missing.")

        if record.batch_id is None:
            return reject("Return issue batch is missing.")

        batch = db.session.get(IssuedBatch, record.batch_id)

        if (
            batch is None
            or batch.invoice_number != invoice_number
            or batch.work_order_id is not None
        ):
            return reject("Return batch mismatch or linked Work Order.")

        if (
            batch.consumed_flag
            or batch.consumed_at
            or batch.consumed_by
            or batch.consumed_note
        ):
            return reject("Return batch has protected history.")

        batch_rows = IssuedPartRecord.query.filter_by(
            batch_id=batch.id
        ).all()

        if (
            len(batch_rows) != 1
            or batch_rows[0].id != record.id
        ):
            return reject("Return batch contains other records.")

        # STOCK returns must have no Supplier Return dependency.
        supplier_headers = SupplierReturnBatch.query.filter_by(
            source_return_record_id=record.id
        ).all()

        supplier_items = SupplierReturnItem.query.filter_by(
            issued_part_record_id=record.id
        ).all()

        if supplier_headers or supplier_items:
            return reject(
                "Supplier Return dependency exists. "
                "STOCK deletion blocked."
            )

        # WCCR STOCK RETURN FIX V3.2
        original_job = (
            (record.reference_job or "").strip()[6:].strip()
        )
        if not original_job:
            return reject("Original job reference is missing.")

        sources = IssuedPartRecord.query.filter_by(
            part_id=record.part_id,
            issued_to=record.issued_to,
            reference_job=original_job,
        ).all()

        sources = [
            item for item in sources
            if int(item.quantity or 0) > 0
        ]

        if record.source_receipt_line_id is not None:
            sources = [
                item for item in sources
                if item.source_receipt_line_id
                == record.source_receipt_line_id
            ]

        if not sources:
            return reject("Original positive issue missing.")

        source = sources[0]

        if sum(int(item.quantity or 0) for item in sources) < abs(qty):
            return reject(
                "Return exceeds verified original issues."
            )

        stock_before = int(part.quantity or 0)
        remove_qty = abs(qty)

        if stock_before < remove_qty:
            return reject(
                "Insufficient stock to cancel return: "
                f"available {stock_before}, required {remove_qty}."
            )

        # Cancellation audit is mandatory.
        audit_exists = db.session.execute(text("""
            SELECT name FROM sqlite_master
            WHERE type = 'table'
              AND name = 'issue_cancellation_audit'
        """)).first()

        if audit_exists is None:
            return reject("Cancellation audit table is missing.")

        previous = db.session.execute(text("""
            SELECT 1 FROM issue_cancellation_audit
            WHERE original_record_id = :rid
            LIMIT 1
        """), {"rid": record.id}).first()

        if previous is not None:
            # WCCR STOCK RETURN DELETE V2.9
            # Distinguish reused IDs from the same cancelled operation.
            audit_rows = db.session.execute(text("""
                SELECT invoice_number, part_id,
                       original_record_json, canceled_at
                FROM issue_cancellation_audit
                WHERE original_record_id = :rid
                ORDER BY id DESC
            """), {"rid": record.id}).all()

            from datetime import datetime as _dt

            def _utc(value):
                if value is None:
                    raise ValueError("Missing audit timestamp")
                if isinstance(value, _dt):
                    result = value
                else:
                    result = _dt.fromisoformat(
                        str(value).replace("Z", "+00:00")
                    )
                if result.tzinfo is None:
                    result = result.replace(tzinfo=timezone.utc)
                return result.astimezone(timezone.utc)

            try:
                current_time = _utc(record.issue_date)

                for old_invoice, old_part, old_json, canceled_at in audit_rows:
                    historical = json.loads(old_json)
                    if not isinstance(historical, dict):
                        raise ValueError("Invalid audit snapshot")

                    old_time = _utc(historical.get("issue_date"))
                    cancel_time = _utc(canceled_at)

                    # WCCR RETURN IDENTITY V3.4
                    # An identical historical operation must
                    # never be reversed twice.
                    if (
                        str(old_invoice) == str(invoice_number)
                        and str(historical.get("issue_date"))
                            == str(record.issue_date)
                    ):
                        return reject(
                            "The same return was already cancelled."
                        )

                    # WCCR STOCK RETURN DELETE V3.5
                    # SQLite may reuse a deleted record ID.
                    # Historical operations from a DIFFERENT
                    # invoice are not this return's identity.
                    # Keep all same-invoice chronology checks.
                    if str(old_invoice) != str(invoice_number):
                        continue

                    if current_time <= cancel_time or current_time <= old_time:
                        return reject(
                            "Return chronology is ambiguous. "
                            "Inventory reconciliation required."
                        )

                    if old_part is None or historical.get("part_id") is None:
                        return reject(
                            "Historical part identity is incomplete."
                        )

                    if str(old_part) != str(historical.get("part_id")):
                        return reject(
                            "Historical audit identity mismatch."
                        )

            except (ValueError, TypeError, KeyError) as exc:
                return reject(
                    "Cannot verify previous cancellation: "
                    + str(exc)[:120]
                )


        snapshot = {}
        for column in record.__table__.columns:
            value = getattr(record, column.name)
            if isinstance(value, datetime):
                value = value.isoformat()
            snapshot[column.name] = value

        snapshot["_wccr_stock_return_source_id"] = source.id
        snapshot["_wccr_stock_return_batch_id"] = batch.id

        stock_after = stock_before - remove_qty

        # Conditional SQL update protects against concurrent changes.
        result = db.session.execute(text("""
            UPDATE part
               SET quantity = quantity - :qty
             WHERE id = :part_id
               AND quantity >= :qty
               AND quantity = :before
        """), {
            "qty": remove_qty,
            "part_id": part.id,
            "before": stock_before,
        })

        if result.rowcount != 1:
            raise RuntimeError(
                "Inventory changed concurrently. Cancellation stopped."
            )

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
                :rid, :invoice, :part_id, :quantity,
                :snapshot, :actor, :at, :before, :after,
                :reason
            )
        """), {
            "rid": record.id,
            "invoice": invoice_number,
            "part_id": record.part_id,
            "quantity": qty,
            "snapshot": json.dumps(
                snapshot, ensure_ascii=False, default=str
            ),
            "actor": str(actor),
            "at": datetime.now(timezone.utc).isoformat(),
            "before": stock_before,
            "after": stock_after,
            "reason": (
                "WCCR Return Delete V2.5 - "
                "Cancel STOCK return"
            ),
        })

        db.session.delete(record)
        db.session.flush()

        db.session.delete(batch)
        db.session.flush()

        db.session.commit()

        return True, (
            f"STOCK return invoice #{invoice_number} deleted. "
            f"Inventory: {stock_before} -> {stock_after}. "
            "Original issue preserved. Audit saved."
        )

    except Exception as exc:
        db.session.rollback()
        return False, (
            "Safe STOCK Return Delete failed: "
            + type(exc).__name__
            + ": "
            + str(exc)[:250]
        )
