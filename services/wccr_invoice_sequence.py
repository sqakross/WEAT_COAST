
"""WCCR Invoice Sequence V1B.

All reservations must occur in the same SQLAlchemy transaction
that creates the corresponding invoice.

The counter is monotonic across committed transactions.
Rolled-back transactions may reuse a number that was never committed.
"""

from sqlalchemy import text


def reserve_invoice_number(session):
    """Reserve one invoice number in the caller's transaction.

    Requires SQLite 3.35+ for RETURNING.
    The caller must commit the transaction.
    """
    connection = session.connection()

    dialect = connection.dialect.name
    if dialect != "sqlite":
        raise RuntimeError(
            "WCCR invoice sequence requires SQLite"
        )

    result = connection.execute(text("""
        UPDATE wccr_invoice_sequence
        SET last_number = (
            SELECT MAX(candidate)
            FROM (
                SELECT last_number + 1 AS candidate
                FROM wccr_invoice_sequence
                WHERE id = 1

                UNION ALL

                SELECT COALESCE(MAX(invoice_number), 0) + 1
                FROM issued_batch

                UNION ALL

                SELECT COALESCE(MAX(invoice_number), 0) + 1
                FROM issued_part_record
            )
        )
        WHERE id = 1
        RETURNING last_number
    """)).scalar_one_or_none()

    if result is None:
        raise RuntimeError(
            "WCCR invoice sequence is not initialized"
        )

    return int(result)
