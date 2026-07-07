from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user

accounting_bp = Blueprint(
    "accounting",
    __name__,
    url_prefix="/accounting",
)


def _accounting_access_required():
    role = (getattr(current_user, "role", "") or "").strip().lower()
    return role in ("admin", "superadmin")


@accounting_bp.get("/technicians")
@login_required
def technician_balances():
    if not _accounting_access_required():
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))

    from services.accounting_service import get_technician_balances

    q = (request.args.get("q") or "").strip().lower()
    sort = (request.args.get("sort") or "remaining_desc").strip()

    rows = get_technician_balances()

    if q:
        rows = [
            r for r in rows
            if q in (r.technician or "").lower()
        ]

    if sort == "tech_asc":
        rows = sorted(rows, key=lambda r: (r.technician or "").lower())
    elif sort == "invoices_desc":
        rows = sorted(rows, key=lambda r: r.open_count, reverse=True)
    else:
        rows = sorted(rows, key=lambda r: r.remaining_amount, reverse=True)

    return render_template(
        "accounting/technician_balances.html",
        rows=rows,
        filters={
            "q": request.args.get("q", ""),
            "sort": sort,
        },
    )

@accounting_bp.get("/technicians/<technician_name>")
@login_required
def technician_ledger(technician_name):
    if not _accounting_access_required():
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))

    from services.accounting_service import (
        get_technician_ledger,
        get_technician_summary,
    )

    rows = get_technician_ledger(technician_name)

    summary = get_technician_summary(technician_name)

    return render_template(
        "accounting/technician_ledger.html",
        technician_name=technician_name,
        rows=rows,
        summary=summary,
    )

@accounting_bp.route(
    "/technicians/<technician_name>/payment/new",
    methods=["GET", "POST"],
)
@login_required
def payment_new(technician_name):
    if not _accounting_access_required():
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))

    from datetime import date
    from services.accounting_service import preview_technician_payment_fifo

    amount_raw = (request.form.get("amount") or request.args.get("amount") or "").strip()

    preview = None
    error = None

    if amount_raw:
        try:
            preview = preview_technician_payment_fifo(
                technician_name=technician_name,
                amount=float(amount_raw),
            )
        except Exception as e:
            error = str(e)

    return render_template(
        "accounting/payment_new.html",
        technician_name=technician_name,
        today=date.today().isoformat(),
        amount_raw=amount_raw,
        preview=preview,
        error=error,
    )

@accounting_bp.post("/technicians/<technician_name>/payment")
@login_required
def payment_create(technician_name):
    if not _accounting_access_required():
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))

    from datetime import datetime
    from services.accounting_service import create_technician_payment_fifo
    from extensions import db

    try:
        amount = float(request.form.get("amount") or 0)
        payment_date_raw = request.form.get("payment_date") or ""
        payment_date = datetime.strptime(payment_date_raw, "%Y-%m-%d").date()

        payment = create_technician_payment_fifo(
            technician_name=technician_name,
            amount=amount,
            payment_date=payment_date,
            method=request.form.get("method"),
            reference=request.form.get("reference"),
            note=request.form.get("note"),
            created_by=getattr(current_user, "id", None),
        )

        flash(f"Payment #{payment.id} posted successfully.", "success")
        return redirect(url_for(
            "accounting.technician_ledger",
            technician_name=technician_name,
        ))

    except Exception as e:
        db.session.rollback()
        flash(str(e), "danger")
        return redirect(url_for(
            "accounting.payment_new",
            technician_name=technician_name,
        ))