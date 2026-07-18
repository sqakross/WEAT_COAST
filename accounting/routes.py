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
        get_technician_payments,
    )

    show_paid = (
        request.args.get("show_paid") or ""
    ).strip().lower() in {"1", "true", "yes", "on"}

    all_rows = get_technician_ledger(technician_name)
    payments = get_technician_payments(technician_name)
    summary = get_technician_summary(technician_name)

    if show_paid:
        rows = all_rows
    else:
        rows = [
            row
            for row in all_rows
            if (row.status or "").strip().lower()
            in {"open", "partial"}
        ]

    payment_total = round(
        sum(
            float(payment.amount or 0.0)
            for payment in payments
            if (payment.status or "").lower() == "posted"
        ),
        2,
    )

    payment_applied = round(
        sum(
            float(payment.applied_amount or 0.0)
            for payment in payments
            if (payment.status or "").lower() == "posted"
        ),
        2,
    )

    payment_unapplied = round(
        sum(
            float(payment.unapplied_amount or 0.0)
            for payment in payments
            if (payment.status or "").lower() == "posted"
        ),
        2,
    )

    return render_template(
        "accounting/technician_ledger.html",
        technician_name=technician_name,
        rows=rows,
        payments=payments,
        summary=summary,
        show_paid=show_paid,
        payment_total=payment_total,
        payment_applied=payment_applied,
        payment_unapplied=payment_unapplied,
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

@accounting_bp.get("/technicians/<technician_name>/adjustment/new")
@login_required
def adjustment_new(technician_name):
    if not _accounting_access_required():
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))

    from services.accounting_service import get_technician_summary

    summary = get_technician_summary(technician_name)

    return render_template(
        "accounting/adjustment_new.html",
        technician_name=technician_name,
        summary=summary,
    )

@accounting_bp.post("/technicians/<technician_name>/adjustment")
@login_required
def adjustment_create(technician_name):
    if not _accounting_access_required():
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))

    from extensions import db
    from services.accounting_service import create_technician_adjustment

    try:
        adjustment_type = (
            request.form.get("adjustment_type") or ""
        ).strip().upper()

        amount = float(request.form.get("amount") or 0)
        reason = (request.form.get("reason") or "").strip()

        if not reason:
            raise ValueError("Adjustment reason is required")

        entry = create_technician_adjustment(
            technician_name=technician_name,
            amount=amount,
            adjustment_type=adjustment_type,
            reason=reason,
            created_by=getattr(current_user, "id", None),
        )

        flash(
            f"Adjustment #{entry.id} posted successfully.",
            "success",
        )

        return redirect(url_for(
            "accounting.technician_ledger",
            technician_name=technician_name,
        ))

    except Exception as e:
        db.session.rollback()
        flash(str(e) or "Adjustment failed.", "danger")

        return redirect(url_for(
            "accounting.adjustment_new",
            technician_name=technician_name,
        ))


@accounting_bp.get(
    "/technicians/<technician_name>/opening-balance/new"
)
@login_required
def opening_balance_new(technician_name):
    if not _accounting_access_required():
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))

    from datetime import date
    from models import TechnicianLedgerEntry

    tech = (technician_name or "").strip()

    existing = (
        TechnicianLedgerEntry.query
        .filter(
            TechnicianLedgerEntry.technician_name == tech,
            TechnicianLedgerEntry.entry_type == "OPENING_BALANCE",
            TechnicianLedgerEntry.voided == False,
        )
        .first()
    )

    return render_template(
        "accounting/opening_balance_new.html",
        technician_name=tech,
        today=date.today().isoformat(),
        existing=existing,
    )


@accounting_bp.post(
    "/technicians/<technician_name>/opening-balance"
)
@login_required
def opening_balance_create(technician_name):
    if not _accounting_access_required():
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))

    from datetime import datetime
    from extensions import db
    from services.accounting_service import (
        create_technician_opening_balance,
    )

    try:
        amount_raw = (request.form.get("amount") or "").strip()
        date_raw = (
                request.form.get("opening_date") or ""
        ).strip()

        if not amount_raw:
            raise ValueError("Opening balance amount is required")

        if not date_raw:
            raise ValueError("Opening balance date is required")

        amount = float(amount_raw)

        opening_date = datetime.strptime(
            date_raw,
            "%Y-%m-%d",
        ).date()

        entry = create_technician_opening_balance(
            technician_name=technician_name,
            amount=amount,
            opening_date=opening_date,
            note=request.form.get("note"),
            created_by=getattr(current_user, "id", None),
        )

        flash(
            f"Opening balance #{entry.id} posted successfully.",
            "success",
        )

        return redirect(url_for(
            "accounting.technician_ledger",
            technician_name=technician_name,
        ))

    except Exception as e:
        db.session.rollback()

        flash(
            str(e) or "Opening balance failed.",
            "danger",
        )

        return redirect(url_for(
            "accounting.opening_balance_new",
            technician_name=technician_name,
        ))

@accounting_bp.post("/payments/<int:payment_id>/void")
@login_required
def payment_void(payment_id):
    if not _accounting_access_required():
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))

    from extensions import db
    from services.accounting_service import (
        void_technician_payment,
    )

    technician_name = (
        request.form.get("technician_name") or ""
    ).strip()

    try:
        reason = (
            request.form.get("void_reason") or ""
        ).strip()

        payment = void_technician_payment(
            payment_id=payment_id,
            void_reason=reason,
            voided_by=getattr(current_user, "id", None),
        )

        flash(
            f"Payment #{payment.id} was voided successfully. "
            "All invoice allocations were reversed.",
            "success",
        )

    except Exception as e:
        db.session.rollback()

        flash(
            str(e) or "Payment void failed.",
            "danger",
        )

    if technician_name:
        return redirect(url_for(
            "accounting.technician_ledger",
            technician_name=technician_name,
        ))

    return redirect(url_for(
        "accounting.technician_balances",
    ))

@accounting_bp.get("/statements")
@login_required
def supplier_statements():
    if not _accounting_access_required():
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))

    from models import SupplierStatement

    rows = (
        SupplierStatement.query
        .order_by(
            SupplierStatement.statement_period.desc(),
            SupplierStatement.created_at.desc(),
        )
        .all()
    )

    return render_template(
        "accounting/supplier_statements.html",
        rows=rows,
    )

@accounting_bp.get("/")
@login_required
def dashboard():
    if not _accounting_access_required():
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))

    return render_template(
        "accounting/dashboard.html",
    )

@accounting_bp.get("/statements/<int:statement_id>")
@login_required
def statement_view(statement_id):
    if not _accounting_access_required():
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))

    from models import SupplierStatement

    from services.statement_matching_service import (
        build_statement_view,
    )

    view = build_statement_view(statement_id)

    return render_template(
        "accounting/statement_view.html",
        view=view,
    )