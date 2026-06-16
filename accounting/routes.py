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
        get_technician_balances,
    )

    rows = get_technician_ledger(technician_name)

    balance = next(
        (
            x for x in get_technician_balances()
            if x.technician == technician_name
        ),
        None,
    )

    return render_template(
        "accounting/technician_ledger.html",
        technician_name=technician_name,
        rows=rows,
        balance=balance,
    )