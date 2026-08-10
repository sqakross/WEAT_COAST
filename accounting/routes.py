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
    from services.statement_matching_helper import (
        find_return_candidates,
        validate_component_total,
    )

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

@accounting_bp.route(
    "/statements/upload",
    methods=["GET", "POST"],
)
@login_required
def statement_upload():
    if not _accounting_access_required():
        flash("Access denied", "danger")
        return redirect(url_for("inventory.wo_list"))

    if request.method == "GET":
        return render_template(
            "accounting/statement_upload.html",
        )

    import os
    import tempfile

    from extensions import db
    from services.statement_reconciliation_service import (
        parse_statement_pdf,
        save_parsed_statement,
    )

    uploaded_file = request.files.get("statement_file")

    if uploaded_file is None or not uploaded_file.filename:
        flash("Please select a statement PDF.", "danger")
        return redirect(url_for("accounting.statement_upload"))

    original_filename = uploaded_file.filename.strip()

    if not original_filename.lower().endswith(".pdf"):
        flash("Only PDF statements are supported.", "danger")
        return redirect(url_for("accounting.statement_upload"))

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            prefix="supplier_statement_",
            suffix=".pdf",
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
            uploaded_file.save(temporary_path)

        parsed_statement = parse_statement_pdf(
            temporary_path
        )

        from sqlalchemy import func
        from models import SupplierStatement

        supplier_name = (
                parsed_statement.supplier_name or ""
        ).strip().lower()

        account_number = (
                parsed_statement.account_number or ""
        ).strip()

        balance_due = round(
            float(parsed_statement.balance_due or 0.0),
            2,
        )

        existing = (
            SupplierStatement.query
            .filter(
                func.lower(
                    func.trim(
                        SupplierStatement.supplier_name
                    )
                ) == supplier_name,

                SupplierStatement.statement_period
                == parsed_statement.statement_period,

                func.coalesce(
                    SupplierStatement.account_number,
                    "",
                ) == account_number,

                func.abs(
                    SupplierStatement.balance_due - balance_due
                ) <= 0.01,
            )
            .first()
        )

        if existing:
            flash(
                f"This supplier statement is already imported "
                f"(Statement #{existing.id}).",
                "warning",
            )

            return redirect(
                url_for(
                    "accounting.statement_view",
                    statement_id=existing.id,
                )
            )

        statement = save_parsed_statement(
            parsed_statement,
            source_file=original_filename,
            created_by=getattr(current_user, "id", None),
        )

        flash(
            f"{statement.supplier_name} statement imported successfully. "
            f"{len(statement.lines)} credit lines were saved.",
            "success",
        )

        return redirect(url_for(
            "accounting.statement_view",
            statement_id=statement.id,
        ))

    except Exception as exc:
        db.session.rollback()

        flash(
            str(exc) or "Statement import failed.",
            "danger",
        )

        return redirect(url_for(
            "accounting.statement_upload",
        ))

    finally:
        if temporary_path and os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass

@accounting_bp.route(
    "/statements/line/<int:line_id>/split",
    methods=["GET", "POST"],
)
@login_required
def statement_line_split(line_id):
    if not _accounting_access_required():
        flash("Access denied", "danger")
        return redirect(
            url_for("inventory.wo_list")
        )

    from extensions import db

    from models import (
        SupplierStatementLine,
        SupplierStatementLineComponent,
    )

    from services.statement_matching_helper import (
        find_return_candidates,
        validate_component_total,
    )

    line = (
        SupplierStatementLine.query
        .get_or_404(line_id)
    )

    supplier_name = (
            line.supplier_name
            or line.statement.supplier_name
            or ""
    ).strip().lower()

    # Convert statement supplier name to the exact
    # ReturnDestination name stored in our database.
    supplier_lookup_name = supplier_name

    if supplier_name in {
        "reliable parts",
        "reliable parts inc",
        "reliable",
        "rel",
    }:
        supplier_lookup_name = "reliable"

    elif supplier_name in {
        "marcone",
        "mar",
    }:
        supplier_lookup_name = "marcone"

    # ========================================================
    # GET
    # Show existing split components and possible candidates.
    # ========================================================
    if request.method == "GET":

        used_by_other_lines = {
            int(record_id)
            for (record_id,) in (
                db.session.query(
                    SupplierStatementLineComponent
                    .matched_issued_part_record_id
                )
                .filter(
                    SupplierStatementLineComponent
                    .matched_issued_part_record_id
                    .isnot(None),

                    SupplierStatementLineComponent
                    .statement_line_id
                    != line.id,
                )
                .all()
            )
            if record_id is not None
        }

        candidate_map = {}

        for component in line.components or []:

            candidate_results = (
                find_return_candidates(
                    supplier_name=supplier_lookup_name,
                    amount=float(
                        component.amount or 0.0
                    ),
                    excluded_ids=used_by_other_lines,
                )
            )

            # statement_line_split.html currently expects
            # IssuedPartRecord objects, not ReturnCandidate DTOs.
            candidate_map[component.id] = [
                candidate.record
                for candidate in candidate_results
            ]

        return render_template(
            "accounting/statement_line_split.html",
            line=line,
            candidate_map=candidate_map,
        )

    # ========================================================
    # POST
    # Validate, replace and save split components.
    # ========================================================
    amount_values = request.form.getlist(
        "component_amount"
    )

    selected_record_values = request.form.getlist(
        "component_record_id"
    )

    try:
        amounts: list[float] = []

        for raw_value in amount_values:
            value = (
                raw_value or ""
            ).strip()

            if not value:
                continue

            amount = round(
                float(value),
                2,
            )

            if amount <= 0:
                raise ValueError(
                    "Every return amount must be "
                    "greater than zero."
                )

            amounts.append(amount)

        if not amounts:
            raise ValueError(
                "Add at least one return amount."
            )

        statement_amount = round(
            abs(
                float(
                    line.credit_amount or 0.0
                )
            ),
            2,
        )

        validate_component_total(
            statement_amount=statement_amount,
            component_amounts=amounts,
        )

        while (
            len(selected_record_values)
            < len(amounts)
        ):
            selected_record_values.append("")

        used_by_other_lines = {
            int(record_id)
            for (record_id,) in (
                db.session.query(
                    SupplierStatementLineComponent
                    .matched_issued_part_record_id
                )
                .filter(
                    SupplierStatementLineComponent
                    .matched_issued_part_record_id
                    .isnot(None),

                    SupplierStatementLineComponent
                    .statement_line_id
                    != line.id,
                )
                .all()
            )
            if record_id is not None
        }

        # Replace existing components only after all basic
        # validation has passed.
        for component in list(
            line.components or []
        ):
            db.session.delete(component)

        db.session.flush()

        selected_in_current_split: set[int] = set()

        for index, amount in enumerate(amounts):

            selected_raw = (
                selected_record_values[index]
                or ""
            ).strip()

            excluded_ids = (
                used_by_other_lines
                | selected_in_current_split
            )

            candidate_results = (
                find_return_candidates(
                    supplier_name=supplier_lookup_name,
                    amount=amount,
                    excluded_ids=excluded_ids,
                )
            )

            candidate_ids = {
                candidate.record.id
                for candidate in candidate_results
            }

            matched_record_id = None
            note = None

            if selected_raw:
                selected_id = int(
                    selected_raw
                )

                if selected_id not in candidate_ids:
                    raise ValueError(
                        "The selected return for "
                        f"${amount:,.2f} is no longer "
                        "available."
                    )

                matched_record_id = selected_id

                selected_in_current_split.add(
                    selected_id
                )

            elif len(candidate_results) == 1:
                matched_record_id = (
                    candidate_results[0]
                    .record
                    .id
                )

                selected_in_current_split.add(
                    matched_record_id
                )

            elif len(candidate_results) == 0:
                note = (
                    "No matching supplier return "
                    "was found."
                )

            else:
                note = (
                    f"{len(candidate_results)} possible "
                    "supplier returns were found for "
                    f"${amount:,.2f}. "
                    "Select the correct return."
                )

            db.session.add(
                SupplierStatementLineComponent(
                    statement_line_id=line.id,
                    amount=amount,
                    matched_issued_part_record_id=(
                        matched_record_id
                    ),
                    note=note,
                    created_by=getattr(
                        current_user,
                        "id",
                        None,
                    ),
                )
            )

        db.session.commit()

        # Reload so relationship reflects the committed rows.
        db.session.refresh(line)

        saved_components = list(
            line.components or []
        )

        matched_count = sum(
            1
            for component in saved_components
            if (
                component
                .matched_issued_part_record_id
            )
        )

        total_count = len(
            saved_components
        )

        if (
            total_count > 0
            and matched_count == total_count
        ):
            flash(
                f"Split saved. All {matched_count} "
                "return amounts were matched.",
                "success",
            )

        else:
            flash(
                f"Split saved. {matched_count} of "
                f"{total_count} return amounts were "
                "matched. Select candidates for the rest.",
                "warning",
            )

        return redirect(
            url_for(
                "accounting.statement_line_split",
                line_id=line.id,
            )
        )

    except Exception as exc:
        db.session.rollback()

        flash(
            str(exc)
            or "Could not save split credit.",
            "danger",
        )

        return redirect(
            url_for(
                "accounting.statement_line_split",
                line_id=line.id,
            )
        )

@accounting_bp.route(
    "/statements/line/<int:line_id>/invoice-match",
    methods=["GET", "POST"],
)
@login_required
def statement_line_invoice_match(
    line_id,
):
    if not _accounting_access_required():
        flash(
            "Access denied",
            "danger",
        )
        return redirect(
            url_for(
                "inventory.wo_list"
            )
        )

    from services.statement_invoice_matching_service import (
        load_page,
        save_components,
    )

    if request.method == "POST":

        try:

            save_components(
                line_id=line_id,
                form=request.form,
                current_user=current_user,
            )

        except Exception as exc:

            flash(
                str(exc),
                "danger",
            )

        return redirect(
            url_for(
                "accounting.statement_line_invoice_match",
                line_id=line_id,
            )
        )

    view = load_page(
        line_id=line_id,
    )

    return render_template(
        "accounting/statement_line_invoice_match.html",
        view=view,
    )