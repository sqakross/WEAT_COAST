from flask import render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from extensions import db
from models import SupplierReturnBatch, SupplierReturnItem
from supplier_returns_services import apply_supplier_return, rollback_supplier_return, SupplierReturnError

from . import supplier_returns_bp


# --- utils ---
def _admins_only():
    role = (getattr(current_user, "role", "") or "").lower()
    return role in ("admin", "superadmin")


# --- list all returns ---
@supplier_returns_bp.route("/", methods=["GET"])
@login_required
def list_returns():
    if not _admins_only():
        flash("Access denied", "danger")
        return redirect(url_for("inventory.dashboard"))

    supplier = (request.args.get("supplier") or "").strip()
    status = (request.args.get("status") or "").strip()

    q = SupplierReturnBatch.query
    if supplier:
        q = q.filter(SupplierReturnBatch.supplier_name.ilike(f"%{supplier}%"))
    if status:
        q = q.filter(SupplierReturnBatch.status == status)

    rows = q.order_by(SupplierReturnBatch.id.desc()).limit(200).all()
    return render_template("supplier_returns/list.html", rows=rows, supplier=supplier, status=status)


# --- create new batch ---
@supplier_returns_bp.route("/new")
@login_required
def new_return():
    if not _admins_only():
        flash("Access denied", "danger")
        return redirect(url_for("inventory.dashboard"))

    b = SupplierReturnBatch()
    db.session.add(b)
    db.session.commit()
    flash("Created new supplier return draft.", "success")
    return redirect(url_for("supplier_returns.edit_return", batch_id=b.id))


# --- edit batch ---
@supplier_returns_bp.route("/<int:batch_id>/edit", methods=["GET", "POST"])
@login_required
def edit_return(batch_id):
    if not _admins_only():
        flash("Access denied", "danger")
        return redirect(url_for("inventory.dashboard"))

    b = SupplierReturnBatch.query.get_or_404(batch_id)

    if request.method == "POST":
        b.supplier_name = (request.form.get("supplier_name") or "").strip()
        part_numbers = request.form.getlist("part_number[]")
        part_names = request.form.getlist("part_name[]")
        qtys = request.form.getlist("qty_returned[]")
        costs = request.form.getlist("unit_cost[]")
        locs = request.form.getlist("location[]")

        b.items.clear()
        total_value = 0.0

        for i in range(len(part_numbers)):
            pn = (part_numbers[i] or "").strip()
            if not pn:
                continue
            qty = int(qtys[i] or 0)
            cost = float(costs[i] or 0.0)
            total = qty * cost

            item = SupplierReturnItem(
                part_number=pn,
                part_name=(part_names[i] or "").strip(),
                qty_returned=qty,
                unit_cost=cost,
                total_cost=total,
                location=(locs[i] or "").strip(),
            )
            b.items.append(item)
            total_value += total

        b.total_items = len(b.items)
        b.total_value = total_value
        db.session.commit()
        flash("Saved.", "success")

    return render_template("supplier_returns/edit.html", b=b)


# --- post / unpost ---
@supplier_returns_bp.route("/<int:batch_id>/toggle", methods=["POST"])
@login_required
def toggle_return(batch_id):
    if not _admins_only():
        flash("Access denied", "danger")
        return redirect(url_for("inventory.dashboard"))

    try:
        b = SupplierReturnBatch.query.get_or_404(batch_id)
        if b.status == "posted":
            rollback_supplier_return(b.id, getattr(current_user, "username", None))
            flash("Unposted and stock restored.", "warning")
        else:
            apply_supplier_return(b.id, getattr(current_user, "username", None))
            flash("Supplier return posted (stock decremented).", "success")
    except SupplierReturnError as e:
        db.session.rollback()
        flash(str(e), "danger")
    except Exception as e:
        db.session.rollback()
        flash(f"Error: {e}", "danger")

    return redirect(url_for("supplier_returns.edit_return", batch_id=batch_id))


# --- delete draft ---
@supplier_returns_bp.route("/<int:batch_id>/delete", methods=["POST"])
@login_required
def delete_return(batch_id):
    if not _admins_only():
        flash("Access denied", "danger")
        return redirect(url_for("inventory.dashboard"))

    b = SupplierReturnBatch.query.get_or_404(batch_id)
    if b.status != "draft":
        flash("Only draft returns can be deleted.", "danger")
        return redirect(url_for("supplier_returns.edit_return", batch_id=batch_id))

    db.session.delete(b)
    db.session.commit()
    flash("Deleted.", "success")
    return redirect(url_for("supplier_returns.list_returns"))
