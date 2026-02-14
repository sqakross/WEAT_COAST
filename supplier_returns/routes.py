# supplier_returns/routes.py
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from sqlalchemy import func as sa_func, and_, or_
from sqlalchemy.sql import exists
from extensions import db
from models import SupplierReturnBatch, SupplierReturnItem, Part
from services.supplier_returns_services import (
    recalc_batch_totals, post_batch, unpost_batch, SupplierReturnError
)

supplier_returns_bp = Blueprint("supplier_returns", __name__, url_prefix="/supplier_returns")


def _require_admin():
    role = (getattr(current_user, "role", "") or "").lower()
    return role in ("superadmin", "admin")


# ---------- API: lookup by Part # ----------
@supplier_returns_bp.route("/api/lookup")
@login_required
def api_part_lookup():
    if not _require_admin():
        return {"ok": False, "error": "forbidden"}, 403

    pn = (request.args.get("pn") or "").strip()
    if not pn:
        return {"ok": False, "error": "empty"}, 400

    p = Part.query.filter(sa_func.lower(Part.part_number) == pn.lower()).first()
    if not p:
        return {"ok": False, "error": "not_found"}, 404

    return {
        "ok": True,
        "data": {
            "part_number": p.part_number,
            "name": p.name,
            "unit_cost": float(p.unit_cost or 0.0),
            "location": p.location or "",
        },
    }

# ------ LIST ----------
@supplier_returns_bp.route("/", methods=["GET"])
@login_required
def list_returns():
    if not _require_admin():
        flash("Access denied", "danger")
        return redirect(url_for("inventory.dashboard"))

    supplier = (request.args.get("supplier") or "").strip()
    status   = (request.args.get("status") or "").strip()
    qtext    = (request.args.get("q") or "").strip()   # <-- NEW общий поиск

    q = SupplierReturnBatch.query

    # Supplier filter (как было)
    if supplier:
        q = q.filter(SupplierReturnBatch.supplier_name.ilike(f"%{supplier}%"))

    # Status filter (как было)
    if status in ("draft", "posted"):
        q = q.filter(SupplierReturnBatch.status == status)

    # NEW: full-text-ish search across batch + items (Part#/Name/Tech-Job)
    if qtext:
        like = f"%{qtext}%"

        # если ввели "#39" или "39" — дадим поиск по ID
        q_digits = (qtext or "").strip().lstrip("#")
        id_val = None
        if q_digits.isdigit():
            try:
                id_val = int(q_digits)
            except Exception:
                id_val = None

        item_match = exists().where(and_(
            SupplierReturnItem.batch_id == SupplierReturnBatch.id,
            or_(
                SupplierReturnItem.part_number.ilike(like),
                SupplierReturnItem.part_name.ilike(like),
                SupplierReturnItem.tech_note.ilike(like),
            )
        ))

        batch_match = or_(
            SupplierReturnBatch.supplier_name.ilike(like),
            item_match
        )

        if id_val is not None:
            batch_match = or_(SupplierReturnBatch.id == id_val, batch_match)

        q = q.filter(batch_match)

    rows = q.order_by(SupplierReturnBatch.id.desc()).all()

    return render_template(
        "supplier_returns/list.html",
        rows=rows,
        supplier=supplier,
        status=status,
        q=qtext,   # <-- NEW чтобы поле не очищалось
    )

# ---------- NEW ----------
@supplier_returns_bp.route("/new", methods=["GET", "POST"])
@login_required
def new_return():
    if not _require_admin():
        flash("Access denied", "danger")
        return redirect(url_for("inventory.dashboard"))

    if request.method == "POST":
        b = SupplierReturnBatch(
            supplier_name=(request.form.get("supplier_name") or "").strip(),
            status="draft",
            created_by=current_user.username if current_user.is_authenticated else None,
        )
        db.session.add(b)
        db.session.commit()
        flash("Created new supplier return draft.", "success")
        return redirect(url_for(".edit_return", batch_id=b.id))

    return render_template("supplier_returns/new.html")


# ---------- helpers ----------
def _get_action_from_form() -> str:
    """
    Возвращает 'post' | 'unpost' | 'save' на основе отправленной формы.
    Устойчиво к дубликатам и порядку полей.
    """
    vals = [v.lower().strip() for v in request.form.getlist("action")]
    # если в форме несколько action (скрытый + кнопка), смотрим последним
    for v in reversed(vals):
        if v in ("post", "unpost", "save"):
            return v
    # запасной путь — вдруг было одно поле
    single = (request.form.get("action") or "").lower().strip()
    if single in ("post", "unpost", "save"):
        return single
    # финальный дефолт
    return "save"


def _save_rows_from_request(b: SupplierReturnBatch) -> dict[int, str]:
    """
    Перечитывает строки из формы -> b.items.
    Возвращает {row_index: error_text} ТОЛЬКО по индексам (без ошибок сервиса).
    """
    # Сначала очищаем старые строки этого батча
    SupplierReturnItem.query.filter_by(batch_id=b.id).delete()
    db.session.flush()

    pns   = request.form.getlist("part_number[]")
    names = request.form.getlist("part_name[]")
    qtys  = request.form.getlist("qty_returned[]")
    costs = request.form.getlist("unit_cost[]")
    locs  = request.form.getlist("location[]")
    techs = request.form.getlist("tech_note[]")   # per-row Tech / Job

    errors_by_index: dict[int, str] = {}

    for idx, (pn, nm, q, c, loc) in enumerate(zip(pns, names, qtys, costs, locs)):
        pn = (pn or "").strip()
        if not pn:
            # пустую строку не сохраняем вообще
            continue

        part = Part.query.filter(sa_func.lower(Part.part_number) == pn.lower()).first()

        # числа
        try:
            qv = max(0, int(q or "0"))
        except Exception:
            qv = 0
        try:
            cv = float(c or "0")
        except Exception:
            cv = 0.0

        # локация (нормализация)
        loc_norm = (loc or "").strip()
        if loc_norm.lower() == "auto":
            loc_norm = ""
        else:
            import re
            loc_norm = re.sub(r"\s*\/\s*", "/", loc_norm)  # A2 / B2 -> A2/B2
            loc_norm = re.sub(r"\s+", " ", loc_norm)       # множественные пробелы -> один

        # Tech / Job (per row)
        tech_raw = techs[idx] if idx < len(techs) else None
        tech_note = (tech_raw or "").strip() or None

        # автозаполнение/нормализация
        if part:
            if not (cv > 0):
                cv = float(part.unit_cost or 0.0)
            if not loc_norm:
                loc_norm = part.location or ""
            part_name = (nm or "").strip() or (part.name or "")
            pn = part.part_number  # нормализуем регистр
        else:
            errors_by_index[idx] = "Part not found"
            part_name = (nm or "").strip()

        db.session.add(
            SupplierReturnItem(
                batch_id=b.id,
                part_number=pn,
                part_name=part_name,
                qty_returned=qv,
                unit_cost=cv,
                location=loc_norm,
                tech_note=tech_note,
            )
        )

    db.session.commit()
    return errors_by_index


# ---------- EDIT ----------
@supplier_returns_bp.route("/<int:batch_id>/edit", methods=["GET", "POST"])
@login_required
def edit_return(batch_id: int):
    if not _require_admin():
        flash("Access denied", "danger")
        return redirect(url_for("inventory.dashboard"))

    b = SupplierReturnBatch.query.get_or_404(batch_id)

    if request.method == "POST":
        action = _get_action_from_form()
        # Запомним, был ли батч уже POSTED до любых действий
        was_posted = (b.status or "draft") == "posted"

        # Если батч был POSTED и жмём Save/Post —
        # сначала делаем UNPOST, чтобы вернуть старые количества на склад,
        # потом пересохраним строки и (для Save) заново POST.
        if was_posted and action in ("save", "post"):
            try:
                res_un = unpost_batch(batch_id=b.id, actor=current_user.username)
            except SupplierReturnError as e:
                flash(str(e), "danger")
                return render_template(
                    "supplier_returns/edit.html",
                    b=b,
                    errors_idx=set(),
                )

            if not res_un.get("ok"):
                flash(
                    "; ".join(res_un.get("errors", {}).values()) or "Cannot unpost",
                    "danger",
                )
                return render_template(
                    "supplier_returns/edit.html",
                    b=b,
                    errors_idx=set(),
                )

            # обновим объект батча в сессии
            db.session.refresh(b)

        # --- 0) supplier ---
        b.supplier_name = (request.form.get("supplier_name") or "").strip()

        # --- 0.1) per-row Tech / Job (массив) ---
        tech_notes_raw = request.form.getlist("tech_note[]")  # ["Tech 1", "Job 123", ...]

        # 1) сохраняем строки (получаем ТОЛЬКО индексные ошибки)
        errors_by_index = _save_rows_from_request(b)  # {0:"...", 2:"..."}

        # 1.1) после того, как _save_rows_from_request актуализировал b.items,
        #      развешиваем tech_note по строкам
        if tech_notes_raw:
            for idx, it in enumerate(b.items):
                if idx >= len(tech_notes_raw):
                    break
                note = (tech_notes_raw[idx] or "").strip()
                it.tech_note = note or None

        # 2) единоразово считаем агрегаты/валидацию сервиса (ошибки по ID)
        svc_errs_by_id = recalc_batch_totals(b) or {}  # {item_id: "..."}

        # 3) строим объединённый набор индексов ошибок для шаблона
        errors_idx = set(errors_by_index.keys())
        if svc_errs_by_id:
            id_to_index = {it.id: i for i, it in enumerate(b.items)}
            for item_id in svc_errs_by_id.keys():
                i = id_to_index.get(item_id)
                if i is not None:
                    errors_idx.add(i)

        # ----- ветки действий -----

        # SAVE
        if action == "save":
            # Случай обычного черновика — логика как раньше
            if not was_posted:
                db.session.commit()
                flash(
                    "Draft saved." + (" Fix rows before posting." if errors_idx else ""),
                    "warning" if errors_idx else "success",
                )
                return redirect(url_for(".edit_return", batch_id=b.id))

            # Сюда попадаем, если батч БЫЛ posted и мы его редактируем.
            # Мы уже сделали UNPOST выше. Теперь нам нужно:
            #   - если есть ошибки — показать их, НЕ постить снова (батч остаётся draft)
            #   - если ошибок нет — вызвать post_batch ещё раз, чтобы заново списать склад.
            if errors_idx:
                flash("Cannot re-post until you fix highlighted rows.", "danger")
                return render_template(
                    "supplier_returns/edit.html",
                    b=b,
                    errors_idx=errors_idx,
                )

            try:
                db.session.flush()
                res = post_batch(batch_id=b.id, actor=current_user.username)
            except SupplierReturnError as e:
                flash(str(e), "danger")
                return render_template(
                    "supplier_returns/edit.html",
                    b=b,
                    errors_idx=set(),
                )

            if not res.get("ok"):
                # ошибки по ID → маппим в индексы
                svc_errs_by_id = res.get("errors") or {}
                id_to_index = {it.id: i for i, it in enumerate(b.items)}
                errors_idx = {
                    id_to_index[i]
                    for i in svc_errs_by_id.keys()
                    if i in id_to_index
                }
                flash("Cannot re-post until you fix highlighted rows.", "danger")
                return render_template(
                    "supplier_returns/edit.html",
                    b=b,
                    errors_idx=errors_idx,
                )

            flash("Posted supplier return updated: stock adjusted.", "success")
            return redirect(url_for(".edit_return", batch_id=b.id))

        # POST (списание со склада из draft-состояния)
        if action == "post":
            if errors_idx:
                # показать подсветку проблемных строк
                return render_template(
                    "supplier_returns/edit.html",
                    b=b,
                    errors_idx=errors_idx,
                )

            try:
                db.session.flush()
                res = post_batch(batch_id=b.id, actor=current_user.username)
            except SupplierReturnError as e:
                flash(str(e), "danger")
                return render_template(
                    "supplier_returns/edit.html",
                    b=b,
                    errors_idx=set(),
                )

            if not res.get("ok"):
                # ошибки по ID → маппим в индексы
                svc_errs_by_id = res.get("errors") or {}
                id_to_index = {it.id: i for i, it in enumerate(b.items)}
                errors_idx = {
                    id_to_index[i]
                    for i in svc_errs_by_id.keys()
                    if i in id_to_index
                }
                return render_template(
                    "supplier_returns/edit.html",
                    b=b,
                    errors_idx=errors_idx,
                )

            from datetime import date
            today_str = date.today().strftime("%Y-%m-%d")

            flash("Posted: stock decremented.", "success")
            return redirect(
                url_for(
                    "inventory.reports_grouped",
                    start_date=today_str,
                    end_date=today_str,
                    recipient=b.supplier_name or "",
                )
            )

        # UNPOST (вернуть на склад вручную кнопкой UNPOST)
        if action == "unpost":
            try:
                res = unpost_batch(batch_id=b.id, actor=current_user.username)
            except SupplierReturnError as e:
                flash(str(e), "danger")
                return render_template(
                    "supplier_returns/edit.html",
                    b=b,
                    errors_idx=set(),
                )

            if not res.get("ok"):
                flash(
                    "; ".join(res.get("errors", {}).values()) or "Cannot unpost",
                    "danger",
                )
            else:
                flash("Unposted: stock restored.", "success")
            return redirect(url_for(".edit_return", batch_id=b.id))

        # fallback -> save (на всякий случай)
        db.session.commit()
        flash(
            "Draft saved." + (" Fix rows before posting." if errors_idx else ""),
            "warning" if errors_idx else "success",
        )
        return redirect(url_for(".edit_return", batch_id=b.id))

    # GET
    recalc_batch_totals(b)
    return render_template("supplier_returns/edit.html", b=b, errors_idx=set())


# ---------- DELETE ----------
@supplier_returns_bp.route("/<int:batch_id>/delete", methods=["POST"])
@login_required
def delete_return(batch_id: int):
    if not _require_admin():
        flash("Access denied", "danger")
        return redirect(url_for("inventory.dashboard"))

    b = SupplierReturnBatch.query.get_or_404(batch_id)
    if (b.status or "draft") != "draft":
        flash("Only draft can be deleted.", "warning")
        return redirect(url_for(".edit_return", batch_id=batch_id))

    SupplierReturnItem.query.filter_by(batch_id=b.id).delete()
    SupplierReturnBatch.query.filter_by(id=b.id).delete()
    db.session.commit()
    flash("Draft deleted.", "success")
    return redirect(url_for(".list_returns"))
