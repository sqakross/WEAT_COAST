import pdfplumber
from flask import (
    Blueprint, render_template, request, redirect, url_for,
    flash, send_file, jsonify, after_this_request,
    current_app,                     # NEW
)
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import or_
from urllib.parse import urlencode

import os
import pandas as pd
from io import BytesIO
from datetime import datetime
from collections import defaultdict

from config import Config
from extensions import db
from models import Part, IssuedPartRecord, User
from utils.invoice_generator import generate_invoice_pdf

# PDF (ReportLab)
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# Сравнение корзин / экспорт
from compare_cart.run_compare import get_marcone_items, check_cart_items, export_to_docx
from compare_cart.run_compare_reliable import get_reliable_items

# Импорт на склад (наш новый функционал)
from .import_rules import load_table, normalize_table, build_receive_movements
from .import_ledger import has_key, add_key
                              # NEW




UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)




from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet



inventory_bp = Blueprint('inventory', __name__)
EPORT_PATH = os.path.join(os.path.dirname(__file__), '..', 'marcone_inventory_report.docx')

def dataframe_from_pdf(path):
    """
    Пытается вытащить таблицы из 'живого' PDF.
    Возвращает pandas.DataFrame со СТРОКАМИ, объединёнными со всех страниц.
    Если таблиц нет (вероятно, скан) — возвращает пустой DataFrame.
    """
    import pandas as pd
    frames = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            # 1) пробуем «по линиям»
            tables = page.extract_tables(table_settings={
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "intersection_tolerance": 5,
            }) or []
            # 2) если не получилось — «по тексту»
            if not tables:
                tables = page.extract_tables(table_settings={
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "snap_tolerance": 3,
                    "intersection_tolerance": 3,
                    "join_tolerance": 3,
                }) or []
            for t in tables:
                # нормализуем: первая непустая строка — заголовок
                rows = [r for r in t if any((c or "").strip() for c in r)]
                if len(rows) < 2:
                    continue
                header = rows[0]
                body = rows[1:]
                w = max(len(r) for r in rows)
                norm = [list(r) + [""]*(w-len(r)) for r in rows]
                df = pd.DataFrame(norm[1:], columns=[(h or "").strip() for h in norm[0]])
                # выбрасываем полностью пустые строки
                df = df[~(df.apply(lambda r: "".join(map(lambda x: str(x or "").strip(), r)), axis=1)=="")]
                if not df.empty:
                    frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # лёгкая чистка странных значений "nan"
    for c in out.columns:
        out[c] = out[c].astype(str).str.replace("\xa0"," ").str.strip()
        out.loc[out[c].str.lower()=="nan", c] = ""
    return out


# ----------------- Dashboard -----------------

@inventory_bp.route('/api/part/<part_number>')
@login_required
def get_part_by_number(part_number):
    part = Part.query.filter_by(part_number=part_number.upper()).first()
    if part:
        return jsonify({
            'id': part.id,                 # Нужен для Issue Part
            'name': part.name,
            'location': part.location,
            'unit_cost': part.unit_cost,
            'quantity': part.quantity      # Нужен для проверки остатков
        })
    return jsonify({'error': 'Not found'}), 404



@inventory_bp.route('/api/part_lookup')
@login_required
def part_lookup():
    part_number = request.args.get('part_number', '').strip().upper()
    part = Part.query.filter_by(part_number=part_number).first()
    if part:
        return {
            'found': True,
            'id': part.id,
            'name': part.name,
            'quantity': part.quantity
        }
    return { 'found': False }

@inventory_bp.route('/')
@login_required
def dashboard():
    search_query = request.args.get('search', '').strip()
    parts = Part.query

    if search_query:
        parts = parts.filter(
            or_(
                Part.part_number.ilike(f"%{search_query}%"),
                Part.name.ilike(f"%{search_query}%")
            )
        )

    parts = parts.all()
    return render_template('index.html', parts=parts, search_query=search_query)

# @inventory_bp.route('/inventory_summary')
# @login_required
# def inventory_summary():
#     parts = Part.query.filter(Part.quantity > 0).all()
#
#     locations = defaultdict(lambda: {
#         'total_quantity': 0,
#         'total_value': 0.0,
#     })
#
#     for part in parts:
#         loc = part.location or 'Unknown'
#         locations[loc]['total_quantity'] += part.quantity
#         locations[loc]['total_value'] += part.quantity * part.unit_cost
#
#     grand_total_quantity = sum(data['total_quantity'] for data in locations.values())
#     grand_total_value = sum(data['total_value'] for data in locations.values())
#
#     return render_template('inventory_summary.html',
#                            locations=locations,
#                            grand_total_quantity=grand_total_quantity,
#                            grand_total_value=grand_total_value)

@inventory_bp.route('/dashboard/location_report')
@login_required
def location_report():
    parts = Part.query.filter(Part.quantity > 0).all()

    locations = defaultdict(lambda: {
        'parts': [],
        'total_quantity': 0,
        'total_value': 0.0,
    })

    for part in parts:
        loc = part.location or 'Unknown'
        locations[loc]['parts'].append(part)
        locations[loc]['total_quantity'] += part.quantity
        locations[loc]['total_value'] += part.quantity * part.unit_cost

    grand_total_quantity = sum(data['total_quantity'] for data in locations.values())
    grand_total_value = sum(data['total_value'] for data in locations.values())

    return render_template('location_report.html',
                           locations=locations,
                           grand_total_quantity=grand_total_quantity,
                           grand_total_value=grand_total_value)

# --- Печать детального отчёта по локациям (весь список товаров) ---
@inventory_bp.route('/dashboard/location_report/print')
@login_required
def print_location_report():
    parts = Part.query.filter(Part.quantity > 0).all()

    locations = defaultdict(lambda: {
        'parts': [],
        'total_quantity': 0,
        'total_value': 0.0,
    })

    for part in parts:
        loc = part.location or 'Unknown'
        locations[loc]['parts'].append(part)
        locations[loc]['total_quantity'] += part.quantity
        locations[loc]['total_value'] += part.quantity * part.unit_cost

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
    elements = []

    styles = getSampleStyleSheet()
    title_style = styles['Heading2']

    for loc, data in locations.items():
        elements.append(Paragraph(f"Location: {loc}", title_style))
        elements.append(Paragraph(f"Total Quantity: {data['total_quantity']}", styles['Normal']))
        elements.append(Paragraph(f"Total Value: ${data['total_value']:.2f}", styles['Normal']))
        elements.append(Spacer(1, 12))

        table_data = [["Part Number", "Name", "Quantity", "Unit Cost", "Total Cost"]]
        for part in data['parts']:
            table_data.append([
                part.part_number,
                part.name,
                str(part.quantity),
                f"${part.unit_cost:.2f}",
                f"${part.quantity * part.unit_cost:.2f}"
            ])
        # Итог по локации
        table_data.append([
            "Location Total", "",
            str(data['total_quantity']), "", f"${data['total_value']:.2f}"
        ])

        table = Table(table_data, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (2, 1), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, -1), (-1, -1), colors.lightblue),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 20))

    doc.build(elements)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="location_report.pdf", mimetype='application/pdf')

# --- Печать сводного отчёта (итог по локациям без деталей) ---
@inventory_bp.route('/dashboard/inventory_summary/print')
@login_required
def print_inventory_summary():
    parts = Part.query.filter(Part.quantity > 0).all()

    locations = defaultdict(lambda: {
        'total_quantity': 0,
        'total_value': 0.0,
    })

    for part in parts:
        loc = part.location or 'Unknown'
        locations[loc]['total_quantity'] += part.quantity
        locations[loc]['total_value'] += part.quantity * part.unit_cost

    grand_total_quantity = sum(data['total_quantity'] for data in locations.values())
    grand_total_value = sum(data['total_value'] for data in locations.values())

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    elements.append(Paragraph("Inventory Summary Report by Location", styles['Heading2']))
    elements.append(Spacer(1, 12))

    table_data = [["Location", "Total Quantity", "Total Value"]]
    for loc, data in locations.items():
        table_data.append([
            loc,
            str(data['total_quantity']),
            f"${data['total_value']:.2f}"
        ])
    # Итог
    table_data.append([
        "Grand Total",
        str(grand_total_quantity),
        f"${grand_total_value:.2f}"
    ])

    table = Table(table_data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
    ]))

    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="inventory_summary.pdf", mimetype='application/pdf')



@inventory_bp.route('/add', methods=['GET', 'POST'])
@login_required
def add_part():
    if request.method == 'POST':
        part_number = request.form['part_number'].strip().upper()
        name = request.form['name'].strip().upper()
        quantity = int(request.form['quantity'])
        unit_cost = float(request.form['unit_cost'])
        location = request.form['location'].strip().upper()

        existing = Part.query.filter_by(part_number=part_number).first()
        if existing:
            existing.quantity += quantity
            existing.unit_cost = unit_cost
            existing.name = name
            existing.location = location
        else:
            new_part = Part(
                name=name,
                part_number=part_number,
                quantity=quantity,
                unit_cost=unit_cost,
                location=location
            )
            db.session.add(new_part)

        db.session.commit()
        flash('Part saved successfully.', 'success')
        return redirect(url_for('inventory.dashboard'))

    return render_template('add_part.html')


# ----------------- Issue Part -----------------
@inventory_bp.route('/issue', methods=['GET', 'POST'])
@login_required
def issue_part():
    parts = Part.query.filter(Part.quantity > 0).all()

    if request.method == 'POST':
        import json
        try:
            all_parts = json.loads(request.form.get('all_parts_json', '[]'))
        except json.JSONDecodeError:
            flash('Invalid part data.', 'danger')
            return redirect(url_for('.issue_part'))

        if not all_parts:
            flash('No parts to issue.', 'warning')
            return redirect(url_for('.issue_part'))

        for item in all_parts:
            part = Part.query.get(item['part_id'])
            if not part or part.quantity < item['quantity']:
                flash(f"Not enough stock for {item.get('part_number', 'UNKNOWN')}", 'danger')
                return redirect(url_for('.issue_part'))

            part.quantity -= item['quantity']

            record = IssuedPartRecord(
                part_id=part.id,
                quantity=item['quantity'],
                issued_to=item['recipient'],
                reference_job=item['reference_job'],
                issued_by=current_user.username,
                issue_date=datetime.utcnow(),
                unit_cost_at_issue=part.unit_cost  # фиксируем цену на момент выдачи
            )
            db.session.add(record)

        db.session.commit()
        flash('All parts issued successfully.', 'success')

        # >>> ЖЁСТКИЙ РЕДИРЕКТ СРАЗУ В ОТЧЁТ (без url_for, чтобы исключить любые конфликты)
        today = datetime.utcnow().date().isoformat()
        first = all_parts[0]
        recipient = (first.get('recipient') or '').strip()
        reference_job = (first.get('reference_job') or '').strip()

        params = {'start_date': today, 'end_date': today}
        if recipient:
            params['recipient'] = recipient
        if reference_job:
            params['reference_job'] = reference_job

        # Итог: /reports_grouped?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&recipient=...&reference_job=...
        return redirect('/reports_grouped?' + urlencode(params), code=303)

    return render_template('issue_part.html', parts=parts)



# ----------------- Reports -----------------



@inventory_bp.route('/reports', methods=['GET', 'POST'])
@login_required
def reports():
    query = IssuedPartRecord.query.join(Part)
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    recipient = request.form.get('recipient')
    reference_job = request.form.get('reference_job')

    if start_date:
        query = query.filter(IssuedPartRecord.issue_date >= start_date)
    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        query = query.filter(IssuedPartRecord.issue_date <= end_dt)
    if recipient:
        query = query.filter(IssuedPartRecord.issued_to.ilike(f'%{recipient}%'))
    if reference_job:
        query = query.filter(IssuedPartRecord.reference_job.ilike(f'%{reference_job}%'))

    records = query.order_by(IssuedPartRecord.issue_date.desc()).all()

    # Группируем записи по ключу
    invoices_map = defaultdict(lambda: {
        'issued_to': '',
        'reference_job': '',
        'issued_by': '',
        'issue_date': None,
        'items': [],
        'total_value': 0.0,
    })

    grand_total = 0.0

    for r in records:
        key = (r.issued_to, r.reference_job or '', r.issued_by, r.issue_date.date())
        inv = invoices_map[key]
        inv['issued_to'] = r.issued_to
        inv['reference_job'] = r.reference_job
        inv['issued_by'] = r.issued_by
        inv['issue_date'] = r.issue_date
        inv['items'].append(r)
        line_total = r.quantity * r.unit_cost_at_issue
        inv['total_value'] += line_total
        grand_total += line_total

    invoices = list(invoices_map.values())

    return render_template(
        'reports_grouped.html',
        invoices=invoices,
        total=grand_total,
        start_date=start_date,
        end_date=end_date,
        recipient=recipient,
        reference_job=reference_job
    )

@inventory_bp.route('/reports_grouped', methods=['GET', 'POST'])
@login_required
def reports_grouped():
    query = IssuedPartRecord.query.join(Part)

    # ← читаем и из GET (?start_date=...) и из POST-форм
    params = request.values
    start_date = params.get('start_date')
    end_date = params.get('end_date')
    recipient = params.get('recipient')
    reference_job = params.get('reference_job')

    # ← ВАЖНО: приводим строки дат к datetime, чтобы сравнивать корректно
    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        query = query.filter(IssuedPartRecord.issue_date >= start_dt)

    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, microsecond=999999
        )
        query = query.filter(IssuedPartRecord.issue_date <= end_dt)

    if recipient:
        query = query.filter(IssuedPartRecord.issued_to.ilike(f'%{recipient}%'))
    if reference_job:
        query = query.filter(IssuedPartRecord.reference_job.ilike(f'%{reference_job}%'))

    records = query.order_by(IssuedPartRecord.issue_date.desc()).all()

    # Группировка — как у тебя было
    grouped = defaultdict(list)
    for r in records:
        key = (r.issued_to, r.reference_job, r.issued_by, r.issue_date.date())
        grouped[key].append(r)

    invoices = []
    grand_total = 0
    for key, items in grouped.items():
        total_value = sum(item.quantity * item.unit_cost_at_issue for item in items)
        grand_total += total_value
        invoices.append({
            'issued_to': key[0],
            'reference_job': key[1],
            'issued_by': key[2],
            'issue_date': key[3],
            'items': items,
            'total_value': total_value
        })

    return render_template(
        'reports_grouped.html',
        invoices=invoices,
        total=grand_total,
        start_date=start_date,
        end_date=end_date,
        recipient=recipient,
        reference_job=reference_job
    )


@inventory_bp.route('/invoice/view')
@login_required
def view_invoice_pdf():
    # Получаем параметры, которые идентифицируют группу инвойса
    issued_to = request.args.get('issued_to')
    reference_job = request.args.get('reference_job')
    issued_by = request.args.get('issued_by')
    issue_date_str = request.args.get('issue_date')

    from datetime import datetime

    # Попытка распарсить дату
    try:
        issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        try:
            issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d')

    # Получаем все записи этой накладной
    records = IssuedPartRecord.query.filter(
        IssuedPartRecord.issued_to == issued_to,
        IssuedPartRecord.reference_job == reference_job,
        IssuedPartRecord.issued_by == issued_by,
        IssuedPartRecord.issue_date.between(
            datetime.combine(issue_date.date(), datetime.min.time()),
            datetime.combine(issue_date.date(), datetime.max.time())
        )
    ).all()

    from utils.invoice_generator import generate_view_pdf

    pdf_data = generate_view_pdf(records)  # Передаём список записей

    from io import BytesIO
    from flask import send_file

    return send_file(BytesIO(pdf_data),
                     as_attachment=True,
                     download_name=f"INVOICE_{issued_to}_{issue_date.strftime('%Y%m%d')}.pdf",
                     mimetype="application/pdf")



# Обновление отдельной позиции
@inventory_bp.route('/reports/update/<int:record_id>', methods=['POST'])
@login_required
def update_report_record(record_id):
    if current_user.role not in ['superadmin', 'admin']:
        flash("Access denied", "danger")
        return redirect(url_for('inventory.reports'))

    record = IssuedPartRecord.query.get_or_404(record_id)

    issued_to = request.form.get('issued_to', '').strip()
    reference_job = request.form.get('reference_job', '').strip()
    unit_cost_str = request.form.get('unit_cost', '').strip()
    issue_date_str = request.form.get('issue_date', '').strip()

    if not issued_to:
        flash("Issued To field cannot be empty.", "danger")
        return redirect(url_for('inventory.reports'))

    try:
        unit_cost = float(unit_cost_str)
        if unit_cost < 0:
            raise ValueError()
    except ValueError:
        flash("Invalid Unit Cost value.", "danger")
        return redirect(url_for('inventory.reports'))

    # Сохраняем дату только для superadmin
    if issue_date_str and current_user.role == 'superadmin':
        try:
            new_issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d')
            record.issue_date = new_issue_date
        except ValueError:
            flash("Invalid Issue Date format.", "danger")
            return redirect(url_for('inventory.reports'))

    record.issued_to = issued_to
    record.reference_job = reference_job if reference_job else None
    record.unit_cost_at_issue = unit_cost

    db.session.commit()
    flash("Issued record updated successfully.", "success")
    return redirect(url_for('inventory.reports'))


# Отмена отдельной позиции
@inventory_bp.route('/reports/cancel/<int:record_id>', methods=['POST'])
@login_required
def cancel_issued_record(record_id):
    if current_user.role not in ['superadmin', 'admin']:
        flash("Access denied", "danger")
        return redirect(url_for('inventory.reports'))

    record = IssuedPartRecord.query.get_or_404(record_id)
    part = Part.query.get(record.part_id)
    if part:
        part.quantity += record.quantity

    db.session.delete(record)
    db.session.commit()

    flash(f"Issued record #{record.id} canceled and stock restored.", "success")
    return redirect(url_for('inventory.reports'))


# Обновление данных накладной (issued_to, reference_job) для всей группы позиций
@inventory_bp.route('/reports/update_invoice', methods=['POST'])
@login_required
def update_invoice():
    if current_user.role not in ['superadmin', 'admin']:
        flash("Access denied", "danger")
        return redirect(url_for('inventory.reports'))

    action = request.form.get('action')
    issued_to_old = request.args.get('issued_to')
    reference_job_old = request.args.get('reference_job')
    issued_by = request.args.get('issued_by')
    issue_date_str = request.args.get('issue_date')

    # Попытка распарсить дату с разными форматами
    try:
        issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        try:
            issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d')

    if action == "save":
        new_issued_to = request.form.get('issued_to', '').strip()
        new_reference_job = request.form.get('reference_job', '').strip()

        if not new_issued_to:
            flash("Issued To field cannot be empty.", "danger")
            return redirect(url_for('inventory.reports'))

        records = IssuedPartRecord.query.filter(
            IssuedPartRecord.issued_to == issued_to_old,
            IssuedPartRecord.reference_job == reference_job_old,
            IssuedPartRecord.issued_by == issued_by,
            IssuedPartRecord.issue_date.between(
                datetime.combine(issue_date.date(), datetime.min.time()),
                datetime.combine(issue_date.date(), datetime.max.time())
            )
        ).all()

        for r in records:
            r.issued_to = new_issued_to
            r.reference_job = new_reference_job if new_reference_job else None

        db.session.commit()
        flash("Invoice updated successfully.", "success")
        return redirect(url_for('inventory.reports'))

    elif action == "cancel":
        records = IssuedPartRecord.query.filter(
            IssuedPartRecord.issued_to == issued_to_old,
            IssuedPartRecord.reference_job == reference_job_old,
            IssuedPartRecord.issued_by == issued_by,
            IssuedPartRecord.issue_date.between(
                datetime.combine(issue_date.date(), datetime.min.time()),
                datetime.combine(issue_date.date(), datetime.max.time())
            )
        ).all()

        for r in records:
            part = Part.query.get(r.part_id)
            if part:
                part.quantity += r.quantity
            db.session.delete(r)

        db.session.commit()
        flash("Invoice canceled and stock restored.", "success")
        return redirect(url_for('inventory.reports'))

    else:
        flash("Invalid action.", "danger")
        return redirect(url_for('inventory.reports'))


# Отмена всей накладной (группы записей)
@inventory_bp.route('/reports/cancel_invoice', methods=['POST'])
@login_required
def cancel_invoice():
    if current_user.role not in ['superadmin', 'admin']:
        flash("Access denied", "danger")
        return redirect(url_for('inventory.reports'))

    issued_to = request.form.get('issued_to')
    reference_job = request.form.get('reference_job')
    issued_by = request.form.get('issued_by')
    issue_date_str = request.form.get('issue_date')

    try:
        issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        try:
            issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d')

    records = IssuedPartRecord.query.filter(
        IssuedPartRecord.issued_to == issued_to,
        IssuedPartRecord.reference_job == reference_job,
        IssuedPartRecord.issued_by == issued_by,
        IssuedPartRecord.issue_date.between(
            datetime.combine(issue_date.date(), datetime.min.time()),
            datetime.combine(issue_date.date(), datetime.max.time())
        )
    ).all()

    for r in records:
        part = Part.query.get(r.part_id)
        if part:
            part.quantity += r.quantity
        db.session.delete(r)

    db.session.commit()
    flash("Invoice canceled and stock restored.", "success")
    return redirect(url_for('inventory.reports'))



# ----------------- Download Invoice -----------------

@inventory_bp.route('/invoice/<int:record_id>')
@login_required
def invoice(record_id):
    record = IssuedPartRecord.query.get(record_id)
    if not record:
        return "Record not found", 404

    pdf_data = generate_invoice_pdf(record)
    return send_file(BytesIO(pdf_data), as_attachment=True, download_name='invoice.pdf')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@inventory_bp.route('/import', methods=['GET', 'POST'])
@login_required
def import_parts():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            file.save(filepath)

            try:
                # Read Excel or CSV
                if filename.endswith('.csv'):
                    df = pd.read_csv(filepath)
                else:
                    df = pd.read_excel(filepath)

                for _, row in df.iterrows():
                    part_number = str(row['Part Number']).strip()
                    name = str(row['Name']).strip()
                    quantity = int(row['Quantity'])
                    unit_cost = float(row['Unit Cost'])
                    location = str(row['Location']).strip()

                    existing = Part.query.filter_by(part_number=part_number).first()
                    if existing:
                        existing.quantity += quantity
                    else:
                        new_part = Part(
                            name=name,
                            part_number=part_number,
                            quantity=quantity,
                            unit_cost=unit_cost,
                            location=location
                        )
                        db.session.add(new_part)

                db.session.commit()
                flash('Parts imported successfully.', 'success')

            except Exception as e:
                flash(f'Import failed: {str(e)}', 'danger')

            os.remove(filepath)
        else:
            flash('Invalid file type. Use .xlsx, .xls or .csv', 'danger')

    return render_template('import_parts.html')


@inventory_bp.route('/api/part/<part_number>')
def api_get_part(part_number):
    part = Part.query.filter_by(part_number=part_number.upper()).first()
    if part:
        return {
            'id': part.id,
            'name': part.name,
            'quantity': part.quantity,
            'unit_cost': part.unit_cost,
            'location': part.location
        }
    return {}, 404


@inventory_bp.route('/reports/download')
@login_required
def download_report_pdf():
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from datetime import datetime
    from io import BytesIO

    start = request.args.get('start_date')
    end = request.args.get('end_date')
    recipient = request.args.get('recipient')
    reference_job = request.args.get('reference_job')

    query = IssuedPartRecord.query.join(Part)
    if start:
        query = query.filter(IssuedPartRecord.issue_date >= start)
    if end:
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        query = query.filter(IssuedPartRecord.issue_date <= end_dt)
    if recipient:
        query = query.filter(IssuedPartRecord.issued_to.ilike(f"%{recipient}%"))
    if reference_job:
        query = query.filter(IssuedPartRecord.reference_job.ilike(f"%{reference_job}%"))

    records = query.order_by(IssuedPartRecord.issue_date.desc()).all()

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), leftMargin=30, rightMargin=30, topMargin=30, bottomMargin=20)
    elements = []
    styles = getSampleStyleSheet()

    title = Paragraph("Issued Parts Report", styles["Heading1"])
    elements.append(title)
    elements.append(Spacer(1, 12))

    filter_text = ""
    if start and end:
        filter_text = f"Filters: from {datetime.strptime(start, '%Y-%m-%d').strftime('%m/%d/%Y')} to {datetime.strptime(end, '%Y-%m-%d').strftime('%m/%d/%Y')}"
    elif start:
        filter_text = f"Filters: from {datetime.strptime(start, '%Y-%m-%d').strftime('%m/%d/%Y')}"
    elif end:
        filter_text = f"Filters: up to {datetime.strptime(end, '%Y-%m-%d').strftime('%m/%d/%Y')}"

    if filter_text:
        elements.append(Paragraph(filter_text, styles["Normal"]))
        elements.append(Spacer(1, 12))

    data = [["Date", "Part #", "Name", "Qty", "Unit Cost", "Total", "Issued To", "Job Ref."]]

    total_sum = 0
    for r in records:
        total = r.quantity * r.part.unit_cost
        total_sum += total
        row = [
            r.issue_date.strftime('%m/%d/%Y'),
            Paragraph(r.part.part_number, styles['Normal']),
            Paragraph(r.part.name, styles['Normal']),
            str(r.quantity),
            f"${r.unit_cost_at_issue:.2f}",
            f"${total:.2f}",
            Paragraph(r.issued_to, styles['Normal']),
            Paragraph(r.reference_job or '—', styles['Normal'])
        ]
        data.append(row)

    # Итоговая строка
    data.append([
        "", "", "", "", "TOTAL:",
        f"${total_sum:.2f}", "", ""
    ])

    col_widths = [65, 130, 150, 35, 65, 65, 90, 110]

    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#d3d3d3")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (3, 1), (5, -2), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),

        # Здесь рисуем сетку по всей таблице
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),

        # Вертикальная линия слева от колонки "Unit Cost"
        ('LINEBEFORE', (5, 0), (5, -2), 0.25, colors.grey),

        # Линия над итоговой строкой
        ('LINEABOVE', (4, -1), (5, -1), 0.25, colors.grey),
        # Итоговая строка - светлый фон, отступы, выравнивание
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#f0f0f0")),  # светло-серый фон
        ('TOPPADDING', (0, -1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, -1), (-1, -1), 8),

        # Итоговые шрифты
        ('FONTNAME', (4, -1), (4, -1), 'Helvetica-Bold'),
        ('FONTNAME', (5, -1), (5, -1), 'Helvetica-Bold'),

        # Выравнивание итога по правому краю
        ('ALIGN', (5, -1), (5, -1), 'RIGHT'),


    ]))

    elements.append(table)
    doc.build(elements)

    buffer.seek(0)
    return send_file(buffer,
                     as_attachment=True,
                     download_name=f"report_{start or 'all'}_{end or 'all'}.pdf",
                     mimetype='application/pdf')


@inventory_bp.route('/users')
@login_required
def users():
    # superadmin видит всех, admin — только пользователей с ролью user и себя
    if current_user.role == 'superadmin':
        users_list = User.query.all()
    elif current_user.role == 'admin':
        users_list = User.query.filter(
            (User.role == 'user') | (User.id == current_user.id)
        ).all()
    else:
        flash("Access denied", "danger")
        return redirect(url_for('inventory.dashboard'))

    return render_template('users.html', users=users_list)


@inventory_bp.route('/users/add', methods=['GET', 'POST'])
@login_required
def add_user():
    if current_user.role not in ['superadmin', 'admin']:
        flash("Access denied", "danger")
        return redirect(url_for('inventory.dashboard'))

    if request.method == 'POST':
        username = request.form['username'].strip()
        role = request.form['role']
        password = request.form['password']

        # Админ может создавать только user
        if current_user.role == 'admin' and role != 'user':
            flash("Admins can only create users with role 'user'.", "danger")
            return redirect(url_for('inventory.add_user'))

        if User.query.filter_by(username=username).first():
            flash("Username already exists", "danger")
            return redirect(url_for('inventory.add_user'))

        new_user = User(
            username=username,
            role=role,
            password_hash=generate_password_hash(password)
        )
        db.session.add(new_user)
        db.session.commit()
        flash("User added successfully", "success")
        return redirect(url_for('inventory.users'))

    allowed_roles = ['user'] if current_user.role == 'admin' else ['user', 'admin', 'superadmin']
    return render_template('add_user.html', allowed_roles=allowed_roles)


@inventory_bp.route('/users/edit/<int:user_id>', methods=['GET', 'POST'])
@login_required
def edit_user(user_id):
    user = User.query.get_or_404(user_id)

    # superadmin может редактировать всех
    # admin — только пользователей с ролью user и себя
    if current_user.role == 'admin':
        if user.role != 'user' and user.id != current_user.id:
            flash("Admins can only edit users with role 'user' or themselves.", "danger")
            return redirect(url_for('inventory.dashboard'))
    elif current_user.role != 'superadmin':
        flash("Access denied", "danger")
        return redirect(url_for('inventory.dashboard'))

    if request.method == 'POST':
        user.username = request.form['username'].strip()
        user.role = request.form['role']
        db.session.commit()
        flash("User updated successfully", "success")
        return redirect(url_for('inventory.users'))

    return render_template('edit_user.html', user=user)


@inventory_bp.route('/users/change_password/<int:user_id>', methods=['GET', 'POST'])
@login_required
def change_password(user_id):
    user = User.query.get_or_404(user_id)

    # superadmin меняет любой пароль
    # admin меняет только пароли пользователей с ролью user и свой (для своего - проверка текущего пароля)
    if current_user.role == 'admin':
        if user.role != 'user' and user.id != current_user.id:
            flash("Admins can only change passwords for users with role 'user' or themselves.", "danger")
            return redirect(url_for('inventory.dashboard'))
    elif current_user.role != 'superadmin' and current_user.id != user_id:
        flash("Access denied", "danger")
        return redirect(url_for('inventory.dashboard'))

    if request.method == 'POST':
        # Если admin меняет свой пароль - проверяем текущий пароль
        if current_user.role == 'admin' and current_user.id == user_id:
            current_password = request.form.get('current_password')
            if not user.check_password(current_password):
                flash("Current password is incorrect.", "danger")
                return redirect(url_for('inventory.change_password', user_id=user_id))

        new_password = request.form['password']
        confirm_password = request.form.get('confirm_password')

        if new_password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(url_for('inventory.change_password', user_id=user_id))

        user.password_hash = generate_password_hash(new_password)

        db.session.commit()
        flash("Password changed successfully", "success")

        if current_user.role == 'superadmin':
            return redirect(url_for('inventory.users'))
        else:
            return redirect(url_for('inventory.dashboard'))

    return render_template('change_password.html', user=user)


@inventory_bp.route('/users/delete/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)

    # Только superadmin может удалять
    if current_user.role != 'superadmin':
        flash("Access denied", "danger")
        return redirect(url_for('inventory.dashboard'))

    if user.id == current_user.id:
        flash("You cannot delete yourself!", "danger")
        return redirect(url_for('inventory.users'))

    db.session.delete(user)
    db.session.commit()
    flash("User deleted successfully", "success")
    return redirect(url_for('inventory.users'))


@inventory_bp.route('/clear_issued_records')
@login_required
def clear_issued_records():
    if current_user.role != 'superadmin':
        flash('Only superadmin can clear records.', 'danger')
        return redirect(url_for('inventory.dashboard'))

    from models import IssuedPartRecord
    from extensions import db

    IssuedPartRecord.query.delete()
    db.session.commit()
    flash('All issued records cleared.', 'success')
    return redirect(url_for('inventory.dashboard'))

@inventory_bp.route('/update_part/<int:part_id>', methods=['POST'])
@login_required
def update_part_field(part_id):
    if current_user.role != 'superadmin':
        flash("Access denied", "danger")
        return redirect(url_for('inventory.dashboard'))

    part = Part.query.get_or_404(part_id)

    quantity = request.form.get('quantity')
    unit_cost = request.form.get('unit_cost')

    if quantity is not None:
        try:
            part.quantity = int(quantity)
        except ValueError:
            flash("Invalid quantity value", "danger")
            return redirect(url_for('inventory.dashboard'))

    if unit_cost is not None:
        try:
            part.unit_cost = float(unit_cost)
        except ValueError:
            flash("Invalid unit cost value", "danger")
            return redirect(url_for('inventory.dashboard'))

    db.session.commit()
    flash("Part updated successfully.", "success")
    return redirect(url_for('inventory.dashboard'))

@inventory_bp.route('/delete/<int:part_id>', methods=['POST'])
@login_required
def delete_part(part_id):
    if current_user.role != 'superadmin':
        flash('Only Superadmin can delete parts.', 'danger')
        return redirect(url_for('inventory.dashboard'))

    part = Part.query.get_or_404(part_id)
    db.session.delete(part)
    db.session.commit()
    flash(f'Part {part.part_number} deleted.', 'success')
    return redirect(url_for('inventory.dashboard'))

@inventory_bp.route('/edit/<int:part_id>', methods=['GET', 'POST'])
@login_required
def edit_part(part_id):
    if current_user.role != 'superadmin':
        flash("Access denied", "danger")
        return redirect(url_for('inventory.dashboard'))

    part = Part.query.get_or_404(part_id)

    if request.method == 'POST':
        part.name = request.form['name'].strip().upper()
        part.part_number = request.form['part_number'].strip().upper()
        try:
            part.quantity = int(request.form['quantity'])
            part.unit_cost = float(request.form['unit_cost'])
        except ValueError:
            flash("Invalid quantity or unit cost", "danger")
            return redirect(url_for('inventory.edit_part', part_id=part_id))

        part.location = request.form['location'].strip().upper()

        db.session.commit()
        flash("Part updated successfully.", "success")
        return redirect(url_for('inventory.dashboard'))

    return render_template('edit_part.html', part=part)


@inventory_bp.route('/reports/delete/<int:record_id>', methods=['POST'])
@login_required
def delete_report_record(record_id):
    if current_user.role != 'superadmin':
        flash('Access denied', 'danger')
        return redirect(url_for('inventory.reports'))

    record = IssuedPartRecord.query.get_or_404(record_id)
    db.session.delete(record)
    db.session.commit()
    flash('Issued record deleted successfully.', 'success')
    return redirect(url_for('inventory.reports'))

@inventory_bp.route('/clear_parts')
@login_required
def clear_parts():
    if current_user.role != 'superadmin':
        flash('Only superadmin can clear parts.', 'danger')
        return redirect(url_for('inventory.dashboard'))

    from models import Part
    from extensions import db

    Part.query.delete()
    db.session.commit()
    flash('All parts cleared.', 'success')
    return redirect(url_for('inventory.dashboard'))

@inventory_bp.route('/compare_cart')
def compare_cart():
    try:
        flash("🔍 Collecting Marcone cart...", "info")
        items = get_marcone_items()

        flash("📦 Comparing with inventory...", "info")
        result = check_cart_items(items)

        filepath = os.path.join(UPLOAD_DIR, "marcone_inventory_report.docx")
        export_to_docx(result, filename=filepath)

        flash("✅ Marcone report generated! Click below to download.", "success")
        return redirect(url_for("inventory.dashboard"))

    except Exception as e:
        flash(f"❌ Error (Marcone): {str(e)}", "danger")
        return redirect(url_for("inventory.dashboard"))

# @inventory_bp.route('/download_marcone_report')
# def download_marcone_report():
#     filepath = os.path.join(UPLOAD_DIR, "marcone_inventory_report.docx")
#     if os.path.exists(filepath):
#         return send_file(filepath, as_attachment=True)
#     flash("❌ Marcone report not found!", "danger")
#     return redirect(url_for("inventory.dashboard"))


@inventory_bp.route('/compare_reliable')
def compare_reliable():
    try:
        flash("🔍 Collecting Reliable cart...", "info")
        items = get_reliable_items()

        flash("📦 Comparing with inventory...", "info")
        result = check_cart_items(items)

        filepath = os.path.join(UPLOAD_DIR, "reliable_inventory_report.docx")
        export_to_docx(result, filename=filepath)

        flash("✅ Reliable report generated! Click below to download.", "success")
        return redirect(url_for("inventory.dashboard"))

    except Exception as e:
        flash(f"❌ Error (Reliable): {str(e)}", "danger")
        return redirect(url_for("inventory.dashboard"))

# @inventory_bp.route('/download_reliable_report')
# def download_reliable_report():
#     filepath = os.path.join(UPLOAD_DIR, "reliable_inventory_report.docx")
#     if os.path.exists(filepath):
#         return send_file(filepath, as_attachment=True)
#     flash("❌ Reliable report not found!", "danger")
#     return redirect(url_for("inventory.dashboard"))

@inventory_bp.route('/download_marcone_report')
@login_required
def download_marcone_report():
    from compare_cart.run_compare import get_marcone_items, check_cart_items, export_to_docx

    filepath = os.path.join(UPLOAD_DIR, "marcone_inventory_report.docx")

    # 1. Получаем данные из корзины Marcone
    items = get_marcone_items()

    # 2. Сравниваем с базой данных
    result = check_cart_items(items)

    # 3. Генерируем новый отчет
    export_to_docx(result, filename=filepath)

    # 4. Отдаем файл пользователю
    if os.path.exists(filepath):
        @after_this_request
        def remove_file(response):
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Failed to delete file: {e}")
            return response

        return send_file(filepath, as_attachment=True)

    flash("❌ Marcone report generation failed!", "danger")
    return redirect(url_for("inventory.dashboard"))


@inventory_bp.route('/download_reliable_report')
@login_required
def download_reliable_report():
    from compare_cart.run_compare_reliable import get_reliable_items, check_cart_items, export_to_docx

    filepath = os.path.join(UPLOAD_DIR, "reliable_inventory_report.docx")

    items = get_reliable_items()
    result = check_cart_items(items)
    export_to_docx(result, filename=filepath)

    if os.path.exists(filepath):
        @after_this_request
        def remove_file(response):
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Failed to delete file: {e}")
            return response

        return send_file(filepath, as_attachment=True)

    flash("❌ Reliable report generation failed!", "danger")
    return redirect(url_for("inventory.dashboard"))

@inventory_bp.route("/import-parts", methods=["GET", "POST"], endpoint="import_parts_upload")
def import_parts_upload():
    enabled = current_app.config.get("WCCR_IMPORT_ENABLED", 0)
    dry     = current_app.config.get("WCCR_IMPORT_DRY_RUN", 1)

    # -------- 1) Нажали "Применить импорт" с предпросмотра --------
    if request.method == "POST" and "apply" in request.form:
        saved_path = request.form.get("saved_path", "")
        if not saved_path or not os.path.exists(saved_path):
            flash("Не найден сохранённый файл. Загрузите его заново.", "danger")
            return redirect(url_for("inventory.import_parts_upload"))

        ext = os.path.splitext(saved_path)[1].lower()

        # Читаем и нормализуем тот же файл (PDF/Excel/CSV)
        if ext == ".pdf":
            df = dataframe_from_pdf(saved_path)
            if df.empty:
                flash("В этом PDF не удалось распознать таблицы (скорее всего скан).", "danger")
                rows = []
                return render_template("import_preview.html", rows=rows, saved_path=saved_path)
        else:
            df = load_table(saved_path)

        norm, issues = normalize_table(df, supplier_hint=None, source_file=saved_path, default_location="MAIN")
        for msg in issues:
            flash(msg, "warning")

        # Если DRY или выключено — просто показать предпросмотр снова
        if dry or not enabled:
            rows = norm.to_dict(orient="records")
            return render_template("import_preview.html", rows=rows, saved_path=saved_path)

        # Применяем: создаём приходы, подавляя дубли по row_key
        def duplicate_exists(rk: str) -> bool:
            return has_key(rk)

        def make_movement(m: dict) -> None:
            PartModel = Part
            session = db.session

            PN_FIELDS   = ["part_number", "number", "sku", "code", "partnum", "pn"]
            NAME_FIELDS = ["name", "part_name", "descr", "description", "title"]
            QTY_FIELDS  = ["quantity", "qty", "on_hand", "stock", "count"]
            LOC_FIELDS  = ["location", "bin", "shelf", "place", "loc"]
            COST_FIELDS = ["unit_cost", "cost", "price", "unitprice", "last_cost"]
            SUP_FIELDS  = ["supplier", "vendor", "provider"]

            def pick_field(model, candidates):
                for f in candidates:
                    if hasattr(model, f):
                        return f
                return None

            pn_field   = pick_field(PartModel, PN_FIELDS)
            name_field = pick_field(PartModel, NAME_FIELDS)
            qty_field  = pick_field(PartModel, QTY_FIELDS)
            loc_field  = pick_field(PartModel, LOC_FIELDS)
            cost_field = pick_field(PartModel, COST_FIELDS)
            sup_field  = pick_field(PartModel, SUP_FIELDS)

            if pn_field is None or qty_field is None:
                raise RuntimeError("Не найдено поле PART # или QTY в модели Part — уточни имена полей.")

            filters = {pn_field: m["part_number"]}
            if loc_field:
                filters[loc_field] = m["location"]

            part = PartModel.query.filter_by(**filters).first()

            if not part:
                kwargs = dict(filters)
                kwargs[qty_field] = 0
                if name_field and m["part_name"]:
                    kwargs[name_field] = m["part_name"]
                if cost_field and (m["unit_cost"] is not None):
                    kwargs[cost_field] = float(m["unit_cost"])
                if sup_field and m.get("supplier"):
                    kwargs[sup_field] = m["supplier"]
                part = PartModel(**kwargs)
                session.add(part)
                session.flush()

            if name_field and not getattr(part, name_field) and m["part_name"]:
                setattr(part, name_field, m["part_name"])
            if cost_field and (m["unit_cost"] is not None):
                setattr(part, cost_field, float(m["unit_cost"]))

            current_qty = getattr(part, qty_field) or 0
            setattr(part, qty_field, current_qty + int(m["qty"]))

            session.commit()
            add_key(m["row_key"], {"file": m["source_file"]})

        built, errors = build_receive_movements(
            norm,
            duplicate_exists_func=duplicate_exists,
            make_movement_func=make_movement
        )
        for e in errors:
            flash(e, "danger")
        flash(f"Создано приходов: {len(built)}", "success")
        return redirect(url_for("inventory.import_parts_upload"))

    # -------- 2) Первая загрузка файла → показать предпросмотр --------
    if request.method == "POST":
        f = request.files.get("file")
        if not f or f.filename == "":
            flash("Выберите файл (.pdf, .xlsx, .xls или .csv)", "warning")
            return redirect(request.url)

        filename = secure_filename(f.filename)
        upload_dir = current_app.config.get("UPLOAD_FOLDER", os.path.join(current_app.instance_path, "uploads"))
        os.makedirs(upload_dir, exist_ok=True)
        path = os.path.join(upload_dir, filename)
        f.save(path)

        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf":
            df = dataframe_from_pdf(path)
            if df.empty:
                flash("В этом PDF не удалось распознать таблицы (скорее всего скан).", "danger")
                rows = []
                return render_template("import_preview.html", rows=rows, saved_path=path)
        else:
            df = load_table(path)

        norm, issues = normalize_table(df, supplier_hint=None, source_file=path, default_location="MAIN")
        for msg in issues:
            flash(msg, "warning")

        rows = norm.to_dict(orient="records")
        return render_template("import_preview.html", rows=rows, saved_path=path)

    # -------- 3) GET → показать твою форму загрузки --------
    return render_template("import_parts.html")










