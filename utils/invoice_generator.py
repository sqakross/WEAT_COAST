def generate_invoice_pdf(records, invoice_number=None):
    """
    Групповая печать инвойса (read-only).
    Приоритет номера: param -> records[0].invoice_number -> legacy id.
    НИЧЕГО в БД не меняет.
    """
    from io import BytesIO
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import Paragraph
    from reportlab.lib.enums import TA_RIGHT
    import os
    from datetime import datetime as _dt

    if not records:
        return b""

    first = records[0]

    # ---------- resolve number ----------
    inv_no = None
    if invoice_number is not None:
        try:
            inv_no = int(invoice_number)
        except Exception:
            inv_no = None
    if inv_no is None:
        n = getattr(first, "invoice_number", None)
        if n is not None:
            try:
                inv_no = int(n)
            except Exception:
                inv_no = None
    if inv_no is None:
        try:
            inv_no = int(getattr(first, "id"))
        except Exception:
            inv_no = None

    inv_title = f"INVOICE-{inv_no:06d}" if inv_no is not None else "INVOICE"

    issued_to  = getattr(first, "issued_to", "") or ""
    issued_by  = getattr(first, "issued_by", "") or ""
    ref_job    = getattr(first, "reference_job", "") or ""
    issue_date = getattr(first, "issue_date", None)
    try:
        issue_date_s = issue_date.strftime("%m-%d-%Y") if isinstance(issue_date, _dt) else ""
    except Exception:
        issue_date_s = ""

    # ---------- canvas ----------
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    width, height = LETTER
    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]
    right_style  = ParagraphStyle("right", parent=normal_style, alignment=TA_RIGHT)

    # === ЛОГО (СТАРАЯ ПОЗИЦИЯ, как в самом первом варианте) ===
    try:
        logo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logo", "logo.png"))
        if os.path.exists(logo_path):
            c.drawImage(logo_path, 40, height - 330, width=140, preserveAspectRatio=True, mask="auto")
    except Exception:
        pass

    # Company block (справа вверху)
    c.setFont("Helvetica-Bold", 12)
    c.drawRightString(width - 40, height - 40, "WEST COAST CHIEF REPAIR")
    c.setFont("Helvetica", 10)
    c.drawRightString(width - 40, height - 55, "3300 N. SAN FERNANDO BLVD.")
    c.drawRightString(width - 40, height - 70, "SUITE 101")
    c.drawRightString(width - 40, height - 85, "BURBANK, CA 91504")
    c.drawRightString(width - 40, height - 100, "parts@chiafappliance.com")
    c.drawRightString(width - 40, height - 115, "Phone:(323) 782-3922")

    # Заголовок
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, height - 140, inv_title)
    c.line(width / 2 - 80, height - 143, width / 2 + 80, height - 143)

    # Шапка данных
    c.setFont("Helvetica", 11)
    y = height - 160
    for line in [
        f"Issue Date: {issue_date_s}",
        f"Issued To: {issued_to}",
        f"Issued By: {issued_by}",
        f"Reference Job: {ref_job or 'N/A'}",
    ]:
        c.drawString(40, y, line)
        y -= 15

    # Заголовок таблицы
    y -= 40
    c.setFillColor(colors.grey)
    c.rect(40, y, width - 80, 20, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 11)
    headers = [(50, "Part Number"), (160, "Part Name"), (300, "Qty"),
               (350, "Unit Cost"), (450, "Total"), (520, "Location")]
    for x, text in headers:
        c.drawString(x, y + 5, text)

    # ---------- Строки (ровное выравнивание всех колонок) ----------
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 10)
    y -= 25
    total_sum = 0.0

    # координаты и ширины колонок (все как в твоём исходнике)
    COLS = {
        "pnum":  {"x": 50,  "w": 100, "style": normal_style},
        "pname": {"x": 160, "w": 130, "style": normal_style},
        "qty":   {"x": 300, "w": 40,  "style": right_style},
        "ucost": {"x": 350, "w": 70,  "style": right_style},
        "total": {"x": 450, "w": 70,  "style": right_style},
        "loc":   {"x": 520, "w": 80,  "style": normal_style},
    }

    def _draw_table_header_only():
        c.setFont("Helvetica-Bold", 11)
        yy = height - 60
        c.setFillColor(colors.grey)
        c.rect(40, yy, width - 80, 20, fill=1)
        c.setFillColor(colors.white)
        for x, text in headers:
            c.drawString(x, yy + 5, text)
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 10)
        return height - 85  # новая рабочая Y после заголовка таблицы

    for r in records:
        pnum = getattr(getattr(r, "part", None), "part_number", "") or ""
        pname = getattr(getattr(r, "part", None), "name", "") or ""
        qty   = getattr(r, "quantity", 0) or 0
        ucost = getattr(r, "unit_cost_at_issue", None)
        if ucost is None:
            ucost = getattr(getattr(r, "part", None), "unit_cost", 0.0) or 0.0
        loc   = getattr(r, "location", None) or getattr(getattr(r, "part", None), "location", "") or ""

        line_total = (qty or 0) * float(ucost or 0.0)

        # перенос страницы при нехватке места
        min_row_h = 15
        if y < (100 + min_row_h):
            c.showPage()
            y = _draw_table_header_only()

        # готовим параграфы для всех колонок (чтобы высота считалась одинаково)
        cells = {
            "pnum":  Paragraph(str(pnum), COLS["pnum"]["style"]),
            "pname": Paragraph(pname,      COLS["pname"]["style"]),
            "qty":   Paragraph(str(qty),   COLS["qty"]["style"]),
            "ucost": Paragraph(f"${float(ucost):.2f}", COLS["ucost"]["style"]),
            "total": Paragraph(f"${float(line_total):.2f}", COLS["total"]["style"]),
            "loc":   Paragraph(str(loc),   COLS["loc"]["style"]),
        }

        # вычисляем единую высоту строки
        max_h = 15
        sizes = {}
        for key, para in cells.items():
            _, h = para.wrap(COLS[key]["w"], 1000)
            sizes[key] = h
            if h > max_h:
                max_h = h

        # рисуем ячейки, прижимая к верхней кромке строки
        for key, para in cells.items():
            x = COLS[key]["x"]
            h = sizes[key]
            para.drawOn(c, x, y + (max_h - h))

        y -= (max_h + 5)
        total_sum += float(line_total)

    # Итог
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawRightString(width - 40, max(y, 60), f"TOTAL: ${total_sum:.2f}")

    # Спасибо
    y -= 50
    c.setFont("Helvetica-Oblique", 10)
    c.drawCentredString(width / 2, max(y, 40), "Thank you for your business!")

    c.showPage()
    c.save()
    pdf = buf.getvalue()
    buf.close()
    return pdf



