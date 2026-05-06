def generate_invoice_pdf(records, invoice_number=None):
    """
    Read-only invoice PDF. Does not write anything to DB.

    Location:
      - If IssuedPartRecord.location exists => use it (snapshot)
      - Else => use current Part.location

    INV#:
      - separate column from IssuedPartRecord.inv_ref
      - must be fully visible (shrink font to fit, no ellipsis)
    """
    from io import BytesIO
    import os
    from datetime import datetime as _dt

    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import Paragraph
    from reportlab.lib.enums import TA_RIGHT, TA_LEFT
    from reportlab.pdfbase.pdfmetrics import stringWidth

    if not records:
        return b""

    first = records[0]

    # ---------- resolve invoice number ----------
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

    issued_to = getattr(first, "issued_to", "") or ""
    issued_by = getattr(first, "issued_by", "") or ""
    ref_job = getattr(first, "reference_job", "") or ""
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
    base = styles["Normal"]

    left_style = ParagraphStyle(
        "left", parent=base, alignment=TA_LEFT,
        fontName="Helvetica", fontSize=9, leading=11,
        spaceBefore=0, spaceAfter=0
    )
    right_style = ParagraphStyle(
        "right", parent=base, alignment=TA_RIGHT,
        fontName="Helvetica", fontSize=9, leading=11,
        spaceBefore=0, spaceAfter=0
    )
    small_left = ParagraphStyle(
        "small_left", parent=left_style, fontSize=8.5, leading=10
    )
    small_right = ParagraphStyle(
        "small_right", parent=right_style, fontSize=8.5, leading=10
    )

    # ---------- logo ----------
    try:
        logo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logo", "logo.png"))
        if os.path.exists(logo_path):
            c.drawImage(
                logo_path,
                40, height - 330,
                width=140,
                preserveAspectRatio=True,
                mask="auto",
            )
    except Exception:
        pass

    # ---------- company block ----------
    c.setFont("Helvetica-Bold", 12)
    c.drawRightString(width - 40, height - 40, "WEST COAST CHIEF REPAIR")
    c.setFont("Helvetica", 10)
    c.drawRightString(width - 40, height - 55, "3300 N. SAN FERNANDO BLVD.")
    c.drawRightString(width - 40, height - 70, "SUITE 101")
    c.drawRightString(width - 40, height - 85, "BURBANK, CA 91504")
    c.drawRightString(width - 40, height - 100, "parts@chiafappliance.com")
    c.drawRightString(width - 40, height - 115, "Phone:(323) 782-3922")

    # ---------- title ----------
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, height - 140, inv_title)
    c.line(width / 2 - 80, height - 143, width / 2 + 80, height - 143)

    # ---------- header info (2-column layout) ----------
    c.setFont("Helvetica", 11)

    y = height - 160

    LEFT_X = 40
    RIGHT_X = width / 2 + 40

    LINE_GAP = 15

    # Row 1
    c.drawString(LEFT_X, y, f"Issue Date: {issue_date_s}")
    c.drawString(RIGHT_X, y, f"Issued By: {issued_by}")

    y -= LINE_GAP

    # Row 2
    c.drawString(LEFT_X, y, f"Issued To: {issued_to}")
    c.drawString(RIGHT_X, y, f"Reference Job: {ref_job or 'N/A'}")

    y -= LINE_GAP

    # ================================================================
    # TABLE LAYOUT
    # ================================================================
    TABLE_X = 40
    TABLE_W = width - 80
    HEADER_H = 20

    PAD = 3
    GAP = 2
    is_return_invoice = any((getattr(r, "quantity", 0) or 0) < 0 for r in records)

    if is_return_invoice:
        col_defs = [
            ("pnum", "Part Number", 72, left_style, "L"),
            ("pname", "Part Name", 88, left_style, "L"),
            ("qty", "Qty", 24, right_style, "R"),
            ("ucost", "Unit Cost", 50, right_style, "R"),
            ("total", "Total", 50, right_style, "R"),
            ("inv", "INV#", 90, small_right, "R"),
            ("returnto", "Return To", 52, small_left, "L"),
            ("company", "Company", 70, small_left, "L"),
        ]
    else:
        col_defs = [
            ("pnum", "Part Number", 85, left_style, "L"),
            ("pname", "Part Name", 130, left_style, "L"),
            ("qty", "Qty", 30, right_style, "R"),
            ("ucost", "Unit Cost", 60, right_style, "R"),
            ("total", "Total", 60, right_style, "R"),
            ("inv", "INV#", 70, small_right, "R"),
            ("company", "Company", 110, small_left, "L"),
        ]

    # AUTO SCALE TABLE WIDTH
    total_cols_width = sum(w for _, _, w, _, _ in col_defs) + GAP * (len(col_defs) - 1)

    if total_cols_width > TABLE_W:
        scale = TABLE_W / total_cols_width
        col_defs = [
            (k, l, max(20, int(w * scale)), s, a)
            for (k, l, w, s, a) in col_defs
        ]

    # x positions
    COLS = []
    x_cursor = TABLE_X
    for key, label, w, style, align in col_defs:
        COLS.append({"key": key, "label": label, "x": x_cursor, "w": w, "style": style, "align": align})
        x_cursor += w + GAP

    # ---------- helpers ----------
    def _one_line(s: str) -> str:
        return (s or "").replace("\r", " ").replace("\n", " ").strip()

    def _ellipsize(text: str, font_name: str, font_size: float, max_width: float) -> str:
        text = text or ""
        if stringWidth(text, font_name, font_size) <= max_width:
            return text
        dots = "…"
        w_dots = stringWidth(dots, font_name, font_size)
        max_w = max_width - w_dots
        if max_w <= 0:
            return dots
        out = ""
        for ch in text:
            if stringWidth(out + ch, font_name, font_size) > max_w:
                break
            out += ch
        return out + dots

    def _draw_cell_oneline(x_left, y_bottom, col_w, text, align="L", font="Helvetica", size=9):
        # draw 1 line + ellipsis (PN/Name only)
        text = _one_line(text)
        max_w = max(1, col_w - (PAD * 2))
        s = _ellipsize(text, font, size, max_w)
        c.setFont(font, size)
        if align == "R":
            c.drawRightString(x_left + col_w - PAD, y_bottom, s)
        else:
            c.drawString(x_left + PAD, y_bottom, s)

    def _draw_fit_left(x_left, y_bottom, col_w, text, base_size=8.0, min_size=6.5):
        # LOC full (shrink font to fit, no ellipsis)
        text = _one_line(text)
        max_w = max(1, col_w - (PAD * 2))
        font = "Helvetica"
        size = float(base_size)
        while size >= min_size and stringWidth(text, font, size) > max_w:
            size -= 0.25
        c.setFont(font, size)
        c.drawString(x_left + PAD, y_bottom, text)

    def _draw_fit_right(x_left, y_bottom, col_w, text, base_size=8.5, min_size=6.5):
        # INV# full (shrink font to fit, no ellipsis)
        text = _one_line(text)
        max_w = max(1, col_w - (PAD * 2))
        font = "Helvetica"
        size = float(base_size)
        while size >= min_size and stringWidth(text, font, size) > max_w:
            size -= 0.25
        c.setFont(font, size)
        c.drawRightString(x_left + col_w - PAD, y_bottom, text)

    def _draw_table_header(y_top):
        c.setFillColor(colors.grey)
        c.rect(TABLE_X, y_top, TABLE_W, HEADER_H, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 9)

        for col in COLS:
            x = col["x"]
            w = col["w"]
            label = col["label"]
            key = col["key"]

            if key in ("qty", "ucost", "total", "inv"):
                c.drawRightString(x + w - PAD, y_top + 5, label)
            else:
                c.drawString(x + PAD, y_top + 5, label)

        c.setFillColor(colors.black)
        c.setFont("Helvetica", 9)

    # header row
    y -= 40
    _draw_table_header(y)

    y -= 25
    total_sum = 0.0
    BOTTOM_MARGIN = 80
    ROW_PADDING = 5

    def _new_page_header():
        c.showPage()
        yy = height - 80
        _draw_table_header(yy)
        return yy - 25

    for r in records:
        part = getattr(r, "part", None)

        pnum = _one_line(getattr(part, "part_number", "") or "").replace(" ", "")
        pname = _one_line(getattr(part, "name", "") or "")

        qty = getattr(r, "quantity", 0) or 0

        ucost = getattr(r, "unit_cost_at_issue", None)
        if ucost is None:
            ucost = getattr(part, "unit_cost", 0.0) or 0.0

        return_to = _one_line(getattr(r, "return_to", "") or "")
        inv_ref_raw = _one_line(getattr(r, "inv_ref", None) or "")

        src_line = getattr(r, "source_receipt_line", None)
        receipt = getattr(src_line, "goods_receipt", None) if src_line else None
        receipt_inv = _one_line(getattr(receipt, "invoice_number", "") if receipt else "")

        if is_return_invoice:
            dest = getattr(r, "return_destination", None)
            company = _one_line(getattr(dest, "name", "") if dest else "")

            # RETURN: show receiving invoice number, not issue invoice number
            inv_ref = receipt_inv or inv_ref_raw or "—"
        else:
            if not inv_ref_raw:
                company = "STOCK"
            else:
                company = _one_line(getattr(receipt, "supplier_name", "") if receipt else "")

            inv_ref = inv_ref_raw or "—"

        if not company:
            company = "—"

        line_total = (qty or 0) * float(ucost or 0.0)

        cell_map = {
            "pnum": pnum,
            "pname": pname,
            "inv": inv_ref,
            "company": company,

            "qty": Paragraph(str(qty), right_style),
            "ucost": Paragraph(f"${float(ucost):.2f}", right_style),
            "total": Paragraph(f"${float(line_total):.2f}", right_style),
        }



        if is_return_invoice:
            cell_map["returnto"] = return_to

        # row height
        min_row_h = 14
        row_h = min_row_h
        sizes = {}

        for col in COLS:
            key = col["key"]
            ww = max(10, col["w"] - (PAD * 2))
            if key in ("pnum", "pname", "inv", "returnto", "company"):
                h = 10
            else:
                para = cell_map[key]
                _, h = para.wrap(ww, 1000)
            sizes[key] = h
            if h > row_h:
                row_h = h

        if y - row_h < BOTTOM_MARGIN:
            y = _new_page_header()

        # draw cells
        for col in COLS:
            key = col["key"]
            x0 = col["x"]
            w = col["w"]
            y_text = y - 10

            if key == "pnum":
                _draw_fit_left(
                    x0,
                    y_text,
                    w,
                    cell_map["pnum"],
                    base_size=8.5,
                    min_size=5.0
                )

            elif key == "pname":
                _draw_cell_oneline(x0, y_text, w, cell_map["pname"], align="L", font="Helvetica", size=8.5)

            elif key == "inv":
                _draw_fit_right(x0, y_text, w, cell_map["inv"], base_size=7.2, min_size=4.8)

            elif key == "returnto":
                _draw_fit_left(x0, y_text, w, cell_map["returnto"], base_size=7.5, min_size=5.5)


            elif key == "company":
                _draw_fit_left(x0, y_text, w, cell_map["company"].upper(), base_size=7.5, min_size=5.5)

            else:
                para = cell_map[key]
                ww = max(10, w - (PAD * 2))
                _, h = para.wrap(ww, 1000)
                para.drawOn(c, x0 + PAD, y - h)

        y -= (row_h + ROW_PADDING)
        total_sum += float(line_total)

    table_bottom_y = y
    # total (привязан к таблице)
    total_y = max(table_bottom_y - 25, 60)

    c.setStrokeColor(colors.black)
    c.setLineWidth(1)
    c.line(TABLE_X, total_y + 10, TABLE_X + TABLE_W, total_y + 10)

    c.setFont("Helvetica-Bold", 13)
    c.drawRightString(width - 40, total_y - 10, f"TOTAL: ${total_sum:.2f}")

    # footer
    y -= 50
    c.setFont("Helvetica-Oblique", 10)
    c.drawCentredString(width / 2, max(y, 40), "Thank you for your business!")

    c.showPage()
    c.save()
    pdf = buf.getvalue()
    buf.close()
    return pdf

