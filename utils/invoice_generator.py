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

    # ---------- header info ----------
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

    # ================================================================
    # TABLE LAYOUT
    # ================================================================
    TABLE_X = 40
    TABLE_W = width - 80
    HEADER_H = 20

    PAD = 6
    GAP = 4

    # tuned widths: PN + INV readable; LOC fully visible by shrinking font
    col_defs = [
        ("pnum",  "Part Number",  95, left_style,  "L"),
        ("pname", "Part Name",   110, left_style,  "L"),
        ("qty",   "Qty",          30, right_style, "R"),
        ("ucost", "Unit Cost",    60, right_style, "R"),
        ("total", "Total",        60, right_style, "R"),
        ("loc",   "LOC",          42, small_left,  "L"),
        ("inv",   "INV#",        102, small_right, "R"),
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
            if col["key"] == "inv":
                c.drawCentredString(x + (w / 2), y_top + 5, label)
            elif col["align"] == "R":
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

        stored_loc = getattr(r, "location", None)
        if stored_loc not in (None, "", " ", "  "):
            loc = str(stored_loc).strip()
        else:
            loc = (getattr(part, "location", "") or "").strip()

        # LOC: up to 6 chars, but ALWAYS display fully (shrink font if needed)
        loc_short = _one_line(loc).upper()[:6]

        inv_ref = _one_line(getattr(r, "inv_ref", None) or "") or "—"

        line_total = (qty or 0) * float(ucost or 0.0)

        cell_map = {
            "pnum":  pnum,
            "pname": pname,
            "loc":   loc_short,
            "inv":   inv_ref,

            "qty":   Paragraph(str(qty), right_style),
            "ucost": Paragraph(f"${float(ucost):.2f}", right_style),
            "total": Paragraph(f"${float(line_total):.2f}", right_style),
        }

        # row height
        min_row_h = 14
        row_h = min_row_h
        sizes = {}

        for col in COLS:
            key = col["key"]
            ww = max(10, col["w"] - (PAD * 2))
            if key in ("pnum", "pname", "loc", "inv"):
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
                _draw_cell_oneline(x0, y_text, w, cell_map["pnum"], align="L", font="Helvetica", size=8.5)

            elif key == "pname":
                _draw_cell_oneline(x0, y_text, w, cell_map["pname"], align="L", font="Helvetica", size=8.5)

            elif key == "loc":
                _draw_fit_left(x0, y_text, w, cell_map["loc"], base_size=8.0, min_size=6.5)

            elif key == "inv":
                _draw_fit_right(x0, y_text, w, cell_map["inv"], base_size=8.5, min_size=6.5)

            else:
                para = cell_map[key]
                ww = max(10, w - (PAD * 2))
                _, h = para.wrap(ww, 1000)
                para.drawOn(c, x0 + PAD, y - h)

        y -= (row_h + ROW_PADDING)
        total_sum += float(line_total)

    # total
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawRightString(width - 40, max(y, 60), f"TOTAL: ${total_sum:.2f}")

    # footer
    y -= 50
    c.setFont("Helvetica-Oblique", 10)
    c.drawCentredString(width / 2, max(y, 40), "Thank you for your business!")

    c.showPage()
    c.save()
    pdf = buf.getvalue()
    buf.close()
    return pdf

