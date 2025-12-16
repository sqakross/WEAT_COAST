def generate_invoice_pdf(records, invoice_number=None):
    """
    Read-only invoice PDF. Does not write anything to DB.

    Location:
      - If IssuedPartRecord.location exists => use it (snapshot)
      - Else => use current Part.location

    INV#:
      - separate column from IssuedPartRecord.inv_ref
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
    # TABLE LAYOUT (fits on page — no overlaps, pro alignment)
    # ================================================================
    TABLE_X = 40
    TABLE_W = width - 80
    HEADER_H = 20

    PAD = 6   # inner padding in each cell
    GAP = 4   # space between columns (prevents "TotalLOC" glue)

    # IMPORTANT: these widths + gaps MUST FIT in TABLE_W
    # widths sum = 480, gaps = 6*4=24 => 504, plus left padding margin fits into 532
    # widths sum = 480, gaps = 24 => 504 (fits)
    col_defs = [
        ("pnum", "Part Number", 75, left_style, "L"),
        ("pname", "Part Name", 135, left_style, "L"),  # было 150 -> стало 135 (минус 15)
        ("qty", "Qty", 30, right_style, "R"),
        ("ucost", "Unit Cost", 60, right_style, "R"),
        ("total", "Total", 60, right_style, "R"),
        ("loc", "LOC", 55, small_left, "L"),  # было 40 -> стало 55 (плюс 15)
        ("inv", "INV#", 80, small_right, "R"),  # было 65 -> стало 80 (плюс 15)
    ]

    # x positions
    COLS = []
    x_cursor = TABLE_X
    for key, label, w, style, align in col_defs:
        COLS.append({
            "key": key, "label": label, "x": x_cursor, "w": w, "style": style, "align": align
        })
        x_cursor += w + GAP

    def _draw_table_header(y_top):
        c.setFillColor(colors.grey)
        c.rect(TABLE_X, y_top, TABLE_W, HEADER_H, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 9)

        for col in COLS:
            x = col["x"]
            w = col["w"]
            label = col["label"]

            # headers: text left, numeric right (with padding) => looks "invoice-like"
            # headers: INV# по центру, остальные numeric можно оставить справа
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

        pnum = (getattr(part, "part_number", "") or "").strip()
        pname = (getattr(part, "name", "") or "").strip()

        qty = getattr(r, "quantity", 0) or 0

        ucost = getattr(r, "unit_cost_at_issue", None)
        if ucost is None:
            ucost = getattr(part, "unit_cost", 0.0) or 0.0

        stored_loc = getattr(r, "location", None)
        if stored_loc not in (None, "", " ", "  "):
            loc = str(stored_loc).strip()
        else:
            loc = (getattr(part, "location", "") or "").strip()

        inv_ref = (getattr(r, "inv_ref", None) or "").strip()

        line_total = (qty or 0) * float(ucost or 0.0)

        cell_map = {
            "pnum":  Paragraph(pnum, left_style),
            "pname": Paragraph(pname, left_style),
            "qty":   Paragraph(str(qty), right_style),
            "ucost": Paragraph(f"${float(ucost):.2f}", right_style),
            "total": Paragraph(f"${float(line_total):.2f}", right_style),
            "loc":   Paragraph(loc, small_left),
            "inv":   Paragraph(inv_ref or "—", small_right),
        }

        # row height
        min_row_h = 14
        row_h = min_row_h
        sizes = {}
        for col in COLS:
            key = col["key"]
            para = cell_map[key]
            # wrap inside width minus padding
            ww = max(10, col["w"] - (PAD * 2))
            _, h = para.wrap(ww, 1000)
            sizes[key] = h
            if h > row_h:
                row_h = h

        if y - row_h < BOTTOM_MARGIN:
            y = _new_page_header()

        # draw cells with padding
        for col in COLS:
            key = col["key"]
            para = cell_map[key]
            x = col["x"] + PAD
            ww = max(10, col["w"] - (PAD * 2))
            h = sizes[key]
            para.wrap(ww, 1000)
            para.drawOn(c, x, y - h)

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
