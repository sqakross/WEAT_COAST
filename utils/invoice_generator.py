from reportlab.lib.pagesizes import landscape, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
import os
from datetime import datetime


def generate_issued_report_pdf(records):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), rightMargin=20, leftMargin=20, topMargin=20, bottomMargin=20)

    styles = getSampleStyleSheet()
    elements = []

    # Заголовок
    elements.append(Paragraph("Issued Parts Report", styles['Title']))
    elements.append(Spacer(1, 12))

    # Заголовки таблицы
    data = [[
        "Date", "Part Number", "Part Name", "Quantity",
        "Unit Cost", "Total", "Issued To", "Reference Job"
    ]]

    total_quantity = 0
    total_value = 0.0

    for r in records:
        line_total = r.quantity * r.part.unit_cost
        total_quantity += r.quantity
        total_value += line_total

        data.append([
            r.issue_date.strftime('%Y-%m-%d'),
            r.part.part_number,
            r.part.name,
            str(r.quantity),
            f"${r.part.unit_cost:.2f}",
            f"${line_total:.2f}",
            r.issued_to,
            r.reference_job or "N/A"
        ])

    # Добавляем итоговую строку
    data.append([
        "", "", "Grand Total:",
        str(total_quantity),
        "",  # пустая ячейка для Unit Cost
        f"${total_value:.2f}",
        "", ""
    ])

    # Таблица
    table = Table(data, repeatRows=1, colWidths=[70, 80, 160, 60, 80, 80, 100, 100])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (3, 1), (-2, -2), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),

        # Стиль для итоговой строки — жирный шрифт и светло-серый фон
        ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
    ]))

    elements.append(table)
    doc.build(elements)

    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data


def generate_invoice_pdf(record):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=LETTER)
    width, height = LETTER
    styles = getSampleStyleSheet()
    normal_style = styles['Normal']

    # Logo
    logo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logo', 'logo.png'))
    if os.path.exists(logo_path):
        c.drawImage(logo_path, 40, height - 100, width=100, preserveAspectRatio=True, mask='auto')

    # Company Info (top right)
    c.setFont("Helvetica-Bold", 12)
    c.drawRightString(width - 40, height - 40, "WEST COAST CHIEF REPAIR")
    c.setFont("Helvetica", 10)
    c.drawRightString(width - 40, height - 55, "www.westcoastchief.com")
    c.drawRightString(width - 40, height - 70, "support@westcoastchief.com")

    # Invoice Number
    invoice_num = f"INVOICE-{str(record.id).zfill(6)}"
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, height - 120, invoice_num)
    c.line(width / 2 - 80, height - 123, width / 2 + 80, height - 123)

    # Invoice Details (with wrapped text)
    y = height - 160
    details = [
        f"Issue Date: {record.issue_date.strftime('%Y-%m-%d')}",
        f"Issued To: {record.issued_to}",
        f"Issued By: {record.issued_by}",
        f"Reference Job: {record.reference_job or 'N/A'}"
    ]

    for line in details:
        p = Paragraph(line, normal_style)
        w, h = p.wrap(width - 80, 100)
        p.drawOn(c, 40, y - h)
        y -= h + 5

    # Table header
    y -= 10
    c.setFillColor(colors.grey)
    c.rect(40, y, width - 80, 20, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 11)
    headers = [(50, "Part Number"), (160, "Part Name"), (300, "Qty"), (350, "Unit Cost"), (450, "Total")]
    for x, text in headers:
        c.drawString(x, y + 5, text)

    # Table row with wrapped Part Name
    y -= 25
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 10)
    c.drawString(50, y, record.part.part_number)

    p_name = Paragraph(record.part.name, normal_style)
    w, h = p_name.wrap(130, 100)
    p_name.drawOn(c, 160, y - h + 10)  # adjust vertical position for alignment

    c.drawString(300, y, str(record.quantity))
    c.drawString(350, y, f"${record.part.unit_cost:.2f}")

    total = record.quantity * record.part.unit_cost
    c.drawString(450, y, f"${total:.2f}")

    # Footer total
    y -= max(h, 15) + 30
    c.setFont("Helvetica-Bold", 12)
    c.drawRightString(width - 40, y, f"TOTAL: ${total:.2f}")

    # Thank you message
    y -= 50
    c.setFont("Helvetica-Oblique", 10)
    c.drawCentredString(width / 2, y, "Thank you for your business!")

    c.showPage()
    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# def generate_view_pdf(record):
#     buffer = BytesIO()
#     c = canvas.Canvas(buffer, pagesize=LETTER)
#     width, height = LETTER
#     styles = getSampleStyleSheet()
#     normal_style = styles['Normal']
#
#     # Logo
#     logo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logo', 'logo.png'))
#     if os.path.exists(logo_path):
#         c.drawImage(logo_path, 40, height - 330, width=140, preserveAspectRatio=True, mask='auto')
#
#     # Company Info (top right)
#     c.setFont("Helvetica-Bold", 12)
#     c.drawRightString(width - 40, height - 40, "WEST COAST CHIEF REPAIR")
#     c.setFont("Helvetica", 10)
#     c.drawRightString(width - 40, height - 55, "3300 N. SAN FERNANDO BLVD.")
#     c.drawRightString(width - 40, height - 70, "SUITE 101")
#     c.drawRightString(width - 40, height - 85, "BURBANK, CA 91504")
#     c.drawRightString(width - 40, height - 100, "parts@chiafappliance.com")
#     c.drawRightString(width - 40, height - 115, "Phone:(323) 782-3922")
#
#     # Invoice Title and number
#     invoice_num = f"INVOICE-{str(record.id).zfill(6)}"
#     c.setFont("Helvetica-Bold", 20)
#     c.drawCentredString(width / 2, height - 140, invoice_num)
#     c.line(width / 2 - 80, height - 143, width / 2 + 80, height - 143)
#
#     # Invoice Info Box
#     c.setFont("Helvetica", 11)
#     y = height - 160
#     info_lines = [
#         f"Issue Date: {record.issue_date.strftime('%m-%d-%Y')}",
#         f"Issued To: {record.issued_to}",
#         f"Issued By: {record.issued_by}",
#         f"Reference Job: {record.reference_job or 'N/A'}"
#     ]
#     for line in info_lines:
#         c.drawString(40, y, line)
#         y -= 15
#
#     # Table Header
#     y -= 40
#     c.setFillColor(colors.grey)
#     c.rect(40, y, width - 80, 20, fill=1)
#     c.setFillColor(colors.white)
#     c.setFont("Helvetica-Bold", 11)
#     headers = [(50, "Part Number"), (160, "Part Name"), (300, "Qty"), (350, "Unit Cost"), (450, "Total")]
#     for x, text in headers:
#         c.drawString(x, y + 5, text)
#
#     # Prepare the Part Name as Paragraph to wrap text
#     part_name_para = Paragraph(record.part.name, normal_style)
#     w, h = part_name_para.wrap(130, 100)  # width limit for Part Name
#
#     # Calculate max height for row (other columns are single line ~15)
#     row_height = max(h, 15)
#
#     y -= row_height
#
#     # Draw cells with vertical alignment
#     c.setFillColor(colors.black)
#     c.setFont("Helvetica", 10)
#     c.drawString(50, y + (row_height - 15), record.part.part_number)
#     part_name_para.drawOn(c, 160, y + (row_height - h))
#     c.drawString(300, y + (row_height - 15), str(record.quantity))
#     c.drawString(350, y + (row_height - 15), f"${record.part.unit_cost:.2f}")
#     total = record.quantity * record.part.unit_cost
#     c.drawString(450, y + (row_height - 15), f"${total:.2f}")
#
#     # Footer - Total
#     y -= 40
#     c.setFont("Helvetica-Bold", 12)
#     c.drawRightString(width - 40, y, f"TOTAL: ${total:.2f}")
#
#     # Thank you message
#     y -= 50
#     c.setFont("Helvetica-Oblique", 10)
#     c.drawCentredString(width / 2, y, "Thank you for your business!")
#
#     # Finish
#     c.showPage()
#     c.save()
#     pdf = buffer.getvalue()
#     buffer.close()
#     return pdf

def generate_view_pdf(records):
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph
    from reportlab.lib import colors
    from io import BytesIO
    import os

    if not records:
        return b''

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=LETTER)
    width, height = LETTER
    styles = getSampleStyleSheet()
    normal_style = styles['Normal']

    # Логотип
    logo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logo', 'logo.png'))
    if os.path.exists(logo_path):
        c.drawImage(logo_path, 40, height - 330, width=140, preserveAspectRatio=True, mask='auto')

    # Общие данные инвойса (берём из первой записи)
    first_record = records[0]

    c.setFont("Helvetica-Bold", 12)
    c.drawRightString(width - 40, height - 40, "WEST COAST CHIEF REPAIR")
    c.setFont("Helvetica", 10)
    c.drawRightString(width - 40, height - 55, "3300 N. SAN FERNANDO BLVD.")
    c.drawRightString(width - 40, height - 70, "SUITE 101")
    c.drawRightString(width - 40, height - 85, "BURBANK, CA 91504")
    c.drawRightString(width - 40, height - 100, "parts@chiafappliance.com")
    c.drawRightString(width - 40, height - 115, "Phone:(323) 782-3922")

    invoice_num = f"INVOICE-{str(first_record.id).zfill(6)}"
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, height - 140, invoice_num)
    c.line(width / 2 - 80, height - 143, width / 2 + 80, height - 143)

    # Информация об инвойсе
    c.setFont("Helvetica", 11)
    y = height - 160
    info_lines = [
        f"Issue Date: {first_record.issue_date.strftime('%m-%d-%Y')}",
        f"Issued To: {first_record.issued_to}",
        f"Issued By: {first_record.issued_by}",
        f"Reference Job: {first_record.reference_job or 'N/A'}"
    ]
    for line in info_lines:
        c.drawString(40, y, line)
        y -= 15

    # Заголовок таблицы
    y -= 40
    c.setFillColor(colors.grey)
    c.rect(40, y, width - 80, 20, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 11)
    headers = [(50, "Part Number"), (160, "Part Name"), (300, "Qty"), (350, "Unit Cost"), (450, "Total")]
    for x, text in headers:
        c.drawString(x, y + 5, text)

    # Отрисовка позиций
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 10)
    y -= 25

    total_sum = 0

    for record in records:
        # Рисуем строку с данными
        part_name_para = Paragraph(record.part.name, normal_style)
        w, h = part_name_para.wrap(130, 100)
        row_height = max(h, 15)

        c.drawString(50, y + (row_height - 15), record.part.part_number)
        part_name_para.drawOn(c, 160, y + (row_height - h))
        c.drawString(300, y + (row_height - 15), str(record.quantity))
        c.drawString(350, y + (row_height - 15), f"${record.part.unit_cost:.2f}")

        line_total = record.quantity * record.part.unit_cost
        c.drawString(450, y + (row_height - 15), f"${line_total:.2f}")

        y -= row_height + 5
        total_sum += line_total

        # Если мало места, начинаем новую страницу
        if y < 100:
            c.showPage()
            y = height - 50
            # Можно повторить заголовки таблицы (опционально)

    # Итог
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawRightString(width - 40, y, f"TOTAL: ${total_sum:.2f}")

    # Спасибо
    y -= 50
    c.setFont("Helvetica-Oblique", 10)
    c.drawCentredString(width / 2, y, "Thank you for your business!")

    c.showPage()
    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf



