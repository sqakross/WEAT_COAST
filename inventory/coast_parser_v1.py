
"""WCCR Coast invoice PDF parser V1."""
import re
from decimal import Decimal
import pandas as pd

INVOICE_RX = re.compile(
    r"Invoice\s*#\s*:\s*(\d{6,}-\d{2})", re.I
)

ROW_RX = re.compile(
    r"^\s*\d+\s+"
    r"([A-Z0-9][A-Z0-9./-]*)\s+"
    r"(.+?)\s+"
    r"(\d+)\s+(\d+)\s+(\d+)\s+"
    r"(?:each|ea)\s+"
    r"\$?([\d,]+\.\d{2})\s+"
    r"\$?([\d,]+\.\d{2})"
    r"(?:\s+Part)?\s*$",
    re.I
)

def is_coast_pdf(path):
    if not str(path).lower().endswith(".pdf"):
        return False
    import fitz
    with fitz.open(path) as doc:
        head = doc[0].get_text() if doc.page_count else ""
    return bool(
        re.search(r"Coast\s+Appliance\s+Parts", head, re.I)
        and INVOICE_RX.search(head)
    )

def parse_coast_text(text, source_file=""):
    invoice_match = INVOICE_RX.search(text)
    if not invoice_match:
        raise ValueError("COAST invoice number not found")

    invoice = invoice_match.group(1)
    freight_match = re.search(
        r"(?im)^\s*Freight\s*:\s*\$?([\d,]+\.\d{2})",
        text
    )
    subtotal_match = re.search(
        r"(?im)^\s*TOTAL\s*:\s*\$?([\d,]+\.\d{2})",
        text
    )
    total_match = re.search(
        r"(?im)^\s*INVOICE\s+TOTAL\s*:\s*\$?([\d,]+\.\d{2})",
        text
    )

    if not (freight_match and subtotal_match and total_match):
        raise ValueError("COAST invoice totals incomplete")

    freight = Decimal(freight_match.group(1).replace(",", ""))
    subtotal = Decimal(subtotal_match.group(1).replace(",", ""))
    invoice_total = Decimal(total_match.group(1).replace(",", ""))

    if subtotal + freight != invoice_total:
        raise ValueError("COAST invoice totals do not reconcile")

    rows = []
    seen = set()

    for line in text.splitlines():
        match = ROW_RX.match(line)
        if not match:
            continue

        pn, description, ordered, backorder, shipped, price, extended = match.groups()
        ordered = int(ordered)
        backorder = int(backorder)
        shipped = int(shipped)
        price = Decimal(price.replace(",", ""))
        extended = Decimal(extended.replace(",", ""))

        if shipped > ordered or backorder > ordered:
            raise ValueError("COAST invalid quantities: " + pn)

        if price * shipped != extended:
            raise ValueError("COAST extended price mismatch: " + pn)

        if pn in seen:
            raise ValueError("COAST duplicate part: " + pn)
        seen.add(pn)

        if shipped == 0:
            continue

        rows.append({
            "part_number": pn.upper(),
            "part_name": description.strip(),
            "quantity": shipped,
            "unit_cost": float(price),
            "supplier": "Coast Appliance Parts",
            "location": "COAST",
            "order_no": invoice,
            "invoice_no": invoice,
            "date": "",
            "source_file": source_file,
        })

    if not rows:
        raise ValueError("COAST no shipped parts recognized")

    calculated = sum(
        Decimal(str(r["unit_cost"])) * r["quantity"]
        for r in rows
    )
    if calculated != subtotal:
        raise ValueError(
            "COAST parts subtotal mismatch: "
            + str(calculated) + " vs " + str(subtotal)
        )

    return pd.DataFrame(rows), invoice, float(freight)

def parse_coast_pdf(path):
    import fitz
    with fitz.open(path) as doc:
        text = "\n".join(
            page.get_text(sort=True) for page in doc
        )
    return parse_coast_text(text, str(path))
