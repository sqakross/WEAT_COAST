from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class ParsedStatementLine:
    line_type: str

    document_number: str
    document_date: object | None
    due_date: object | None

    description: str

    invoice_amount: float
    credit_amount: float
    open_balance: float

    raw_text: str


@dataclass(frozen=True)
class ParsedStatement:
    supplier_name: str
    statement_period: object
    account_number: str | None
    balance_due: float
    lines: list[ParsedStatementLine]


def _money_to_float(value: str | None) -> float:
    """
    Converts:
        ($82.29)   -> 82.29
        $1,234.56 -> 1234.56
        123.45    -> 123.45
    """
    text = (value or "").strip()

    if not text:
        return 0.0

    text = (
        text.replace("$", "")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
        .strip()
    )

    return round(float(text), 2)


def _parse_mmddyyyy(value: str):
    return datetime.strptime(
        value,
        "%m/%d/%Y",
    ).date()


def parse_marcone_statement_text(
    text: str,
) -> ParsedStatement:
    """
    Parse a Marcone CUSTOMER STATEMENT.

    Parses both sections:
    - OPEN CREDITS & PAYMENTS
    - OPEN INVOICES

    This parser only extracts data. It does not:
    - modify inventory;
    - modify employee ledger;
    - create payments;
    - create returns;
    - perform reconciliation.
    """

    raw_text = text or ""

    if "marcone" not in raw_text.lower():
        raise ValueError(
            "This does not appear to be a Marcone statement."
        )

    # ---------------------------------------------------------
    # Statement period
    #
    # Example:
    # June 2026 CUSTOMER STATEMENT
    # ---------------------------------------------------------
    period_match = re.search(
        r"\b("
        r"January|February|March|April|May|June|"
        r"July|August|September|October|November|December"
        r")\s+(\d{4})\s+CUSTOMER\s+STATEMENT\b",
        raw_text,
        flags=re.IGNORECASE,
    )

    if not period_match:
        raise ValueError(
            "Could not determine statement period."
        )

    month_name = period_match.group(1)
    year = int(period_match.group(2))

    statement_period = datetime.strptime(
        f"{month_name} {year}",
        "%B %Y",
    ).date().replace(day=1)

    # ---------------------------------------------------------
    # Account number
    #
    # Example:
    # Account *To Be Applied...
    # 965767 ($3,005.49)
    # ---------------------------------------------------------
    account_number = None

    account_match = re.search(
        r"\bAccount\b.*?\n\s*(\d{4,})\b",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if account_match:
        account_number = (
            account_match.group(1) or ""
        ).strip() or None

    # ---------------------------------------------------------
    # Balance Due
    #
    # Example:
    # Balance Due 07/20/2026
    # $12,067.51
    # ---------------------------------------------------------
    balance_due = 0.0

    balance_match = re.search(
        r"Balance\s+Due\s+\d{2}/\d{2}/\d{4}\s*"
        r"\$([\d,]+\.\d{2})",
        raw_text,
        flags=re.IGNORECASE,
    )

    if balance_match:
        balance_due = _money_to_float(
            balance_match.group(1)
        )

    # All parsed statement lines are collected here.
    lines: list[ParsedStatementLine] = []

    # =========================================================
    # OPEN CREDITS & PAYMENTS
    # =========================================================
    credits_section_match = re.search(
        r"OPEN\s+CREDITS\s*&\s*PAYMENTS"
        r"(.*?)"
        r"OPEN\s+INVOICES",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not credits_section_match:
        raise ValueError(
            "OPEN CREDITS & PAYMENTS section was not found."
        )

    credits_section = credits_section_match.group(1)

    credit_pattern = re.compile(
        r"(?P<document_date>\d{2}/\d{2}/\d{4})\s+"
        r"(?P<document_number>\d+)\s+"
        r"(?P<description>.*?)\s+"
        r"(?P<due_date>\d{2}/\d{2}/\d{4})\s+"
        r"\(\$(?P<credit_amount>[\d,]+\.\d{2})\)\s+"
        r"\(\$(?P<remaining_credit>[\d,]+\.\d{2})\)\s+"
        r"\$[\d,]+\.\d{2}",
        flags=re.IGNORECASE,
    )

    for match in credit_pattern.finditer(
        credits_section
    ):
        document_number = (
            match.group("document_number") or ""
        ).strip()

        description = (
            match.group("description") or ""
        ).strip()

        description_upper = description.upper()

        if description_upper.startswith("RETURN"):
            line_type = "return"
        elif "PAYMENT" in description_upper:
            line_type = "payment"
        else:
            line_type = "credit"

        lines.append(
            ParsedStatementLine(
                line_type=line_type,

                document_number=document_number,

                document_date=_parse_mmddyyyy(
                    match.group("document_date")
                ),

                due_date=_parse_mmddyyyy(
                    match.group("due_date")
                ),

                description=description,

                invoice_amount=0.0,

                credit_amount=_money_to_float(
                    match.group("credit_amount")
                ),

                open_balance=_money_to_float(
                    match.group("remaining_credit")
                ),

                raw_text=match.group(0).strip(),
            )
        )

    # =========================================================
    # OPEN INVOICES
    # =========================================================
    invoice_section_match = re.search(
        r"OPEN\s+INVOICES"
        r"(.*?)"
        r"Aged\s+Summary",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not invoice_section_match:
        raise ValueError(
            "OPEN INVOICES section was not found."
        )

    invoice_section = invoice_section_match.group(1)

    invoice_pattern = re.compile(
        r"(?P<document_date>\d{2}/\d{2}/\d{4})\s+"
        r"(?P<document_number>\d+)\s+"
        r"(?P<description>.*?)\s+"
        r"(?P<due_date>\d{2}/\d{2}/\d{4})\s+"
        r"\$(?P<invoice_amount>[\d,]+\.\d{2})\s+"
        r"\$(?P<open_balance>[\d,]+\.\d{2})\s+"
        r"\$(?P<balance_due>[\d,]+\.\d{2})",
        flags=re.IGNORECASE,
    )

    for match in invoice_pattern.finditer(
        invoice_section
    ):
        lines.append(
            ParsedStatementLine(
                line_type="invoice",

                document_number=(
                    match.group("document_number") or ""
                ).strip(),

                document_date=_parse_mmddyyyy(
                    match.group("document_date")
                ),

                due_date=_parse_mmddyyyy(
                    match.group("due_date")
                ),

                description=(
                    match.group("description") or ""
                ).strip(),

                invoice_amount=_money_to_float(
                    match.group("invoice_amount")
                ),

                credit_amount=0.0,

                open_balance=_money_to_float(
                    match.group("open_balance")
                ),

                raw_text=match.group(0).strip(),
            )
        )

    credit_count = sum(
        1
        for line in lines
        if line.line_type in {
            "credit",
            "return",
            "payment",
        }
    )

    invoice_count = sum(
        1
        for line in lines
        if line.line_type == "invoice"
    )

    if credit_count == 0:
        raise ValueError(
            "No credit/payment lines could be parsed "
            "from the Marcone statement."
        )

    if invoice_count == 0:
        raise ValueError(
            "No open invoice lines could be parsed "
            "from the Marcone statement."
        )

    return ParsedStatement(
        supplier_name="Marcone",
        statement_period=statement_period,
        account_number=account_number,
        balance_due=balance_due,
        lines=lines,
    )


def extract_pdf_text(
    pdf_path: str | Path,
) -> str:
    """
    Extract text from a text-based PDF.

    Scanned/image-only PDFs are not handled here.
    """

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is required. "
            "Install it with: pip install pypdf"
        ) from exc

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Statement PDF was not found: {path}"
        )

    reader = PdfReader(str(path))

    pages_text: list[str] = []

    for page in reader.pages:
        page_text = page.extract_text() or ""

        if page_text.strip():
            pages_text.append(page_text)

    text = "\n".join(pages_text).strip()

    if not text:
        raise ValueError(
            "No text could be extracted from the PDF. "
            "The file may be a scanned/image-only PDF."
        )

    return text


def parse_statement_pdf(
    pdf_path: str | Path,
    supplier_name: str | None = None,
) -> ParsedStatement:
    """
    Main parser entry point.
    """

    text = extract_pdf_text(pdf_path)

    supplier = (
        supplier_name or ""
    ).strip().lower()

    if (
        supplier == "marcone"
        or "marcone" in text.lower()
    ):
        return parse_marcone_statement_text(text)

    raise ValueError(
        "Unsupported supplier statement format."
    )


def save_parsed_statement(
    parsed_statement,
    *,
    source_file: str | None = None,
    created_by: int | None = None,
):
    """
    Save the complete parsed supplier statement.

    Saves:
    - statement header;
    - OPEN CREDITS & PAYMENTS;
    - OPEN INVOICES.

    Does not:
    - modify inventory;
    - modify employee ledger;
    - create adjustments;
    - reconcile automatically.
    """

    from extensions import db
    from models import (
        SupplierStatement,
        SupplierStatementLine,
    )

    supplier_name = (
        getattr(
            parsed_statement,
            "supplier_name",
            None,
        )
        or ""
    ).strip()

    statement_period = getattr(
        parsed_statement,
        "statement_period",
        None,
    )

    account_number = (
        getattr(
            parsed_statement,
            "account_number",
            None,
        )
        or ""
    ).strip() or None

    balance_due = float(
        getattr(
            parsed_statement,
            "balance_due",
            0.0,
        )
        or 0.0
    )

    parsed_lines = list(
        getattr(
            parsed_statement,
            "lines",
            None,
        )
        or []
    )

    if not supplier_name:
        raise ValueError(
            "Supplier name is missing."
        )

    if statement_period is None:
        raise ValueError(
            "Statement period is missing."
        )

    if not parsed_lines:
        raise ValueError(
            "Statement does not contain parsed lines."
        )

    clean_source_file = None

    if source_file:
        clean_source_file = Path(
            source_file
        ).name

    try:
        statement = SupplierStatement(
            supplier_name=supplier_name,
            statement_period=statement_period,
            account_number=account_number,
            balance_due=balance_due,
            source_file=clean_source_file,
            status="draft",
            created_by=created_by,
        )

        db.session.add(statement)
        db.session.flush()

        for parsed_line in parsed_lines:
            document_number = (
                getattr(
                    parsed_line,
                    "document_number",
                    None,
                )
                or ""
            ).strip()

            if not document_number:
                raise ValueError(
                    "Parsed statement line has "
                    "no document number."
                )

            line_type = (
                getattr(
                    parsed_line,
                    "line_type",
                    None,
                )
                or ""
            ).strip().lower()

            if line_type not in {
                "credit",
                "return",
                "payment",
                "invoice",
            }:
                raise ValueError(
                    f"Unsupported statement line type: "
                    f"{line_type!r}."
                )

            description = (
                getattr(
                    parsed_line,
                    "description",
                    None,
                )
                or ""
            ).strip() or None

            raw_line_text = (
                getattr(
                    parsed_line,
                    "raw_text",
                    None,
                )
                or ""
            ).strip() or None

            invoice_amount = abs(
                float(
                    getattr(
                        parsed_line,
                        "invoice_amount",
                        0.0,
                    )
                    or 0.0
                )
            )

            credit_amount = abs(
                float(
                    getattr(
                        parsed_line,
                        "credit_amount",
                        0.0,
                    )
                    or 0.0
                )
            )

            open_balance = abs(
                float(
                    getattr(
                        parsed_line,
                        "open_balance",
                        0.0,
                    )
                    or 0.0
                )
            )

            statement_line = SupplierStatementLine(
                statement_id=statement.id,
                supplier_name=supplier_name,
                line_type=line_type,
                document_number=document_number,

                document_date=getattr(
                    parsed_line,
                    "document_date",
                    None,
                ),

                due_date=getattr(
                    parsed_line,
                    "due_date",
                    None,
                ),

                description=description,

                invoice_amount=invoice_amount,
                credit_amount=credit_amount,
                open_balance=open_balance,

                raw_text=raw_line_text,
            )

            db.session.add(statement_line)

        db.session.commit()

        return statement

    except Exception:
        db.session.rollback()
        raise