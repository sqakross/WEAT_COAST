from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class ParsedStatementLine:
    document_number: str
    document_date: object | None
    description: str
    credit_amount: float
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
        ($82.29) -> 82.29
        $1,234.56 -> 1234.56
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
    return datetime.strptime(value, "%m/%d/%Y").date()


def parse_marcone_statement_text(text: str) -> ParsedStatement:
    """
    Parse Marcone CUSTOMER STATEMENT text.

    V1 intentionally parses only OPEN CREDITS & PAYMENTS.

    It does NOT:
    - modify ledger;
    - create payments;
    - create returns;
    - touch inventory.

    It only extracts statement data for reconciliation.
    """

    raw_text = text or ""

    if "MARCONe".lower() not in raw_text.lower():
        raise ValueError("This does not appear to be a Marcone statement.")

    # ---------------------------------------------------------
    # Statement period
    # Example:
    # May 2026 CUSTOMER STATEMENT
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
        raise ValueError("Could not determine statement period.")

    month_name = period_match.group(1)
    year = int(period_match.group(2))

    statement_period = datetime.strptime(
        f"{month_name} {year}",
        "%B %Y",
    ).date().replace(day=1)

    # ---------------------------------------------------------
    # Account number
    # Marcone sample:
    # Account *To Be Applied...
    # 965767 ($3,353.42)
    # ---------------------------------------------------------
    account_number = None

    account_match = re.search(
        r"\bAccount\b.*?\n\s*(\d{4,})\b",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if account_match:
        account_number = account_match.group(1).strip()

    # ---------------------------------------------------------
    # Balance Due
    # Example:
    # Balance Due 06/20/2026
    # $12,387.49
    # ---------------------------------------------------------
    balance_due = 0.0

    balance_match = re.search(
        r"Balance\s+Due\s+\d{2}/\d{2}/\d{4}\s*"
        r"\$([\d,]+\.\d{2})",
        raw_text,
        flags=re.IGNORECASE,
    )

    if balance_match:
        balance_due = _money_to_float(balance_match.group(1))

    # ---------------------------------------------------------
    # Extract only OPEN CREDITS & PAYMENTS section
    # ---------------------------------------------------------
    section_match = re.search(
        r"OPEN\s+CREDITS\s*&\s*PAYMENTS"
        r"(.*?)"
        r"OPEN\s+INVOICES",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not section_match:
        raise ValueError(
            "OPEN CREDITS & PAYMENTS section was not found."
        )

    section = section_match.group(1)

    lines: list[ParsedStatementLine] = []

    # Typical Marcone line:
    #
    # 05/08/2026 73863995 RETURN 04302026
    # 05/08/2026 ($217.80) ($217.80) $0.00
    #
    # Or:
    #
    # 05/04/2026 73713681 LEONARD
    # 05/04/2026 ($82.29) ($82.29) $0.00
    #
    line_pattern = re.compile(
        r"(?P<document_date>\d{2}/\d{2}/\d{4})\s+"
        r"(?P<document_number>\d+)\s+"
        r"(?P<description>.*?)\s+"
        r"(?P<due_date>\d{2}/\d{2}/\d{4})\s+"
        r"\(\$(?P<credit_amount>[\d,]+\.\d{2})\)\s+"
        r"\(\$(?P<remaining_credit>[\d,]+\.\d{2})\)\s+"
        r"\$[\d,]+\.\d{2}",
        flags=re.IGNORECASE,
    )

    for match in line_pattern.finditer(section):
        document_number = (
            match.group("document_number") or ""
        ).strip()

        description = (
            match.group("description") or ""
        ).strip()

        credit_amount = _money_to_float(
            match.group("credit_amount")
        )

        document_date = _parse_mmddyyyy(
            match.group("document_date")
        )

        raw_line = match.group(0).strip()

        lines.append(
            ParsedStatementLine(
                document_number=document_number,
                document_date=document_date,
                description=description,
                credit_amount=credit_amount,
                raw_text=raw_line,
            )
        )

    if not lines:
        raise ValueError(
            "No credit/payment lines could be parsed "
            "from the Marcone statement."
        )

    return ParsedStatement(
        supplier_name="Marcone",
        statement_period=statement_period,
        account_number=account_number,
        balance_due=balance_due,
        lines=lines,
    )


def extract_pdf_text(pdf_path: str | Path) -> str:
    """
    Extract text from a text-based PDF.

    Uses pypdf because it is lightweight and sufficient
    for normal supplier statement PDFs.

    Scanned/image-only PDFs are intentionally not handled
    in V1.
    """

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is required. Install it with: pip install pypdf"
        ) from exc

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Statement PDF was not found: {path}"
        )

    reader = PdfReader(str(path))

    pages_text = []

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

    Later we can add other supplier parsers here without
    changing Accounting or reconciliation logic.
    """

    text = extract_pdf_text(pdf_path)

    supplier = (supplier_name or "").strip().lower()

    if supplier == "marcone" or "marcone" in text.lower():
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
    Сохраняет уже распарсенный supplier statement в БД.

    V1:
    - сохраняет header statement;
    - сохраняет OPEN CREDITS & PAYMENTS;
    - не изменяет Employee Ledger;
    - не создаёт adjustments;
    - не выполняет reconciliation автоматически.
    """
    from pathlib import Path

    from extensions import db
    from models import SupplierStatement, SupplierStatementLine

    supplier_name = (
        getattr(parsed_statement, "supplier_name", None) or ""
    ).strip()

    statement_period = getattr(
        parsed_statement,
        "statement_period",
        None,
    )

    account_number = (
        getattr(parsed_statement, "account_number", None) or ""
    ).strip() or None

    balance_due = float(
        getattr(parsed_statement, "balance_due", 0.0) or 0.0
    )

    parsed_lines = list(
        getattr(parsed_statement, "lines", None) or []
    )

    if not supplier_name:
        raise ValueError("Supplier name is missing.")

    if statement_period is None:
        raise ValueError("Statement period is missing.")

    if not parsed_lines:
        raise ValueError(
            "Statement does not contain parsed credit/payment lines."
        )

    clean_source_file = None

    if source_file:
        clean_source_file = Path(source_file).name

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
                getattr(parsed_line, "document_number", None) or ""
            ).strip()

            if not document_number:
                raise ValueError(
                    "Parsed statement line has no document number."
                )

            credit_amount = float(
                getattr(parsed_line, "credit_amount", 0.0) or 0.0
            )

            description = (
                getattr(parsed_line, "description", None) or ""
            ).strip() or None

            raw_text = (
                getattr(parsed_line, "raw_text", None) or ""
            ).strip() or None

            description_upper = (description or "").upper()

            if description_upper.startswith("RETURN"):
                line_type = "return"
            elif "PAYMENT" in description_upper:
                line_type = "payment"
            else:
                line_type = "credit"

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
                invoice_amount=0.0,
                credit_amount=abs(credit_amount),
                open_balance=0.0,
                raw_text=raw_text,
            )

            db.session.add(statement_line)

        db.session.commit()

        return statement

    except Exception:
        db.session.rollback()
        raise