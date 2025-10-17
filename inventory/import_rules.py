from __future__ import annotations
import os, re, hashlib
import numpy as np
import pandas as pd
import logging
log = logging.getLogger(__name__)

_RELIABLE_SKIP = {
    "REMIT PAYMENT","BILL TO:","SHIP TO:","CUSTOMER NAME","CUSTOMER NUMBER",
    "ORDER LINE TOTAL","ORDER TOTAL","ORDERTOTAL","INVOICE TOTAL","INVOICETOTAL","TOTAL",
    "SUBTOTAL","SUB-TOTAL","SUB TOTAL",
    "SALES TAX","TAX","AMOUNT","AMOUNT DUE","AMT DUE","BALANCE","BALANCE DUE","CREDIT",
    "FREIGHT","SHIPPING","DELIVERY","HANDLING","FEE","SURCHARGE","FUEL SURCHARGE"
}


# ------------------------------- #
# Column name hints (case-tolerant)
# ------------------------------- #
COLUMN_SYNONYMS = {
    "part_number": [
        "PART #","PART","PART NO","SKU","ITEM","ITEM #","ITEM NO",
        "PN","PART NUMBER","MFR PART #",
        "PRODUCT"              # <- ОСТАЁТСЯ здесь, это PN
    ],
    "part_name":   [
        "DESCR.","DESCRIPTION","ITEM DESCRIPTION","DESC","NAME",
        "ITEM NAME","PART NAME"  # ВАЖНО: БЕЗ "PRODUCT"
    ],
    "quantity":    [
        "QTY","QUANTITY","QTY ORDERED","ORDERED QTY","Q-ty","ORDER QTY",
        "SHIPPED"             # Reliable: shipped = фактическое
    ],
    "unit_cost":   ["UNIT PRICE","UNITPRICE","PRICE","UNIT COST","COST","PRICE $","PRICE USD",
                    "UNIT $","UNIT NET","PRICE EA","PRICE EACH","EACH","EA"],
    "supplier":    ["SUPPLIER","VENDOR","SELLER"],
    "order_no":    ["ORDER #","ORDER","ORDER NO","WO","WORK ORDER","SO #","PO #","PO","SO"],
    "date":        ["DATE","ORDER DATE","INVOICE DATE","DOC DATE","BILL DATE"],
    "location":    ["LOCATION","SHELF","BIN","PLACE","LOC"],
}


# Optional: map supplier -> default location if vendor didn’t specify any
SUPPLIER_LOCATION_MAP = {
    # короткие коды, которых ты ожидаешь в базе/интерфейсе
    "reliable": "REL",
    "reliableparts": "REL",
    "reliable parts": "REL",
    "marcone": "MAR",
    "marcone supply": "MAR",
}

# Rows we never want as line items
_NOISE_PREFIXES = {
    # header/sections
    "REMIT PAYMENT", "BILL TO:", "SHIP TO:", "CUSTOMER NAME", "CUSTOMER NUMBER",
    # non-item labels
    "REF", "TRACK", "TRACKING", "BACKORDER", "SHIPPED PRODUCT", "INTERNAL", "INTERNAL -", "F/S",
    # totals/charges
    "ORDER LINE TOTAL", "ORDER TOTAL", "INVOICE TOTAL", "TOTAL",
    "SUBTOTAL", "SUB-TOTAL", "SUB TOTAL",
    "SALES TAX", "TAX", "AMOUNT", "AMOUNT DUE", "AMT DUE",
    "BALANCE", "BALANCE DUE", "CREDIT",
    "FREIGHT", "SHIPPING", "DELIVERY", "HANDLING", "FEE", "SURCHARGE", "FUEL SURCHARGE",
}

_RELIABLE_SKIP_WORDS = (
    "ORDER TOTAL", "INVOICE TOTAL", "SUBTOTAL",
    "FREIGHT", "SHIPPING", "SALES TAX", "AMOUNT DUE", "BALANCE",
    "REMIT PAYMENT", "BILL TO", "SHIP TO"
)

_MARCONE_SKIP_WORDS = (
    "BILL TO", "SHIP TO", "REMIT PAYMENT", "SUBTOTAL", "SALES TAX",
    "DELIVERY", "HANDLING", "C.O.D.", "INVOICE TOTAL", "AMOUNT DUE",
    "TRACKING", "REF:", "INTERNAL", "PAGE", "ORD PART", "NARDA #"
)

NUMBER_RE  = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
MONEY_RE   = re.compile(r"\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d{1,4})|[0-9]+(?:\.[0-9]{1,4}))")

_PN_RE = re.compile(r"\b[A-Z0-9]{3,}(?:[-/][A-Z0-9]+)*\b")
_MONEY2_RE = re.compile(r"\$?\s*\d+(?:,\d{3})*(?:\.\d{2})")

# strict money (has $ or .cc), used by the “last resort” extractor
MONEY_TOK_STRICT = re.compile(r"""
    ^\s*
    \$?\s*
    (?:\d{1,3}(?:,\d{3})*|\d+)
    \.\d{2}
    \s*(?:USD)?\s*$
""", re.I | re.X)

# At file top (near other regexes)
UNIT_TOKENS = r"(EA|EACH|PC|PCS|PK|PKG|BOX|BG|CS)"
QTY_NEAR_UNIT = re.compile(rf"(\d{{1,4}})\s*{UNIT_TOKENS}\b", re.I)
MONEY_ANY = re.compile(r"\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})")

def _looks_like_marcone(df: pd.DataFrame) -> bool:
    """Heuristic: does this table look like a Marcone invoice page?"""
    if df is None or df.empty:
        return False
    heads = [_normh(c) for c in df.columns]
    head_text = " ".join(heads)
    # Typical headers present in Marcone: ORD, SHIP, B/O, PART, DESCRIPTION, MSRP
    has_ord  = "ORD" in head_text or "ORDERED" in head_text
    has_ship = "SHIP" in head_text
    has_part = "PART" in head_text or "MFR PART" in head_text
    has_desc = "DESCR" in head_text or "DESCRIPTION" in head_text
    has_msrp = "MSRP" in head_text or "PRICE" in head_text
    # Or: many money-looking cells across columns
    moneyish = 0
    for c in df.columns:
        s = df[c].astype(str)
        moneyish += s.str.contains(r"\$?\d{1,3}(?:,\d{3})*\.\d{2}", regex=True, na=False).sum()
    many_money = moneyish >= max(6, len(df) // 2)
    return (has_ord and has_ship and (has_part or has_desc)) or (has_ship and has_msrp) or many_money

def _marcone_headerless_row_to_item(line: str, *, supplier_hint: str, default_location: str):
    """
    Headerless fallback for Marcone: collapse a row to one string and extract
    PN, DESCRIPTION, UNIT PRICE, SHIP.
    - Unit price: smallest monetary value on the line
    - Ship: last integer AFTER the last money; if not found, assume 1
    - Skip obvious service/footer lines and Ship==0
    """
    if not line:
        return None
    U = line.upper()
    if any(w in U for w in _MARCONE_SKIP_WORDS):
        return None

    # money tokens
    money_pat = re.compile(r"\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})")
    moneys = list(money_pat.finditer(line))
    if not moneys:
        return None

    # unit price: smallest numeric among all money tokens
    def _m2f(m): return float(m.group(0).replace("$","").replace(",",""))
    unit_cost = min(_m2f(m) for m in moneys if 0 < _m2f(m) < 10000)

    # part number: pick a plausible token
    pn = _pick_pn_from_tokens(line.split())
    if not pn:
        return None

    # Ship qty: last integer after the LAST money token; if nothing → default to 1
    tail = line[moneys[-1].end():]
    ints_tail = [int(x) for x in re.findall(r"\b\d{1,5}\b", tail)]
    ship = ints_tail[-1] if ints_tail else 1
    if ship <= 0:
        return None

    # description: take text up to the FIRST money token, cut PN out, cleanup
    left = line[:moneys[0].start()].strip()
    i = left.upper().find(pn.upper())
    raw_desc = (left[:i] + left[i+len(pn):]).strip(" -.,;") if i >= 0 else left
    desc = _cleanup_description(raw_desc, pn=pn)

    # minimal signal
    if len(re.sub(r"[^A-Za-z0-9]+", "", desc)) < 3:
        return None

    return {
        "part_number": pn,
        "part_name": desc,
        "quantity": int(ship),
        "unit_cost": unit_cost,
        "supplier": supplier_hint or "Marcone",
        "location": SUPPLIER_LOCATION_MAP.get("marcone", default_location),
        "order_no": "",
        "date": "",
    }

def _pick_unit_price(money_strs: list[str]) -> float | None:
    """Choose the most plausible *unit* price for a line.
       Heuristic: the smallest positive monetary value on the line."""
    vals = []
    for m in money_strs:
        try:
            vals.append(float(m.replace(",", "").replace("$", "")))
        except:
            pass
    return min(vals) if vals else None

def _token_is_numeric_pn(tok: str) -> bool:
    """Accept strictly-numeric part numbers (Reliable has many).
       Guardrails so we don't confuse with qty or amounts."""
    t = tok.strip().replace(" ", "")
    # pure digits with reasonable length
    if t.isdigit() and 5 <= len(t) <= 12:   # было 6 <= len(t) <= 12
        return True
    # digit groups with dashes like "30-3132-48"
    if re.fullmatch(r"\d{2,}(?:-\d{2,})+", t):
        return True
    return False

def _pick_pn_from_tokens(tokens: list[str]) -> str:
    """Pick first plausible PN; fall back to last if needed."""
    def _clean(t: str) -> str:
        return re.sub(r"[^A-Za-z0-9\-./]", "", t).upper()

    toks = [_clean(t) for t in tokens if t.strip()]
    # 1) alphanumeric with both letters and digits
    for t in toks:
        if _looks_good_pn(t):
            return t
    # 2) numeric-only PN or dashy numeric PN (e.g., 240343803, 30-3132-48)
    for t in toks:
        if _token_is_numeric_pn(t):
            return t
    # 3) try from the end
    for t in reversed(toks):
        if _looks_good_pn(t) or _token_is_numeric_pn(t):
            return t
    return ""

def _cleanup_description(desc: str, pn: str = "") -> str:
    """
    Clean vendor noise and fix glued words.
    Guarantees that a leading SIZE fraction like 1/2 (from text or PN) is preserved at the start.
    Keeps output UPPERCASE.
    """
    s0 = str(desc or "")

    # >>> FIX: убрать мусорные псевдостроки до любой другой обработки
    s0 = re.sub(r'\b(?:nan|none|null|n/?a)\b', ' ', s0, flags=re.I)
    # <<<

    # --- 1) Extract any fraction from text or PN (1/2", 1/2 in, 1/2)
    FRACTION_ANY = re.compile(r'\b(\d{1,3})\s*/\s*(\d{1,3})\s*(?:["”“″]|IN(?:CH(?:ES)?)?)?\b', re.I)
    m = FRACTION_ANY.search(s0) or FRACTION_ANY.search(str(pn or ""))
    frac = None
    if m:
        # normalized bare form, e.g. "1/2"
        frac = f"{int(m.group(1))}/{int(m.group(2))}"
        # remove ALL fraction tokens from the text; we’ll reinsert once at the end
        s0 = FRACTION_ANY.sub(" ", s0)

    # remove PN token itself if present (do not touch the fraction we already removed)
    if pn:
        s0 = re.sub(rf'\b{re.escape(str(pn))}\b', ' ', s0, flags=re.I)

    s = s0.upper()

    # --- 2) Remove money, units, qty noise (these often nuke the fraction, so we do it BEFORE reinserting frac)
    s = re.sub(r'\bUSD\b', ' ', s)
    s = re.sub(r'\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})\b', ' ', s)   # $131.83 / 131.83
    s = re.sub(r'\b\d+(?:\.\d{2})\b', ' ', s)                      # 17.02 etc.
    s = re.sub(r'\bQTY\b\s*\d{1,4}\b', ' ', s)
    s = re.sub(r'\b(?:EA|EACH|PC|PCS|PK|PKG|BOX|BG|CS)\b', ' ', s)
    s = re.sub(r'\b(?:GEN|WPL|WCI)\b(?:\s*\d+(?:\.\d{2})?)?', ' ', s)

    # --- 3) DO remove plain leading qty like "1 " or "7 X 2 ", but not fractions (we already pulled them out)
    s = re.sub(r'^\s*\d{1,4}(?:\s*X\s*\d{1,4})?\b\s*', ' ', s)

    # --- 4) Normalize MXF -> M X F (also handles M/X/F or glued MXF)
    s = re.sub(r'\bM\s*[\/\-]?\s*X\s*[\/\-]?\s*F\b', 'M X F', s)
    s = re.sub(r'\bMXF\b', 'M X F', s)

    # drop small flags like F/S and inch quotes right after a fraction (1/2")
    s = re.sub(r"\bF/?S\b", " ", s)             # drop F/S
    s = re.sub(r'(?<=\d/\d)"', "", s)           # 1/2" -> 1/2

    # --- 5) Unglue common tokens (extend list as needed)
    TOKENS = [
        "BLADE","EVAP","FAN","ASM","KNOB","THERMOSTAT","ICING","KIT","GROMMET",
        "ROUND","DUAL","RUN","CAP","CAPAC","CAPACITOR","TRAY","LID","LOCK",
        "VALVE","PROBE","OVEN","TEMP","SCREW","GAS","FLEX","SS","O-RING","OVN",
        "CRISPER","PAN","DRAWER","SHELF","RACK","BIN","DOOR","HANDLE",
    ]
    for w in TOKENS:
        # insert a space before token if it's glued to previous letters
        s = re.sub(rf'(?<=[A-Z]){w}(?=[A-Z])', f' {w}', s)

    # --- 6) Final tidy (keep '/', '+', '.', '-' so fractions survive)
    s = re.sub(r'[^\w/+.-]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip(' -,')

    # --- 7) Reinsert the fraction at the very start if we had it
    if frac:
        # if a fraction is already present (somehow), don't duplicate
        if not re.search(r'\b\d{1,3}/\d{1,3}\b', s):
            s = f"{frac} {s}"
        else:
            # ensure there's a space after it if it’s glued
            s = re.sub(r'^(\d{1,3}/\d{1,3})(?=\S)', r'\1 ', s)

    return s

def _row_to_text(row_vals) -> str:
    toks = [str(x) for x in row_vals if str(x).strip() and str(x).lower() != "nan"]
    return " ".join(toks).strip()

def _looks_good_pn(tok: str) -> bool:
    if not _PN_RE.fullmatch(tok):
        return False
    if tok.startswith(("REF", "TRACK", "TRACKING")):
        return False
    return any(ch.isalpha() for ch in tok) and any(ch.isdigit() for ch in tok)

def _parse_reliable_by_columns(df: pd.DataFrame, supplier_hint, default_location) -> pd.DataFrame:
    """
    Надёжный column-first для Reliable.
    - Поднимаем заголовки из первых 10 строк, если текущие бесполезные.
    - PN берём из колонки PRODUCT/PART#, но если там мусор — извлекаем PN из всей строки.
    - SHIPPED → QTY, UNIT PRICE → unit_cost.
    - Жёстко скипаем строки с TERMS/REMIT/TOTAL/… до разбора.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    def _is_noise_line(vals) -> bool:
        s = _row_to_text(vals).upper()
        return any(w in s for w in _RELIABLE_SKIP)

    df0 = df.copy()

    def _rebind_headers_if_needed(df_in: pd.DataFrame) -> pd.DataFrame:
        heads_now = " ".join([_normh(c) for c in df_in.columns])
        have_key_now = (("PRODUCT" in heads_now or "PART" in heads_now) and
                        ("DESCR"   in heads_now or "DESCRIPTION" in heads_now) and
                        ("PRICE"   in heads_now or "UNIT PRICE"  in heads_now) and
                        ("SHIPPED" in heads_now or "QTY" in heads_now or "QUANTITY" in heads_now))
        if have_key_now:
            return df_in
        limit = min(10, len(df_in))
        for r in range(limit):
            row = [_normh(x) for x in df_in.iloc[r].tolist()]
            row_text = " ".join(row)
            looks = (("PRODUCT" in row_text or "PART" in row_text) and
                     ("DESCR"   in row_text or "DESCRIPTION" in row_text) and
                     ("PRICE"   in row_text or "UNIT PRICE"  in row_text) and
                     ("SHIPPED" in row_text or "QTY" in row_text or "QUANTITY" in row_text))
            if looks:
                df2 = df_in.iloc[r+1:].copy()
                headers = []
                for j, val in enumerate(df_in.iloc[r].tolist()):
                    h = str(val).strip() if str(val).strip() else f"C{j}"
                    headers.append(h)
                df2.columns = headers
                return df2
        return df_in

    df = _rebind_headers_if_needed(df0)

    part_col  = _find_col(df, ["PRODUCT","PART #","PART NO","ITEM","ITEM #","MFR PART #","PRODUCT"])
    desc_col  = _find_col(df, ["DESCRIPTION","ITEM DESCRIPTION","DESCR.","DESC"])
    qty_col   = _find_col(df, ["SHIPPED","QTY","QUANTITY"])
    price_col = _find_col(df, ["PRICE","UNIT PRICE","UNIT COST","PRICE USD","UNIT $"])

    if not part_col and not desc_col:
        return pd.DataFrame()  # уйдём в line-fallback

    if not desc_col and part_col:
        try:
            pi = df.columns.get_loc(part_col)
            for j in range(pi + 1, min(pi + 4, df.shape[1])):
                h = _normh(df.columns[j])
                if "MAKE" in h or "NARDA" in h:
                    continue
                desc_col = df.columns[j]; break
        except Exception:
            pass

    pre_cols = []
    if part_col:
        try:
            pre_cols = list(df.columns[: df.columns.get_loc(part_col)])
        except Exception:
            pre_cols = []

    rows = []
    for i in range(len(df)):
        row_vals = df.iloc[i, :].tolist()
        if _is_noise_line(row_vals):
            continue

        line_text = _row_to_text(row_vals)

        # PN из колонки или из строки
        pn_cell = _norm(df.at[i, part_col]) if part_col and i in df.index else ""
        pn = pn_cell if (pn_cell and (_looks_good_pn(pn_cell) or _token_is_numeric_pn(pn_cell))) else _pick_pn_from_tokens(line_text.split())
        if not pn:
            continue

        # QTY (Ship)
        ship = _to_int(df.at[i, qty_col]) if qty_col and i in df.index else None
        if not ship or ship <= 0:
            ints = []
            for c in pre_cols:
                v = _to_int(df.at[i, c]) if i in df.index else None
                if v is not None:
                    ints.append(v)
            ship = ints[1] if len(ints) >= 2 else (ints[0] if ints else 0)
        if ship is None or ship <= 0:
            continue

        # UNIT COST
        unit_cost = _to_float(df.at[i, price_col]) if price_col else None
        if unit_cost is None or pd.isna(unit_cost):
            money = _row_money_values_strict(row_vals)
            unit_cost = min(money) if money else None

        # DESCRIPTION: из колонки → если пусто/‘nan’ → из всей строки до первой суммы
        raw_desc = _norm(df.at[i, desc_col]) if desc_col and i in df.index else ""
        if not raw_desc or raw_desc.lower() == "nan" or len(re.sub(r"[^A-Za-z0-9]+", "", raw_desc)) < 3:
            s = re.sub(rf"\b{re.escape(pn)}\b", " ", line_text, flags=re.I)
            mfirst = MONEY_ANY.search(s)
            if mfirst:
                s = s[:mfirst.start()]
            raw_desc = s

        desc = _cleanup_description(raw_desc, pn)
        if len(re.sub(r"[^A-Za-z0-9]+", "", desc)) < 3:
            continue

        rows.append({
            "part_number": pn,
            "part_name": desc,
            "quantity": int(ship),
            "unit_cost": unit_cost if unit_cost is not None else np.nan,
            "supplier": supplier_hint or "ReliableParts",
            "location": SUPPLIER_LOCATION_MAP.get("reliable", default_location),
            "order_no": "",
            "date": "",
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=[
        "part_number","part_name","quantity","unit_cost","supplier","location","order_no","date"
    ])


def _parse_reliable_wide(df: pd.DataFrame, supplier_hint, default_location) -> pd.DataFrame:
    """
    Reliable wide-table fallback:
    - собираем строку из всех ячеек
    - выкидываем служебные TOTAL/FREIGHT/...
    - PN ищем сначала по ячейкам (схлопывая мусор/пробелы), потом по всей строке (тоже схлопывая)
    - цену берём как минимальную реалистичную money-цифру в строке
    """
    # >>> NEW: try clean column-based parse first
    try:
        by_cols = _parse_reliable_by_columns(df, supplier_hint, default_location)
        if by_cols is not None and not by_cols.empty:
            return by_cols
    except Exception as e:
        try:
            log.warning("[IMPORT/Reliable] by_columns failed: %s", e)
        except Exception:
            pass

    rows = []

    for i in range(len(df)):
        row_vals = df.iloc[i, :].tolist()

        # текст строки
        text_raw = _row_to_text(row_vals)
        if not text_raw:
            continue
        up_raw = text_raw.upper()

        # отбрасываем явные служебные строки
        if any(w in up_raw for w in _RELIABLE_SKIP_WORDS):
            continue

        # ---------- PN detection ----------
        pn = None

        # (A) пробуем найти PN в отдельных ячейках, предварительно "схлопывая" мусор/пробелы
        for cell in row_vals:
            cs = str(cell)
            if not cs or cs.lower() == "nan":
                continue
            cand = re.sub(r"[^A-Za-z0-9/-]+", "", cs).upper()  # оставляем только буквы/цифры/-/
            if _looks_good_pn(cand):
                pn = cand
                break

        # (B) если не нашли — схлопываем ПРОБЕЛЫ во всей строке и ищем ещё раз
        if not pn:
            up_compact = re.sub(r"(?<=\w)\s+(?=\w)", "", up_raw)  # убираем пробелы между \w
            for cand in _PN_RE.findall(up_compact):
                cand = cand.upper()
                if _looks_good_pn(cand):
                    pn = cand
                    break

        # ---------- Price detection ----------
        money_vals = []
        for m in _MONEY2_RE.findall(text_raw):
            try:
                x = float(m.replace("$", "").replace(",", ""))
                if 0 < x < 10000:
                    money_vals.append(x)
            except:
                pass

        if not pn or not money_vals:
            # без PN или цены — не товар
            continue

        price = min(money_vals)

        # ---------- Qty detection (очень простая) ----------
        qty = 1
        # ищем небольшое целое рядом с EA/EACH; если нет — просто любое небольшое целое
        mqty = re.search(r"\b([1-9]\d{0,2})\b(?:\s*(?:EA|EACH))?", up_raw)
        if mqty:
            try:
                qi = int(mqty.group(1))
                if 1 <= qi <= 999:
                    qty = qi
            except:
                pass

        # ---------- Description ----------
        desc = text_raw
        desc = desc.replace(pn, " ")
        desc = _MONEY2_RE.sub(" ", desc)
        desc = re.sub(r"\b(?:EA|EACH)\b", " ", desc, flags=re.I)
        desc = re.sub(r"\s+", " ", desc).strip(" -,:")
        # ДОБАВЬ ЭТУ СТРОКУ:
        desc = _cleanup_description(desc, pn)
        if len(desc) < 3:
            continue

        rows.append({
            "part_number": pn,
            "part_name": desc,
            "quantity": qty,
            "unit_cost": price,
            "supplier": supplier_hint or "ReliableParts",
            "location": SUPPLIER_LOCATION_MAP.get("reliable", default_location),
            "order_no": "",
            "date": "",
        })

    return pd.DataFrame(rows)

def _parse_marcone_table_like(
    df: pd.DataFrame,
    *,
    supplier_hint: str,
    source_file: str,
    default_location: str
) -> pd.DataFrame:
    """
    Robust Marcone parser that does NOT rely on column headers.
    Row shape (typical, but tolerant):
        ORD  SHIP  [B/O]  PN  <DESCRIPTION ...>  $UNIT  [$MSRP ...]  ...  [B/O  SHIP]
    Rules:
      - QTY := Ship = last small integer RIGHT BEFORE PN; if not found, use last integer at line tail.
      - Skip if Ship==0.
      - PN may be alphanumeric or numeric-only (6–12 digits, or 2+ digit groups with dashes).
      - unit_cost = smallest monetary value on the row (proxy for unit).
      - Description = tokens after PN and before first money (if no money → after PN, trimming trailing ints).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    money_pat = re.compile(r"\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})")
    is_int    = re.compile(r"^\d+$")
    rows = []

    for i in range(len(df)):
        # 1) Collapse row to one line
        raw = _row_to_text(df.iloc[i, :].tolist())
        if not raw:
            continue
        U = raw.upper()
        if any(w in U for w in _MARCONE_SKIP_WORDS):
            continue

        # 2) Tokenize (keep ., /, - so PN survives)
        toks_raw = raw.split()
        toks = [re.sub(r"[^\w./\-]+", "", t) for t in toks_raw]
        toks = [t for t in toks if t]

        # 3) Find start run of integers (ORD/SHIP/[B/O] etc.)
        start_int_run_end = 0
        for t in toks:
            if is_int.match(t):
                start_int_run_end += 1
            else:
                break
        leading_ints = [int(t) for t in toks[:start_int_run_end] if is_int.match(t)]

        # 4) Find PN: earliest token after that run that "looks like PN"
        pn_idx = -1
        pn = ""
        for idx in range(start_int_run_end, len(toks)):
            t = toks[idx].upper()
            if (_PN_RE.fullmatch(t) and any(c.isalpha() for c in t) and any(c.isdigit() for c in t)) or _token_is_numeric_pn(t):
                pn = t
                pn_idx = idx
                break
        if pn_idx < 0:
            # fallback: scan all tokens
            for idx in range(len(toks)-1, -1, -1):
                t = toks[idx].upper()
                if (_PN_RE.fullmatch(t) and any(c.isalpha() for c in t) and any(c.isdigit() for c in t)) or _token_is_numeric_pn(t):
                    pn = t
                    pn_idx = idx
                    break
        if pn_idx < 0:
            continue  # no PN → not an item

        # 5) Ship = last integer RIGHT BEFORE PN (robust for "Ord Ship B/O")
        head_tokens = toks[:pn_idx]
        head_ints = [int(t) for t in head_tokens if is_int.match(t)]
        ship = head_ints[-1] if head_ints else None

        # If still None/0, try tail “… B/O  SHIP”
        if ship is None or ship <= 0:
            tail_ints = [int(t) for t in toks[-3:] if is_int.match(t)]  # usually "... 0 2"
            if tail_ints:
                ship = tail_ints[-1]
        if ship is None:
            ship = 0
        if ship <= 0:
            continue  # strictly skip not shipped

        # 6) unit_cost = smallest money on the row (unit tends to be the smallest)
        money_vals = [float(m.group(0).replace("$", "").replace(",", "")) for m in money_pat.finditer(raw)]
        unit_cost = min([x for x in money_vals if 0 < x < 10000], default=np.nan)

        # 7) Description = tokens after PN up to first money-token (if any)
        #    Build a string from PN onward, then trim at first money position if present
        after_pn_text = " ".join(toks[pn_idx+1:]).strip()
        # cut before first money, if any
        mfirst = money_pat.search(after_pn_text)
        if mfirst:
            after_pn_text = after_pn_text[:mfirst.start()].strip()
        # trim trailing integers (sometimes B/O/SHIP are at end)
        while after_pn_text and after_pn_text.split() and is_int.match(after_pn_text.split()[-1]):
            after_pn_text = " ".join(after_pn_text.split()[:-1]).strip()

        desc = _cleanup_description(after_pn_text, pn)

        # Minimal signal in desc
        if len(re.sub(r"[^A-Za-z0-9]+", "", desc)) < 3:
            continue

        rows.append({
            "part_number": pn,
            "part_name": desc,
            "quantity": int(ship),
            "unit_cost": unit_cost,
            "supplier": supplier_hint or "Marcone",
            "location": SUPPLIER_LOCATION_MAP.get("marcone", default_location),
            "order_no": "",
            "date": "",
        })

    cols = ["part_number","part_name","quantity","unit_cost","supplier","location","order_no","date"]
    return pd.DataFrame(rows, columns=cols)


def _parse_marcone_from_pdf_text(pdf_path: str, *, supplier_hint: str,
                                 default_location: str) -> pd.DataFrame:
    """
    Fallback when table extraction fails: parse page text lines.
    Matches lines like:
      '1 DA97-14474C ASSY TRAY ICE ... $131.83 ... $0.00  0  1'
                               ^unit                ^B/O  ^Ship
    """
    try:
        import pdfplumber
    except Exception:
        return pd.DataFrame()

    pat = re.compile(r"""
        ^\s*
        (?P<ord>\d+)\s+                              # ordered qty (ignored)
        (?P<pn>[A-Z0-9][A-Z0-9\-]+)\s+               # part number
        (?P<desc>.+?)                                # description (lazy)
        \s+\$?\d{1,3}(?:,\d{3})*\.\d{2}              # unit price
        .*?\$?\d{1,3}(?:,\d{3})*\.\d{2}              # total
        \s+(?P<bo>\d+)\s+(?P<ship>\d+)\s*$           # B/O and Ship at tail
    """, re.X)

    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.splitlines():
                s = _norm(line)
                if not s:
                    continue
                U = s.upper()
                if any(tag in U for tag in ("REF:", "TRACKING", "REMIT PAYMENT", "BILL TO", "SHIP TO", "INVOICE TOTAL", "SUBTOTAL")):
                    continue
                m = pat.match(s)
                if not m:
                    continue
                ship = int(m.group("ship"))
                if ship <= 0:
                    continue

                pn   = m.group("pn").upper()
                desc = _cleanup_description(m.group("desc"), pn)

                # unit price: grab the smallest money on the line to approximate unit
                money_vals = re.findall(r"\$?\d{1,3}(?:,\d{3})*\.\d{2}", s)
                unit_cost = np.nan
                if money_vals:
                    vs = [float(x.replace("$","").replace(",","")) for x in money_vals]
                    unit_cost = min(v for v in vs if 0 < v < 10000)

                rows.append({
                    "part_number": pn,
                    "part_name": desc,
                    "quantity": ship,
                    "unit_cost": unit_cost,
                    "supplier": supplier_hint or "Marcone",
                    "location": SUPPLIER_LOCATION_MAP.get("marcone", default_location),
                    "order_no": "",
                    "date": "",
                })

    return pd.DataFrame(rows, columns=["part_number","part_name","quantity","unit_cost","supplier","location","order_no","date"])



# ---------- small helpers ----------
def _norm(s: str) -> str:
    return str(s or "").replace("\xa0", " ").strip()

def _normh(s: str) -> str:
    # header-ish normalization
    return _norm(s).replace("$","USD").replace("#","NO").upper()

def _find_col(df: pd.DataFrame, variants) -> str | None:
    """Exact-ish header match, then contains fallback."""
    cols = {_normh(c): c for c in df.columns}
    for v in variants:
        key = _normh(v)
        if key in cols:
            return cols[key]
    for c in df.columns:
        if any(_normh(v) in _normh(c) for v in variants):
            return c
    return None

def _to_int(x) -> int | None:
    try:
        s = _norm(x)
        m = NUMBER_RE.search(s)
        if not m: return None
        return int(float(m.group(0).replace(",", ".")))
    except Exception:
        return None

def _to_float(x) -> float | None:
    try:
        s = _norm(x)
        m = NUMBER_RE.search(s)
        if not m: return None
        n = m.group(0).replace(",", ".")
        return float(n)
    except Exception:
        return None

def _row_money_values_strict(seq) -> list[float]:
    """Return only realistic unit-like amounts from a row:
       - must look like $xx.xx or xx.xx
       - 0 < x < 10_000
    """
    out = []
    for cell in seq:
        s = str(cell)
        if MONEY_TOK_STRICT.search(s):
            num = re.search(r"\d+(?:,\d{3})*(?:\.\d{2})", s)
            if num:
                x = float(num.group(0).replace(",", ""))
                if 0 < x < 10000:
                    out.append(x)
    return out

def _looks_like_part_number(s: str) -> bool:
    s = (s or "").strip()
    if not s or len(s) < 3 or len(s) > 32:
        return False
    up = s.upper()
    if up.startswith(("REF", "TRACK", "TRACKING")):
        return False
    has_digit = any(ch.isdigit() for ch in s)
    has_alpha = any(ch.isalpha() for ch in s)
    return has_alpha and has_digit

def _is_noise_row(pn: str, name: str) -> bool:
    pn = (pn or "").strip().upper()
    name = (name or "").strip().upper()
    if any(name.startswith(p) for p in _NOISE_PREFIXES): return True
    if any(pn.startswith(p) for p in _NOISE_PREFIXES):   return True
    return False

def _guess_unit_cost_col(df: pd.DataFrame) -> str | None:
    """Pick the column that *looks* like price (share of money-looking values)."""
    def _bad(name: str) -> bool:
        h = _normh(name)
        return any(tag in h for tag in ("TOTAL","MSRP","SUBTOTAL","TAX","DELIVERY","HANDLING","FEE","EXT","EXTENDED"))
    best_col, best_ratio = None, 0.0
    for c in df.columns:
        if _bad(str(c)): continue
        s = df[c].astype(str)
        hits = s.str.contains(r"\$?\s*\d+(?:,\d{3})*(?:\.\d{1,4})\b", regex=True, na=False)
        ratio = hits.mean() if len(s) else 0.0
        if ratio > best_ratio:
            best_col, best_ratio = c, ratio
    return best_col if best_ratio >= 0.25 else None

def _guess_qty_col(df: pd.DataFrame) -> str | None:
    """Pick the integer-like column (most values are small integers)."""
    best_col, best_ratio = None, 0.0
    for c in df.columns:
        s = df[c].astype(str)
        nums = s.apply(lambda x: NUMBER_RE.search(str(x)).group(0) if NUMBER_RE.search(str(x)) else "")
        ok = nums.apply(lambda x: x.isdigit() and int(x) <= 99999)
        ratio = ok.mean() if len(ok) else 0.0
        if ratio > best_ratio:
            best_col, best_ratio = c, ratio
    return best_col if best_ratio >= 0.25 else None

# -----------------------------
# Public: used by routes.py
# -----------------------------
def load_table(path_or_buf):
    """Read .xlsx/.xls or .csv into a DataFrame."""
    ext = os.path.splitext(str(path_or_buf))[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path_or_buf)
    if ext in (".csv", ""):
        return pd.read_csv(path_or_buf)
    # fallback to Excel
    return pd.read_excel(path_or_buf)

# ------------------------------------------
# Vendor fast path: Reliable Parts (1–2 cols)
# ------------------------------------------
_RELIABLE_SKIP = tuple(k for k in _NOISE_PREFIXES) + (
    "INVOICE", "PAGE", "TERMS", "ACCOUNT", "PLEASE REMIT", "PHONE", "FAX",
)

_UNIT_TOKENS = r"(EA|EACH|PC|PCS|PK|PKG|BOX|BG|CS)"

def _parse_marcone_table_like_headered(
    df: pd.DataFrame,
    *,
    supplier_hint: str,
    source_file: str,
    default_location: str
) -> pd.DataFrame:
    """
    Parse Marcone when pdf->table returned a wide DataFrame.
    - QTY comes from the 'Ship' column (from numeric cols BEFORE PART #)
    - Rows with Ship == 0 are skipped
    """

    if df is None or df.empty:
        return pd.DataFrame()

    # Ensure positional indexing works even if df has non-RangeIndex
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    df.reset_index(drop=True, inplace=True)

    part_col = _find_col(df, ["PART #", "PART", "PART NO", "MFR PART #"])
    desc_col = _find_col(df, ["DESCRIPTION", "DESCR.", "ITEM DESCRIPTION"])
    price_col = _guess_unit_cost_col(df)  # may be None

    if part_col is None or desc_col is None:
        return pd.DataFrame()

    # Columns before PART # (usually: Ord | Ship | B/O)
    part_idx = df.columns.get_loc(part_col)
    pre_cols = list(df.columns[:part_idx])

    rows = []
    for i in range(len(df)):
        # Safe cell access by position
        pn       = _norm(df.iloc[i, df.columns.get_loc(part_col)])
        desc_raw = _norm(df.iloc[i, df.columns.get_loc(desc_col)])

        up = desc_raw.upper()
        if up in {"F/S"} or up.startswith("REF:") or up.startswith("INTERNAL"):
            continue

        # Ship from preceding numeric columns (Ord, Ship, B/O)
        ints = []
        for c in pre_cols:
            try:
                v = _to_int(df.iloc[i, df.columns.get_loc(c)])
            except Exception:
                v = None
            if v is not None:
                ints.append(v)
        ship = ints[1] if len(ints) >= 2 else (ints[0] if len(ints) == 1 else 0)
        if ship <= 0:
            continue  # skip unshipped

        # Unit price
        unit_cost = None
        if price_col:
            unit_cost = _to_float(df.iloc[i, df.columns.get_loc(price_col)])
        if unit_cost is None:
            money = _row_money_values_strict(df.iloc[i, :].tolist())
            unit_cost = min(money) if money else None

        if not pn and not desc_raw:
            continue

        desc = _cleanup_description(desc_raw, pn)

        rows.append({
            "part_number": pn,
            "part_name": desc,
            "quantity": int(ship),
            "unit_cost": unit_cost,
            "supplier": supplier_hint or "Marcone",
            "location": SUPPLIER_LOCATION_MAP.get("marcone", default_location),
            "order_no": "",
            "date": "",
        })

    cols = ["part_number","part_name","quantity","unit_cost","supplier","location","order_no","date"]
    return pd.DataFrame(rows, columns=cols)

def _parse_marcone_from_pdf_text(
    pdf_path: str,
    *,
    supplier_hint: str,
    default_location: str
) -> pd.DataFrame:
    """
    Parse Marcone by reading raw PDF text lines:
    line looks like:
    '1 0 1 DA97-14474C ASSY TRAY ICE;FDR,T-TYPE $131.83 $200.38 $0.00'
      Ord Ship B/O  PART#    DESCRIPTION                Unit      MSRP     Total
    We take Ship as QTY; skip rows with Ship==0.
    """
    try:
        import pdfplumber
    except Exception:
        return pd.DataFrame()

    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            t = (p.extract_text() or "").splitlines()
            lines.extend([_norm(x) for x in t if _norm(x)])

    # найти начало таблицы
    started = False
    items = []
    money_pat = re.compile(r"\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})")

    for ln in lines:
        U = ln.upper()
        if not started:
            if U.startswith("ORD SHIP") or " ORD " in U and " SHIP " in U and " PART " in U:
                started = True
            continue

        # отсекаем служебные строки
        if U in {"F/S"} or U.startswith("REF:") or U.startswith("INTERNAL"):
            continue
        if U.startswith("INVOICE") or U.startswith("REMIT PAYMENT"):
            # на всякий случай — конец зоны позиций
            break

        # --- извлекаем три ведущих числа + PN + хвост
        # прим: Ord Ship B/O PN <desc...> $xx.xx [дальше суммы]
        m = re.match(r"^\s*(\d+)\s+(\d+)\s+\d+\s+([A-Z0-9\-]+)\s+(.*)$", U)
        if not m:
            # иногда B/O отсутствует → допускаем два числа
            m = re.match(r"^\s*(\d+)\s+(\d+)\s+([A-Z0-9\-]+)\s+(.*)$", U)
        if not m:
            continue

        ord_n, ship_n, pn, rest = m.groups()
        try:
            ship = int(ship_n)
        except:
            ship = 0
        if ship <= 0:
            continue  # строго пропускаем Ship==0

        # --- цена: первая денежная сумма в строке
        unit_cost = None
        mprice = money_pat.search(rest)
        if mprice:
            unit_cost = float(mprice.group(0).replace("$", "").replace(",", ""))
            desc_raw = rest[:mprice.start()].strip()
        else:
            desc_raw = rest

        # подчистка описания
        desc = _cleanup_description(desc_raw, pn)

        items.append({
            "part_number": pn,
            "part_name": desc,
            "quantity": ship,
            "unit_cost": unit_cost,
            "supplier": supplier_hint or "Marcone",
            "location": SUPPLIER_LOCATION_MAP.get("marcone", default_location),
            "order_no": "",
            "date": "",
        })

    cols = ["part_number","part_name","quantity","unit_cost","supplier","location","order_no","date"]
    return pd.DataFrame(items, columns=cols)


def _parse_reliable_lines(df: pd.DataFrame, *, supplier_hint: str,
                          source_file: str, default_location: str) -> pd.DataFrame:
    """
    Join each row into a single line and extract:
      PN, DESCRIPTION, QTY(shipped), UNIT PRICE
    Rules:
      - PN may be alphanumeric or numeric-only (6–12 digits, or digit-groups with dashes)
      - QTY = last integer right before unit token (EA/EACH/...)
      - unit_cost = smallest monetary value on the line
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Build candidate lines and skip obvious headers/footers
    lines: list[str] = []
    for _, row in df.iterrows():
        s = " ".join([_norm(v) for v in row.tolist() if _norm(v)]).strip()
        if not s:
            continue
        U = s.upper()
        if any(tag in U for tag in _RELIABLE_SKIP):
            continue
        if len(s) < 6:
            continue
        lines.append(s)

    items = []
    for ln in lines:
        # --- all monetary values on the line
        money_strs = MONEY_ANY.findall(ln)
        price = _pick_unit_price(money_strs)
        if price is None:
            continue

        # --- shipped qty = last "<int> <EA/EACH/...>" on the line
        qty = 1
        mqty = None
        for m in QTY_NEAR_UNIT.finditer(ln):
            mqty = m
        if mqty:
            try:
                qty = int(mqty.group(1))
            except:
                qty = 1

        # --- part number: search over whole line tokens
        tokens = ln.split()
        pn = _pick_pn_from_tokens(tokens)
        if not pn:
            continue

        # --- description: scrub PN/money/units/qty and tidy
        desc = ln
        # remove every occurrence of the PN
        desc = re.sub(rf"\b{re.escape(pn)}\b", " ", desc, flags=re.I)
        # remove money and unit markers
        desc = MONEY_ANY.sub(" ", desc)
        desc = re.sub(rf"\b{UNIT_TOKENS}\b", " ", desc, flags=re.I)
        # remove loose qtys
        desc = re.sub(r"\b\d{1,4}\b", " ", desc)
        # final cleanup via your normalizer (also de-glues tokens)
        desc = _cleanup_description(desc, pn)

        # keep only meaningful descriptions
        if len(re.sub(r"[^A-Za-z0-9]+", "", desc)) < 3:
            continue

        items.append({
            "part_number": pn,
            "part_name": desc,
            "quantity": qty,
            "unit_cost": price,
            "supplier": supplier_hint or "ReliableParts",
            "location": SUPPLIER_LOCATION_MAP.get("reliable", default_location),
            "order_no": "",
            "date": "",
        })

    cols = ["part_number","part_name","quantity","unit_cost","supplier","location","order_no","date"]
    return pd.DataFrame(items, columns=cols)

def _parse_marcone_lines(
    df: pd.DataFrame, *, supplier_hint: str, source_file: str, default_location: str
) -> pd.DataFrame:
    """
    Line-based Marcone parser with robust PN and QTY extraction.

    - Finds dashed PNs like DA82-01415A / DC61-04406A with optional NV2/NW2 prefix.
    - Allows punctuation/parentheses immediately after a PN (e.g., DA62-03441A(2)).
    - QTY: prefer last integer AFTER the last $; otherwise pick the last integer
      on the line that is not inside any PN span and is not a tiny packaging
      count directly after '('.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    # Normalize unicode hyphens to ASCII '-' to stabilize matching.
    HYPHENS = "\u2010\u2011\u2012\u2013\u2014\u2212\uFE58\uFE63\uFF0D"
    HNORM = re.compile(f"[{HYPHENS}]")

    # Money and integers.
    INT = re.compile(r"\b\d+\b")

    # PN detector:
    # - optional "N V/W 2" prefix with spaces allowed: NV2, NW2, N V2, N W2
    # - core dashed PN like DA82-01415A, DC61-04406A, DA97-12345B
    # - allow punctuation/parenthesis right after: use a lookahead for non-word/dash or end
    PN_ALL = re.compile(
        r"(?:\bN\s*[VW]\s*2\s*[,/|-]?\s*)?([A-Z]{2,4}\d{2,}-\d{3,}[A-Z0-9]*)"
        r"(?=(?:[^A-Za-z0-9-]|$))",
        re.I,
    )

    def _overlap(a, b):
        if not a or not b:
            return False
        return not (a[1] <= b[0] or b[1] <= a[0])

    items = []

    for _, row in df.iterrows():
        # Collapse visible cells into one line.
        parts = []
        for v in row.tolist():
            s = str(v).strip()
            if s and s.lower() != "nan":
                parts.append(s)
        line_raw = " ".join(parts)
        if not line_raw:
            continue

        line = HNORM.sub("-", line_raw)
        U = line.upper()

        if any(w in U for w in _MARCONE_SKIP_WORDS):
            continue

        # All money tokens on the line (we'll use them for unit price and tail).
        m_all = list(MONEY_ANY.finditer(line))
        if not m_all:
            continue
        last_money = m_all[-1]

        # ---- Part number(s) ----
        pn = ""
        pn_spans = []  # spans of all matched PNs to exclude their digits later

        cand = list(PN_ALL.finditer(line))
        if cand:
            # Pick the earliest PN on the line as the main PN.
            cand.sort(key=lambda m: m.start(1))
            pn = cand[0].group(1).upper()
            pn_spans = [m.span(1) for m in cand]  # keep all PN spans to ignore their digits
        else:
            # Fallback: token-based picker (previous behavior).
            toks = [t for t in re.split(r"\s+", line) if t]
            toks = [re.sub(r"[^A-Za-z0-9.\-]", "", t) for t in toks if t]
            pn = _pick_pn_from_tokens(toks) or ""
            if pn:
                m = re.search(rf"\b{re.escape(pn)}\b", line, flags=re.I)
                if m:
                    pn_spans = [m.span()]

        if not pn:
            continue

        # ---- Quantity (Ship) ----
        ints_all = list(INT.finditer(line))
        # Prefer tail ints after the last money.
        tail_ints = [m for m in ints_all if m.start() >= last_money.end()]

        def _is_packaging_small(m):
            # Treat "(...2)" right after a PN as packaging, not ship qty.
            # If the digit is preceded by '(' with no more than 1 whitespace, consider it packaging.
            start = m.start()
            return start > 0 and line[max(0, start - 2):start].strip().startswith("(") and int(m.group(0)) <= 9

        ship_val = None
        if tail_ints:
            # Use the last tail integer that is not inside a PN and not a packaging "(2)".
            for m in reversed(tail_ints):
                if any(_overlap(m.span(), ps) for ps in pn_spans):
                    continue
                if _is_packaging_small(m):
                    continue
                ship_val = int(m.group(0))
                break
        if ship_val is None:
            # Fallback: last integer on the line, excluding those inside PNs and packaging "(2)".
            for m in reversed(ints_all):
                if any(_overlap(m.span(), ps) for ps in pn_spans):
                    continue
                if _is_packaging_small(m):
                    continue
                ship_val = int(m.group(0))
                break

        if ship_val is None or ship_val <= 0:
            continue

        # ---- Unit price: pick the smallest money token (heuristic) ----
        unit_cost = _pick_unit_price([m.group(0) for m in m_all])

        # ---- Description: text before last money, cleaned ----
        left = line[:last_money.start()]
        left = re.sub(rf"\b{re.escape(pn)}\b", " ", left, flags=re.I)
        # Remove *other* PNs as well to avoid noise in description.
        for ps in pn_spans[1:]:
            other = line[ps[0]:ps[1]]
            left = left.replace(other, " ")
        left = re.sub(rf"\b{UNIT_TOKENS}\b", " ", left, flags=re.I)
        left = re.sub(r"\bQTY\b\s*\d{1,4}\b", " ", left, flags=re.I)
        left = re.sub(r"^\s*(\d+\s+){1,3}", " ", left)
        desc = _cleanup_description(left, pn)

        if len(re.sub(r"[^A-Za-z0-9]+", "", desc)) < 3:
            continue

        items.append({
            "part_number": pn,
            "part_name": desc,
            "quantity": int(ship_val),
            "unit_cost": unit_cost if unit_cost is not None else np.nan,
            "supplier": supplier_hint or "Marcone",
            "location": SUPPLIER_LOCATION_MAP.get("marcone", default_location),
            "order_no": "",
            "date": "",
        })

    cols = ["part_number","part_name","quantity","unit_cost","supplier","location","order_no","date"]
    return pd.DataFrame(items, columns=cols)


def _parse_reliable_from_pdf_text(pdf_path: str, *, supplier_hint: str,
                                  default_location: str) -> pd.DataFrame:
    """
    Если табличный парсинг провалился (остались только TOTAL/футеры),
    читаем весь текст страниц через pdfplumber и парсим строки тем же механизмом.
    """
    try:
        import pdfplumber  # должен быть, раз уже работает OCR-парсер
    except Exception:
        log.warning("[IMPORT] pdfplumber not available for text fallback")
        return pd.DataFrame()

    texts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text() or ""
                for line in t.splitlines():
                    s = _norm(line)
                    if not s:
                        continue
                    U = s.upper()
                    if any(tag in U for tag in _RELIABLE_SKIP):
                        continue
                    texts.append(s)
    except Exception as e:
        log.warning(f"[IMPORT] pdf text fallback failed: {e}")
        return pd.DataFrame()

    if not texts:
        return pd.DataFrame()

    tmp = pd.DataFrame({"col": texts})
    return _parse_reliable_lines(tmp, supplier_hint=supplier_hint,
                                 source_file=pdf_path, default_location=default_location)


# ------------------------------------------
# General normalizer (covers Marcone & others)
# ------------------------------------------
def normalize_table(df: pd.DataFrame, supplier_hint=None, *, source_file: str = "", default_location: str = "MAIN"):
    """
    Normalize vendor tables to a canonical schema and always return (out_df, issues).

    Fast paths:
      - Marcone: QTY must come from 'Ship' and rows with Ship==0 are skipped.
                 Order of attempts:
                   1) _parse_marcone_lines                (row-to-line text when headers are messy)
                   2) _parse_marcone_table_like_headered  (header-based table; fallback to old name)
                   3) _parse_marcone_from_pdf_text        (raw PDF text)
      - Reliable: wide-table or explicit hint  → if few/empty → line parser → raw PDF text
    """
    issues: list[str] = []
    final_cols = ["part_number","part_name","quantity","unit_cost","supplier","location","order_no","date"]

    # --- Fast path: MARCONE (QTY from Ship; Ship==0 -> skip)
    is_marcone = (
        (supplier_hint and "marcone" in str(supplier_hint).lower()) or
        ("marcone" in os.path.basename(str(source_file)).lower()) or
        _looks_like_marcone(df)
    )

    if df is not None and len(df) > 0 and is_marcone:
        # 1) Сначала строковый парсер (надежнее при "склеенных" ячейках)
        parsed = _parse_marcone_lines(
            df,
            supplier_hint=supplier_hint or "Marcone",
            source_file=source_file,
            default_location=default_location,
        )

        # 2) Если пусто — колоночный/header-based.
        #    Поддерживаем два имени: новое и старое.
        if parsed is None or parsed.empty:
            headered_parser = globals().get("_parse_marcone_table_like_headered") or globals().get("_parse_marcone_table_like")
            if headered_parser:
                parsed = headered_parser(
                    df,
                    supplier_hint=supplier_hint or "Marcone",
                    source_file=source_file,
                    default_location=default_location,
                )

        # 3) Если всё ещё пусто и это PDF — fallback по сырому тексту
        if (parsed is None or parsed.empty) and str(source_file).lower().endswith(".pdf"):
            parsed = _parse_marcone_from_pdf_text(
                source_file,
                supplier_hint=supplier_hint or "Marcone",
                default_location=default_location,
            )

        # Если что-то распарсили — завершаем рано
        if parsed is not None and not parsed.empty:
            # Safety: если вдруг пришёл только qty — приведём к quantity
            if "quantity" not in parsed.columns and "qty" in parsed.columns:
                parsed["quantity"] = pd.to_numeric(parsed["qty"], errors="coerce").fillna(0).astype(int)

            # Гарантируем наличие date
            if "date" not in parsed.columns:
                parsed["date"] = ""

            parsed["source_file"] = source_file

            # build stable row_key (vectorized)
            src    = os.path.basename(source_file)
            pn_u   = parsed["part_number"].astype(str).str.upper()
            name_u = parsed["part_name"].astype(str).str.upper()
            qty_s  = pd.to_numeric(parsed["quantity"], errors="coerce").fillna(0).astype(int).astype(str)
            cost_s = parsed["unit_cost"].apply(lambda v: f"{float(v):.4f}" if pd.notna(v) else "NA")
            loc_u  = parsed["location"].astype(str).str.upper()
            sup_u  = parsed["supplier"].astype(str).str.upper()

            payload = (src + "|" + pn_u + "|" + name_u + "|" + qty_s + "|" + cost_s + "|" + loc_u + "|" + sup_u)
            parsed["row_key"] = payload.apply(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest())

            parsed = parsed[["part_number","part_name","quantity","unit_cost","supplier","location","order_no","date","source_file","row_key"]]
            return parsed, issues

    # --- Fast path: RELIABLE (wide table or explicit hint)
    try:
        is_reliable = bool(supplier_hint and "reliable" in str(supplier_hint).lower())
        has_total_col = any("TOTAL" in _normh(c) for c in df.columns)
        part_col = _find_col(df, COLUMN_SYNONYMS["part_number"])
        looks_wide = (df is not None) and (len(df) > 0) and (df.shape[1] >= 10) and has_total_col and (part_col is None)
    except Exception:
        looks_wide = False
        is_reliable = False

    if df is not None and len(df) and (is_reliable or looks_wide):
        # 1) wide-парсер
        parsed = _parse_reliable_wide(df, supplier_hint, default_location)

        # 2) Если пусто ИЛИ подозрительно мало строк — безопасно пробуем строковый парсер
        if parsed is None or parsed.empty or len(parsed) < 3:
            try:
                parsed2 = _parse_reliable_lines(
                    df,
                    supplier_hint=supplier_hint or "ReliableParts",
                    source_file=source_file,
                    default_location=default_location,
                )
                if parsed2 is not None and not parsed2.empty:
                    parsed = parsed2
            except Exception:
                pass

        # 3) Если всё ещё пусто — для PDF читаем сырой текст
        if (parsed is None or parsed.empty) and source_file and str(source_file).lower().endswith(".pdf"):
            try:
                parsed_txt = _parse_reliable_from_pdf_text(
                    source_file,
                    supplier_hint=supplier_hint or "ReliableParts",
                    default_location=default_location,
                )
                if parsed_txt is not None and not parsed_txt.empty:
                    parsed = parsed_txt
            except Exception:
                pass

        if parsed is not None and not parsed.empty:
            # Гарантируем наличие date
            if "date" not in parsed.columns:
                parsed["date"] = ""

            parsed["source_file"] = source_file

            # build stable row_key (vectorized)
            src    = os.path.basename(source_file)
            pn_u   = parsed["part_number"].astype(str).str.upper()
            name_u = parsed["part_name"].astype(str).str.upper()
            qty_s  = pd.to_numeric(parsed["quantity"], errors="coerce").fillna(0).astype(int).astype(str)
            cost_s = parsed["unit_cost"].apply(lambda v: f"{float(v):.4f}" if pd.notna(v) else "NA")
            loc_u  = parsed["location"].astype(str).str.upper()
            sup_u  = parsed["supplier"].astype(str).str.upper()

            payload = (src + "|" + pn_u + "|" + name_u + "|" + qty_s + "|" + cost_s + "|" + loc_u + "|" + sup_u)
            parsed["row_key"] = payload.apply(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest())

            parsed = parsed[["part_number","part_name","quantity","unit_cost","supplier","location","order_no","date","source_file","row_key"]]
            return parsed, issues

    # --- Empty input guard
    if df is None or len(df) == 0:
        out = pd.DataFrame(columns=final_cols + ["source_file","row_key"])
        return out, issues

    # --- Generic path (other vendors)
    m = {k: _find_col(df, v) for k, v in COLUMN_SYNONYMS.items()}
    if not m.get("unit_cost"): m["unit_cost"] = _guess_unit_cost_col(df)
    if not m.get("quantity"):  m["quantity"]  = _guess_qty_col(df)

    n = len(df)
    out = pd.DataFrame(index=range(n))
    out["part_number"] = df[m["part_number"]] if m.get("part_number") else np.nan
    out["part_name"]   = df[m["part_name"]]   if m.get("part_name")   else np.nan
    out["quantity"]    = df[m["quantity"]]    if m.get("quantity")    else 1
    out["unit_cost"]   = df[m["unit_cost"]]   if m.get("unit_cost")   else np.nan
    out["supplier"]    = df[m["supplier"]]    if m.get("supplier")    else np.nan
    out["order_no"]    = df[m["order_no"]]    if m.get("order_no")    else np.nan
    out["date"]        = df[m["date"]]        if m.get("date")        else np.nan
    out["location_explicit"] = df[m["location"]] if m.get("location") else np.nan

    # string-ish NaN -> ""
    out[["part_number","part_name","supplier","order_no","date","location_explicit"]] = \
        out[["part_number","part_name","supplier","order_no","date","location_explicit"]].replace({np.nan: ""})

    # normalize values
    out["part_number"] = out["part_number"].astype(str).replace({"nan": ""}).str.replace("\xa0"," ", regex=False).str.strip()
    out["part_name"]   = out["part_name"].astype(str).replace({"nan": ""}).str.replace("\xa0"," ", regex=False).str.strip()
    out["supplier"]    = out["supplier"].astype(str).replace({"nan": ""}).str.strip()
    out["order_no"]    = out["order_no"].astype(str).replace({"nan": ""}).str.strip()
    out["date"]        = out["date"].astype(str).replace({"nan": ""}).str.strip()
    out["location_explicit"] = out["location_explicit"].astype(str).replace({"nan": ""}).str.strip()

    # qty / price coercion
    out["quantity"]  = out["quantity"].apply(_to_int).fillna(1).astype(int)
    out["unit_cost"] = out["unit_cost"].apply(_to_float)

    # row-level sanitation
    keep = []
    for i in out.index:
        pn   = (out.at[i, "part_number"] or "").strip()
        name = (out.at[i, "part_name"]   or "").strip()
        price = out.at[i, "unit_cost"]

        if _is_noise_row(pn, name):
            keep.append(False); continue

        letters_digits = re.sub(r"[^A-Za-z0-9]+", "", name)
        if pn == "" and (name == "" or len(letters_digits) < 3):
            keep.append(False); continue

        if pn == "" and name == "" and pd.to_numeric(pd.Series([price]), errors="coerce").notna().iloc[0]:
            keep.append(False); continue

        if (pd.isna(price) or price is None) and (_looks_like_part_number(pn) or name):
            row_vals = df.iloc[i, :].tolist() if i < len(df) else []
            money = _row_money_values_strict(row_vals)
            if money:
                out.at[i, "unit_cost"] = min(money)

        keep.append(True)

    out = out[pd.Series(keep, index=out.index)].reset_index(drop=True)

    # hard cut: "empty + price" and "empty"
    pn_clean = out["part_number"].astype(str).str.strip().replace({"nan": ""})
    nm_clean = out["part_name"].astype(str).str.strip().replace({"nan": ""})
    price_ok = pd.to_numeric(out["unit_cost"], errors="coerce").notna()
    mask_bad = (pn_clean.eq("") & nm_clean.eq("") & price_ok) | (pn_clean.eq("") & nm_clean.eq(""))
    out = out[~mask_bad].reset_index(drop=True)

    # validation messages
    for i, r in out.iterrows():
        if not r["part_number"]:
            issues.append(f"Row {i+1}: missing PART #")
        if not r["part_name"]:
            issues.append(f"Row {i+1}: missing DESCRIPTION")
        if (r["quantity"] or 0) < 1:
            issues.append(f"Row {i+1}: bad QTY {r['quantity']}")

    # supplier fallback from hint
    if supplier_hint:
        out["supplier"] = out["supplier"].apply(lambda s: s if str(s).strip() else supplier_hint)

    # LOCATION: explicit -> supplier map -> default
    loc_exp = out.get("location_explicit", pd.Series([""] * len(out))).astype(str).str.strip()
    sup_ser = out.get("supplier", pd.Series([""] * len(out))).astype(str).str.strip()
    sup_filled = sup_ser.where(sup_ser.ne(""), (supplier_hint or "")).fillna("")
    sup_key = sup_filled.str.replace(r"\s+", "", regex=True).str.lower()
    mapped = sup_key.map(SUPPLIER_LOCATION_MAP)
    out["location"] = np.where(loc_exp.ne(""), loc_exp, mapped.fillna(default_location))

    # ensure final column set
    for c in final_cols:
        if c not in out.columns:
            if c == "quantity": out[c] = 1
            elif c == "unit_cost": out[c] = np.nan
            else: out[c] = ""
    out = out[final_cols]
    out["source_file"] = source_file

    # stable row_key
    out.columns = out.columns.map(lambda x: str(x).strip())
    out = out.loc[:, ~out.columns.duplicated()].reset_index(drop=True)

    src    = os.path.basename(source_file)
    pn_u   = out["part_number"].astype(str).str.upper()
    name_u = out["part_name"].astype(str).str.upper()
    qty_s  = pd.to_numeric(out["quantity"], errors="coerce").fillna(0).astype(int).astype(str)
    cost_s = out["unit_cost"].apply(lambda v: f"{float(v):.4f}" if pd.notna(v) else "NA")
    loc_u  = out["location"].astype(str).str.upper()
    sup_u  = out["supplier"].astype(str).str.upper()

    payload = (src + "|" + pn_u + "|" + name_u + "|" + qty_s + "|" + cost_s + "|" + loc_u + "|" + sup_u)
    out["row_key"] = payload.apply(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest())

    return out, issues

# ------------------------------------------
# Movement builder (unchanged behavior)
# ------------------------------------------
def build_receive_movements(normalized: pd.DataFrame, *, duplicate_exists_func, make_movement_func):
    built, errors = [], []
    for i, r in normalized.iterrows():
        rk = str(r["row_key"])
        if duplicate_exists_func(rk):
            continue
        qty = int(r["quantity"] or 0)
        if qty <= 0:
            errors.append(f"Row {i+1}: non-positive qty")
            continue
        mov = {
            "movement_type": "RECEIVE",
            "row_key": rk,
            "part_number": r["part_number"],
            "part_name": r["part_name"],
            "qty": qty,
            "unit_cost": float(r["unit_cost"]) if pd.notna(r["unit_cost"]) else None,
            "location": r["location"],
            "supplier": r["supplier"],
            "source_file": r["source_file"],
            "ref_order_no": r["order_no"],
            "ref_date": r["date"],
        }
        try:
            make_movement_func(mov)
            built.append(mov)
        except Exception as e:
            errors.append(f"Row {i+1}: DB error: {e}")
    return built, errors


