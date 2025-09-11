# import_rules.py (minimal)
# Maps columns, validates, assigns location by supplier, builds RECEIVE movements with row_key

import os, hashlib
import pandas as pd

COLUMN_SYNONYMS = {
    "part_number": ["PART #","PART","PART NO","SKU","ITEM","ITEM #","ITEM NO","PN","PART NUMBER"],
    "part_name":   ["DESCR.","DESCRIPTION","ITEM DESCRIPTION","DESC","NAME","PRODUCT"],
    "quantity":    ["QTY","QUANTITY","QTY ORDERED","ORDERED QTY","Q-ty"],
    "unit_cost":   ["PRICE","UNIT PRICE","COST","UNIT COST","PRICE $","PRICE USD"],
    "supplier":    ["SUPPLIER","VENDOR","SELLER"],
    "order_no":    ["ORDER #","ORDER","ORDER NO","WO","WORK ORDER"],
    "date":        ["DATE","ORDER DATE","INVOICE DATE"],
    "location":    ["LOCATION","SHELF","BIN","PLACE"],
}

SUPPLIER_LOCATION_MAP = {
    "reliable": "RELIABLE",
    "reliableparts": "RELIABLE",
    "marcone": "MARCONE",
    "marcone supply": "MARCONE",
}

def _norm(s): return str(s or "").replace("\xa0"," ").strip()
def _normh(s): return _norm(s).replace("$","USD").replace("#","NO").upper()

def _find_col(df, variants):
    cols = {_normh(c): c for c in df.columns}
    for v in variants:
        key = _normh(v)
        if key in cols: return cols[key]
    # contains-fallback
    for c in df.columns:
        if any(_normh(v) in _normh(c) for v in variants): return c
    return None

def _to_int(x):
    try:
        s = _norm(x)
        if s == "": return None
        return int(float(s.replace(",","")))
    except: return None

def _to_float(x):
    try:
        s = _norm(x).replace("$","").replace(",","")
        if s == "": return None
        return float(s)
    except: return None

def load_table(path_or_buf):
    ext = os.path.splitext(str(path_or_buf))[1].lower()
    if ext in [".xlsx",".xls"]: return pd.read_excel(path_or_buf)
    if ext in [".csv",""]:      return pd.read_csv(path_or_buf)
    return pd.read_excel(path_or_buf)

def normalize_table(df, supplier_hint=None, *, source_file="", default_location="MAIN"):
    issues = []
    m = {k:_find_col(df, v) for k,v in COLUMN_SYNONYMS.items()}
    out = pd.DataFrame()
    out["part_number"] = df[m["part_number"]] if m["part_number"] else ""
    out["part_name"]   = df[m["part_name"]]   if m["part_name"]   else ""
    out["quantity"]    = df[m["quantity"]]    if m["quantity"]    else 1
    out["unit_cost"]   = df[m["unit_cost"]]   if m["unit_cost"]   else None
    out["supplier"]    = df[m["supplier"]]    if m["supplier"]    else ""
    out["order_no"]    = df[m["order_no"]]    if m["order_no"]    else ""
    out["date"]        = df[m["date"]]        if m["date"]        else ""
    out["location_explicit"] = df[m["location"]] if m["location"] else ""

    out["part_number"] = out["part_number"].astype(str).str.strip()
    out["part_name"]   = out["part_name"].astype(str).str.strip()
    out["quantity"]    = out["quantity"].apply(_to_int).fillna(1)
    out["unit_cost"]   = out["unit_cost"].apply(_to_float)
    out["supplier"]    = out["supplier"].astype(str).str.strip()
    out["order_no"]    = out["order_no"].astype(str).str.strip()
    out["date"]        = out["date"].astype(str).str.strip()
    out["location_explicit"] = out["location_explicit"].astype(str).str.strip()

    # validate
    for i, r in out.iterrows():
        if not r["part_number"]: issues.append(f"Row {i+1}: missing PART #")
        if not r["part_name"]:   issues.append(f"Row {i+1}: missing DESCR.")
        if (r["quantity"] or 0) < 1: issues.append(f"Row {i+1}: bad QTY {r['quantity']}")

    # supplier→location
    locs = []
    sups = []
    for i, r in out.iterrows():
        sup = r["supplier"] or (supplier_hint or "")
        sups.append(sup)
        loc = r["location_explicit"]
        if not loc:
            key = _norm(sup).lower().replace(" ", "")
            loc = SUPPLIER_LOCATION_MAP.get(key, default_location)
        locs.append(loc)
    out["supplier"] = sups
    out["location"] = locs

    out = out[["part_number","part_name","quantity","unit_cost","supplier","location","order_no","date"]]
    out["source_file"] = source_file
    # stable idempotent key
    def rowkey(r):
        payload = "|".join([os.path.basename(source_file), r.part_number.upper(), r.part_name.upper(),
                            str(int(r.quantity or 0)), f"{r.unit_cost:.4f}" if r.unit_cost is not None else "NA",
                            r.location.upper(), r.supplier.upper()])
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()
    out["row_key"] = out.apply(rowkey, axis=1)
    # drop empty
    out = out[~((out["part_number"]=="") & (out["part_name"]==""))].reset_index(drop=True)
    return out, issues

def build_receive_movements(normalized, *, duplicate_exists_func, make_movement_func):
    built, errors = [], []
    for i, r in normalized.iterrows():
        rk = str(r["row_key"])
        if duplicate_exists_func(rk):  # skip duplicates
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
            "unit_cost": float(r["unit_cost"]) if r["unit_cost"] is not None else None,
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
