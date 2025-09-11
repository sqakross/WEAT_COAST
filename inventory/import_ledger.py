# import_ledger.py (minimal JSON ledger for dedupe without touching your DB)
import os, json, time

DEFAULT_LEDGER_PATH = os.environ.get("WCCR_IMPORT_LEDGER", os.path.join("instance","import_ledger.json"))

def _load(path=DEFAULT_LEDGER_PATH):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        return {"applied": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"applied": {}}

def _save(data, path=DEFAULT_LEDGER_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def has_key(row_key, path=DEFAULT_LEDGER_PATH):
    data = _load(path)
    return row_key in data.get("applied", {})

def add_key(row_key, meta=None, path=DEFAULT_LEDGER_PATH):
    data = _load(path)
    data.setdefault("applied", {})[row_key] = meta or {"ts": time.time()}
    _save(data, path)
