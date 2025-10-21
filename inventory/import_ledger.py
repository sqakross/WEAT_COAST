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

# --- Adapter API for dedup cleanup ------------------------------------------

def iter_keys(path=DEFAULT_LEDGER_PATH):
    """
    Yield (key, meta) for all applied entries.
    """
    data = _load(path)
    for k, meta in (data.get("applied") or {}).items():
        yield k, (meta or {})

def del_key(row_key, path=DEFAULT_LEDGER_PATH):
    """
    Remove a single key from the ledger.
    """
    data = _load(path)
    applied = data.get("applied", {})
    if row_key in applied:
        applied.pop(row_key, None)
        _save(data, path)
        return True
    return False

def del_keys_by_batch(batch_id: int, path=DEFAULT_LEDGER_PATH):
    """
    Remove all keys where meta.batch_id == batch_id (string/int match).
    """
    data = _load(path)
    applied = data.get("applied", {})
    to_del = []
    for k, meta in list(applied.items()):
        meta = meta or {}
        if str(meta.get("batch_id", "")) == str(batch_id):
            to_del.append(k)
    for k in to_del:
        applied.pop(k, None)
    if to_del:
        _save(data, path)
    return len(to_del)
