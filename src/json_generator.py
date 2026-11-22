"""Build, validate and save the structured JSON output."""
import json
from pathlib import Path

PARTY_KEYS = ("name", "address", "email")
ITEM_KEYS = ("description", "quantity", "amount")
FINANCIAL_KEYS = ("subtotal", "tax", "total", "currency")
TOP_LEVEL_KEYS = ("invoice_id", "invoice_number", "invoice_date", "due_date",
                  "seller", "buyer", "items", "financial", "ocr_text")


def _text(value):
    return "" if value is None else str(value)


def _number(value):
    """Return an int/float, or None when the value is missing or not numeric."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(value) if isinstance(value, int) else number


def build_invoice_record(invoice_id, fields, ocr_text):
    """Create one JSON-ready record with a consistent structure.

    Missing text -> "" and missing numbers -> None, so every record has the same keys.
    """
    fields = fields or {}
    seller = fields.get("seller") or {}
    buyer = fields.get("buyer") or {}
    financial = fields.get("financial") or {}
    items = [
        {
            "description": _text(it.get("description")),
            "quantity": _number(it.get("quantity")),
            "amount": _number(it.get("amount")),
        }
        for it in (fields.get("items") or [])
    ]
    return {
        "invoice_id": _text(invoice_id),
        "invoice_number": _text(fields.get("invoice_number")),
        "invoice_date": _text(fields.get("invoice_date")),
        "due_date": _text(fields.get("due_date")),
        "seller": {k: _text(seller.get(k)) for k in PARTY_KEYS},
        "buyer": {k: _text(buyer.get(k)) for k in PARTY_KEYS},
        "items": items,
        "financial": {
            "subtotal": _number(financial.get("subtotal")),
            "tax": _number(financial.get("tax")),
            "total": _number(financial.get("total")),
            "currency": _text(financial.get("currency")),
        },
        "ocr_text": _text(ocr_text),
    }


def validate_record(record):
    """Return a list of problems (an empty list means the record is valid)."""
    problems = []
    if not isinstance(record, dict):
        return ["record is not a dictionary"]
    for key in TOP_LEVEL_KEYS:
        if key not in record:
            problems.append(f"missing key: {key}")
    if problems:
        return problems
    for key in ("invoice_id", "invoice_number", "invoice_date", "due_date", "ocr_text"):
        if not isinstance(record[key], str):
            problems.append(f"{key} must be a string")
    for party in ("seller", "buyer"):
        if not isinstance(record[party], dict) or set(record[party]) != set(PARTY_KEYS):
            problems.append(f"{party} must contain exactly {PARTY_KEYS}")
        elif not all(isinstance(v, str) for v in record[party].values()):
            problems.append(f"{party} values must be strings")
    if not isinstance(record["items"], list):
        problems.append("items must be a list")
    else:
        for n, item in enumerate(record["items"]):
            if not isinstance(item, dict) or set(item) != set(ITEM_KEYS):
                problems.append(f"items[{n}] must contain exactly {ITEM_KEYS}")
                continue
            if not isinstance(item["description"], str):
                problems.append(f"items[{n}].description must be a string")
            for k in ("quantity", "amount"):
                if item[k] is not None and not isinstance(item[k], (int, float)):
                    problems.append(f"items[{n}].{k} must be a number or null")
    fin = record["financial"]
    if not isinstance(fin, dict) or set(fin) != set(FINANCIAL_KEYS):
        problems.append(f"financial must contain exactly {FINANCIAL_KEYS}")
    else:
        for k in ("subtotal", "tax", "total"):
            if fin[k] is not None and not isinstance(fin[k], (int, float)):
                problems.append(f"financial.{k} must be a number or null")
        if not isinstance(fin["currency"], str):
            problems.append("financial.currency must be a string")
    return problems


def _write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    # Re-read to prove the file is valid, readable JSON.
    with open(path, "r", encoding="utf-8") as f:
        json.load(f)
    return path


def save_invoice_json(record, json_dir):
    """Save one invoice as <json_dir>/<invoice_id>.json."""
    return _write_json(record, Path(json_dir) / f"{record['invoice_id']}.json")


def save_all_invoices(records, output_file):
    """Save every record in one combined JSON file (a list of records)."""
    return _write_json(list(records), output_file)


def save_raw_text(invoice_id, text, raw_text_dir):
    """Save the raw OCR text as <raw_text_dir>/<invoice_id>.txt."""
    path = Path(raw_text_dir) / f"{invoice_id}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")
    return path


def load_json(path):
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
