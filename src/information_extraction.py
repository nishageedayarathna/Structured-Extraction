"""
Rule-based information extraction from raw OCR text.

Only regular expressions and keyword rules are used (no machine learning).
Missing text fields are returned as "" and missing numbers as None - nothing is invented.
"""
import re

# ----------------------------------------------------------------------------
# Regular expressions
# ----------------------------------------------------------------------------
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")

_MONTHS = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
           r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
DATE_RE = re.compile(
    r"(?<![\d/])(?:"
    r"\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}"                                   # 2024-03-25
    r"|\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}"                                # 25/03/2024, 03-25-24
    rf"|\d{{1,2}}(?:st|nd|rd|th)?[-\s]+{_MONTHS}\.?[-\s,]+\d{{4}}"
    rf"|{_MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}"            # March 25, 2024
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTHS}\.?[-\s,]+\d{{4}}"            # 25 March 2024
    r")(?![\d/])",
    re.IGNORECASE,
)

DUE_LABEL_RE = re.compile(
    r"(?:payment\s+)?due\s*(?:date|on)?|date\s+due|pay(?:able)?\s+by|pay\s+before", re.IGNORECASE)
INVOICE_DATE_LABEL_RE = re.compile(
    r"invoice\s*date|date\s+of\s+issue|issue\s*date|issued(?:\s+on)?|billing\s*date|"
    r"date\s+issued|\bdate\b", re.IGNORECASE)

INVOICE_NO_RES = [
    re.compile(r"(?:invoice|inv|bill|receipt)\s*(?:number|no|num|nr|id)\b\.?\s*[:#\-]?\s*(?P<v>[A-Za-z0-9][A-Za-z0-9\-/_.]*)", re.I),
    re.compile(r"(?:invoice|inv|bill|receipt)\s*#\s*[:\-]?\s*(?P<v>[A-Za-z0-9][A-Za-z0-9\-/_.]*)", re.I),
    re.compile(r"(?:invoice|inv)\s*[:\-]\s*(?P<v>[A-Za-z0-9][A-Za-z0-9\-/_.]*)", re.I),
    re.compile(r"\binvoice\s+(?P<v>[A-Za-z0-9\-/_]*\d[A-Za-z0-9\-/_]*)", re.I),
    re.compile(r"\b(?P<v>INV[\-_/]?\d[A-Za-z0-9\-/_]*)", re.I),
]

# Money: an amount always has exactly two decimals (1,234.50 or 12,50).
AMOUNT_RE = re.compile(r"\d[\d,]*\.\d{2}(?!\d)|\d+,\d{2}(?![\d,])")
_M = r"[$€£₹¥]?\s?\d[\d,]*[.,]\d{2}"
_Q = r"\d+(?:\.\d+)?"
_UNIT = r"(?:x|pcs?|units?|hrs?|hours?|days?)?"
ITEM_PATTERNS = [
    # OCR-damaged quantity token; quantity can be recovered from amount / unit price
    re.compile(rf"^\s*(?P<desc>[A-Za-z].*?)\s+(?P<qty_raw>[^\s]+)\s+(?P<price>{_M})\s+(?P<tax>{_M})\s+(?P<amount>{_M})\s*$", re.I),
    # description  qty  unit-price  tax  amount
    re.compile(rf"^\s*(?P<desc>[A-Za-z].*?)\s+(?P<qty>{_Q})\s+(?P<price>{_M})\s+(?P<tax>{_M})\s+(?P<amount>{_M})\s*$", re.I),
    # description  qty  unit-price  amount
    re.compile(rf"^\s*(?:\d{{1,3}}[.)]?\s+)?(?P<desc>[^\d$€£₹¥].*?)\s+(?P<qty>{_Q})\s*{_UNIT}\s+(?P<price>{_M})\s+(?P<amount>{_M})\s*$", re.I),
    # qty  description  unit-price  amount
    re.compile(rf"^\s*(?P<qty>{_Q})\s*x?\s+(?P<desc>[A-Za-z].*?)\s+(?P<price>{_M})\s+(?P<amount>{_M})\s*$", re.I),
    # description  amount
    re.compile(rf"^\s*(?:\d{{1,3}}[.)]?\s+)?(?P<desc>[A-Za-z].*?)\s+(?P<amount>{_M})\s*$", re.I),
]

CURRENCY_CODES = ["USD", "EUR", "GBP", "INR", "JPY", "CAD", "AUD", "CHF", "CNY", "LKR",
                  "SEK", "NOK", "DKK", "NZD", "SGD", "HKD", "ZAR", "AED", "MXN", "BRL"]
CURRENCY_CODE_RE = re.compile(r"\b(" + "|".join(CURRENCY_CODES) + r")\b")
CURRENCY_SYMBOL_RE = re.compile(r"[$€£₹¥]")

# ----------------------------------------------------------------------------
# Keyword rules
# ----------------------------------------------------------------------------
SUBTOTAL_RE = re.compile(r"\bsub\s*[-.]?\s*total\b", re.I)
TAX_RE = re.compile(r"\b(?:sales\s+tax|tax|vat|gst|hst|igst|cgst|sgst)\b", re.I)
TAX_EXCLUDE_RE = re.compile(r"\btax\s*(?:id|no|number|invoice|reg\w*|code)\b|\btaxable\b", re.I)
STRONG_TOTAL_RE = re.compile(
    r"\b(?:grand\s+total|total\s+due|amount\s+due|balance\s+due|total\s+amount|amount\s+payable|total\s+payable)\b", re.I)
WEAK_TOTAL_RE = re.compile(r"\btotal\b", re.I)

BUYER_LABEL_RES = [
    re.compile(r"\b(?:bill(?:ed)?\s*to|invoice\s*to|sold\s*to|bill\s+recipient)\b\s*(?:name)?\s*[:\-]?", re.I),
    re.compile(r"\b(?:customer|client|buyer|recipient)\b(?!\s*(?:id|no|number|#|code|ref|po)\b)\s*(?:name)?\s*[:\-]?", re.I),
    re.compile(r"\b(?:ship(?:ped)?\s*to|deliver\s*to)\b\s*[:\-]?", re.I),
]
SELLER_LABEL_RE = re.compile(
    r"^\W*(?:bill\s*from|from|seller|vendor|supplier|sold\s*by|issued\s*by|payee)\b\s*(?:name)?\s*[:\-]?", re.I)

TITLE_RE = re.compile(
    r"^\W*(?:tax\s+|sales\s+|commercial\s+|proforma\s+|final\s+)?(?:invoice|receipt|bill|statement)\W*$", re.I)
COMPANY_RE = re.compile(
    r"\b(?:inc|llc|ltd|limited|corp|corporation|co|company|gmbh|group|plc|pvt|llp|sons|associates|"
    r"enterprises|solutions|services|industries|holdings|partners|trading|studio|agency)\b\.?", re.I)

# Words that end a name/address block
STOP_RE = re.compile(
    r"\b(?:invoice|inv\s*(?:no|#)|date|due|bill\s*to|billed\s*to|ship\s*to|shipped\s*to|sold\s*to|from|"
    r"description|item|items|qty|quantity|unit|price|rate|amount|subtotal|sub-total|total|tax|vat|"
    r"terms|payment|email|e-mail|p\.?o\.?\s*(?:no|#|number)?|notes?)\b", re.I)
CUT_RE = re.compile(
    r"\b(?:ship(?:ped)?\s*to|deliver\s*to|invoice\s*(?:no|number|date|#)|date|due|from)\b", re.I)
PHONE_LINE_RE = re.compile(r"^\W*(?:tel|telephone|phone|ph|fax|mobile|mob|cell|contact)\b|^\W*\+?[\d\s\-().]{7,}$", re.I)
WEB_LINE_RE = re.compile(r"\b(?:www\.|https?://)|^\W*[\w\-]+\.(?:com|net|org|io|co)\b", re.I)

TABLE_DESC_RE = re.compile(r"\b(?:description|item|items|product|products|service|services|particulars|details)\b", re.I)
TABLE_NUM_RE = re.compile(r"\b(?:qty|quantity|price|amount|rate|total|unit|cost)\b", re.I)
TABLE_QTY_RE = re.compile(r"\b(?:qty|quantity)\b", re.I)
DISCOUNT_RE = re.compile(r"^\s*(?:discount|coupon|rebate)\b", re.I)
TABLE_END_RE = re.compile(
    r"\b(?:sub\s*[-.]?\s*total|total|grand\s+total|tax|vat|gst|balance|amount\s+due|thank\s*you|"
    r"payment\s+(?:terms|info|method)|notes?|terms)\b", re.I)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def normalize_lines(text):
    """Split OCR text into stripped, non-empty lines with single spaces."""
    lines = []
    for raw in (text or "").splitlines():
        line = re.sub(r"[ \t\u00a0]+", " ", raw).strip()
        if line:
            lines.append(line)
    return lines


def parse_amount(value):
    """'$1,234.50' -> 1234.5 ; '12,50' -> 12.5 ; anything else -> None."""
    if value is None:
        return None
    s = re.sub(r"[^\d.,]", "", str(value))
    if not s:
        return None
    if re.fullmatch(r"\d+,\d{2}", s):          # European decimal comma
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _num(value):
    """Return int for whole numbers, float otherwise."""
    return int(value) if float(value).is_integer() else float(value)


def _clean(s):
    return re.sub(r"\s+", " ", s or "").strip(" \t:;-|_,.")


def _is_noise(line):
    return sum(ch.isalnum() for ch in line) < 3


def _last_amount(line):
    """Last money amount on a line (percentages like '8.25%' are ignored)."""
    line = re.sub(r"\d+(?:[.,]\d+)?\s*%", " ", line)
    found = AMOUNT_RE.findall(line)
    return parse_amount(found[-1]) if found else None


def _amount_for_line(lines, i):
    """Amount on line i, or - if the line has none - on a purely numeric next line."""
    amount = _last_amount(lines[i])
    if amount is not None:
        return amount
    if i + 1 < len(lines) and not re.search(r"[A-Za-z]{2,}", lines[i + 1]):
        return _last_amount(lines[i + 1])
    return None


# ----------------------------------------------------------------------------
# Invoice number / dates
# ----------------------------------------------------------------------------
def extract_invoice_number(lines):
    for pattern in INVOICE_NO_RES:
        for line in lines:
            for m in pattern.finditer(line):
                value = m.group("v").strip(".-/_:")
                if any(ch.isdigit() for ch in value):
                    return value
    # Label on one line, value on the next ("Invoice No." / "12345")
    for i, line in enumerate(lines[:-1]):
        if re.search(r"(?:invoice|inv)\s*(?:number|no|num|#)\.?\s*[:\-]?\s*$", line, re.I):
            m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9\-/_.]*)", lines[i + 1])
            if m and any(ch.isdigit() for ch in m.group(1)):
                return m.group(1).strip(".-/_:")
    return ""


def _labeled_date(lines, label_re, exclude_spans_re=None):
    for i, line in enumerate(lines):
        for m in label_re.finditer(line):
            if exclude_spans_re is not None:
                if any(m.start() >= d.start() and m.end() <= d.end() for d in exclude_spans_re.finditer(line)):
                    continue                       # "Date" inside "Due Date" - not an invoice date label
            rest = line[m.end():]
            if exclude_spans_re is not None:       # stop at a following "Due Date" on the same line
                cut = exclude_spans_re.search(rest)
                if cut:
                    rest = rest[:cut.start()]
            d = DATE_RE.search(rest)
            if d:
                return d.group(0).strip()
            if not rest.strip(" :-") and i + 1 < len(lines):   # value on the next line
                d = DATE_RE.search(lines[i + 1])
                if d:
                    return d.group(0).strip()
    return ""


def extract_dates(lines):
    """Return (invoice_date, due_date)."""
    # Correct a frequent OCR confusion: "Jui" for "Jul" in date tokens.
    lines = [re.sub(r"(?i)(?<=-)Jui(?=-)", "Jul", line) for line in lines]
    due = _labeled_date(lines, DUE_LABEL_RE)
    invoice_date = _labeled_date(lines, INVOICE_DATE_LABEL_RE, exclude_spans_re=DUE_LABEL_RE)
    if not invoice_date:
        # Fallback: first date in the document that is not the due date.
        for line in lines:
            for d in DATE_RE.finditer(line):
                if d.group(0).strip() != due:
                    return d.group(0).strip(), due
    return invoice_date, due


# ----------------------------------------------------------------------------
# Seller / buyer
# ----------------------------------------------------------------------------
def _find_table_header(lines):
    for i, line in enumerate(lines):
        if (TABLE_DESC_RE.search(line) and TABLE_NUM_RE.search(line)) or \
           (TABLE_QTY_RE.search(line) and TABLE_NUM_RE.search(line)):
            return i
    return None


def _find_buyer_label(lines):
    for pattern in BUYER_LABEL_RES:
        for i, line in enumerate(lines):
            m = pattern.search(line)
            if m:
                return i, m.end(), line
    return None


def _find_seller_label(lines):
    for i, line in enumerate(lines):
        m = SELLER_LABEL_RE.match(line)
        if m:
            return i, m.end(), line
    return None


def _party_from_block(lines, start_idx, remainder, max_lines=4, max_scan=7):
    """Collect name + address lines after a label. Returns (name, address)."""
    block = []
    rem = EMAIL_RE.sub("", remainder or "")
    cut = CUT_RE.search(rem)
    if cut:
        rem = rem[:cut.start()]
    rem = _clean(rem)
    if rem and not _is_noise(rem):
        block.append(rem)
    i, scanned = start_idx + 1, 0
    while i < len(lines) and len(block) < max_lines and scanned < max_scan:
        line = lines[i]
        i += 1
        scanned += 1
        if STOP_RE.search(line):
            break
        cleaned = _clean(EMAIL_RE.sub("", line))
        if "@" in cleaned or not cleaned or _is_noise(cleaned) \
                or PHONE_LINE_RE.match(cleaned) or WEB_LINE_RE.search(cleaned):
            continue                      # skip e-mail (even OCR-garbled), phone, web and noise lines
        block.append(cleaned)
    name = block[0] if block else ""
    address = ", ".join(block[1:])
    return name, address


def _seller_from_header(lines, limit):
    """Fallback when there is no 'From' label: the top of the invoice is usually the seller."""
    cand = [(i, l) for i, l in enumerate(lines[:limit])
            if not TITLE_RE.match(l) and not STOP_RE.search(l)
            and "@" not in l and not PHONE_LINE_RE.match(l)
            and not WEB_LINE_RE.search(l) and not _is_noise(l)]
    if not cand:
        return "", ""
    name_pos = 0
    for pos, (_, l) in enumerate(cand[:6]):
        if COMPANY_RE.search(l):
            name_pos = pos
            break
    else:
        # Some invoice templates place the seller address before the seller name.
        # For "Address: street", "City, ST ZIP", "Company", prefer the third line.
        if re.match(r"^address\s*:", cand[0][1], re.I):
            if len(cand) > 2:
                name_pos = 2
            elif len(cand) > 1:
                name_pos = 1
    name = _clean(cand[name_pos][1])
    if name_pos == 2 and re.match(r"^address\s*:", cand[0][1], re.I):
        address = ", ".join(_clean(l) for _, l in cand[:2])
        address = re.sub(r"^Address\s*:\s*", "", address, flags=re.I)
    else:
        address = ", ".join(_clean(l) for _, l in cand[name_pos + 1:name_pos + 4])
    return name, address


def extract_parties(lines):
    """Return (seller, buyer) dicts with name / address / email."""
    seller = {"name": "", "address": "", "email": ""}
    buyer = {"name": "", "address": "", "email": ""}

    table_idx = _find_table_header(lines)
    buyer_hit = _find_buyer_label(lines)
    seller_hit = _find_seller_label(lines)

    if buyer_hit:
        i, end, line = buyer_hit
        buyer["name"], buyer["address"] = _party_from_block(lines, i, line[end:])
    if seller_hit:
        i, end, line = seller_hit
        seller["name"], seller["address"] = _party_from_block(lines, i, line[end:])
    else:
        limit_candidates = [x for x in (buyer_hit[0] if buyer_hit else None, table_idx) if x is not None]
        limit = min(limit_candidates) if limit_candidates else min(len(lines), 10)
        seller["name"], seller["address"] = _seller_from_header(lines, max(limit, 1))

    # ---- e-mail addresses -------------------------------------------------
    emails = [(i, e) for i, line in enumerate(lines) for e in EMAIL_RE.findall(line)]
    if buyer_hit is None:
        # No buyer section: first e-mail -> seller, second distinct e-mail -> buyer
        distinct = []
        for _, e in emails:
            if e not in distinct:
                distinct.append(e)
        if distinct:
            seller["email"] = distinct[0]
        if len(distinct) > 1:
            buyer["email"] = distinct[1]
    else:
        starts = {"seller": seller_hit[0] if seller_hit else 0, "buyer": buyer_hit[0]}
        for i, e in emails:
            if table_idx is not None and i >= table_idx:
                owner = "seller"                                   # footer e-mail
            else:
                owner = max((p for p in starts if starts[p] <= i), key=lambda p: (starts[p], p == "seller"),
                            default="seller")
            target = seller if owner == "seller" else buyer
            if not target["email"] and e != (buyer if owner == "seller" else seller)["email"]:
                target["email"] = e
    return seller, buyer


# ----------------------------------------------------------------------------
# Items
# ----------------------------------------------------------------------------
def parse_item_line(line):
    """Parse one table row into {description, quantity, amount} or return None."""
    if DISCOUNT_RE.match(line):
        return None
    for pattern in ITEM_PATTERNS:
        m = pattern.match(line)
        if not m:
            continue
        desc = _clean(m.group("desc"))
        amount = parse_amount(m.group("amount"))
        if not desc or amount is None or _is_noise(desc) or TABLE_END_RE.match(desc):
            continue
        groups = m.groupdict()
        qty = groups.get("qty")
        if qty is None and groups.get("qty_raw"):
            qty = None
        quantity = None
        if qty:
            try:
                quantity = _num(float(qty))
            except (TypeError, ValueError):
                quantity = None

        # If OCR damaged the quantity (e.g. "§.00"), recover it only when
        # amount / unit price is a whole number and agrees with the row amount.
        if quantity is None and groups.get("price"):
            unit_price = parse_amount(groups.get("price"))
            if unit_price not in (None, 0):
                inferred = amount / unit_price
                nearest = round(inferred)
                if nearest >= 1 and abs(inferred - nearest) <= 0.015:
                    quantity = int(nearest)

        return {"description": desc, "quantity": quantity, "amount": amount}
    return None


def extract_items(lines):
    header = _find_table_header(lines)
    if header is None:
        return []
    items = []
    for line in lines[header + 1:]:
        if TABLE_END_RE.match(line) or SUBTOTAL_RE.search(line) or STRONG_TOTAL_RE.search(line):
            break
        item = parse_item_line(line)
        if item:
            items.append(item)
    return items


# ----------------------------------------------------------------------------
# Financial summary
# ----------------------------------------------------------------------------
def extract_currency(text):
    m = CURRENCY_CODE_RE.search(text or "")
    if m:
        return m.group(1)
    m = CURRENCY_SYMBOL_RE.search(text or "")
    return m.group(0) if m else ""


def extract_financial(lines, text):
    subtotal = tax = None
    strong_total = weak_total = None
    for i, line in enumerate(lines):
        if SUBTOTAL_RE.search(line):
            value = _amount_for_line(lines, i)
            if value is not None:
                subtotal = value
        elif TAX_RE.search(line) and not TAX_EXCLUDE_RE.search(line):
            value = _amount_for_line(lines, i)
            if value is not None:
                tax = value
        elif STRONG_TOTAL_RE.search(line):
            value = _amount_for_line(lines, i)
            if value is not None:
                strong_total = value
        elif WEAK_TOTAL_RE.search(line):
            value = _amount_for_line(lines, i)
            if value is not None:
                weak_total = value
    total = strong_total if strong_total is not None else weak_total
    return {"subtotal": subtotal, "tax": tax, "total": total, "currency": extract_currency(text)}


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------
def extract_invoice_fields(text):
    """Turn raw OCR text into a dictionary of invoice fields."""
    lines = normalize_lines(text)
    invoice_date, due_date = extract_dates(lines)
    seller, buyer = extract_parties(lines)
    return {
        "invoice_number": extract_invoice_number(lines),
        "invoice_date": invoice_date,
        "due_date": due_date,
        "seller": seller,
        "buyer": buyer,
        "items": extract_items(lines),
        "financial": extract_financial(lines, text or ""),
    }
