"""
Tests for JSON generation (structure, validity, saving).

Run from the project folder with either:
    python -m unittest discover -s tests -v
    pytest tests
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import json_generator as jg  # noqa: E402

SAMPLE_FIELDS = {
    "invoice_number": "INV-1",
    "invoice_date": "01/01/2024",
    "due_date": "",
    "seller": {"name": "Seller Co", "address": "1 Main St", "email": "s@x.com"},
    "buyer": {"name": "Buyer", "address": "", "email": ""},
    "items": [{"description": "Widget", "quantity": 2, "amount": 20.0}],
    "financial": {"subtotal": 20.0, "tax": None, "total": 20.0, "currency": "$"},
}


class TestRecordStructure(unittest.TestCase):
    def test_record_has_required_keys_and_is_valid(self):
        rec = jg.build_invoice_record("inv_001", SAMPLE_FIELDS, "raw text")
        self.assertEqual(list(rec), list(jg.TOP_LEVEL_KEYS))
        self.assertEqual(jg.validate_record(rec), [])
        self.assertEqual(rec["invoice_id"], "inv_001")
        self.assertEqual(rec["ocr_text"], "raw text")

    def test_missing_values_use_empty_string_and_null(self):
        rec = jg.build_invoice_record("x", {}, "")
        self.assertEqual(rec["invoice_number"], "")
        self.assertEqual(rec["seller"], {"name": "", "address": "", "email": ""})
        self.assertEqual(rec["items"], [])
        self.assertEqual(rec["financial"], {"subtotal": None, "tax": None, "total": None, "currency": ""})
        self.assertEqual(jg.validate_record(rec), [])

    def test_none_input_is_handled(self):
        rec = jg.build_invoice_record("x", None, None)
        self.assertEqual(rec["ocr_text"], "")
        self.assertEqual(jg.validate_record(rec), [])

    def test_non_numeric_amounts_become_null(self):
        rec = jg.build_invoice_record("x", {"financial": {"total": "abc"},
                                            "items": [{"description": "a", "quantity": "?", "amount": None}]}, "")
        self.assertIsNone(rec["financial"]["total"])
        self.assertIsNone(rec["items"][0]["quantity"])

    def test_validator_detects_problems(self):
        rec = jg.build_invoice_record("x", SAMPLE_FIELDS, "t")
        del rec["buyer"]
        self.assertTrue(any("buyer" in p for p in jg.validate_record(rec)))
        rec = jg.build_invoice_record("x", SAMPLE_FIELDS, "t")
        rec["financial"]["total"] = "40"
        self.assertTrue(any("financial.total" in p for p in jg.validate_record(rec)))
        self.assertTrue(jg.validate_record("not a dict"))


class TestSaving(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_single_and_combined_files_are_valid_json(self):
        records = [jg.build_invoice_record(f"inv_{i}", SAMPLE_FIELDS, f"text {i}") for i in range(3)]
        for rec in records:
            path = jg.save_invoice_json(rec, self.tmp / "json")
            self.assertEqual(path.name, f"{rec['invoice_id']}.json")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), rec)
        combined = jg.save_all_invoices(records, self.tmp / "all_invoices.json")
        loaded = jg.load_json(combined)
        self.assertEqual(len(loaded), 3)
        self.assertEqual([r["invoice_id"] for r in loaded], ["inv_0", "inv_1", "inv_2"])

    def test_empty_combined_file_is_valid(self):
        path = jg.save_all_invoices([], self.tmp / "all.json")
        self.assertEqual(jg.load_json(path), [])

    def test_unicode_and_newlines_survive(self):
        text = "Total: €1.234,50\nKäse & Brot"
        rec = jg.build_invoice_record("u", {}, text)
        path = jg.save_invoice_json(rec, self.tmp)
        self.assertEqual(jg.load_json(path)["ocr_text"], text)

    def test_save_raw_text(self):
        path = jg.save_raw_text("inv_9", "hello\nworld", self.tmp / "raw")
        self.assertEqual(path.read_text(encoding="utf-8"), "hello\nworld")


if __name__ == "__main__":
    unittest.main()
