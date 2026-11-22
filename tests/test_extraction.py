"""
Tests for preprocessing, OCR integration, extraction rules and the full pipeline.

Run from the project folder with either:
    python -m unittest discover -s tests -v
    pytest tests

NOTE: the image tests use SYNTHETIC invoices drawn with Pillow (created on the fly), because the
real FATURA images are not shipped with the project. They prove the code works end to end; they do
not measure accuracy on real FATURA invoices.
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import main as main_module  # noqa: E402
from src import image_processing as ip  # noqa: E402
from src import information_extraction as ie  # noqa: E402
from src import ocr_engine as ocr  # noqa: E402

try:
    ocr.configure_tesseract(config.TESSERACT_CMD)
    TESSERACT_OK = True
except ocr.TesseractNotFoundError:
    TESSERACT_OK = False


# ----------------------------------------------------------------------------
# Synthetic invoice helper
# ----------------------------------------------------------------------------
def _font(size):
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_invoice_image(path, lines, width=1240, height=1000):
    """Draw simple invoice text lines on a white page and save it."""
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = _font(30)
    y = 40
    for line in lines:
        draw.text((60, y), line, fill="black", font=font)
        y += 48
    img.save(path)


INVOICE_A = [
    "Acme Widgets Ltd",
    "12 Market Street, Springfield",
    "billing@acmewidgets.com",
    "INVOICE",
    "Invoice No: INV-2024-0042",
    "Invoice Date: 15/03/2024",
    "Due Date: 14/04/2024",
    "Bill To:",
    "Jane Roberts",
    "88 Lake Road, Portland",
    "jane.roberts@example.org",
    "Description Qty Price Amount",
    "Blue Widget 2 10.00 20.00",
    "Red Gadget 3 5.50 16.50",
    "Subtotal $36.50",
    "Tax (10%) $3.65",
    "Total $40.15",
]

INVOICE_B = [
    "Delta Consulting GmbH",
    "Hauptstrasse 5, Berlin",
    "info@delta-consulting.de",
    "Invoice #: 7788",
    "Date: March 3, 2024",
    "Bill To: Orion Foods Inc",
    "9 Harbor Way, Boston",
    "Item Quantity Rate Total",
    "Consulting service 10 50.00 500.00",
    "Subtotal USD 500.00",
    "VAT 8.25% 41.25",
    "Grand Total USD 541.25",
]


# ----------------------------------------------------------------------------
# Image preprocessing
# ----------------------------------------------------------------------------
class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.color = rng.integers(0, 255, (300, 200, 3), dtype=np.uint8)

    def test_grayscale(self):
        gray = ip.to_grayscale(self.color)
        self.assertEqual(gray.ndim, 2)
        self.assertEqual(gray.shape, (300, 200))

    def test_resize_keeps_aspect_ratio(self):
        out = ip.resize_image(ip.to_grayscale(self.color), 400)
        self.assertEqual(out.shape, (600, 400))

    def test_threshold_is_binary(self):
        binary = ip.apply_threshold(ip.to_grayscale(self.color), "otsu")
        self.assertTrue(set(np.unique(binary)).issubset({0, 255}))
        binary = ip.apply_threshold(ip.to_grayscale(self.color), "adaptive")
        self.assertTrue(set(np.unique(binary)).issubset({0, 255}))

    def test_full_preprocess(self):
        steps = ip.preprocess_steps(self.color, target_width=400)
        self.assertEqual(set(steps), {"gray", "resized", "denoised", "binary"})
        self.assertEqual(steps["binary"].shape, (600, 400))

    def test_noise_kernel_even_value_is_fixed(self):
        out = ip.remove_noise(ip.to_grayscale(self.color), 4)
        self.assertEqual(out.shape, (300, 200))


class TestImageLoading(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_loads_jpg_jpeg_png(self):
        for name in ("a.jpg", "b.jpeg", "c.png"):
            Image.new("RGB", (50, 40), "white").save(self.tmp / name)
        (self.tmp / "notes.txt").write_text("not an image")
        paths = ip.find_images(self.tmp, num_images=10)
        self.assertEqual([p.name for p in paths], ["a.jpg", "b.jpeg", "c.png"])
        for p in paths:
            self.assertIsNotNone(ip.load_image(p))

    def test_limits_number_of_images(self):
        for i in range(15):
            Image.new("RGB", (20, 20), "white").save(self.tmp / f"img{i:02d}.png")
        self.assertEqual(len(ip.find_images(self.tmp, 10)), 10)
        self.assertEqual(len(ip.find_images(self.tmp, 10, mode="spread")), 10)

    def test_unreadable_image_returns_none(self):
        bad = self.tmp / "broken.jpg"
        bad.write_bytes(b"this is not a jpeg")
        self.assertIsNone(ip.load_image(bad))
        self.assertIsNone(ip.load_image(self.tmp / "does_not_exist.jpg"))
        empty = self.tmp / "empty.png"
        empty.write_bytes(b"")
        self.assertIsNone(ip.load_image(empty))

    def test_missing_folder_raises_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            ip.find_images(self.tmp / "nope")


# ----------------------------------------------------------------------------
# Extraction rules (plain text - no OCR needed)
# ----------------------------------------------------------------------------
class TestExtractionRules(unittest.TestCase):
    def test_parse_amount(self):
        self.assertEqual(ie.parse_amount("$1,234.50"), 1234.5)
        self.assertEqual(ie.parse_amount("12,50"), 12.5)
        self.assertEqual(ie.parse_amount("EUR 99.00"), 99.0)
        self.assertIsNone(ie.parse_amount("abc"))
        self.assertIsNone(ie.parse_amount(None))

    def test_layout_a(self):
        f = ie.extract_invoice_fields("\n".join(INVOICE_A))
        self.assertEqual(f["invoice_number"], "INV-2024-0042")
        self.assertEqual(f["invoice_date"], "15/03/2024")
        self.assertEqual(f["due_date"], "14/04/2024")
        self.assertEqual(f["seller"]["name"], "Acme Widgets Ltd")
        self.assertIn("12 Market Street", f["seller"]["address"])
        self.assertEqual(f["seller"]["email"], "billing@acmewidgets.com")
        self.assertEqual(f["buyer"]["name"], "Jane Roberts")
        self.assertIn("88 Lake Road", f["buyer"]["address"])
        self.assertEqual(f["buyer"]["email"], "jane.roberts@example.org")
        self.assertEqual(f["items"], [
            {"description": "Blue Widget", "quantity": 2, "amount": 20.0},
            {"description": "Red Gadget", "quantity": 3, "amount": 16.5},
        ])
        self.assertEqual(f["financial"], {"subtotal": 36.5, "tax": 3.65, "total": 40.15, "currency": "$"})

    def test_layout_b(self):
        f = ie.extract_invoice_fields("\n".join(INVOICE_B))
        self.assertEqual(f["invoice_number"], "7788")
        self.assertEqual(f["invoice_date"], "March 3, 2024")
        self.assertEqual(f["due_date"], "")
        self.assertEqual(f["seller"]["name"], "Delta Consulting GmbH")
        self.assertEqual(f["seller"]["email"], "info@delta-consulting.de")
        self.assertEqual(f["buyer"]["name"], "Orion Foods Inc")
        self.assertEqual(f["buyer"]["email"], "")
        self.assertEqual(f["items"][0]["description"], "Consulting service")
        self.assertEqual(f["items"][0]["quantity"], 10)
        self.assertEqual(f["financial"]["tax"], 41.25)          # 8.25% must not be read as the tax
        self.assertEqual(f["financial"]["total"], 541.25)       # Grand Total wins over Subtotal
        self.assertEqual(f["financial"]["currency"], "USD")

    def test_dates_on_one_line(self):
        inv, due = ie.extract_dates(["Invoice Date: 01/02/2024 Due Date: 15/02/2024"])
        self.assertEqual((inv, due), ("01/02/2024", "15/02/2024"))

    def test_date_formats(self):
        for text in ("2024-03-25", "25.03.2024", "25 March 2024", "Mar 25, 2024", "25th Mar 2024"):
            inv, _ = ie.extract_dates([f"Date: {text}"])
            self.assertEqual(inv, text)

    def test_missing_values_are_not_invented(self):
        f = ie.extract_invoice_fields("hello world\nnothing useful here")
        self.assertEqual(f["invoice_number"], "")
        self.assertEqual(f["invoice_date"], "")
        self.assertEqual(f["due_date"], "")
        self.assertEqual(f["buyer"], {"name": "", "address": "", "email": ""})
        self.assertEqual(f["seller"]["email"], "")
        self.assertEqual(f["items"], [])
        self.assertEqual(f["financial"], {"subtotal": None, "tax": None, "total": None, "currency": ""})

    def test_empty_text(self):
        f = ie.extract_invoice_fields("")
        self.assertEqual(f["items"], [])
        self.assertIsNone(f["financial"]["total"])

    def test_item_line_variants(self):
        self.assertEqual(ie.parse_item_line("Widget 2 10.00 20.00"),
                         {"description": "Widget", "quantity": 2, "amount": 20.0})
        self.assertEqual(ie.parse_item_line("1 Widget 2 $10.00 $20.00"),
                         {"description": "Widget", "quantity": 2, "amount": 20.0})
        self.assertEqual(ie.parse_item_line("3 Gadget 4.00 12.00"),
                         {"description": "Gadget", "quantity": 3, "amount": 12.0})
        self.assertEqual(ie.parse_item_line("Delivery fee 15.00"),
                         {"description": "Delivery fee", "quantity": None, "amount": 15.0})
        self.assertIsNone(ie.parse_item_line("Thank you for your business"))

    def test_garbled_email_does_not_leak_into_address(self):
        text = "Northwind Traders Pvt Ltd\n45 Galle Road, Colombo\naccounts@northwind.l|k\nInvoice No: 5"
        f = ie.extract_invoice_fields(text)
        self.assertEqual(f["seller"]["name"], "Northwind Traders Pvt Ltd")
        self.assertNotIn("@", f["seller"]["address"])
        self.assertEqual(f["seller"]["email"], "")           # an unreadable e-mail is left empty

    def test_total_on_next_line(self):
        f = ie.extract_invoice_fields("Total\n$ 99.00")
        self.assertEqual(f["financial"]["total"], 99.0)


# ----------------------------------------------------------------------------
# Tesseract + full pipeline (needs a working Tesseract installation)
# ----------------------------------------------------------------------------
@unittest.skipUnless(TESSERACT_OK, "Tesseract is not installed / not found")
class TestTesseractAndPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.images = self.tmp / "images"
        self.images.mkdir()
        make_invoice_image(self.images / "invoice_a.png", INVOICE_A)
        make_invoice_image(self.images / "invoice_b.jpg", INVOICE_B)
        (self.images / "broken.jpg").write_bytes(b"corrupt")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_tesseract_runs(self):
        self.assertTrue(ocr.get_tesseract_version())

    def test_ocr_reads_synthetic_invoice(self):
        img = ip.load_image(self.images / "invoice_a.png")
        text = ocr.extract_text(ip.preprocess_image(img), config.TESSERACT_LANG, config.TESSERACT_CONFIG)
        self.assertIn("INV-2024-0042", text.replace(" ", ""))
        self.assertIn("Acme", text)

    def test_main_pipeline_end_to_end(self):
        out = self.tmp / "out"
        records = main_module.run(
            images_dir=self.images, num_images=10,
            json_dir=out / "json", raw_text_dir=out / "raw_text",
            all_file=out / "all_invoices.json", print_text=False)
        self.assertEqual(len(records), 2)                       # the corrupt image is skipped
        self.assertEqual(len(list((out / "json").glob("*.json"))), 2)
        self.assertEqual(len(list((out / "raw_text").glob("*.txt"))), 2)
        combined = json.loads((out / "all_invoices.json").read_text(encoding="utf-8"))
        self.assertEqual(len(combined), 2)
        by_id = {r["invoice_id"]: r for r in combined}
        self.assertEqual(by_id["invoice_a"]["financial"]["total"], 40.15)
        self.assertEqual(by_id["invoice_a"]["seller"]["email"], "billing@acmewidgets.com")


if __name__ == "__main__":
    unittest.main()
