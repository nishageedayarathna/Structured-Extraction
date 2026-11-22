"""
STRUCTURED EXTRACTION
Invoice Images -> Image Preprocessing -> OCR -> Rule-Based Information Extraction -> Structured JSON

Run with:  python main.py
Optional:  python main.py --images-dir "D:\\FATURA\\images" --num-images 10
"""
import argparse
import logging
import sys
from pathlib import Path

import config
from src import image_processing as ip
from src import information_extraction as ie
from src import json_generator as jg
from src import ocr_engine as ocr

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def run(images_dir=None, num_images=None, tesseract_cmd=None, json_dir=None,
        raw_text_dir=None, all_file=None, print_text=None):
    """Run the whole pipeline and return the list of generated records."""
    images_dir = Path(images_dir or config.IMAGES_DIR)
    num_images = num_images or config.NUM_IMAGES
    json_dir = Path(json_dir or config.JSON_DIR)
    raw_text_dir = Path(raw_text_dir or config.RAW_TEXT_DIR)
    all_file = Path(all_file or config.ALL_INVOICES_FILE)
    print_text = config.PRINT_OCR_TEXT if print_text is None else print_text

    print("=" * 70)
    print("STRUCTURED EXTRACTION  -  messy invoice images -> structured JSON")
    print("=" * 70)

    # Tesseract
    used = ocr.configure_tesseract(tesseract_cmd or config.TESSERACT_CMD)
    print(f"Tesseract : {used} (version {ocr.get_tesseract_version()})")

    # 1. Load
    paths = ip.find_images(images_dir, num_images, config.SUPPORTED_EXTENSIONS, config.SELECTION_MODE)
    print(f"Images    : {len(paths)} selected from {images_dir}\n")

    records, failed = [], []
    for n, path in enumerate(paths, start=1):
        invoice_id = path.stem
        print(f"[{n}/{len(paths)}] {path.name}")

        img = ip.load_image(path)
        if img is None:
            print("    ! could not read this image - skipped")
            failed.append(path.name)
            continue

        # 2. Preprocess
        processed = ip.preprocess_image(img, config.RESIZE_TARGET_WIDTH,
                                        config.DENOISE_KERNEL, config.THRESHOLD_METHOD)
        # 3. OCR
        text = ocr.extract_text(processed, config.TESSERACT_LANG, config.TESSERACT_CONFIG)
        jg.save_raw_text(invoice_id, text, raw_text_dir)
        if print_text:
            print("    ---- raw OCR text ----")
            print("\n".join("    " + line for line in text.splitlines()))
            print("    ----------------------")

        # 4. Extract
        fields = ie.extract_invoice_fields(text)

        # 5. JSON
        record = jg.build_invoice_record(invoice_id, fields, text)
        problems = jg.validate_record(record)
        if problems:
            print("    ! JSON structure problems:", problems)
        jg.save_invoice_json(record, json_dir)
        records.append(record)

        fin = record["financial"]
        print(f"    invoice no: {record['invoice_number'] or '-'} | date: {record['invoice_date'] or '-'} | "
              f"items: {len(record['items'])} | total: {fin['total']} {fin['currency']}")

    jg.save_all_invoices(records, all_file)

    print("\n" + "=" * 70)
    print(f"Done. {len(records)} invoice(s) processed, {len(failed)} skipped.")
    if failed:
        print("Skipped:", ", ".join(failed))
    print(f"Individual JSON : {json_dir}")
    print(f"Raw OCR text    : {raw_text_dir}")
    print(f"Combined JSON   : {all_file}")
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description="STRUCTURED EXTRACTION - invoice images to JSON")
    parser.add_argument("--images-dir", help="folder with invoice images (default: config.IMAGES_DIR)")
    parser.add_argument("--num-images", type=int, help="how many images to process (default: config.NUM_IMAGES)")
    parser.add_argument("--tesseract-cmd", help="path to tesseract executable (default: config.TESSERACT_CMD)")
    args = parser.parse_args(argv)
    try:
        run(args.images_dir, args.num_images, args.tesseract_cmd)
    except (FileNotFoundError, ocr.TesseractNotFoundError) as exc:
        print(f"\nERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
