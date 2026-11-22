# STRUCTURED EXTRACTION

**Transforming Unstructured Invoice Images into Structured JSON Using OCR and Rule-Based Information Extraction.**

```text
Invoice Images → Image Preprocessing → OCR → Rule-Based Information Extraction → Structured JSON
```

---

## 1. Project Introduction

Invoices usually arrive as scanned pages or photos. A person can read them, but a computer sees only pixels.

**STRUCTURED EXTRACTION** is a small Data Science project that converts invoice images into structured JSON data that can be searched, stored and analysed.

## 2. Problem Statement

Information locked inside invoice images (invoice number, dates, seller, buyer, items and totals) cannot be used by software until it is converted into a structured format. Typing it in by hand is slow and error-prone.

## 3. Objectives

- Load invoice images and prepare them for OCR.
- Read text from each image using Tesseract OCR.
- Extract key invoice fields using regular expressions and keyword-based rules.
- Produce one JSON file per invoice and one combined JSON file.
- Keep the project simple, readable and runnable with one command.

## 4. Dataset

**FATURA** – a public invoice dataset containing approximately 10,000 invoice images.

```text
FATURA/
    images/            <- used as input
    Annotations/       (not used for extraction)
    strat1_train.csv   (not used)
    strat1_dev.csv     (not used)
    strat1_test.csv    (not used)
    Strat2_Split/      (not used)
```

This project processes **10 invoice images** from the `images` folder. Annotation files are not used by the extraction pipeline; the JSON is generated from the text read by OCR. The dataset is not included in this repository. Set the dataset path in `config.py`.

## 5. Technologies

Python · OpenCV · Tesseract OCR · pytesseract · Pillow · NumPy · Regular Expressions · JSON.

This is a rule-based OCR project. It does not use machine learning, a database, a dashboard or a web application.

## 6. Methodology

1. **Image loading** (`src/image_processing.py`) – loads the configured number of invoice images. Unreadable images are reported and skipped.
2. **Image preprocessing** – converts images to grayscale, resizes them, applies median-blur noise removal and uses Otsu thresholding.
3. **OCR** (`src/ocr_engine.py`) – Tesseract (`--psm 6`) reads the processed image. The raw OCR text is saved for each invoice.
4. **Rule-based information extraction** (`src/information_extraction.py`) – keyword and regular-expression rules identify:
   - Invoice numbers and dates using labels and date/number patterns.
   - Seller and buyer details using invoice text labels and block-based rules.
   - Item descriptions, quantities and amounts using table headers and row patterns.
   - Financial values and currency when they can be identified from the OCR text.
   - Common OCR variations, including selected character corrections and damaged quantity values where the available row information supports recovery.
5. **JSON generation** (`src/json_generator.py`) – creates a consistent record, saves individual JSON files and a combined JSON file, and validates the generated JSON.

Missing information is represented using an empty string or `null`, as appropriate. The system does not calculate replacement invoice totals or silently invent missing values.

## 7. System Workflow

```text
config.py ─► main.py
               │
               ├─ find_images()             load invoice images
               ├─ preprocess_image()        grayscale → resize → denoise → threshold
               ├─ extract_text()            Tesseract OCR → output/raw_text/<id>.txt
               ├─ extract_invoice_fields()  keyword + regex rules
               └─ build_invoice_record()    → output/json/<id>.json
                                            → output/all_invoices.json
```

## 8. JSON Output

Each invoice record follows a consistent structure. The actual output also includes the complete `ocr_text` read from the invoice.

```json
{
  "invoice_id": "Template10_Instance0",
  "invoice_number": "INV/30-14/832",
  "invoice_date": "03-Jan-1994",
  "due_date": "06-Feb-2007",
  "seller": {
    "name": "Mclean-Cochran",
    "address": "4894 Melanie Pass Apt. 742, Tammyland, SD 42587 US",
    "email": "cjames@example.com"
  },
  "buyer": {
    "name": "...",
    "address": "...",
    "email": "..."
  },
  "items": [
    {
      "description": "...",
      "quantity": 1,
      "amount": 0.0
    }
  ],
  "financial": {
    "subtotal": null,
    "tax": null,
    "total": 0.0,
    "currency": "EUR"
  },
  "ocr_text": "..."
}
```

The values above illustrate the output structure; ellipses and example numeric values are placeholders, not a complete extracted invoice. `invoice_id` is based on the image filename without its extension. The combined file `output/all_invoices.json` contains a list of invoice records.

## 9. Evaluation Results

The pipeline was run on **10 FATURA invoice images**. The latest generated `output/all_invoices.json` was reviewed.

| Evaluation measure | Result |
|---|---:|
| Invoice images processed | 10 |
| Successfully processed | 10 |
| Skipped images | 0 |
| Invoice numbers extracted | 10/10 |
| Invoice dates extracted | 10/10 |
| Due dates extracted | 10/10 |
| Seller names extracted | 10/10 |
| Buyer names extracted | 10/10 |
| Item records extracted | 32 |

The evaluation confirms that the pipeline produced the expected key fields for this 10-image sample. These counts are **field extraction coverage**, not a formal accuracy score. Accuracy, precision and recall have not been calculated against ground-truth annotations.

Some financial fields, such as subtotal and tax, remain `null` when they are not reliably identified in the OCR text. The extracted printed total is retained rather than being replaced by a total calculated from item rows.

## 10. Installation (Windows + VS Code)

1. Install Python 3.9 or later and open the `STRUCTURED_EXTRACTION` folder in VS Code.
2. Install Tesseract OCR for Windows. Set the executable path using `TESSERACT_CMD` in `config.py`. The current development setup uses:

   ```python
   TESSERACT_CMD = r"D:\Software\tesseract.exe"
   ```

   Make sure the English language data (`eng.traineddata`) is installed for Tesseract.
3. Download the FATURA dataset and set `DATASET_DIR` in `config.py` to the folder containing the `images` directory. For example:

   ```python
   DATASET_DIR = Path(r"D:\04th_Year Uni\Data Science\Project\invoices_dataset_final")
   ```

4. Install the Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## 11. Execution

Run the project from the root folder:

```bash
python main.py
```

The program loads the configured images, preprocesses them, runs OCR, extracts invoice fields and writes the results to `output/`.

| Output | Content |
|---|---|
| `output/raw_text/<id>.txt` | Raw OCR text for each invoice |
| `output/json/<id>.json` | Structured JSON for each invoice |
| `output/all_invoices.json` | All invoice records in one JSON file |

Optional command-line overrides, if supported by the current `main.py`:

```bash
python main.py --images-dir "D:\FATURA\images" --num-images 10 --tesseract-cmd "D:\Software\tesseract.exe"
```

**Notebook:** `notebooks/structured_extraction.ipynb` can be opened in VS Code with the Jupyter extension. Select the project Python environment/kernel before running the cells.

**Tests:** Run the available tests with:

```bash
python -m unittest discover -s tests -v
```

The tests check program behaviour using synthetic test data; they do not establish extraction accuracy on the FATURA images.

## 12. Limitations

- Only 10 images were used in this project run. Results may differ across the full FATURA dataset.
- FATURA contains different invoice templates. Keyword and regular-expression rules may work on some layouts and miss fields on others.
- OCR quality affects extraction. Low contrast, coloured backgrounds, stamps, small fonts and unusual layouts can cause recognition errors.
- Side-by-side columns (for example, “Bill To” beside “Ship To”) may be read as a single line, which can cause seller and buyer details to be mixed.
- Multi-line item descriptions and tables with additional columns may not be fully extracted.
- OCR can misread digits, letters, dates and email addresses. Rule-based corrections can address selected patterns but cannot resolve every error.
- Some invoice totals do not reconcile with the extracted item amounts and discounts. The printed total is preserved; the project does not assume that the source invoice or OCR is always consistent.
- Subtotal and tax are left as `null` when they cannot be reliably extracted.
- Currency symbols such as `$` can be ambiguous, so a symbol is not automatically treated as a specific currency code.
- The current extraction rules are designed for English-language invoices.
- A formal accuracy evaluation against the FATURA annotations has not been performed. The annotations were not used by the extraction pipeline or the reported evaluation.

## 13. Conclusion

The project demonstrates a complete, simple pipeline for converting invoice images into structured JSON using image preprocessing, Tesseract OCR and rule-based information extraction. In the tested sample, all 10 images were processed and the key invoice identifiers, dates and party names were extracted. The results also show the limitations of OCR and fixed extraction rules when invoice layouts and source data vary.
