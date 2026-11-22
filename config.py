"""
STRUCTURED EXTRACTION - configuration.

Edit the values below (or set the environment variables shown) to match your
computer. Everything else in the project reads its settings from this file.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Tesseract OCR executable
# --------------------------------------------------------------------------
# Windows path requested for this project. If this file does not exist the
# program falls back to a "tesseract" found on the system PATH.
TESSERACT_CMD = os.environ.get("TESSERACT_CMD", r"D:\Software\tesseract.exe")
TESSERACT_LANG = "eng"
TESSERACT_CONFIG = "--psm 6"   # psm 6 = "a single uniform block of text" (keeps line order)

# --------------------------------------------------------------------------
# FATURA dataset
# --------------------------------------------------------------------------
# DATASET_DIR is the folder that contains: images/, Annotations/, strat1_train.csv ...
# Change it here, or set the FATURA_DATASET_DIR environment variable.
# Example (Windows):  DATASET_DIR = Path(r"D:\Datasets\FATURA")
DATASET_DIR = Path(r"D:\04th_Year Uni\Data Science\Project\invoices_dataset_final")
IMAGES_DIR = DATASET_DIR / "images"      # only the images folder is used as input

NUM_IMAGES = 10                          # how many invoice images to process
SELECTION_MODE = "first"                 # "first" = first N (sorted) | "spread" = N evenly spaced
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")

# --------------------------------------------------------------------------
# Image preprocessing
# --------------------------------------------------------------------------
RESIZE_TARGET_WIDTH = 1800               # every image is scaled to this width (aspect ratio kept)
DENOISE_KERNEL = 3                       # odd number; median blur kernel for noise removal
THRESHOLD_METHOD = "otsu"                # "otsu" | "adaptive"

# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "output"
JSON_DIR = OUTPUT_DIR / "json"
RAW_TEXT_DIR = OUTPUT_DIR / "raw_text"
ALL_INVOICES_FILE = OUTPUT_DIR / "all_invoices.json"

PRINT_OCR_TEXT = True                    # show the raw OCR text in the console
