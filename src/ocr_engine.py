"""OCR with Tesseract (pytesseract)."""
import logging
import shutil
from pathlib import Path

import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


class TesseractNotFoundError(RuntimeError):
    """Raised when the Tesseract executable cannot be located."""


def configure_tesseract(tesseract_cmd=None):
    """Point pytesseract at the Tesseract executable and return the path used.

    1. the configured path (e.g. D:\\Software\\tesseract.exe) if it exists
    2. otherwise a 'tesseract' found on the system PATH
    """
    if tesseract_cmd and Path(tesseract_cmd).is_file():
        pytesseract.pytesseract.tesseract_cmd = str(tesseract_cmd)
        return str(tesseract_cmd)
    found = shutil.which("tesseract")
    if found:
        if tesseract_cmd:
            logger.warning("Configured Tesseract path not found (%s); using PATH: %s", tesseract_cmd, found)
        pytesseract.pytesseract.tesseract_cmd = found
        return found
    raise TesseractNotFoundError(
        f"Tesseract executable not found at '{tesseract_cmd}' and not on PATH.\n"
        "Install Tesseract and set TESSERACT_CMD in config.py to the full path of tesseract.exe."
    )


def get_tesseract_version():
    """Return the Tesseract version string (also proves the executable really runs)."""
    return str(pytesseract.get_tesseract_version())


def extract_text(image, lang="eng", config="--psm 6"):
    """Run OCR on a numpy image (grayscale or BGR) or PIL image and return the raw text."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    return pytesseract.image_to_string(image, lang=lang, config=config)
