"""Image loading and preprocessing (OpenCV)."""
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def find_images(images_dir, num_images=10, extensions=(".jpg", ".jpeg", ".png"), mode="first"):
    """Return up to `num_images` image paths from `images_dir` (searched recursively).

    mode="first"  -> the first N files in sorted order
    mode="spread" -> N files evenly spaced through the sorted list (more template variety)
    """
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Images folder not found: {images_dir}\n"
            "Edit DATASET_DIR in config.py (or set FATURA_DATASET_DIR) so it points to your FATURA folder."
        )
    exts = {e.lower() for e in extensions}
    files = sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"No {', '.join(sorted(exts))} images found in: {images_dir}")
    if len(files) <= num_images:
        return files
    if mode == "spread":
        step = len(files) / num_images
        return [files[int(i * step)] for i in range(num_images)]
    return files[:num_images]


def load_image(path):
    """Load an image as a BGR numpy array. Returns None if it cannot be read.

    np.fromfile + imdecode is used because cv2.imread fails on some Windows paths.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            logger.warning("Empty file: %s", path)
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Unreadable image (not a valid image file): %s", path)
        return img
    except Exception as exc:  # noqa: BLE001 - we want to survive any I/O problem
        logger.warning("Could not load %s: %s", path, exc)
        return None


def to_grayscale(img):
    """Convert a BGR / BGRA / already-gray image to grayscale."""
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_image(img, target_width=1800):
    """Scale to `target_width` keeping the aspect ratio."""
    h, w = img.shape[:2]
    if w == target_width or w == 0:
        return img
    scale = target_width / float(w)
    new_size = (target_width, max(1, int(round(h * scale))))
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(img, new_size, interpolation=interp)


def remove_noise(gray, kernel=3):
    """Basic noise removal with a median blur (kernel must be odd)."""
    kernel = max(1, int(kernel))
    if kernel % 2 == 0:
        kernel += 1
    if kernel == 1:
        return gray
    return cv2.medianBlur(gray, kernel)


def apply_threshold(gray, method="otsu"):
    """Convert grayscale to a black & white image."""
    if method == "adaptive":
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
        )
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def preprocess_steps(img, target_width=1800, kernel=3, method="otsu"):
    """Run the whole pipeline and return every intermediate stage (handy for the notebook)."""
    gray = to_grayscale(img)
    resized = resize_image(gray, target_width)
    denoised = remove_noise(resized, kernel)
    binary = apply_threshold(denoised, method)
    return {"gray": gray, "resized": resized, "denoised": denoised, "binary": binary}


def preprocess_image(img, target_width=1800, kernel=3, method="otsu"):
    """Grayscale -> resize -> denoise -> threshold. Returns the final binary image."""
    return preprocess_steps(img, target_width, kernel, method)["binary"]
