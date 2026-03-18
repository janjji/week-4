import cv2 # type: ignore
import numpy as np # type: ignore
import pytesseract # type: ignore
from pathlib import Path
import tensorflow as tf # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore

#  constants 
IMG_SIZE        = (224, 224)
# Use absolute path relative to this file's location
# forgery_check.py is in src/, so go up 1 level to project root
_SRC_DIR   = Path(__file__).resolve().parent
_ROOT_DIR  = _SRC_DIR.parent
MODEL_PATH = _ROOT_DIR / "outputs" / "module6" / "mobilenetv2_truck_classifier.keras"
OCR_CONF_THRESH = 20.0
# Lowered thresholds for license image domain
# (CNN was trained on trucks, so scores are naturally lower)
HIGH_THRESH   = 0.50   # catches forged_blur (0.618+) as HIGH
MEDIUM_THRESH = 0.338

# ensemble weights
W_OCR       = 0.30
W_CNN       = 0.40
W_FORENSICS = 0.30

#  model (loaded once at import time) 

def _get_model():
    global _model
    try:
        if _model is None:
            pass
    except NameError:
        _model = None

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}.\n"
                "Run notebooks/06_forgery_detection.ipynb first."
            )
        _model = load_model(str(MODEL_PATH))
    return _model


#  Step 2: image forensics 
def _blur_score(gray):
    """Laplacian variance — lower = blurrier."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _noise_score(gray):
    """Mean high-frequency residual."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return float(cv2.absdiff(gray, blurred).astype(np.float32).mean())

def _forensics_flag(img, blur_thresh=654.3, noise_thresh=16.6):
    """
    Returns True if image looks anomalous (too blurry or too noisy).
    Thresholds are mean ± 2σ from the training set analysis.
    """
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur  = _blur_score(gray)
    noise = _noise_score(gray)
    flagged = (blur < blur_thresh) or (noise > noise_thresh)
    return flagged, round(blur, 2), round(noise, 2)


#  Step 3: OCR plate extraction 
def _detect_plate_roi(img):
    """Contour-based license plate region detector."""
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur    = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blur, 50, 200)
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours     = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    h, w = img.shape[:2]
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect     = cw / (ch + 1e-6)
        area_ratio = (cw * ch) / (w * h)
        if 2.0 < aspect < 6.0 and 0.005 < area_ratio < 0.15:
            return img[y:y+ch, x:x+cw], (x, y, cw, ch)

    return None, None

def _run_ocr(img):
    """
    Detect plate ROI then run Tesseract OCR.
    Returns (text, confidence, roi_found).
    """
    roi, bbox = _detect_plate_roi(img)
    target    = roi if (roi is not None and roi.size > 0) else img
    roi_found = roi is not None

    gray     = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    resized  = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        cfg  = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        data = pytesseract.image_to_data(
            thresh, config=cfg, output_type=pytesseract.Output.DICT
        )
        confs     = [c for c in data["conf"] if c != -1]
        texts     = [t.strip() for t in data["text"] if t.strip()]
        avg_conf  = float(np.mean(confs)) if confs else 0.0
        full_text = "".join(texts).upper().strip()
        return full_text, round(avg_conf, 1), roi_found
    except Exception:
        return "", 0.0, roi_found


# Step 4: CNN classification 
def _cnn_score(img):
    rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE).astype(np.float32)
    arr     = preprocess_input(resized[np.newaxis, ...])
    score   = float(_get_model().predict(arr, verbose=0)[0, 0])
    return round(score, 4)


#  Step 5: ensemble 
def _ensemble(ocr_conf, cnn_prob, forensics_flagged):
    """Combine three signals into a single suspicion score."""
    ocr_score       = 1.0 - min(ocr_conf / 100.0, 1.0)   # high conf = low suspicion
    forensics_score = float(forensics_flagged)
    score = W_OCR * ocr_score + W_CNN * cnn_prob + W_FORENSICS * forensics_score
    return round(score, 4)

def _risk_tier(score):
    if score >= HIGH_THRESH:   return "HIGH"
    if score >= MEDIUM_THRESH: return "MEDIUM"
    return "LOW"


#  public API 
def check_vehicle(image_path: str) -> dict:
    """
    Run the full vehicle verification pipeline on a single image.

    Parameters
    ----------
    image_path : str or Path
        Path to the truck image (.jpg / .png).

    Returns
    -------
    dict with keys:
        image_path      - input path
        ocr_text        - extracted plate text (may be empty)
        ocr_conf        - Tesseract confidence (0–100)
        roi_found       - whether a plate region was detected
        blur_score      - Laplacian variance (lower = blurrier)
        noise_score     - mean residual noise
        forensics_flag  - True if image quality is anomalous
        cnn_score       - CNN suspicion probability (0–1)
        suspicion_score - final ensemble score (0–1)
        risk_level      - "LOW" / "MEDIUM" / "HIGH"
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # run all three stages
    ocr_text, ocr_conf, roi_found        = _run_ocr(img)
    flagged, blur, noise                 = _forensics_flag(img)
    cnn_prob                             = _cnn_score(img)
    suspicion                            = _ensemble(ocr_conf, cnn_prob, flagged)

    return {
        "image_path"     : str(image_path),
        "ocr_text"       : ocr_text,
        "ocr_conf"       : ocr_conf,
        "roi_found"      : roi_found,
        "blur_score"     : blur,
        "noise_score"    : noise,
        "forensics_flag" : flagged,
        "cnn_score"      : cnn_prob,
        "suspicion_score": suspicion,
        "risk_level"     : _risk_tier(suspicion),
    }


def check_batch(image_paths: list) -> list:
    """Run check_vehicle on a list of image paths. Returns list of result dicts."""
    return [check_vehicle(p) for p in image_paths]


#  quick CLI test 
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python forgery_check.py <image_path> [image_path ...]")
        sys.exit(1)

    paths   = sys.argv[1:]
    results = check_batch(paths)

    for r in results:
        print(json.dumps(r, indent=2))