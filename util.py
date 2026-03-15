"""
util.py — ANPR Utility Functions
=================================
Provides reusable, modular helper functions for the ANPR pipeline:

  - read_license_plate()     : Run PaddleOCR on a cropped plate image (primary, local)
  - post_process_plate()     : Character correction + format validation (Regex).
  - draw_border()            : Draw a colored bounding box with label on a frame.
  - get_car()                : Map a license plate detection to a vehicle track.
  - write_csv()              : Persist results to a CSV log file.
  - interpolate_bounding_boxes() : SciPy-based smoothing of missed detections.
"""

import re
import csv
import os
import warnings
import logging
import numpy as np
from scipy.interpolate import interp1d

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 .* doesn't match a supported version!",
)

# Silence PaddleOCR internal debug logs (e.g., "ppocr DEBUG: dt_boxes ...").
for _logger_name in ("ppocr", "paddleocr"):
    _logger = logging.getLogger(_logger_name)
    _logger.setLevel(logging.ERROR)
    _logger.propagate = False

# ==============================================================================
# Global OCR Reader — initialised once to avoid repeated model loads
# ==============================================================================
_paddle_reader = None

def get_paddle_ocr_reader():
    """
    Lazily initialise PaddleOCR reader on first call.
    PaddleOCR is faster and more accurate for license plates than EasyOCR.
    """
    global _paddle_reader
    if _paddle_reader is None:
        # Skip external model-source connectivity checks during startup.
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")
        from paddleocr import PaddleOCR
        base_kwargs = {"use_angle_cls": False, "lang": "en"}

        # PaddleOCR argument names differ across versions. Try supported forms
        # in order, then fall back to CPU-only mode.
        device_attempts = [
            {"device": "gpu"},
            {"use_gpu": True},
            {"use_cuda": True},
            {},
        ]
        log_attempts = [
            {"show_log": False},
            {},
        ]

        last_exc = None
        for log_kwargs in log_attempts:
            for extra_kwargs in device_attempts:
                try:
                    _paddle_reader = PaddleOCR(**base_kwargs, **log_kwargs, **extra_kwargs)
                    break
                except (TypeError, ValueError) as exc:
                    last_exc = exc
            if _paddle_reader is not None:
                break

        if _paddle_reader is None and last_exc is not None:
            raise last_exc
    return _paddle_reader


# ==============================================================================
# Phase 4 Helper — Character Mapping for Common OCR Errors
# ==============================================================================

# Maps visually ambiguous characters in the LETTER positions of a plate.
# E.g., '0' is often misread as 'O', '1' as 'I' or 'L', etc.
DIGIT_MAP = {
    'O': '0', 'I': '1', 'L': '1', 'Z': '2',
    'S': '5', 'G': '6', 'B': '8', 'A': '4',
    'J': '7', 'T': '7', 'D': '0', 'Q': '0'
}

# Maps visually ambiguous characters in the DIGIT positions of a plate.
LETTER_MAP = {
    '0': 'O', '1': 'I', '2': 'Z',
    '5': 'S', '6': 'G', '8': 'B',
    '4': 'A', '7': 'T'
}

# ==============================================================================
# License Plate Format Validators (Regex)
# ==============================================================================

VALID_PLATE_PATTERNS = [
    # Standard modern: MH12AB1234, DL8C1234, KA01XYZ9999
    re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$'), 
    # Older or specific commercial: UP141234
    re.compile(r'^[A-Z]{2}\d{2}\d{4}$'),          
    # New Bharat (BH) Series: 21BH2345AA
    re.compile(r'^\d{2}BH\d{4}[A-Z]{1,2}$')       
]

# Valid two-letter Indian state / UT registration prefixes.
VALID_STATE_CODES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 'GA', 'GJ',
    'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN', 'MP',
    'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UK', 'UP',
    'WB',
}

# Indian plate positional templates: L=letter expected, D=digit expected
# Used for character correction BEFORE regex validation.
PLATE_TEMPLATES = [
    'LLDDLLDDDD',   # Standard 10-char: KA03HW9382
    'LLDDLDDDD',    # Short 9-char:     DL8C12345 → actually DL08C1234
    'LLDDLLLDDDD',  # Extended series:  MH12ABC1234
    'LLDDDDDD',     # Old commercial:   UP141234
    'DDLLDDDDLL',   # Bharat (BH):      21BH2345AA
]


def _strip_ind_noise(text: str) -> str:
    """
    Remove common non-plate artifacts from Indian plates before validation.
    Most common offender is the blue-strip text "IND" (or OCR variants like 1ND).
    """
    t = text.upper().strip()

    # Keep only alnum for robust token cleanup.
    t = re.sub(r'[^A-Z0-9]', '', t)

    # Remove leading IND-like prefixes that OCR often prepends.
    t = re.sub(r'^(IND|1ND|I1ND|IN0|INO)+', '', t)

    return t


def _apply_positional_correction(text: str) -> str:
    """
    Try each Indian plate template. For the best-matching template,
    swap characters using DIGIT_MAP / LETTER_MAP at each position.
    Returns the corrected string (may still fail regex validation).
    """
    if not text:
        return text

    best_corrected = text
    best_match_score = -1

    for template in PLATE_TEMPLATES:
        if len(template) != len(text):
            continue

        corrected = list(text)
        match_score = 0

        for i, expected in enumerate(template):
            ch = corrected[i]
            if expected == 'L':
                # Position expects a letter
                if ch.isalpha():
                    match_score += 1
                elif ch in LETTER_MAP:
                    corrected[i] = LETTER_MAP[ch]
                    match_score += 1
            elif expected == 'D':
                # Position expects a digit
                if ch.isdigit():
                    match_score += 1
                elif ch in DIGIT_MAP:
                    corrected[i] = DIGIT_MAP[ch]
                    match_score += 1

        if match_score > best_match_score:
            best_match_score = match_score
            best_corrected = ''.join(corrected)

    return best_corrected


def post_process_plate(text: str) -> tuple[str, bool]:
    """
    Cleans up raw OCR text, applies positional character correction,
    and validates against strict Indian plate formats.
    """
    # 1. Strip separators and known IND strip artifacts
    clean_text = _strip_ind_noise(text)
    
    # 2. Length Filter: Indian plates are strictly 8 to 10 chars (without spaces)
    if len(clean_text) < 8 or len(clean_text) > 10:
        return clean_text, False

    # 3. Positional character correction using templates
    corrected = _apply_positional_correction(clean_text)

    # 4. Validate strict format first.
    is_valid = any(pattern.match(corrected) for pattern in VALID_PLATE_PATTERNS)
    if not is_valid:
        return corrected, False

    # 5. Enforce valid state/UT prefix for non-BH plates.
    if 'BH' not in corrected[2:4]:
        state_code = corrected[:2]
        if state_code not in VALID_STATE_CODES:
            return corrected, False
    
    return corrected, is_valid


def _ocr_results_to_text_conf(results) -> tuple[str | None, float | None]:
    """Normalize PaddleOCR outputs across legacy/new response formats."""
    if not results:
        return None, None

    # Legacy format: [[(poly, (text, conf)), ...]]
    if isinstance(results, list) and results and isinstance(results[0], list):
        if not results[0]:
            return None, None
        sorted_results = sorted(results[0], key=lambda r: r[0][0][1])
        combined_text = ''.join([text for _, (text, _) in sorted_results]).upper().strip()
        avg_confidence = sum(conf for _, (_, conf) in sorted_results) / len(sorted_results)
        return combined_text, float(avg_confidence)

    # Newer format: [dict(..., rec_texts=[...], rec_scores=[...], rec_polys=[...])]
    if isinstance(results, list) and results and isinstance(results[0], dict):
        item = results[0]
        rec_texts = item.get('rec_texts') or []
        rec_scores = item.get('rec_scores') or []
        rec_polys = item.get('rec_polys') or item.get('dt_polys') or []

        if not rec_texts:
            return None, None

        order = list(range(len(rec_texts)))
        if rec_polys and len(rec_polys) == len(rec_texts):
            try:
                order = sorted(order, key=lambda i: float(rec_polys[i][0][1]))
            except Exception:
                pass

        texts = [str(rec_texts[i]) for i in order]
        scores = [float(rec_scores[i]) for i in order if i < len(rec_scores)]
        combined_text = ''.join(texts).upper().strip()
        avg_confidence = (sum(scores) / len(scores)) if scores else 0.0
        return combined_text, float(avg_confidence)

    return None, None


def _run_ocr(reader, image) -> tuple[str | None, float | None]:
    """Call PaddleOCR across API variants and normalize output."""
    try:
        results = reader.ocr(image, cls=False)
    except TypeError:
        results = reader.ocr(image)
    return _ocr_results_to_text_conf(results)


# ==============================================================================
# Phase 3 Pre-step: Image Enhancement for OCR
# ==============================================================================

def preprocess_plate_crop(plate_crop: np.ndarray) -> np.ndarray:
    """
    Enhance a license plate crop before OCR to improve character legibility.

    Pipeline:
      1. Upscale  — ensures sufficient resolution (PaddleOCR struggles < ~60px height)
      2. Grayscale — removes colour noise, simplifies subsequent steps
      3. CLAHE     — adaptive contrast boost; lifts faded/overexposed characters
      4. Bilateral filter — edge-preserving denoise (keeps char strokes sharp)
      5. Unsharp mask — sharpens character edges for cleaner glyph boundaries

    Args:
        plate_crop: BGR image crop of the license plate.

    Returns:
        Preprocessed grayscale image (H, W) as a numpy uint8 array.
    """
    import cv2

    # 1. Upscale: target 2× current size, but enforce a minimum height of 64 px
    h, w = plate_crop.shape[:2]
    target_h = max(h * 2, 64)
    scale = target_h / h
    resized = cv2.resize(
        plate_crop,
        (int(w * scale), target_h),
        interpolation=cv2.INTER_CUBIC,
    )

    # 2. Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 3. CLAHE — adaptive histogram equalisation (better than global equalise)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 4. Bilateral filter — smooth noise while preserving character edges
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # 5. Unsharp mask — sharpens character strokes
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    return gray


# ==============================================================================
# Phase 3: OCR — Read a License Plate Crop
# ==============================================================================

def read_license_plate(plate_crop: np.ndarray) -> tuple[str | None, float | None]:
    """
    Run PaddleOCR on a cropped license plate image.
    PaddleOCR is specifically optimized for license plates and runs locally (no API).
    
    For multi-line plates (e.g. two-wheeler), all detected text fragments are
    concatenated in top-to-bottom order to form a single plate string.

    Args:
        plate_crop: Cropped BGR image of the license plate (numpy array).

    Returns:
        (text, confidence): Combined plate string and average confidence,
                            or (None, None) if no text was found.
    """
    reader = get_paddle_ocr_reader()
    enhanced = preprocess_plate_crop(plate_crop)

    import cv2

    # Build a small ensemble of preprocessing variants and keep best OCR result.
    variants = []
    base_gray = enhanced if enhanced.ndim == 2 else cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    variants.append(base_gray)

    # Adaptive threshold variants (both polarities) improve difficult glare/night frames.
    th_bin = cv2.adaptiveThreshold(
        base_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    th_inv = cv2.adaptiveThreshold(
        base_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    variants.extend([th_bin, th_inv])

    best_text = None
    best_conf = -1.0
    best_score = -1.0

    for v in variants:
        ocr_in = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR) if v.ndim == 2 else v
        raw_text, conf = _run_ocr(reader, ocr_in)
        if raw_text is None or conf is None:
            continue

        # Post-process for candidate ranking.
        cleaned_text = _strip_ind_noise(raw_text)
        corrected_text, is_valid = post_process_plate(cleaned_text)

        # Prefer regex-valid readings, then confidence.
        # Small length prior favors realistic Indian plate lengths.
        len_bonus = 0.05 if 8 <= len(corrected_text) <= 10 else 0.0
        score = float(conf) + (0.35 if is_valid else 0.0) + len_bonus

        if score > best_score:
            best_score = score
            best_conf = float(conf)
            best_text = corrected_text if corrected_text else cleaned_text

    if best_text is None:
        return None, None
    return best_text, best_conf


# ==============================================================================
# Phase 2 Helper — Map Plate Detection to Vehicle Track
# ==============================================================================

def get_car(plate_bbox: list, vehicle_track_ids: np.ndarray):
    """
    Given a license plate bounding box, find which tracked vehicle it belongs to
    by checking if the plate centre lies within any vehicle's bounding box.

    This ensures we associate each plate reading with the correct vehicle ID.

    Args:
        plate_bbox:        [x1, y1, x2, y2] of the detected license plate.
        vehicle_track_ids: np.array from SORT of shape (M, 5)
                           columns: [x1, y1, x2, y2, track_id]

    Returns:
        (x1, y1, x2, y2, car_id): Bounding box + track ID of the matched vehicle,
                                    or (-1, -1, -1, -1, -1) if no match is found.
    """
    px1, py1, px2, py2 = plate_bbox

    # Compute the centre of the plate bounding box
    pcx = (px1 + px2) / 2.0
    pcy = (py1 + py2) / 2.0

    for track in vehicle_track_ids:
        vx1, vy1, vx2, vy2, vid = track
        # Check if plate centre is inside the vehicle bounding box
        if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
            return int(vx1), int(vy1), int(vx2), int(vy2), int(vid)

    # No vehicle contains this plate — could be a false positive
    return -1, -1, -1, -1, -1


# ==============================================================================
# Phase 5 Helper — Draw Annotations on Frame
# ==============================================================================

def draw_border(
    img: np.ndarray,
    top_left: tuple, bottom_right: tuple,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    line_length_x: int = 30, line_length_y: int = 30
) -> np.ndarray:
    """
    Draw a stylised corner-bracket bounding box on the frame.
    More visually distinctive than a full rectangle for ANPR overlays.

    Args:
        img:           BGR frame to draw on.
        top_left:      (x, y) top-left corner.
        bottom_right:  (x, y) bottom-right corner.
        color:         BGR colour tuple.
        thickness:     Line thickness in pixels.
        line_length_x: Length of horizontal bracket segments.
        line_length_y: Length of vertical bracket segments.

    Returns:
        Annotated copy of the frame.
    """
    import cv2
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Top-left corner
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)

    # Top-right corner
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    # Bottom-left corner
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)

    # Bottom-right corner
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)

    return img


# ==============================================================================
# Phase 5 Helper — CSV Output
# ==============================================================================

def write_csv(results: dict, output_path: str) -> None:
    """
    Write the final recognised plate records to a CSV file.

    CSV Schema:
        frame_nmr   — Frame number when the detection occurred
        car_id      — Unique vehicle track ID from SORT
        car_bbox    — Vehicle bounding box [x1, y1, x2, y2]
        license_plate_bbox        — Plate bounding box [x1, y1, x2, y2]
        license_plate_bbox_score  — Plate detection confidence from YOLO
        license_number            — Corrected OCR text
        license_number_score      — OCR confidence from PaddleOCR

    Args:
        results:     Dict structured as:
                     { frame_nmr: { car_id: { 'car': {...}, 'license_plate': {...} } } }
        output_path: Path to the output CSV file.
    """
    header = [
        'frame_nmr', 'car_id',
        'car_bbox',
        'license_plate_bbox', 'license_plate_bbox_score',
        'license_number', 'license_number_score',
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for frame_nmr in sorted(results.keys()):
            for car_id in results[frame_nmr]:
                entry = results[frame_nmr][car_id]

                # Only write rows with successful OCR
                if 'license_plate' not in entry:
                    continue
                lp = entry['license_plate']
                car = entry.get('car', {})

                writer.writerow({
                    'frame_nmr':                frame_nmr,
                    'car_id':                   car_id,
                    'car_bbox':                 car.get('bbox', ''),
                    'license_plate_bbox':       lp.get('bbox', ''),
                    'license_plate_bbox_score': lp.get('bbox_score', ''),
                    'license_number':           lp.get('text', ''),
                    'license_number_score':     lp.get('text_score', ''),
                })


# ==============================================================================
# Phase 3 Helper — SciPy Interpolation for Missed Detections
# ==============================================================================

def interpolate_bounding_boxes(
    data: list[dict],
    smooth_car_bbox_window: int = 1,
    min_track_frames: int = 0,
) -> list[dict]:
    """
    Use SciPy 1D interpolation to fill in missing license plate bounding boxes
    across frames where the plate was not detected (e.g., due to motion blur,
    occlusion, or failed OCR).

    This maintains visual continuity in the output video and prevents
    "blinking" annotations for fast-moving vehicles.

    Algorithm:
      For each unique car_id:
        1. Identify frames where the plate was detected (anchor points).
        2. Use scipy.interpolate.interp1d to linearly interpolate plate bbox
           coordinates between anchor frames.
        3. Propagate the last-seen OCR text to interpolated frames.

    Args:
        data: List of dicts loaded from the raw CSV output of the pipeline.
              Each dict has keys: frame_nmr, car_id, car_bbox,
              license_plate_bbox, license_plate_bbox_score,
              license_number, license_number_score.

    Args:
        smooth_car_bbox_window: Centered moving-average window for car_bbox
            coordinate smoothing. Set to 1 to disable.
        min_track_frames: Drop tracks shorter than this number of frames after
            interpolation/smoothing. Set to 0 to keep all tracks.

    Returns:
        List of dicts with interpolated rows added in-place, sorted by
        (frame_nmr, car_id).
    """
    def parse_bbox(bbox_str):
        """Parse bbox string with spaces/commas into [x1,y1,x2,y2] floats."""
        nums = re.findall(r'[-+]?\d*\.?\d+', str(bbox_str))
        return [float(n) for n in nums] if len(nums) == 4 else None

    def smooth_centered(values: list[float], window: int) -> list[float]:
        """Centered moving average with edge handling for short sequences."""
        if window <= 1 or len(values) <= 1:
            return values
        half = window // 2
        out = []
        for i in range(len(values)):
            lo = max(0, i - half)
            hi = min(len(values), i + half + 1)
            out.append(float(np.mean(values[lo:hi])))
        return out

    # Group rows by car_id
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids       = np.array([int(row['car_id'])    for row in data])

    interpolated = []

    for car_id in np.unique(car_ids):
        # Extract all rows for this vehicle
        mask = car_ids == car_id
        car_rows = [data[i] for i in np.where(mask)[0]]
        car_frames = frame_numbers[mask]

        # Identify "anchor" frames: frames where we have a valid plate bbox
        anchor_frames = []
        anchor_bboxes = []  # Each is [x1, y1, x2, y2]
        anchor_texts  = []
        anchor_scores = []

        for row in car_rows:
            bbox = parse_bbox(str(row.get('license_plate_bbox', '')))
            if bbox is not None and row.get('license_number', '').strip():
                anchor_frames.append(int(row['frame_nmr']))
                anchor_bboxes.append(bbox)
                anchor_texts.append(row['license_number'])
                anchor_scores.append(float(row.get('license_number_score', 0)))

        if len(anchor_frames) < 2:
            # Not enough anchors to interpolate — keep original rows as-is
            interpolated.extend(car_rows)
            continue

        # Build interpolation functions for each bbox coordinate
        anchor_frames_arr = np.array(anchor_frames, dtype=float)
        anchor_bboxes_arr = np.array(anchor_bboxes, dtype=float)  # (N, 4)

        interp_fns = [
            interp1d(anchor_frames_arr, anchor_bboxes_arr[:, i],
                     kind='linear', fill_value='extrapolate')
            for i in range(4)
        ]

        # Pick the best OCR result (highest confidence) to use for all frames
        best_idx  = int(np.argmax(anchor_scores))
        best_text = anchor_texts[best_idx]
        best_score = anchor_scores[best_idx]

        # Build a quick lookup of original rows
        original_lookup = {int(r['frame_nmr']): r for r in car_rows}

        MAX_GAP = 15  # Maximum number of missing frames we are willing to interpolate

        # Iterate through anchor pairs to fill gaps safely
        for i in range(len(anchor_frames)):
            f_current = anchor_frames[i]
            
            # Always add the anchor frame itself
            if f_current in original_lookup:
                row = dict(original_lookup[f_current])
                if not row.get('license_number', '').strip():
                    row['license_number']       = best_text
                    row['license_number_score'] = best_score
                interpolated.append(row)
                
            # If there's a next anchor, check the gap
            if i < len(anchor_frames) - 1:
                f_next = anchor_frames[i+1]
                
                # If the gap is too large, DO NOT interpolate. Just let it drop.
                if (f_next - f_current) > MAX_GAP:
                    continue
                    
                # Otherwise, interpolate the frames between f_current and f_next
                for f in range(f_current + 1, f_next):
                    interp_bbox = [float(fn(f)) for fn in interp_fns]
                    new_row = {
                        'frame_nmr':                f,
                        'car_id':                   car_id,
                        'car_bbox':                 original_lookup.get(anchor_frames[0], car_rows[0]).get('car_bbox', ''),
                        'license_plate_bbox':       '[{:.2f} {:.2f} {:.2f} {:.2f}]'.format(*interp_bbox),
                        'license_plate_bbox_score': 0.0,   
                        'license_number':           best_text,
                        'license_number_score':     best_score,
                    }
                    interpolated.append(new_row)

    # Optional car bbox smoothing and optional short-track filtering.
    if interpolated and (smooth_car_bbox_window > 1 or min_track_frames > 0):
        by_car = {}
        for row in interpolated:
            cid = int(row['car_id'])
            by_car.setdefault(cid, []).append(row)

        processed = []
        for cid, rows in by_car.items():
            rows.sort(key=lambda r: int(r['frame_nmr']))

            if min_track_frames > 0 and len(rows) < min_track_frames:
                continue

            if smooth_car_bbox_window > 1:
                coords = [parse_bbox(r.get('car_bbox', '')) for r in rows]
                valid_idx = [i for i, c in enumerate(coords) if c is not None]
                if valid_idx:
                    x1 = [coords[i][0] for i in valid_idx]
                    y1 = [coords[i][1] for i in valid_idx]
                    x2 = [coords[i][2] for i in valid_idx]
                    y2 = [coords[i][3] for i in valid_idx]

                    sx1 = smooth_centered(x1, smooth_car_bbox_window)
                    sy1 = smooth_centered(y1, smooth_car_bbox_window)
                    sx2 = smooth_centered(x2, smooth_car_bbox_window)
                    sy2 = smooth_centered(y2, smooth_car_bbox_window)

                    for k, idx in enumerate(valid_idx):
                        rows[idx]['car_bbox'] = '[{:.2f} {:.2f} {:.2f} {:.2f}]'.format(
                            sx1[k], sy1[k], sx2[k], sy2[k]
                        )

            processed.extend(rows)

        interpolated = processed

    # Sort combined results by (frame, car_id) for downstream processing
    interpolated.sort(key=lambda r: (int(r['frame_nmr']), int(r['car_id'])))
    return interpolated
