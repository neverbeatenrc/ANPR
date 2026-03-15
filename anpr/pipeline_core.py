"""
pipeline.py — ANPR Main Inference Pipeline
===========================================
End-to-end Automatic Number Plate Recognition for real-time traffic monitoring
using pre-recorded video footage.

Pipeline Data Flow (per frame):
  Video Frame
    → [Phase 1a] YOLOv8 Vehicle Detector  → Vehicle bounding boxes
    → [Phase 2]  SORT Tracker             → Vehicles + unique track IDs
    → [Phase 1b] YOLOv8 Plate Localiser   → Plate bbox (within vehicle ROI)
    → [Phase 3]  PaddleOCR                → Raw plate text + confidence
    → [Phase 4]  Post-processing          → Corrected + validated plate text
    → [Phase 5a] Annotated video output   → MP4 with boxes, IDs, plate text
    → [Phase 5b] CSV database             → frame, car_id, plate, timestamp

Usage:
  python pipeline.py --source input.mp4 [--output output.mp4] [--csv results.csv]

  Or edit the CONFIG section below and run: python pipeline.py
"""

import os
import sys
import csv
import time
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 .* doesn't match a supported version!",
)

import cv2
import numpy as np
import torch
from collections import defaultdict, deque, Counter
from ultralytics import YOLO

# Local modules — must be in the same directory as pipeline.py
from sort import Sort
from util import (
    read_license_plate,
    post_process_plate,
    get_car,
    draw_border,
    write_csv,
    interpolate_bounding_boxes,
)

# ==============================================================================
# ░░░░░░░░░░░░░░░░  CONFIGURATION  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ==============================================================================

# ---- Input / Output ---
INPUT_VIDEO   = 'input.mp4'    # Path to the source video file
OUTPUT_VIDEO  = 'output.mp4'   # Path for the annotated output video
OUTPUT_CSV    = 'results.csv'  # Path for the CSV plate recognition log

# ---- Model Paths ---
# Vehicle detector: standard COCO-pretrained YOLOv8 (detects cars, trucks, buses)
VEHICLE_MODEL_PATH = 'yolov8n.pt'    # Will auto-download on first run

# Plate localiser: custom-trained model from train.py
# Set this to 'runs/detect/license_plate_detector/weights/best.pt' after training.
# If you don't have a custom model yet, a generic COCO model is used as fallback.
PLATE_MODEL_PATH = 'runs/detect/license_plate_detector/weights/best.pt'

# ---- YOLO Class IDs for vehicles (COCO dataset) ---
# 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASS_IDS = {2, 3, 5, 7}

# ---- Detection Thresholds ---
VEHICLE_CONF_THRESHOLD = 0.3   # Min confidence for vehicle detection
PLATE_CONF_THRESHOLD   = 0.1  # Very low: maximize plate recall, OCR regex will filter false positives
OCR_CONF_THRESHOLD     = 0.35  # Raised: preprocessing now produces cleaner reads
REQUIRE_VALID_PLATE   = False  # If True, discard OCR unless regex-valid Indian plate
PLATE_BLUR_VAR_THRESHOLD = 80.0  # Skip very blurry plate crops before OCR
PLATE_HISTORY_WINDOW = 12
OCR_MIN_FRAME_GAP = 2           # Run OCR for same track every N frames (temporal voting fills gaps)
MAX_OCR_CALLS_PER_FRAME = 2     # Requested: cap OCR calls/frame to improve speed.
CAR_BBOX_SMOOTH_WINDOW = 5      # Centered moving average window for car bbox smoothing
MIN_TRACK_FRAMES_FOR_OUTPUT = 0 # Set >0 (e.g. 15) to drop short-lived ghost tracks

# ---- SORT Tracker Hyperparameters ---
SORT_MAX_AGE       = 30     # Frames to keep a track alive without a detection
SORT_MIN_HITS      = 3      # Frames needed to confirm a new track (prevents phantom IDs)
SORT_IOU_THRESHOLD = 0.45   # IoU threshold — raised to prevent cross-matching adjacent cars

# ---- GPU / VRAM Optimisation (RTX 2050, 4GB) ---
# half=True:  FP16 inference — halves VRAM, minimal accuracy loss
# imgsz=640:  Standard YOLO input size
USE_HALF_PRECISION = False # Disabled due to PyTorch FP16/FP32 fusion bug
INFERENCE_IMG_SIZE = 640        # Requested: lower inference size for throughput.

# ---- Annotation Style ---
VEHICLE_BOX_COLOR = (0, 255, 0)       # Green — vehicle bounding box
PLATE_BOX_COLOR   = (0, 0, 255)       # Red   — license plate box
TEXT_COLOR        = (255, 255, 255)   # White — overlay text
FONT              = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE        = 0.7
FONT_THICKNESS    = 2
PROGRESS_LOG_EVERY_N_FRAMES = 30

# ==============================================================================
# ░░░░░░░░░░░░░░░░  PIPELINE CLASS  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ==============================================================================

class ANPRPipeline:
    """
    Encapsulates the full ANPR inference pipeline.

    Attributes:
        vehicle_model: YOLOv8 model for vehicle detection (COCO weights).
        plate_model:   YOLOv8 model for license plate localisation (custom weights).
        tracker:       SORT multi-object tracker instance.
        results:       Dict accumulating per-frame recognition data for CSV output.
        device:        Torch device string ('cuda' or 'cpu').
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.device  = self._select_device()
        self.vehicle_model = self._load_vehicle_model()
        self.plate_model   = self._load_plate_model()
        self.tracker = Sort(
            max_age=SORT_MAX_AGE,
            min_hits=SORT_MIN_HITS,
            iou_threshold=SORT_IOU_THRESHOLD,
        )

        # Pre-warm PaddleOCR reader (avoids first-frame latency spike)
        self._info("[OCR ] Warming up PaddleOCR reader...")
        from util import get_paddle_ocr_reader
        get_paddle_ocr_reader()
        self._info("[OCR ] Ready.\n")

        # Results dictionary:
        # { frame_nmr: { car_id: { 'car': {...}, 'license_plate': {...} } } }
        self.results = {}

        # Temporal voting: per-vehicle rolling buffer of recent plate reads.
        # Stores tuples of (text, confidence) for confidence-weighted voting.
        self.plate_history = defaultdict(lambda: deque(maxlen=PLATE_HISTORY_WINDOW))
        # Best-frame OCR per tracked vehicle (highest OCR confidence seen so far).
        self.best_plate_by_car = {}
        # Track-wise OCR throttle state.
        self.last_ocr_frame_by_car = defaultdict(lambda: -10_000)

    def _info(self, message: str) -> None:
        print(message)

    def _debug(self, message: str) -> None:
        if self.verbose:
            print(message)

    # ------------------------------------------------------------------
    # Temporal Voting — Majority Vote over Recent Frames
    # ------------------------------------------------------------------

    def _get_voted_plate(self, car_id: int) -> str | None:
        """
        Return confidence-weighted majority text for this car_id across
        recent frames. Returns None if no usable readings exist.
        """
        history = self.plate_history.get(car_id)
        if not history:
            return None

        valid = [(t, s) for t, s in history if t]
        if not valid:
            return None

        weighted = defaultdict(float)
        counts = Counter()
        for text, score in valid:
            # Floor confidence to keep very low-confidence reads from dominating.
            w = max(float(score), 0.05)
            weighted[text] += w
            counts[text] += 1

        # Primary key: summed confidence. Tie-breaker: frequency.
        return max(weighted.keys(), key=lambda t: (weighted[t], counts[t]))

    def _update_best_plate(self, car_id: int, text: str, conf: float, is_valid: bool) -> None:
        """Persist best OCR candidate for this track based on confidence and validity."""
        existing = self.best_plate_by_car.get(car_id)
        if existing is None:
            self.best_plate_by_car[car_id] = {
                'text': text,
                'conf': conf,
                'is_valid': is_valid,
            }
            return

        # Prefer valid over invalid. Otherwise, keep higher confidence.
        if is_valid and not existing.get('is_valid', False):
            self.best_plate_by_car[car_id] = {'text': text, 'conf': conf, 'is_valid': is_valid}
            return

        if conf > float(existing.get('conf', 0.0)):
            self.best_plate_by_car[car_id] = {'text': text, 'conf': conf, 'is_valid': is_valid}

    # ------------------------------------------------------------------
    # Initialisation Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_device() -> str:
        """Detect and report available compute device."""
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[GPU ] Device: {name} ({vram:.1f} GB VRAM)")
            return 'cuda'
        print("[WARN] No CUDA GPU detected — running on CPU (will be slow).")
        return 'cpu'

    def _load_vehicle_model(self) -> YOLO:
        """Load the vehicle detection model (COCO-pretrained YOLOv8)."""
        self._info(f"[MODEL] Loading vehicle detector: {VEHICLE_MODEL_PATH}")
        model = YOLO(VEHICLE_MODEL_PATH)
        if USE_HALF_PRECISION and self.device == 'cuda':
            model.model.half()  # Convert to FP16 for 4GB VRAM
        return model

    def _load_plate_model(self) -> YOLO:
        """
        Load the license plate localisation model.
        Falls back to COCO model if custom weights are not found.
        """
        plate_path = Path(PLATE_MODEL_PATH)
        if plate_path.exists():
            self._info(f"[MODEL] Loading plate localiser: {plate_path}")
            model = YOLO(str(plate_path))
        else:
            self._info(f"[WARN ] Custom plate model not found at '{plate_path}'.")
            self._info("         Run train.py first to generate it.")
            self._info("         Falling back to generic COCO model (reduced accuracy).")
            model = YOLO(VEHICLE_MODEL_PATH)

        if USE_HALF_PRECISION and self.device == 'cuda':
            model.model.half()
        return model

    # ------------------------------------------------------------------
    # Phase 1a — Vehicle Detection
    # ------------------------------------------------------------------

    def detect_vehicles(self, frame: np.ndarray) -> np.ndarray:
        """
        Run YOLOv8 vehicle detection on a single frame.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            np.array of shape (N, 5) — [[x1, y1, x2, y2, confidence], ...]
            containing only detections with class IDs in VEHICLE_CLASS_IDS.
        """
        detections = self.vehicle_model(
            frame,
            imgsz=INFERENCE_IMG_SIZE,
            conf=VEHICLE_CONF_THRESHOLD,
            device=self.device,
            verbose=False,
            half=USE_HALF_PRECISION,
        )[0]  # Returns a Results object for the single frame

        vehicles = []
        for box in detections.boxes:
            cls_id = int(box.cls[0])
            if cls_id in VEHICLE_CLASS_IDS:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                conf = float(box.conf[0])
                vehicles.append([x1, y1, x2, y2, conf])

        return np.array(vehicles) if vehicles else np.empty((0, 5))

    # ------------------------------------------------------------------
    # Phase 2 — SORT Tracking
    # ------------------------------------------------------------------

    def update_tracker(self, vehicle_detections: np.ndarray) -> np.ndarray:
        """
        Feed current vehicle detections into the SORT tracker.

        Args:
            vehicle_detections: np.array (N, 5) — [[x1, y1, x2, y2, score], ...]

        Returns:
            np.array (M, 5) — [[x1, y1, x2, y2, track_id], ...]
            Tracks that have been confirmed (met min_hits threshold).
        """
        return self.tracker.update(vehicle_detections)

    # ------------------------------------------------------------------
    # Phase 1b — Plate Localisation within Vehicle ROI
    # ------------------------------------------------------------------

    def detect_plates(self, frame: np.ndarray, vehicle_tracks: np.ndarray) -> list:
        """
        For each tracked vehicle, crop its bounding box region from the frame
        and run the plate detection model to find license plates.

        Restricting plate detection to each vehicle's ROI (region of interest):
          - Eliminates false positives from other parts of the frame.
          - Is much faster than running plate detection on the full frame.
          - Automatically associates each plate with the correct vehicle.

        Args:
            frame:          Full BGR video frame.
            vehicle_tracks: np.array (M, 5) from SORT — [[x1,y1,x2,y2,track_id],...]

        Returns:
            List of dicts, each containing:
              {
                'plate_bbox':  [x1, y1, x2, y2] in FULL frame coordinates,
                'plate_score': float,
                'car_bbox':    [x1, y1, x2, y2] of the parent vehicle,
                'car_id':      int track ID,
              }
        """
        plate_detections = []
        frame_h, frame_w = frame.shape[:2]

        for track in vehicle_tracks:
            vx1, vy1, vx2, vy2, car_id = track
            # Clamp to frame boundaries to avoid out-of-bounds crop
            vx1 = max(0, int(vx1))
            vy1 = max(0, int(vy1))
            vx2 = min(frame_w, int(vx2))
            vy2 = min(frame_h, int(vy2))

            if vx2 <= vx1 or vy2 <= vy1:
                continue  # Skip degenerate (zero-area) vehicle boxes

            # Crop the vehicle region from the frame
            vehicle_crop = frame[vy1:vy2, vx1:vx2]

            # Run plate detector on this crop
            # The crop is small, so upscaling it to 1024x1024 ruins accuracy.
            # Using 320 ensures features aren't grossly stretched.
            plate_results = self.plate_model(
                vehicle_crop,
                imgsz=320,  # Explicitly smaller than INFERENCE_IMG_SIZE
                conf=PLATE_CONF_THRESHOLD,
                device=self.device,
                verbose=False,
                half=USE_HALF_PRECISION,
            )[0]

            for box in plate_results.boxes:
                # Plate coordinates are relative to the vehicle crop —
                # convert back to full-frame coordinates by adding the crop offset.
                px1, py1, px2, py2 = box.xyxy[0].tolist()
                plate_detections.append({
                    'plate_bbox':  [
                        vx1 + px1, vy1 + py1,
                        vx1 + px2, vy1 + py2,
                    ],
                    'plate_score': float(box.conf[0]),
                    'car_bbox':    [vx1, vy1, vx2, vy2],
                    'car_id':      int(car_id),
                })

        return plate_detections

    # ------------------------------------------------------------------
    # Phase 3 — License Plate OCR
    # ------------------------------------------------------------------

    def run_ocr(self, frame: np.ndarray, plate_det: dict) -> dict | None:
        """
        Crop the license plate region and run OCR using PaddleOCR.

        Args:
            frame:     Full BGR video frame.
            plate_det: A single plate detection dict from detect_plates().

        Returns:
            Dict with keys:
              'raw_text', 'corrected_text', 'is_valid', 'ocr_confidence'
            Or None if OCR failed or confidence was below threshold.
        """
        px1, py1, px2, py2 = [int(v) for v in plate_det['plate_bbox']]
        plate_crop = frame[py1:py2, px1:px2]

        if plate_crop.size == 0:
            return None

        # Skip low-quality crops from motion blur/compression artifacts.
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_var < PLATE_BLUR_VAR_THRESHOLD:
            self._debug(
                f"[DEBUG] Skip blurry crop (var={blur_var:.1f} < {PLATE_BLUR_VAR_THRESHOLD})"
            )
            return None

        raw_text, ocr_conf = read_license_plate(plate_crop)

        if raw_text is None or ocr_conf is None:
            return None

        if ocr_conf < OCR_CONF_THRESHOLD:
            return None  # Low-confidence reading — discard to reduce noise

        # Phase 4: Post-process and validate
        corrected_text, is_valid = post_process_plate(raw_text)

        return {
            'raw_text':       raw_text,
            'corrected_text': corrected_text,
            'is_valid':       is_valid,
            'ocr_confidence': ocr_conf,
        }

    # ------------------------------------------------------------------
    # Phase 5a — Frame Annotation
    # ------------------------------------------------------------------

    def annotate_frame(
        self,
        frame: np.ndarray,
        vehicle_tracks: np.ndarray,
        plate_detections: list,
        frame_results: dict,
    ) -> np.ndarray:
        """
        Draw bounding boxes and verified plate text annotations on the frame.

        Args:
            frame:            BGR video frame.
            vehicle_tracks:   np.array (M, 5) from SORT.
            plate_detections: List of plate detection dicts.
            frame_results:    Dict of { car_id: { 'license_plate': {...} } }
                              for this frame (used to draw plate text).

        Returns:
            Annotated BGR frame.
        """
        annotated = frame.copy()

        # --- Draw vehicle bounding boxes ---
        for track in vehicle_tracks:
            x1, y1, x2, y2, car_id = [int(v) for v in track]

            # Corner-bracket style box for vehicles (green)
            draw_border(
                annotated,
                (x1, y1), (x2, y2),
                color=VEHICLE_BOX_COLOR,
                thickness=2,
                line_length_x=20, line_length_y=20,
            )

            # --- Draw recognized plate text only when verified valid ---
            car_data = frame_results.get(int(car_id), {})
            plate_info = car_data.get('license_plate', {})
            plate_text = plate_info.get('text', '')
            if plate_text and plate_info.get('is_valid'):
                display = plate_text
                (tw, th), _ = cv2.getTextSize(display, FONT, FONT_SCALE + 0.1, FONT_THICKNESS)
                # Draw a semi-transparent label background below the vehicle box
                overlay = annotated.copy()
                cv2.rectangle(overlay, (x1, y2 + 2), (x1 + tw + 8, y2 + th + 10),
                              (30, 30, 30), -1)
                cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
                cv2.putText(annotated, display, (x1 + 4, y2 + th + 4),
                            FONT, FONT_SCALE + 0.1, (0, 255, 255), FONT_THICKNESS)

        # --- Draw license plate bounding boxes (red) ---
        for pd in plate_detections:
            px1, py1, px2, py2 = [int(v) for v in pd['plate_bbox']]
            cv2.rectangle(annotated, (px1, py1), (px2, py2), PLATE_BOX_COLOR, 2)

        return annotated

    # ------------------------------------------------------------------
    # Phase 5b — Accumulate Results for CSV
    # ------------------------------------------------------------------

    def record_result(
        self,
        frame_nmr: int,
        car_id: int,
        car_bbox: list,
        plate_det: dict,
        ocr_result: dict,
    ) -> None:
        """
        Store a single frame's recognition result in self.results for CSV output.
        Only the best (highest confidence) reading per vehicle per frame is kept.
        """
        if frame_nmr not in self.results:
            self.results[frame_nmr] = {}

        existing = self.results[frame_nmr].get(car_id, {})
        existing_conf = existing.get('license_plate', {}).get('text_score', 0.0)

        # Only update if the new reading has higher OCR confidence
        if ocr_result['ocr_confidence'] > existing_conf:
            self.results[frame_nmr][car_id] = {
                'car': {
                    'bbox': car_bbox,
                },
                'license_plate': {
                    'bbox':       plate_det['plate_bbox'],
                    'bbox_score': plate_det['plate_score'],
                    'text':       ocr_result['corrected_text'],
                    'raw_text':   ocr_result['raw_text'],
                    'text_score': ocr_result['ocr_confidence'],
                    'is_valid':   ocr_result['is_valid'],
                },
            }

    # ------------------------------------------------------------------
    # Main Processing Loop
    # ------------------------------------------------------------------

    def run(self, input_path: str, output_path: str, csv_path: str) -> None:
        """
        Process the entire input video through the ANPR pipeline.

        Args:
            input_path:  Path to the input .mp4 video file.
            output_path: Path to write the annotated output video.
            csv_path:    Path to write the CSV recognition log.
        """
        # --- Open Input Video ---
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            self._info(f"[ERROR] Cannot open video: {input_path}")
            sys.exit(1)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._info(f"[VIDEO] Input:     {input_path}")
        self._info(f"[VIDEO] Resolution:{frame_w}x{frame_h} @ {fps:.1f} FPS")
        self._info(f"[VIDEO] Frames:    {total_frames}")

        # --- Set up Output Video Writer ---
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

        # --- Set up Raw CSV Writer (written frame-by-frame for crash safety) ---
        raw_csv_path = csv_path.replace('.csv', '_raw.csv')
        csv_file = open(raw_csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'frame_nmr', 'car_id', 'car_bbox',
            'license_plate_bbox', 'license_plate_bbox_score',
            'license_number', 'license_number_score',
        ])

        # ==============================================================
        # Main Frame Loop
        # ==============================================================
        frame_nmr = 0
        start_time = time.time()

        self._info("\n[RUN ] Starting inference...\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            frame_nmr += 1

            # --- Phase 1a: Detect Vehicles ---
            vehicle_detections = self.detect_vehicles(frame)

            # --- Phase 2: SORT Tracking ---
            # SORT assigns/maintains unique IDs across frames.
            # vehicle_tracks: [[x1, y1, x2, y2, track_id], ...]
            vehicle_tracks = self.update_tracker(vehicle_detections)

            # --- Phase 1b: Detect License Plates within each vehicle ROI ---
            plate_detections = self.detect_plates(frame, vehicle_tracks)
            
            if frame_nmr == 1:
                self._debug(
                    f"[DEBUG] Frame {frame_nmr}: {len(vehicle_tracks)} vehicles, {len(plate_detections)} plates detected"
                )

            # Per-frame results (for annotation)
            frame_results = {}

            ocr_calls_this_frame = 0
            # Process highest-confidence plate detections first when crowded.
            plate_detections_sorted = sorted(
                plate_detections,
                key=lambda d: float(d.get('plate_score', 0.0)),
                reverse=True,
            )

            for plate_det in plate_detections_sorted:
                car_id = int(plate_det['car_id'])

                if ocr_calls_this_frame >= MAX_OCR_CALLS_PER_FRAME:
                    continue

                if (frame_nmr - self.last_ocr_frame_by_car[car_id]) < OCR_MIN_FRAME_GAP:
                    continue

                # --- Phase 3: OCR ---
                ocr_result = self.run_ocr(frame, plate_det)
                self.last_ocr_frame_by_car[car_id] = frame_nmr
                ocr_calls_this_frame += 1
                # Skip only when OCR itself failed
                if ocr_result is None:
                    self.plate_history[plate_det['car_id']].append((None, 0.0))
                    continue
                # Optionally skip invalid plates (per REQUIRE_VALID_PLATE config)
                if REQUIRE_VALID_PLATE and not ocr_result['is_valid']:
                    self.plate_history[plate_det['car_id']].append((None, 0.0))
                    continue

                car_id = plate_det['car_id']
                car_bbox = plate_det['car_bbox']

                # --- Phase 4: Already applied inside run_ocr → post_process_plate ---
                # Choose display text: corrected when valid, otherwise raw OCR fallback
                display_text = (
                    ocr_result['corrected_text']
                    if ocr_result['is_valid']
                    else ocr_result['raw_text']
                )

                # Push reading into temporal voting buffer and update best-frame memory.
                self.plate_history[car_id].append((display_text, ocr_result['ocr_confidence']))
                self._update_best_plate(
                    car_id=car_id,
                    text=display_text,
                    conf=ocr_result['ocr_confidence'],
                    is_valid=ocr_result['is_valid'],
                )

                voted_text = self._get_voted_plate(car_id)
                stable_text = voted_text or display_text

                # Accumulate for final smoothed CSV (keep strict "valid-only" record behavior)
                if ocr_result['is_valid']:
                    self.record_result(frame_nmr, car_id, car_bbox, plate_det, ocr_result)

                # Store in per-frame results for annotation
                frame_results[car_id] = {
                    'license_plate': {
                        'text':     stable_text,
                        'is_valid': ocr_result['is_valid'],
                        'bbox':     plate_det['plate_bbox'],
                        'bbox_score': plate_det['plate_score'],
                        'text_score': ocr_result['ocr_confidence'],
                    }
                }

            # --- Anti-Vanishing: write ALL tracked vehicles to CSV ---
            # Ensures SciPy interpolation has vehicle bbox data even when OCR fails.
            for track in vehicle_tracks:
                car_id = int(track[4])
                car_bbox = [int(v) for v in track[:4]]
                ocr_data = frame_results.get(car_id, {})
                plate_info = ocr_data.get('license_plate', {})

                # Use temporal voting for display text
                voted_text = self._get_voted_plate(car_id)
                best_text = self.best_plate_by_car.get(car_id, {}).get('text')
                fallback_text = voted_text or best_text
                if fallback_text and car_id not in frame_results:
                    corrected_fb, is_valid_fb = post_process_plate(str(fallback_text))
                    frame_results[car_id] = {
                        'license_plate': {'text': corrected_fb, 'is_valid': is_valid_fb}
                    }

                csv_writer.writerow([
                    frame_nmr,
                    car_id,
                    car_bbox,
                    plate_info.get('bbox', ''),
                    plate_info.get('bbox_score', ''),
                    plate_info.get('text', ''),
                    plate_info.get('text_score', ''),
                ])

            # --- Phase 5a: Annotate and Write Output Frame ---
            annotated_frame = self.annotate_frame(
                frame, vehicle_tracks, plate_detections, frame_results
            )

            # Overlay: frame counter and FPS
            elapsed = time.time() - start_time
            cur_fps = frame_nmr / elapsed if elapsed > 0 else 0
            progress_text = (
                f"Frame {frame_nmr}/{total_frames} | "
                f"FPS: {cur_fps:.1f} | "
                f"Vehicles: {len(vehicle_tracks)} | "
                f"Plates: {len(plate_detections)}"
            )
            cv2.putText(
                annotated_frame, progress_text,
                (10, 30), FONT, 0.6, (0, 255, 255), 2
            )

            writer.write(annotated_frame)

            # Console progress (enabled by default, with extra detail in verbose mode)
            show_progress = frame_nmr % PROGRESS_LOG_EVERY_N_FRAMES == 0
            if show_progress:
                pct = (frame_nmr / total_frames) * 100 if total_frames > 0 else 0
                self._info(
                    f"  [{pct:5.1f}%] Frame {frame_nmr}/{total_frames} | "
                    f"FPS {cur_fps:.1f} | Vehicles {len(vehicle_tracks)} | "
                    f"Plates {len(plate_detections)}"
                )

        # ==============================================================
        # Cleanup
        # ==============================================================
        cap.release()
        writer.release()
        csv_file.close()

        total_time = time.time() - start_time
        self._info(
            f"\n[DONE] Processed {frame_nmr} frames in {total_time:.1f}s "
            f"({frame_nmr / total_time:.1f} FPS average)."
        )
        self._info(f"[OUT ] Annotated video  -> {output_path}")
        self._info(f"[OUT ] Raw CSV log      -> {raw_csv_path}")

        # ==============================================================
        # Phase 3 (Smoothing): SciPy Interpolation Post-Processing
        # ==============================================================
        # Read raw CSV back and apply temporal interpolation to fill
        # gaps where plates were missed due to blur / occlusion.
        self._info("\n[INTERP] Applying SciPy interpolation for missed detections...")

        raw_rows = []
        with open(raw_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_rows.append(row)

        if raw_rows:
            smoothed_rows = interpolate_bounding_boxes(
                raw_rows,
                smooth_car_bbox_window=CAR_BBOX_SMOOTH_WINDOW,
                min_track_frames=MIN_TRACK_FRAMES_FOR_OUTPUT,
            )

            # Write the final, smoothed CSV
            with open(csv_path, 'w', newline='') as f:
                fieldnames = [
                    'frame_nmr', 'car_id', 'car_bbox',
                    'license_plate_bbox', 'license_plate_bbox_score',
                    'license_number', 'license_number_score',
                ]
                writer_final = csv.DictWriter(f, fieldnames=fieldnames)
                writer_final.writeheader()
                for row in smoothed_rows:
                    # Ensure all required keys exist (interpolated rows may lack some)
                    writer_final.writerow({k: row.get(k, '') for k in fieldnames})

            self._info(f"[OUT ] Smoothed CSV log -> {csv_path}")
            self._info(f"       ({len(raw_rows)} raw -> {len(smoothed_rows)} interpolated rows)")
        else:
            self._info("[WARN ] No plate data detected - skipping interpolation.")

        self._info("\n[OK ] Pipeline complete.")


# ==============================================================================
# ░░░░░░░░░░░░░░░░  VIDEO OVERLAY (Post-Processing Pass)  ░░░░░░░░░░░░░░░
# ==============================================================================

def render_final_video(
    input_path: str,
    output_path: str,
    csv_path: str,
    final_output_path: str,
) -> None:
    """
    Optional second pass: re-render the output video using the interpolated
    CSV data for smoother, more complete annotations.

    This overlays the best recognised plate text (after smoothing) for
    all frames, including those where OCR originally failed.

    Args:
        input_path:        Original source video.
        output_path:       First-pass annotated video (from pipeline run).
        csv_path:          Smoothed CSV from interpolation step.
        final_output_path: Path to write the final polished video.
    """
    import pandas as pd

    # Load the smoothed CSV data indexed by frame number
    df = pd.read_csv(csv_path)

    # For each car_id, find its single best (highest-confidence) VERIFIED plate reading.
    # This "best plate" will be displayed consistently across all frames.
    # Filter to rows with actual OCR data (anti-vanishing rows have empty scores).
    best_plates = {}
    for car_id, group in df.groupby('car_id'):
        scores = pd.to_numeric(group['license_number_score'], errors='coerce')
        valid_mask = scores.notna() & group['license_number'].notna() & (group['license_number'].astype(str).str.strip() != '')
        scored = group[valid_mask].copy()
        if not scored.empty:
            scored['verified_plate'] = scored['license_number'].astype(str).map(
                lambda t: post_process_plate(t)[0] if post_process_plate(t)[1] else ''
            )
            scored = scored[scored['verified_plate'].astype(str).str.strip() != '']
        if scored.empty:
            continue
        best_idx = pd.to_numeric(scored['license_number_score'], errors='coerce').idxmax()
        best_row = scored.loc[best_idx]
        best_plates[int(car_id)] = {
            'text':  best_row['verified_plate'],
            'score': float(best_row['license_number_score']),
        }

    # Build a per-frame lookup: { frame_nmr: { car_id: { car_bbox, plate_bbox, text } } }
    frame_lookup = {}
    for _, row in df.iterrows():
        fnr    = int(row['frame_nmr'])
        car_id = int(row['car_id'])
        if fnr not in frame_lookup:
            frame_lookup[fnr] = {}

        # Parse bboxes from string representations
        def parse_bbox_str(s):
            import re
            nums = re.findall(r'[\d.]+', str(s))
            return [float(n) for n in nums] if len(nums) == 4 else None

        frame_lookup[fnr][car_id] = {
            'car_bbox':   parse_bbox_str(row.get('car_bbox', '')),
            'plate_bbox': parse_bbox_str(row.get('license_plate_bbox', '')),
            'text':       best_plates.get(car_id, {}).get('text', ''),
        }

    # Re-render
    cap    = cv2.VideoCapture(input_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fw     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(final_output_path, fourcc, fps, (fw, fh))

    frame_nmr = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_nmr += 1

        # Draw all tracked vehicles and plates from the CSV for this frame
        frame_data = frame_lookup.get(frame_nmr, {})
        for car_id, info in frame_data.items():
            car_bbox   = info.get('car_bbox')
            plate_bbox = info.get('plate_bbox')
            text       = info.get('text', '')

            if car_bbox:
                x1, y1, x2, y2 = [int(v) for v in car_bbox]
                draw_border(frame, (x1, y1), (x2, y2), color=VEHICLE_BOX_COLOR,
                            thickness=2, line_length_x=20, line_length_y=20)

            if plate_bbox:
                px1, py1, px2, py2 = [int(v) for v in plate_bbox]
                cv2.rectangle(frame, (px1, py1), (px2, py2), PLATE_BOX_COLOR, 2)

            if text and car_bbox:
                x1, _, _, y2 = [int(v) for v in car_bbox]
                cv2.putText(frame, text, (x1, y2 + 22),
                            FONT, FONT_SCALE + 0.1, (0, 255, 255), FONT_THICKNESS)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"[OUT ] Final polished video -> {final_output_path}")


# ==============================================================================
# ░░░░░░░░░░░░░░░░  CLI Entry Point  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='ANPR Pipeline — Automatic Number Plate Recognition'
    )
    parser.add_argument(
        '--source', type=str, default=INPUT_VIDEO,
        help='Path to input video file (default: %(default)s)'
    )
    parser.add_argument(
        '--output', type=str, default=OUTPUT_VIDEO,
        help='Path for annotated output video (default: %(default)s)'
    )
    parser.add_argument(
        '--csv', type=str, default=OUTPUT_CSV,
        help='Path for CSV recognition log (default: %(default)s)'
    )
    parser.add_argument(
        '--final', type=str, default='final_output.mp4',
        help='Path for final polished video after interpolation (default: %(default)s)'
    )
    parser.add_argument(
        '--skip-final-render', action='store_true',
        help='Skip the optional second-pass video render'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose per-frame progress logs'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n" + "="*65)
    print("  ANPR — Automatic Number Plate Recognition Pipeline")
    print("="*65)

    # ---- Run main pipeline ----
    pipeline = ANPRPipeline(verbose=args.verbose)
    pipeline.run(
        input_path=args.source,
        output_path=args.output,
        csv_path=args.csv,
    )

    # ---- Optional: re-render with smoothed annotations ----
    if not args.skip_final_render and Path(args.csv).exists():
        print(f"\n[POST] Rendering final video with smoothed annotations...")
        render_final_video(
            input_path=args.source,
            output_path=args.output,
            csv_path=args.csv,
            final_output_path=args.final,
        )

    print("\n  All done! Check the output files:")
    print(f"    Annotated video:           {args.output}")
    print(f"    Raw CSV log:               {args.csv.replace('.csv', '_raw.csv')}")
    print(f"    Smoothed CSV:              {args.csv}")
    if not args.skip_final_render:
        print(f"    Final polished video:      {args.final}")
    print()


if __name__ == '__main__':
    main()
