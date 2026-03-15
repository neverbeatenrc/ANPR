# Architecture & Technical Implementation

This document maps the current repository implementation to the runtime pipeline, helper modules, and data artifacts.

## 1. End-to-End Frame Flow

For each video frame, pipeline execution in [anpr/pipeline_core.py](anpr/pipeline_core.py) follows:

1. Vehicle detection (YOLOv8) on full frame.
2. Vehicle tracking (SORT) to assign stable track IDs.
3. Plate detection (YOLOv8) inside each tracked vehicle ROI.
4. OCR candidate selection with quality gates:
   - blur rejection
   - per-track OCR frame-gap throttle
   - per-frame OCR budget cap
5. OCR and text cleanup in [util.py](util.py):
   - preprocessing ensemble
   - IND-strip cleanup
   - positional correction + regex validation
6. Temporal text stabilization:
   - confidence-weighted voting in rolling history
   - best-read fallback memory per track
7. Annotation + raw CSV row emission.
8. Post-loop interpolation to build smoothed CSV.
9. Optional second-pass polished video render from smoothed CSV.

## 2. Module Responsibilities

### pipeline.py

- Thin root CLI wrapper that calls modular pipeline package entrypoint.

### anpr/pipeline_core.py

- Owns runtime orchestration through the `ANPRPipeline` class.
- Loads and executes vehicle and plate detectors.
- Applies tracker update cycle.
- Applies OCR scheduling controls:
  - OCR_MIN_FRAME_GAP
  - MAX_OCR_CALLS_PER_FRAME
- Maintains temporal memory structures:
  - plate_history (confidence-weighted votes)
  - best_plate_by_car (fallback text)
- Writes first-pass annotated video + raw CSV.
- Runs interpolation and optional polished render.
- Exposes `main()` for CLI entry.

### anpr/cli.py

- Package-level CLI shim for invoking `anpr.pipeline_core.main()`.

### util.py

- Initializes PaddleOCR reader with API compatibility fallback across versions.
- Suppresses noisy OCR dependency logs.
- Handles OCR output normalization for legacy and newer PaddleOCR response shapes.
- Implements preprocessing ensemble for OCR:
  - contrast/denoise/sharpen base preprocessing
  - adaptive threshold variants
- Implements Indian-plate post-processing:
  - IND-prefix noise stripping
  - positional correction with template-aware character swaps
  - regex validation
- Provides drawing, track association, CSV writing, and interpolation helpers.

### sort.py

- Provides SORT tracker (Kalman + assignment) used by main loop.

### train.py

- Fine-tunes plate localizer.
- Auto-resumes from prior best checkpoint when available.
- Runs validation and emits model artifacts/plots.

### download_dataset.py

- Builds merged YOLO dataset from:
  - Hugging Face keremberke license plate dataset
  - Indian Roboflow dataset
- Includes fallback Roboflow export path when SDK download returns empty content.
- Uses `requests` fallback export flow when Roboflow SDK path is unavailable.

### setup_local.py

- Local bootstrap utility for file-based execution.
- Creates/reuses virtual environment.
- Installs dependencies from `requirements.txt`.
- Runs import smoke checks to validate local runtime.

### webapp/api.py

- Defines FastAPI routes for upload/status/results/video/index.
- Uses an app-factory pattern (`create_app`) so server boot and app composition are decoupled.
- Delegates heavy processing to `JobManager` instead of embedding subprocess logic in routes.

### webapp/jobs.py

- Owns background job lifecycle and in-memory state for `/jobs/{run_id}`.
- Serializes processing with an async lock.
- Streams pipeline stdout into bounded job logs.
- Extracts progress percentages from pipeline output.
- Performs stale-artifact cleanup before each run.
- Runs browser-safe video re-encode after successful pipeline completion.

### webapp/video_codec.py

- Resolves ffmpeg binary (system path or `imageio_ffmpeg`).
- Re-encodes `final_output.mp4` to H.264 + yuv420p + faststart for consistent browser playback.

### webapp/config.py

- Central location for backend paths/constants and no-cache headers.

### main.py

- Thin composition root: creates app and runs uvicorn server.

## 3. Key Runtime Data Structures

Per-frame in-memory recognition structure:

```json
{
  "frame_nmr": 124,
  "car_id": 5,
  "car": {
    "bbox": [150.0, 300.0, 450.0, 500.0]
  },
  "license_plate": {
    "bbox": [280.0, 450.0, 350.0, 480.0],
    "bbox_score": 0.895,
    "text": "MH12AB1234",
    "raw_text": "MH12A81234",
    "text_score": 0.94,
    "is_valid": true
  }
}
```

Temporal state:

- plate_history[car_id]: deque of (text, confidence)
- best_plate_by_car[car_id]: best known {text, conf, is_valid}
- last_ocr_frame_by_car[car_id]: last frame OCR was attempted for throttling

## 4. OCR Processing Details

### 4.1 Preprocessing Ensemble

OCR runs against multiple variants of each plate crop:

1. Base enhanced grayscale.
2. Adaptive threshold (binary).
3. Adaptive threshold (inverse).

Best candidate is selected using confidence + validity-biased scoring.

### 4.2 Text Cleanup and Validation

Cleanup and normalization steps in order:

1. Remove non-alphanumeric artifacts.
2. Strip common IND-prefix OCR noise variants.
3. Enforce length guardrails.
4. Apply position-aware character correction using templates.
5. Validate against Indian plate regex patterns.

### 4.3 PaddleOCR Compatibility

The code supports:

- constructor argument differences across PaddleOCR versions
- ocr invocation signature changes
- legacy and new output payload formats

## 5. Throughput Controls

To prevent severe FPS collapse in dense traffic scenes, current implementation includes:

1. Detection-size control (INFERENCE_IMG_SIZE).
2. Per-track OCR frame spacing (OCR_MIN_FRAME_GAP).
3. Per-frame OCR call budget (MAX_OCR_CALLS_PER_FRAME).
4. Blur gating before OCR (PLATE_BLUR_VAR_THRESHOLD).

These controls trade some per-frame OCR coverage for stable throughput and are compensated by temporal voting.

Default values currently configured:

- INFERENCE_IMG_SIZE = 640
- OCR_MIN_FRAME_GAP = 2
- MAX_OCR_CALLS_PER_FRAME = 2
- PLATE_BLUR_VAR_THRESHOLD = 80.0

## 6. Outputs and Post-Processing

Generated outputs:

- output.mp4: first-pass annotated output.
- results_raw.csv: raw per-frame rows.
- results.csv: smoothed/interpolated rows.
- final_output.mp4: optional polished render from smoothed CSV.

Interpolation in [util.py](util.py):

- Fills missing plate boxes between anchor detections.
- Applies max-gap limits to avoid unrealistic long-gap interpolation.
- Keeps best-confidence text when synthesizing intermediate rows.
- Optionally smooths `car_bbox` coordinates with centered moving average.
- Optionally drops short-lived tracks (`min_track_frames`) to suppress ghost IDs.

## 7. Known Runtime Constraints

- GPU OCR on Windows may fail in mixed Torch + Paddle GPU environments due to framework/DLL conflicts.
- CPU OCR remains stable fallback for mixed-stack runtime.
- For high-throughput production, separate-process OCR or Linux/WSL deployment is often easier to stabilize.

## 8. Web Backend Runtime Notes

- `/upload` enqueues work and returns immediately with `run_id`.
- Frontend polls `/jobs/{run_id}` for progress and live logs.
- Final outputs are served with no-cache headers to avoid stale browser artifacts.
- Uvicorn access logs are disabled by default to keep terminal output focused on actionable errors.

## 9. Deployment Topology (Recommended)

- Google Colab: dataset build + optional YOLO training + artifact export only.
- Laptop/desktop: run FastAPI/web app locally via [main.py](main.py).
- Artifact handoff: copy/export trained plate detector weights to local repository path `runs/detect/license_plate_detector/weights/best.pt` (or update `PLATE_MODEL_PATH` in [anpr/pipeline_core.py](anpr/pipeline_core.py)).
- No public tunnel is required for the recommended setup because UI and API are served on local `http://localhost:8000`.

## 10. Execution Modes

### 10.1 Local file-based mode

Run everything locally from scripts:

1. `python setup_local.py`
2. `python download_dataset.py`
3. `python train.py` (optional if reusing existing weights)
4. `python pipeline.py --source input.mp4 --output output.mp4 --csv results.csv`
5. `python main.py`

### 10.2 Colab GPU + local app mode

1. Run `ANPR_Colab.ipynb` to install dependencies and build dataset.
2. Optionally train in Colab GPU.
3. Export weight artifacts from notebook.
4. Copy artifacts to local repository.
5. Run `python main.py` locally for production inference and UI.
