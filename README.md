# ANPR: Automatic Number Plate Recognition Pipeline

End-to-end ANPR for video files using YOLOv8 + SORT + PaddleOCR, tuned for Indian license plate formats.

This repository includes:
- Inference pipeline with tracking, OCR stabilization, CSV outputs, and annotated video generation.
- Dataset builder that merges Hugging Face and Roboflow Indian YOLO datasets.
- YOLO training/fine-tuning script for the plate detector.
- FastAPI web app with background jobs and browser-safe video re-encoding.
- Google Colab notebook for GPU-based dataset/training workflows.
- Local setup bootstrap script for one-command environment creation.

## 1) Local Setup (Recommended First Step)

Run the bootstrap script:

```bash
python setup_local.py
```

What it does:
- Creates `venv` (or reuses it if present).
- Upgrades `pip`, `setuptools`, `wheel`.
- Installs `requirements.txt`.
- Runs a smoke import check for core modules.

Optional flags:

```bash
python setup_local.py --venv-dir .venv
python setup_local.py --python C:/Path/To/python.exe
```

After setup, activate your virtual environment:

- Windows PowerShell: `venv\Scripts\Activate.ps1`
- Linux/macOS: `source venv/bin/activate`

## 2) Local File-Based Workflow (Dataset -> Train -> Run)

### 2.1 Build dataset

Default (HF + Roboflow if key exists):

```bash
python download_dataset.py
```

If `ROBOFLOW_API_KEY` is not set, local interactive runs will prompt for it.
Leave it blank to continue with Hugging Face source only.

With explicit Roboflow key:

```bash
# PowerShell
$env:ROBOFLOW_API_KEY="your_key_here"
python download_dataset.py --roboflow-version 1
```

Useful flags:
- `--no-hf`
- `--no-roboflow`
- `--keep-temp`

Output dataset:
- `data/license_plates/images/train|val|test`
- `data/license_plates/labels/train|val|test`
- `data/license_plates/data.yaml`

### 2.2 Train plate detector

```bash
python train.py
```

Fine-tune behavior:
- If `runs/detect/license_plate_detector/weights/best.pt` exists, training resumes from it.
- Otherwise training starts from `yolov8n.pt`.

Override start weights:

```bash
python train.py --weights runs/detect/license_plate_detector/weights/best.pt
```

### 2.3 Run inference pipeline

```bash
python pipeline.py --source input.mp4 --output output.mp4 --csv results.csv
```

Useful runtime options:

```bash
python pipeline.py --source input.mp4 --verbose
python pipeline.py --source input.mp4 --skip-final-render
```

### 2.4 Run local web app

```bash
python main.py
```

Open: `http://localhost:8000`

## 3) Colab GPU Workflow

Use notebook: [ANPR_Colab.ipynb](ANPR_Colab.ipynb)

Notebook cell order:
1. Cell 1: Overview.
2. Cell 2: Repo detection/clone + dependency install.
3. Cell 3: Roboflow key instructions.
4. Cell 4: Prompt for `ROBOFLOW_API_KEY` (optional).
5. Cell 5: Dataset build from notebook.
6. Cell 6: Training notes.
7. Cell 7: Optional training (`TRAIN_IF_NEEDED = False` by default).
8. Cell 8: Export artifacts zip for local laptop use.

Recommended split:
- Colab GPU: dataset building + optional training.
- Local machine: inference + web app execution.

After artifact export from Colab:
1. Download zip from Colab.
2. Copy trained weights to `runs/detect/license_plate_detector/weights/best.pt` locally.
3. Run local app with `python main.py`.

## 4) Key Runtime Controls

Primary tuning constants are in [anpr/pipeline_core.py](anpr/pipeline_core.py):
- Detection thresholds: `VEHICLE_CONF_THRESHOLD`, `PLATE_CONF_THRESHOLD`, `OCR_CONF_THRESHOLD`
- OCR speed/quality: `PLATE_BLUR_VAR_THRESHOLD`, `OCR_MIN_FRAME_GAP`, `MAX_OCR_CALLS_PER_FRAME`
- Stabilization: `CAR_BBOX_SMOOTH_WINDOW`, `MIN_TRACK_FRAMES_FOR_OUTPUT`
- Throughput: `INFERENCE_IMG_SIZE`

## 5) Outputs

- `output.mp4`: first-pass annotated video
- `results_raw.csv`: per-frame raw OCR/tracking rows
- `results.csv`: smoothed/interpolated rows
- `final_output.mp4`: optional polished render

## 6) Troubleshooting Notes

- If FPS drops, lower `INFERENCE_IMG_SIZE` and/or reduce OCR frequency with `OCR_MIN_FRAME_GAP`.
- If OCR is unstable on Windows GPU, CPU fallback is typically more reliable for PaddleOCR.
- If browser playback fails in web UI, ensure `ffmpeg` is available. The app also attempts `imageio-ffmpeg` fallback.

## 7) Project Map

- [setup_local.py](setup_local.py): local bootstrap (venv + install + smoke check)
- [download_dataset.py](download_dataset.py): dataset download and merge
- [train.py](train.py): YOLO plate detector training/fine-tuning
- [pipeline.py](pipeline.py): root CLI entrypoint
- [anpr/pipeline_core.py](anpr/pipeline_core.py): core ANPR runtime
- [main.py](main.py): web app server entrypoint
- [webapp/api.py](webapp/api.py): API routes
- [webapp/jobs.py](webapp/jobs.py): job lifecycle/progress/log streaming
- [webapp/video_codec.py](webapp/video_codec.py): web-safe H.264 conversion helper
- [util.py](util.py): OCR preprocessing and result post-processing helpers
- [sort.py](sort.py): tracker implementation

## 8) Architecture Docs

- Detailed technical internals: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
