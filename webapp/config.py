from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_VIDEO_PATH = BASE_DIR / "input.mp4"
OUTPUT_VIDEO_PATH = BASE_DIR / "output.mp4"
FINAL_VIDEO_PATH = BASE_DIR / "final_output.mp4"
RESULTS_CSV_PATH = BASE_DIR / "results.csv"
RESULTS_RAW_CSV_PATH = BASE_DIR / "results_raw.csv"
INDEX_PATH = BASE_DIR / "index.html"
PIPELINE_PATH = BASE_DIR / "pipeline.py"

MAX_LOG_LINES = 400

NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}
