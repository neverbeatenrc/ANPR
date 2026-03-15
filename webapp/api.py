import csv
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .config import FINAL_VIDEO_PATH, INDEX_PATH, NO_CACHE_HEADERS, RESULTS_CSV_PATH
from .jobs import JobManager


def _read_results_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with RESULTS_CSV_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_raw = str(row.get("frame_nmr", "")).strip()
            car_raw = str(row.get("car_id", "")).strip()
            number_raw = str(row.get("license_number", "")).strip()
            score_raw = str(row.get("license_number_score", "")).strip()

            if not frame_raw or not car_raw:
                continue

            try:
                frame_nmr = int(float(frame_raw))
                car_id = int(float(car_raw))
            except ValueError:
                continue

            try:
                license_number_score = float(score_raw) if score_raw else None
            except ValueError:
                license_number_score = None

            rows.append(
                {
                    "frame_nmr": frame_nmr,
                    "car_id": car_id,
                    "license_number": number_raw,
                    "license_number_score": license_number_score,
                }
            )
    return rows


def create_app() -> FastAPI:
    app = FastAPI(title="ANPR Web API", version="1.1.0")
    jobs = JobManager()

    @app.get("/")
    def index() -> FileResponse:
        if not INDEX_PATH.exists():
            raise HTTPException(status_code=404, detail="index.html not found")
        return FileResponse(INDEX_PATH, headers=NO_CACHE_HEADERS)

    @app.post("/upload")
    async def upload_video(video: UploadFile = File(...)) -> dict[str, Any]:
        if not video.filename:
            raise HTTPException(status_code=400, detail="Missing filename")
        if not video.filename.lower().endswith(".mp4"):
            raise HTTPException(status_code=400, detail="Only .mp4 uploads are supported")

        content = await video.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        job = jobs.create_job()
        jobs.start(job["run_id"], content)

        return {
            "status": "queued",
            "run_id": job["run_id"],
            "status_url": f"/jobs/{job['run_id']}",
            "video_url": "/video/final",
            "results_url": "/results",
        }

    @app.get("/jobs/{run_id}")
    def get_job_status(run_id: str) -> JSONResponse:
        job = jobs.get_job(run_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown run_id")
        return JSONResponse(content=job, headers=NO_CACHE_HEADERS)

    @app.get("/video/final")
    def get_final_video() -> FileResponse:
        if not FINAL_VIDEO_PATH.exists():
            raise HTTPException(status_code=404, detail="final_output.mp4 not found")
        return FileResponse(
            FINAL_VIDEO_PATH,
            media_type="video/mp4",
            filename="final_output.mp4",
            headers=NO_CACHE_HEADERS,
        )

    @app.get("/results")
    def get_results() -> JSONResponse:
        if not RESULTS_CSV_PATH.exists():
            raise HTTPException(status_code=404, detail="results.csv not found")
        rows = _read_results_rows()
        return JSONResponse(content={"rows": rows, "count": len(rows)}, headers=NO_CACHE_HEADERS)

    return app
