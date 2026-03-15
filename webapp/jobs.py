import asyncio
import csv
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .config import (
    BASE_DIR,
    FINAL_VIDEO_PATH,
    INPUT_VIDEO_PATH,
    MAX_LOG_LINES,
    OUTPUT_VIDEO_PATH,
    PIPELINE_PATH,
    RESULTS_CSV_PATH,
    RESULTS_RAW_CSV_PATH,
)
from .video_codec import reencode_for_web


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    def create_job(self) -> dict[str, Any]:
        run_id = uuid.uuid4().hex[:12]
        job = {
            "run_id": run_id,
            "status": "queued",
            "progress": 0.0,
            "message": "Queued",
            "logs": [],
            "error": None,
            "created_at": time.time(),
            "duration_seconds": None,
            "rows": 0,
            "video_url": "/video/final",
            "results_url": "/results",
        }
        self.jobs[run_id] = job
        return job

    def get_job(self, run_id: str) -> dict[str, Any] | None:
        return self.jobs.get(run_id)

    def start(self, run_id: str, video_bytes: bytes) -> None:
        asyncio.create_task(self._process_job(run_id, video_bytes))

    @staticmethod
    def _append_log(job: dict[str, Any], line: str) -> None:
        line = line.rstrip("\n")
        if not line:
            return

        print(line, flush=True)

        logs = job["logs"]
        logs.append(line)
        if len(logs) > MAX_LOG_LINES:
            del logs[: len(logs) - MAX_LOG_LINES]

    @staticmethod
    def _extract_progress(line: str) -> float | None:
        match = re.search(r"\[\s*(\d+(?:\.\d+)?)%\]", line)
        if not match:
            return None
        try:
            return max(0.0, min(100.0, float(match.group(1))))
        except ValueError:
            return None

    @staticmethod
    def _remove_stale_outputs() -> None:
        for stale in (FINAL_VIDEO_PATH, RESULTS_CSV_PATH, RESULTS_RAW_CSV_PATH, OUTPUT_VIDEO_PATH):
            if stale.exists():
                stale.unlink()

    @staticmethod
    def _build_command() -> list[str]:
        return [
            sys.executable,
            "-u",
            str(PIPELINE_PATH),
            "--source",
            str(INPUT_VIDEO_PATH),
            "--output",
            str(OUTPUT_VIDEO_PATH),
            "--csv",
            str(RESULTS_CSV_PATH),
            "--final",
            str(FINAL_VIDEO_PATH),
        ]

    def _run_pipeline_and_stream(self, job: dict[str, Any], command: list[str]) -> int:
        proc = subprocess.Popen(
            command,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            self._append_log(job, line)
            progress = self._extract_progress(line)
            if progress is not None:
                job["progress"] = progress
                job["message"] = f"Processing... {progress:.1f}%"
        return proc.wait()

    @staticmethod
    def _count_results_rows(path: Path) -> int:
        count = 0
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for _ in reader:
                count += 1
        return count

    async def _process_job(self, run_id: str, video_bytes: bytes) -> None:
        job = self.jobs[run_id]
        job["status"] = "running"
        job["message"] = "Starting pipeline"

        async with self._lock:
            started_at = time.time()
            try:
                INPUT_VIDEO_PATH.write_bytes(video_bytes)
                self._remove_stale_outputs()

                command = self._build_command()
                return_code = await asyncio.to_thread(self._run_pipeline_and_stream, job, command)
                if return_code != 0:
                    job["status"] = "failed"
                    job["error"] = "Pipeline exited with non-zero status"
                    job["message"] = "Pipeline failed"
                    return

                if not FINAL_VIDEO_PATH.exists() or not RESULTS_CSV_PATH.exists():
                    job["status"] = "failed"
                    job["error"] = "Expected outputs not found"
                    job["message"] = "Pipeline failed"
                    return

                ok, msg = reencode_for_web(FINAL_VIDEO_PATH)
                self._append_log(job, f"[VIDEO] {msg}")
                if not ok:
                    self._append_log(job, "[VIDEO] Warning: browser playback may fail with original codec")

                job["rows"] = self._count_results_rows(RESULTS_CSV_PATH)
                job["progress"] = 100.0
                job["status"] = "completed"
                job["message"] = "Processing complete"
            except Exception as exc:
                job["status"] = "failed"
                job["error"] = str(exc)
                job["message"] = "Pipeline failed"
            finally:
                job["duration_seconds"] = round(time.time() - started_at, 2)
