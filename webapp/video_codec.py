import shutil
import subprocess
from pathlib import Path

from .config import BASE_DIR


def find_ffmpeg_exe() -> str | None:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def reencode_for_web(path: Path) -> tuple[bool, str]:
    ffmpeg_exe = find_ffmpeg_exe()
    if not ffmpeg_exe:
        return False, "ffmpeg not found; kept original video encoding"

    tmp_path = path.with_name(f"{path.stem}.webtmp{path.suffix}")
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]

    proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not tmp_path.exists():
        detail = (proc.stderr or proc.stdout or "ffmpeg failed").strip()
        return False, f"web re-encode failed: {detail[-300:]}"

    tmp_path.replace(path)
    return True, "web re-encode successful (H.264 yuv420p)"
