"""
Local bootstrap script for ANPR.

Creates a virtual environment, installs dependencies, and validates core imports
needed to run dataset download, training, pipeline, and web app locally.

Usage:
  python setup_local.py
  python setup_local.py --venv-dir .venv
  python setup_local.py --python C:/Path/To/python.exe
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
REQUIREMENTS = REPO_ROOT / "requirements.txt"


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print("[RUN ]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def venv_pip_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "pip.exe"
    return venv_dir / "bin" / "pip"


def ensure_venv(venv_dir: Path, base_python: str) -> tuple[Path, Path]:
    py_path = venv_python_path(venv_dir)
    pip_path = venv_pip_path(venv_dir)

    if py_path.exists() and pip_path.exists():
        print(f"[INFO] Reusing existing virtual environment: {venv_dir}")
        return py_path, pip_path

    print(f"[INFO] Creating virtual environment at: {venv_dir}")
    run_cmd([base_python, "-m", "venv", str(venv_dir)], cwd=REPO_ROOT)

    if not py_path.exists() or not pip_path.exists():
        raise FileNotFoundError("Virtual environment created, but python/pip were not found.")

    return py_path, pip_path


def install_dependencies(py_exe: Path) -> None:
    if not REQUIREMENTS.exists():
        raise FileNotFoundError(f"requirements.txt not found at {REQUIREMENTS}")

    run_cmd([str(py_exe), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run_cmd([str(py_exe), "-m", "pip", "install", "-r", str(REQUIREMENTS)], cwd=REPO_ROOT)


def validate_runtime(py_exe: Path) -> None:
    smoke = (
        "import cv2, torch, ultralytics, fastapi, pandas;"
        "print('imports-ok')"
    )
    run_cmd([str(py_exe), "-c", smoke], cwd=REPO_ROOT)


def print_next_steps(venv_dir: Path, py_exe: Path) -> None:
    print("\n" + "=" * 72)
    print("Setup complete.")
    print("=" * 72)

    if os.name == "nt":
        activate_cmd = f"{venv_dir}\\Scripts\\Activate.ps1"
        print(f"Activate (PowerShell): {activate_cmd}")
    else:
        activate_cmd = f"source {venv_dir}/bin/activate"
        print(f"Activate (bash/zsh): {activate_cmd}")

    print(f"Python in venv: {py_exe}")
    print("\nRun locally:")
    print("1) Dataset build: python download_dataset.py")
    print("2) Training:      python train.py")
    print("3) Inference:     python pipeline.py --source input.mp4 --output output.mp4 --csv results.csv")
    print("4) Web app:       python main.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap local ANPR environment")
    parser.add_argument(
        "--venv-dir",
        default="venv",
        help="Virtual environment directory (default: venv)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Base python executable used to create the virtual environment",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    venv_dir = (REPO_ROOT / args.venv_dir).resolve()

    py_exe, _ = ensure_venv(venv_dir=venv_dir, base_python=args.python)
    install_dependencies(py_exe)
    validate_runtime(py_exe)
    print_next_steps(venv_dir, py_exe)


if __name__ == "__main__":
    main()
