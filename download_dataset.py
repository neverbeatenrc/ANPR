"""
Unified ANPR dataset downloader.

Downloads and merges:
1) Hugging Face dataset: keremberke/license-plate-object-detection
2) Indian Roboflow dataset: license-plate-detection-khhkb/indian-license-plate-detection-6tmbr

Output:
- data/license_plates/images/{train,val,test}
- data/license_plates/labels/{train,val,test}
- data/license_plates/data.yaml

Notes:
- Roboflow download needs an API key in ROBOFLOW_API_KEY (or --roboflow-api-key).
- If one source fails, the script still succeeds if at least one source is imported.
"""

import argparse
import shutil
import zipfile
import warnings
import os
import sys
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 .* doesn't match a supported version!",
)


HF_REPO_ID = "keremberke/license-plate-object-detection"
HF_CONFIG = "full"

RF_WORKSPACE = "license-plate-detection-khhkb"
RF_PROJECT = "indian-license-plate-detection-6tmbr"

OUTPUT_ROOT = Path("data/license_plates")
TEMP_ROOT = Path("data/_tmp_downloads")


def _log(msg: str) -> None:
    print(msg)


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _resolve_split_name(name: str) -> str:
    n = name.lower().strip()
    if n in {"train", "training"}:
        return "train"
    if n in {"val", "valid", "validation"}:
        return "val"
    if n in {"test", "testing"}:
        return "test"
    return n


def _extract_zip_to_dir(zip_path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def _find_dataset_root(path: Path) -> Path:
    # Prefer a folder that contains YOLO-ish split dirs.
    candidates = [p for p in [path] + list(path.glob("**/*")) if p.is_dir()]
    for c in candidates:
        has_train = (c / "train").exists() or (c / "images" / "train").exists()
        has_valid = (c / "valid").exists() or (c / "val").exists() or (c / "images" / "val").exists()
        if has_train and has_valid:
            return c
    return path


def _detect_layout(root: Path) -> str:
    # layout_a: split/images + split/labels (Roboflow default)
    # layout_b: images/split + labels/split
    if (root / "train" / "images").exists() and (root / "train" / "labels").exists():
        return "layout_a"
    if (root / "images" / "train").exists() and (root / "labels" / "train").exists():
        return "layout_b"
    return "unknown"


def _collect_pairs_from_source(root: Path) -> Dict[str, List[tuple[Path, Path]]]:
    layout = _detect_layout(root)
    pairs: Dict[str, List[tuple[Path, Path]]] = {"train": [], "val": [], "test": []}

    if layout == "layout_a":
        split_dirs = [p for p in root.iterdir() if p.is_dir()]
        for split_dir in split_dirs:
            split = _resolve_split_name(split_dir.name)
            if split not in pairs:
                continue
            images_dir = split_dir / "images"
            labels_dir = split_dir / "labels"
            if not images_dir.exists() or not labels_dir.exists():
                continue
            for label_path in labels_dir.glob("*.txt"):
                stem = label_path.stem
                image_path = None
                for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                    p = images_dir / f"{stem}{ext}"
                    if p.exists():
                        image_path = p
                        break
                if image_path is not None:
                    pairs[split].append((image_path, label_path))

    elif layout == "layout_b":
        for split in pairs:
            source_split = "valid" if split == "val" and (root / "images" / "valid").exists() else split
            images_dir = root / "images" / source_split
            labels_dir = root / "labels" / source_split
            if not images_dir.exists() or not labels_dir.exists():
                continue
            for label_path in labels_dir.glob("*.txt"):
                stem = label_path.stem
                image_path = None
                for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                    p = images_dir / f"{stem}{ext}"
                    if p.exists():
                        image_path = p
                        break
                if image_path is not None:
                    pairs[split].append((image_path, label_path))

    return pairs


def _download_hf(out_dir: Path) -> Path | None:
    _log("[HF  ] Downloading Hugging Face dataset...")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download

        zip_candidates = ["data/yolov8.zip", "yolov8.zip", "dataset.zip"]
        zip_path = None
        for candidate in zip_candidates:
            try:
                zip_path = Path(
                    hf_hub_download(
                        repo_id=HF_REPO_ID,
                        repo_type="dataset",
                        filename=candidate,
                    )
                )
                break
            except Exception:
                continue

        if zip_path is not None:
            _extract_zip_to_dir(zip_path, out_dir)
            root = _find_dataset_root(out_dir)
            pairs = _collect_pairs_from_source(root)
            if sum(len(v) for v in pairs.values()) > 0:
                return root

    except Exception as ex:
        _log(f"[HF  ] Hub zip path failed: {ex}")

    # Some dataset repos no longer load via `datasets.load_dataset` because they rely
    # on deprecated script-style builders. Fallback to direct repo snapshot download.
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
        )
        root = _find_dataset_root(out_dir)
        pairs = _collect_pairs_from_source(root)
        if sum(len(v) for v in pairs.values()) > 0:
            _log("[HF  ] Snapshot fallback succeeded.")
            return root
        _log("[HF  ] Snapshot downloaded but no YOLO image/label pairs were found.")
    except Exception as ex:
        _log(f"[HF  ] Snapshot fallback failed: {ex}")

    try:
        from datasets import load_dataset

        ds = load_dataset(HF_REPO_ID, name=HF_CONFIG)
        for split in ["train", "validation", "test"]:
            if split not in ds:
                continue
            ysplit = "val" if split == "validation" else split
            images_dir = out_dir / "images" / ysplit
            labels_dir = out_dir / "labels" / ysplit
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            for idx, ex in enumerate(ds[split]):
                img = ex["image"]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                image_id = str(ex.get("image_id", idx))
                img_path = images_dir / f"{image_id}.jpg"
                lbl_path = labels_dir / f"{image_id}.txt"
                img.save(img_path)

                width, height = img.size
                objects = ex.get("objects", {})
                bboxes = objects.get("bbox", [])
                categories = objects.get("category", [])

                with open(lbl_path, "w", encoding="utf-8") as f:
                    for bbox, cat in zip(bboxes, categories):
                        x_min, y_min, w, h = bbox
                        x_center = (x_min + w / 2.0) / width
                        y_center = (y_min + h / 2.0) / height
                        norm_w = w / width
                        norm_h = h / height
                        x_center = max(0.0, min(1.0, x_center))
                        y_center = max(0.0, min(1.0, y_center))
                        norm_w = max(0.0, min(1.0, norm_w))
                        norm_h = max(0.0, min(1.0, norm_h))
                        f.write(f"{int(cat)} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        return out_dir

    except Exception as ex:
        _log(f"[HF  ] datasets path failed: {ex}")
        return None


def _download_roboflow(out_dir: Path, api_key: str, version: int) -> Path | None:
    _log("[RF  ] Downloading Indian Roboflow dataset...")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=api_key)
        project = rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
        dataset = project.version(version).download("yolov8", location=str(out_dir))

        location = Path(getattr(dataset, "location", out_dir))
        root = _find_dataset_root(location)
        pairs = _collect_pairs_from_source(root)
        if sum(len(v) for v in pairs.values()) > 0:
            return root
        _log("[RF  ] SDK download returned empty dataset. Falling back to API export link...")
    except Exception as ex:
        _log(f"[RF  ] SDK download failed: {ex}")

    # Fallback: resolve export link from Roboflow REST API and download zip directly.
    try:
        import requests

        meta_url = (
            f"https://api.roboflow.com/{RF_WORKSPACE}/{RF_PROJECT}/{version}/yolov8"
            f"?api_key={api_key}"
        )
        meta = requests.get(meta_url, timeout=60).json()
        export = (meta or {}).get("export", {})
        link = export.get("link")
        if not link:
            _log("[RF  ] API metadata did not include an export link.")
            return None

        zip_path = out_dir / "roboflow_yolov8.zip"
        with requests.get(link, timeout=120, stream=True) as resp:
            resp.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        _extract_zip_to_dir(zip_path, out_dir)
        root = _find_dataset_root(out_dir)
        pairs = _collect_pairs_from_source(root)
        if sum(len(v) for v in pairs.values()) == 0:
            _log("[RF  ] API export downloaded but no YOLO image/label pairs were found.")
            return None
        return root
    except Exception as ex:
        _log(f"[RF  ] API export fallback failed: {ex}")
        return None


def _merge_sources(sources: Dict[str, Path], output_root: Path) -> Path:
    _clean_dir(output_root)
    for split in ["train", "val", "test"]:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_written = {"train": 0, "val": 0, "test": 0}

    def _materialize(src: Path, dst: Path) -> None:
        # Hardlink is much faster than byte-copy for local filesystems.
        # Fall back to copy when linking is unsupported.
        try:
            if dst.exists():
                dst.unlink()
            os.link(src, dst)
        except Exception:
            shutil.copyfile(src, dst)

    for source_name, source_root in sources.items():
        pairs = _collect_pairs_from_source(source_root)
        _log(
            f"[MERGE] {source_name}: "
            f"train={len(pairs['train'])}, val={len(pairs['val'])}, test={len(pairs['test'])}"
        )

        for split in ["train", "val", "test"]:
            for idx, (img_path, lbl_path) in enumerate(pairs[split]):
                dest_stem = f"{source_name}_{idx:06d}"
                dest_img = output_root / "images" / split / f"{dest_stem}{img_path.suffix.lower()}"
                dest_lbl = output_root / "labels" / split / f"{dest_stem}.txt"
                _materialize(img_path, dest_img)
                _materialize(lbl_path, dest_lbl)
                total_written[split] += 1
                if total_written[split] % 2000 == 0:
                    _log(f"[MERGE] {split}: {total_written[split]} files")

    yaml_path = output_root / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_root.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                "nc: 1",
                "names: ['license_plate']",
            ]
        ),
        encoding="utf-8",
    )

    _log(
        f"[DONE] Merged dataset -> {yaml_path} "
        f"(train={total_written['train']}, val={total_written['val']}, test={total_written['test']})"
    )
    return yaml_path


def build_dataset(
    include_hf: bool,
    include_roboflow: bool,
    roboflow_api_key: str | None,
    roboflow_version: int,
    clean_temp: bool,
) -> Path:
    if clean_temp:
        _clean_dir(TEMP_ROOT)
    else:
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    sources: Dict[str, Path] = {}

    if include_hf:
        hf_root = _download_hf(TEMP_ROOT / "hf")
        if hf_root is not None:
            sources["hf"] = hf_root

    if include_roboflow:
        if not roboflow_api_key:
            _log("[RF  ] Skipped: no API key. Set ROBOFLOW_API_KEY or pass --roboflow-api-key.")
        else:
            rf_root = _download_roboflow(
                TEMP_ROOT / "roboflow_india",
                api_key=roboflow_api_key,
                version=roboflow_version,
            )
            if rf_root is not None:
                sources["rf_india"] = rf_root

    if not sources:
        raise RuntimeError("No dataset source succeeded. Nothing to merge.")

    return _merge_sources(sources, OUTPUT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and merge ANPR datasets")
    parser.add_argument("--no-hf", action="store_true", help="Skip Hugging Face source")
    parser.add_argument("--no-roboflow", action="store_true", help="Skip Roboflow source")
    parser.add_argument(
        "--roboflow-api-key",
        type=str,
        default=None,
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
    )
    parser.add_argument(
        "--roboflow-version",
        type=int,
        default=1,
        help="Roboflow version number for indian-license-plate-detection-6tmbr",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary downloaded source folders under data/_tmp_downloads",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    include_hf = not args.no_hf
    include_roboflow = not args.no_roboflow

    import os

    api_key = args.roboflow_api_key or os.getenv("ROBOFLOW_API_KEY")

    if include_roboflow and not api_key:
        if sys.stdin and sys.stdin.isatty():
            try:
                from getpass import getpass

                entered = getpass(
                    "Enter ROBOFLOW_API_KEY (leave blank to skip Roboflow source): "
                ).strip()
            except Exception:
                entered = input(
                    "Enter ROBOFLOW_API_KEY (leave blank to skip Roboflow source): "
                ).strip()

            if entered:
                api_key = entered
                os.environ["ROBOFLOW_API_KEY"] = entered
            else:
                _log("[RF  ] No key entered. Roboflow source will be skipped.")
        else:
            _log(
                "[RF  ] No API key provided and shell is non-interactive. "
                "Set ROBOFLOW_API_KEY or pass --roboflow-api-key to enable Roboflow source."
            )

    _log("=" * 72)
    _log("ANPR Dataset Builder")
    _log("=" * 72)
    _log(f"Sources: HF={include_hf}, Roboflow Indian={include_roboflow}")

    yaml_path = build_dataset(
        include_hf=include_hf,
        include_roboflow=include_roboflow,
        roboflow_api_key=api_key,
        roboflow_version=args.roboflow_version,
        clean_temp=not args.keep_temp,
    )

    _log("\nUse this for training:")
    _log(f"  python train.py  # DATA_YAML already points to: {yaml_path.as_posix()}")


if __name__ == "__main__":
    main()
