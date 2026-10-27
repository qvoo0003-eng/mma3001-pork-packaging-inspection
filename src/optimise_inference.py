"""Compare YOLO validation at imgsz 640, 480 and 800 without training.

Only validation images and labels are copied. Their bytes are preserved.
Ultralytics caches are confined to the temporary dataset. Each experiment uses
a new output directory. No configuration is automatically selected.

Inference timing is Ultralytics forward-pass time per image, not end-to-end
latency. Single-run GPU timings vary with warm-up, temperature and other load.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from shutil import copyfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "runs/detect/main-1eqhuzvr/weights/best.pt"
IMAGES = ROOT / "data/cleaned/v1/valid/images"
LABELS = ROOT / "data/cleaned/v1/valid/labels"
OUTPUT_ROOT = ROOT / "results/optimisation"
RESOLUTIONS = (640, 480, 800)
EXPECTED_IMAGES = 119
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
METRIC_NAMES = ("precision", "recall", "mAP50", "mAP50_95", "inference_ms_per_image")


def file_hash(path: Path) -> str:
    """Calculate SHA-256 without modifying a file.

    Parameters
    ----------
    path : Path
        File to read.

    Returns
    -------
    str
        Hexadecimal digest. Read failures propagate as OSError.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_files(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    """Index validation files by their relative stems.

    Parameters
    ----------
    directory : Path
        Required source directory.
    suffixes : set[str]
        Supported lowercase file extensions.

    Returns
    -------
    dict[str, Path]
        Stem-to-file mapping.

    Raises
    ------
    ValueError
        If symlinks or ambiguous stems are encountered.
    FileNotFoundError
        If the directory does not exist.
    """
    if directory.is_symlink():
        raise ValueError(f"Symbolic-link source directory is unsupported: {directory}")
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing validation directory: {directory}")
    indexed = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symbolic-link dataset entry is unsupported: {path}")
        if path.is_file() and path.suffix.lower() in suffixes:
            key = path.relative_to(directory).with_suffix("").as_posix().casefold()
            if key in indexed:
                raise ValueError(f"Ambiguous image-label stem: {key}")
            indexed[key] = path
    return indexed


def prepare_dataset(destination: Path, names: list[str]) -> list[dict[str, str]]:
    """Copy and verify one isolated validation dataset.

    Parameters
    ----------
    destination : Path
        New temporary dataset directory.
    names : list[str]
        Checkpoint class names in ID order.

    Returns
    -------
    list[dict[str, str]]
        Source-to-copy manifest.

    Raises
    ------
    ValueError
        If counts, pairing, copied bytes or image decoding are invalid.
    OSError
        If copying fails.
    """
    import cv2
    import numpy as np

    images = index_files(IMAGES, IMAGE_SUFFIXES)
    labels = index_files(LABELS, {".txt"})
    if len(images) != EXPECTED_IMAGES or len(labels) != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected 119 images and labels; found {len(images)} and {len(labels)}."
        )
    if images.keys() != labels.keys():
        raise ValueError("Validation image-label pairing is incomplete or ambiguous.")
    (destination / "images").mkdir()
    (destination / "labels").mkdir()
    manifest = []
    for number, key in enumerate(sorted(images)):
        image = images[key]
        stem = f"image_{number:04d}"
        copied_image = destination / "images" / f"{stem}{image.suffix.lower()}"
        copied_label = destination / "labels" / f"{stem}.txt"
        for source, target in ((image, copied_image), (labels[key], copied_label)):
            copyfile(source, target)
            if file_hash(source) != file_hash(target):
                raise ValueError(f"Copied bytes do not match: {source}")
        encoded = np.fromfile(copied_image, dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
        if decoded is None or min(decoded.shape[:2]) < 2:
            raise ValueError(f"Validation image cannot be decoded: {image}")
        manifest.append({
            "source_image": image.relative_to(IMAGES).as_posix(),
            "source_label": labels[key].relative_to(LABELS).as_posix(),
            "copied_image": copied_image.name,
            "copied_label": copied_label.name,
        })
    # JSON is valid YAML. The train compatibility entry is never used.
    configuration = {
        "path": destination.resolve().as_posix(),
        "train": "images", "val": "images", "nc": len(names), "names": names,
    }
    (destination / "data.yaml").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    return manifest


def extract_metrics(metrics: Any) -> dict[str, float]:
    """Read actual aggregate metrics from the Ultralytics result.

    Parameters
    ----------
    metrics : object
        Result of YOLO.val.

    Returns
    -------
    dict[str, float]
        Precision, recall, mAP values and forward-pass milliseconds per image.

    Raises
    ------
    ValueError
        If required metrics are unavailable, nonfinite or outside valid ranges.
    """
    import math

    try:
        values = (
            metrics.box.mp, metrics.box.mr, metrics.box.map50,
            metrics.box.map, metrics.speed["inference"],
        )
        result = dict(zip(METRIC_NAMES, map(float, values)))
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Ultralytics did not return all required metrics.") from exc
    for name, value in result.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid metric {name}: {value}")
        if name != "inference_ms_per_image" and value > 1:
            raise ValueError(f"Metric outside [0, 1]: {name}={value}")
    return result


def run_experiment() -> None:
    """Run the three resolutions and retain metrics and validation outputs.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If CUDA is unavailable or an evaluation omits validation images.
    ValueError
        If the checkpoint, dataset or metrics are invalid.
    OSError
        If input/output operations fail.
    """
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA device 0 is unavailable; CPU fallback is disabled.")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    checkpoint_hash = file_hash(CHECKPOINT)
    model = YOLO(str(CHECKPOINT), task="detect")
    mapping = model.names
    names = [mapping[i] for i in range(len(mapping))]
    if model.task != "detect" or len(names) != 5:
        raise ValueError("Expected a five-class detection checkpoint.")
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    settings = dict(
        batch=4, device=0, workers=0, conf=0.001, iou=0.7, max_det=300,
        half=False, augment=False, rect=True, agnostic_nms=False,
        single_cls=False, seed=42, deterministic=True, split="val",
        cache=False, plots=True, save_json=False, save_txt=False, verbose=False,
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(tempfile.mkdtemp(prefix=f"resolution-{stamp}-", dir=OUTPUT_ROOT))
    print(f"Experiment: {output}\nGPU: {torch.cuda.get_device_name(0)}")
    rows = []
    metadata = {
        "status": "preparing", "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": checkpoint_hash, "classes": names,
        "validation_images": str(IMAGES), "validation_labels": str(LABELS),
        "resolutions_in_order": list(RESOLUTIONS), "parameters": settings,
        "gpu": torch.cuda.get_device_name(0),
        "versions": {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "ultralytics": ultralytics.__version__, "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        "results": rows,
        "timing_limitations": "Single-run forward-pass timing; variable GPU "
                             "conditions; excludes end-to-end pipeline latency.",
    }
    counts = {}

    def verify_evaluation(validator: Any) -> None:
        """Require 119 loaded and processed images at validation completion."""
        counts["loaded"] = len(validator.dataloader.dataset)
        counts["seen"] = int(validator.seen)
        if counts["loaded"] != EXPECTED_IMAGES or counts["seen"] != EXPECTED_IMAGES:
            raise RuntimeError(f"Validation omitted images: {counts}; expected 119.")

    model.add_callback("on_val_end", verify_evaluation)
    fields = ["imgsz", "n_validation_images", *METRIC_NAMES]
    fields += [f"{name}_change_percent_vs_640" for name in METRIC_NAMES]
    try:
        with tempfile.TemporaryDirectory(prefix="_validation_", dir=output) as temporary:
            data = Path(temporary)
            metadata["source_manifest"] = prepare_dataset(data, names)
            metadata["status"] = "evaluating"
            (output / "experiment.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            with (output / "results.csv").open("x", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=fields).writeheader()
            baseline = None
            for resolution in RESOLUTIONS:
                counts.clear()
                print(f"Evaluating imgsz={resolution}...")
                metrics = model.val(
                    data=str(data / "data.yaml"), imgsz=resolution,
                    project=str(output), name=f"imgsz_{resolution}",
                    exist_ok=False, **settings,
                )
                if counts.get("seen") != EXPECTED_IMAGES:
                    raise RuntimeError("Validation image-count callback did not complete.")
                values = extract_metrics(metrics)
                if baseline is None:
                    baseline = values.copy()
                row = {"imgsz": resolution, "n_validation_images": counts["seen"], **values}
                for name in METRIC_NAMES:
                    row[f"{name}_change_percent_vs_640"] = (
                        100.0 * (values[name] - baseline[name]) / baseline[name]
                        if baseline[name] > 0 else ""
                    )
                rows.append(row)
                with (output / "results.csv").open("a", newline="", encoding="utf-8") as handle:
                    csv.DictWriter(handle, fieldnames=fields).writerow(row)
                metadata.setdefault("validation_outputs", {})[str(resolution)] = str(
                    getattr(metrics, "save_dir", output / f"imgsz_{resolution}")
                )
                (output / "experiment.json").write_text(
                    json.dumps(metadata, indent=2), encoding="utf-8"
                )
        if file_hash(CHECKPOINT) != checkpoint_hash:
            raise RuntimeError("Checkpoint hash changed during evaluation.")
        metadata.update(status="complete", checkpoint_hash_unchanged=True,
                        temporary_dataset_removed=True)
    except BaseException as exc:
        metadata.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (output / "experiment.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(f"Available outputs: {output}")
    print("\nValues (percentage change versus 640); inference is ms/image.")
    print("imgsz | Precision | Recall | mAP50 | mAP50-95 | Inference")
    for row in rows:
        cells = []
        for name in METRIC_NAMES:
            change = row[f"{name}_change_percent_vs_640"]
            delta = "undefined" if change == "" else f"{change:+.2f}%"
            cells.append(f"{row[name]:.4f} ({delta})")
        print(f"{row['imgsz']:5d} | " + " | ".join(cells))
    print("Single-run GPU timing varies and is not end-to-end latency.")
    print("No winning configuration has been selected.")


if __name__ == "__main__":
    try:
        run_experiment()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"Experiment failed ({type(error).__name__}): {error}", file=sys.stderr)
        if "out of memory" in str(error).lower():
            print("Close other GPU applications before retrying.", file=sys.stderr)
        raise SystemExit(1)