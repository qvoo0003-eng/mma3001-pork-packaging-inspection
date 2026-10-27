"""Evaluate the fixed YOLO configuration once on the withheld test split.

Original image and label bytes are copied into a temporary dataset. Ultralytics
caches remain isolated. No training, optimisation or configuration selection occurs.

Precision/recall use Ultralytics' reporting convention. Aggregate F1 is calculated
from aggregate precision and recall; it is not the mean of per-class F1 scores.
Inference timing is forward-pass time, not complete end-to-end latency.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from shutil import copyfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "runs/detect/main-1eqhuzvr/weights/best.pt"
IMAGES = ROOT / "data/cleaned/v1/test/images"
LABELS = ROOT / "data/cleaned/v1/test/labels"
OUTPUT_ROOT = ROOT / "results/final_evaluation"
EXPECTED_IMAGES = 80
CLASSES = ["loose-meat", "packaging-error", "twisted-meat", "unsealed", "wrinkle"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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
    """Index test files by relative stem and reject ambiguous paths.

    Parameters
    ----------
    directory : Path
        Required source directory.
    suffixes : set[str]
        Supported lowercase extensions.

    Returns
    -------
    dict[str, Path]
        Relative stems mapped to source files.

    Raises
    ------
    ValueError
        If symbolic links or duplicate stems are encountered.
    FileNotFoundError
        If the source directory is missing.
    """
    if directory.is_symlink():
        raise ValueError(f"Symbolic-link source directory is unsupported: {directory}")
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing test directory: {directory}")
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


def prepare_test_data(destination: Path) -> tuple[list[dict], list[int]]:
    """Copy all test pairs, verify their bytes, and count annotations.

    Parameters
    ----------
    destination : Path
        Temporary dataset directory.

    Returns
    -------
    tuple[list[dict], list[int]]
        Copy manifest and ground-truth box counts in class-ID order.

    Raises
    ------
    ValueError
        If counts, pairing, decoding, annotations or copied bytes are invalid.
    OSError
        If reading or copying fails.
    """
    import cv2
    import numpy as np

    images = index_files(IMAGES, IMAGE_SUFFIXES)
    labels = index_files(LABELS, {".txt"})
    if len(images) != EXPECTED_IMAGES or len(labels) != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected 80 test images and labels; found {len(images)} and {len(labels)}."
        )
    if images.keys() != labels.keys():
        raise ValueError("Test images and labels do not have exact one-to-one pairing.")
    (destination / "images").mkdir()
    (destination / "labels").mkdir()
    manifest, counts = [], [0] * len(CLASSES)
    for number, key in enumerate(sorted(images)):
        stem = f"test_{number:04d}"
        image_target = destination / "images" / f"{stem}{images[key].suffix.lower()}"
        label_target = destination / "labels" / f"{stem}.txt"
        record = {
            "source_image": images[key].relative_to(IMAGES).as_posix(),
            "source_label": labels[key].relative_to(LABELS).as_posix(),
            "copied_image": image_target.relative_to(destination).as_posix(),
            "copied_label": label_target.relative_to(destination).as_posix(),
        }
        for kind, source, target in (
            ("image", images[key], image_target), ("label", labels[key], label_target)
        ):
            digest = file_hash(source)
            copyfile(source, target)
            if file_hash(target) != digest:
                raise ValueError(f"Copied bytes differ from source: {source}")
            record[f"{kind}_sha256"] = digest
        encoded = np.fromfile(image_target, dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
        if decoded is None or min(decoded.shape[:2]) < 2:
            raise ValueError(f"Unreadable test image: {images[key]}")
        for line_number, line in enumerate(
            label_target.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            fields = line.split()
            if not fields:
                continue
            location = f"{labels[key]}, line {line_number}"
            if len(fields) != 5 or fields[0] not in {"0", "1", "2", "3", "4"}:
                raise ValueError(f"Invalid YOLO detection row: {location}")
            coordinates = list(map(float, fields[1:]))
            if (
                not all(math.isfinite(v) and 0 <= v <= 1 for v in coordinates)
                or min(coordinates[2:]) <= 0
            ):
                raise ValueError(f"Invalid normalized bounding box: {location}")
            counts[int(fields[0])] += 1
        manifest.append(record)
    # Compatibility keys refer only to the isolated test copy.
    data = {
        "path": destination.resolve().as_posix(), "train": "images",
        "val": "images", "test": "images", "nc": 5, "names": CLASSES,
    }
    (destination / "data.yaml").write_text(json.dumps(data), encoding="utf-8")
    return manifest, counts


def run_evaluation() -> None:
    """Run one fixed test evaluation and save measured results.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If CUDA is unavailable or evaluation does not include every test image.
    ValueError
        If inputs, checkpoint mapping or returned metrics are invalid.
    OSError
        If required input/output operations fail.
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
    digest = file_hash(CHECKPOINT)
    model = YOLO(str(CHECKPOINT), task="detect")
    names = [model.names[i] for i in range(len(model.names))]
    if model.task != "detect" or names != CLASSES:
        raise ValueError(f"Checkpoint class mapping differs from expected: {names}")
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    settings = dict(
        imgsz=640, batch=4, device=0, workers=0, conf=0.001, iou=0.7,
        split="test", max_det=300, half=False, augment=False, rect=True,
        agnostic_nms=False, single_cls=False, seed=42, deterministic=True,
        cache=False, plots=True, save_json=False, save_txt=False, verbose=False,
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(tempfile.mkdtemp(prefix=f"test-{stamp}-", dir=OUTPUT_ROOT))
    document: dict[str, Any] = {
        "status": "preparing", "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": digest, "test_images": str(IMAGES),
        "test_labels": str(LABELS), "expected_images": EXPECTED_IMAGES,
        "classes": CLASSES, "evaluation_settings": settings,
        "configuration_selected_before_test": True,
        "gpu": torch.cuda.get_device_name(0),
        "software_versions": {
            "python": sys.version.split()[0], "torch": str(torch.__version__),
            "ultralytics": ultralytics.__version__, "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        "limitations": [
            "Small test set; rare-class results have limited evidence.",
            "Classes with no ground-truth boxes are NOT EVALUABLE.",
            "Aggregate F1 is the harmonic mean of aggregate precision and recall.",
            "Ultralytics precision/recall follow its curve-reporting convention; "
            "conf=0.001 is candidate filtering, not a selected deployment threshold.",
            "Single-run GPU inference timing varies and excludes end-to-end latency.",
            "Filename-based split cleaning does not prove temporal independence.",
        ],
    }
    observed = {}

    def verify_counts(validator: Any) -> None:
        """Check loaded images, processed images and ground-truth box counts."""
        observed["loaded_images"] = len(validator.dataloader.dataset)
        observed["evaluated_images"] = int(validator.seen)
        observed["annotation_counts"] = [int(v) for v in validator.metrics.nt_per_class]
        if observed["loaded_images"] != 80 or observed["evaluated_images"] != 80:
            raise RuntimeError(f"Evaluation did not include all 80 images: {observed}")

    model.add_callback("on_val_end", verify_counts)
    print(f"Final fixed-configuration test evaluation\nOutput: {output}")
    try:
        with tempfile.TemporaryDirectory(prefix="_test_copy_", dir=output) as temporary:
            data_root = Path(temporary)
            manifest, counts = prepare_test_data(data_root)
            document.update(
                status="evaluating", actual_copied_images=len(manifest),
                source_manifest=manifest, ground_truth_annotation_counts=counts,
            )
            (output / "evaluation.json").write_text(
                json.dumps(document, indent=2), encoding="utf-8"
            )
            # Exactly one evaluation; default plots include labelled batch examples.
            metrics = model.val(
                data=str(data_root / "data.yaml"), project=str(output),
                name="test_evaluation", exist_ok=False, **settings,
            )
            if observed.get("evaluated_images") != EXPECTED_IMAGES:
                raise RuntimeError("The evaluation image-count check did not complete.")
            if observed["annotation_counts"] != counts:
                raise RuntimeError("Ultralytics ground-truth counts differ from copied labels.")
            for record in manifest:
                for kind in ("image", "label"):
                    if file_hash(data_root / record[f"copied_{kind}"]) != record[f"{kind}_sha256"]:
                        raise RuntimeError(f"Temporary {kind} bytes changed during evaluation.")
            precision, recall = float(metrics.box.mp), float(metrics.box.mr)
            overall = {
                "n_test_images": observed["evaluated_images"],
                "precision": precision, "recall": recall,
                "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
                "inference_ms_per_image": float(metrics.speed["inference"]),
            }
            for key, value in overall.items():
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid returned metric: {key}={value}")
                if key not in {"n_test_images", "inference_ms_per_image"} and value > 1:
                    raise ValueError(f"Detection metric exceeds one: {key}={value}")
            positions = {int(class_id): i for i, class_id in enumerate(metrics.box.ap_class_index)}
            per_class = []
            for class_id, name in enumerate(CLASSES):
                row = dict(class_id=class_id, class_name=name, ground_truth_boxes=counts[class_id])
                row.update(status="NOT EVALUABLE", precision="", recall="", F1="", mAP50="", mAP50_95="")
                if counts[class_id] > 0:
                    row["status"] = "METRICS UNAVAILABLE"
                    if class_id in positions:
                        i = positions[class_id]
                        p, r = float(metrics.box.p[i]), float(metrics.box.r[i])
                        row.update(
                            status="EVALUATED", precision=p, recall=r,
                            F1=2 * p * r / (p + r) if p + r else 0.0,
                            mAP50=float(metrics.box.ap50[i]), mAP50_95=float(metrics.box.ap[i]),
                        )
                per_class.append(row)
            if file_hash(CHECKPOINT) != digest:
                raise RuntimeError("Checkpoint hash changed during evaluation.")
            for filename, rows in (
                ("overall_metrics.csv", [overall]), ("per_class_metrics.csv", per_class)
            ):
                with (output / filename).open("x", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            document.update(
                status="complete", actual_evaluation_counts=observed,
                overall_metrics=overall, per_class_metrics=per_class,
                checkpoint_hash_unchanged=True,
                evaluation_plot_directory=str(
                    getattr(metrics, "save_dir", output / "test_evaluation")
                ),
            )
        document["temporary_dataset_removed"] = True
    except BaseException as exc:
        document.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (output / "evaluation.json").write_text(
            json.dumps(document, indent=2), encoding="utf-8"
        )
        print(f"Available outputs: {output}")
    print("\nMeasured overall test metrics:")
    for key, value in overall.items():
        print(f"  {key}: {value:.6f}")
    for row in per_class:
        print(f"  {row['class_name']}: {row['ground_truth_boxes']} boxes; {row['status']}")
    print("Single-run inference timing is variable and is not end-to-end latency.")


if __name__ == "__main__":
    try:
        run_evaluation()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"Evaluation failed ({type(error).__name__}): {error}", file=sys.stderr)
        if "out of memory" in str(error).lower():
            print("Close other GPU applications before retrying.", file=sys.stderr)
        raise SystemExit(1)