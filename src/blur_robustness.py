"""Evaluate a fixed YOLO detector on three validation-image blur conditions.

All conditions use lossless PNG images decoded from the same source images.
Gaussian blur is applied independently to the original BGR image using:
sigma=1: 7x7 kernel; sigma=2: 13x13 kernel; BORDER_REFLECT_101 padding.

Only cleaned validation images and labels are accessed. Temporary copies isolate
Ultralytics label caches from the original dataset. No training is performed.

Metrics come directly from the Ultralytics validation API. Its inference timing
excludes image decoding, blur generation, preprocessing and postprocessing.
One evaluation per condition is not a statistically robust GPU speed benchmark.

The validation set is expected to contain no wrinkle annotations. Consequently,
it cannot establish wrinkle-class detection performance. Rare-class results have
limited evidence. Edge-strength changes alone do not establish detector accuracy.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = REPOSITORY_ROOT / "runs/detect/main-1eqhuzvr/weights/best.pt"
IMAGE_DIRECTORY = REPOSITORY_ROOT / "data/cleaned/v1/valid/images"
LABEL_DIRECTORY = REPOSITORY_ROOT / "data/cleaned/v1/valid/labels"
RESULTS_ROOT = REPOSITORY_ROOT / "results/yolo_blur"

EXPECTED_IMAGES = 119
SIGMAS = (0, 1, 2)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

CSV_FIELDS = (
    "blur_sigma",
    "n_validation_images",
    "precision",
    "recall",
    "mAP50",
    "mAP50_95",
    "inference_ms_per_image",
    "relative_mAP50_change_percent_vs_sigma0",
)


def sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 digest.

    Parameters
    ----------
    path : Path
        File to read.

    Returns
    -------
    str
        Hexadecimal digest.

    Raises
    ------
    OSError
        If the file cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    """Index files by case-insensitive relative stem without following links.

    Parameters
    ----------
    directory : Path
        Validation image or label directory.
    suffixes : set[str]
        Accepted lowercase suffixes.

    Returns
    -------
    dict[str, Path]
        Relative stems mapped to source files.

    Raises
    ------
    FileNotFoundError
        If the directory is missing.
    ValueError
        If paths use symbolic links, escape the directory, or share a stem.
    """
    if directory.is_symlink():
        raise ValueError(f"Symbolic-link dataset directories are unsupported: {directory}")
    if not directory.is_dir():
        raise FileNotFoundError(f"Required directory does not exist: {directory}")

    root = directory.resolve()
    indexed = {}

    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symbolic-link dataset entries are unsupported: {path}")
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Dataset entry escapes its validation directory: {path}")
        if path.suffix.lower() not in suffixes:
            print(f"WARNING: Ignoring unsupported file: {path}")
            continue

        relative = path.relative_to(directory)
        key = relative.with_suffix("").as_posix().casefold()
        if key in indexed:
            raise ValueError(
                f"Ambiguous image-label stem: {indexed[key]} and {path}"
            )
        indexed[key] = path

    return indexed


def read_label(path: Path, class_count: int) -> tuple[bytes, Counter]:
    """Read a YOLO label without changing its bytes and count annotations.

    Parameters
    ----------
    path : Path
        Source annotation file. Empty annotations are valid.
    class_count : int
        Number of classes in the checkpoint.

    Returns
    -------
    tuple[bytes, Counter]
        Original file bytes and annotation counts by integer class ID.

    Raises
    ------
    OSError
        If the annotation cannot be read.
    ValueError
        If annotation encoding or detection rows are invalid.
    """
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ValueError(f"Annotation is not valid UTF-8: {path}") from exc

    counts = Counter()
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        description = f"{path}, line {line_number}"
        if len(fields) != 5 or not fields[0].isascii() or not fields[0].isdigit():
            raise ValueError(f"Invalid YOLO detection row: {description}")

        class_id = int(fields[0])
        if not 0 <= class_id < class_count:
            raise ValueError(f"Class ID outside checkpoint mapping: {description}")

        try:
            x, y, width, height = map(float, fields[1:])
        except ValueError as exc:
            raise ValueError(f"Nonnumeric box coordinates: {description}") from exc

        coordinates = (x, y, width, height)
        if (
            not all(math.isfinite(value) and 0 <= value <= 1 for value in coordinates)
            or width <= 0
            or height <= 0
        ):
            raise ValueError(f"Invalid normalized bounding box: {description}")

        counts[class_id] += 1

    return payload, counts


def prepare_datasets(
    temporary_root: Path,
    names: list[str],
) -> tuple[dict[int, Path], list[dict[str, Any]], Counter]:
    """Create and verify all three isolated validation datasets.

    Parameters
    ----------
    temporary_root : Path
        New temporary directory owned by this experiment.
    names : list[str]
        Class names in checkpoint ID order.

    Returns
    -------
    tuple[dict, list, Counter]
        Dataset YAML paths by sigma, source-to-copy manifest, and class counts.

    Raises
    ------
    ValueError
        If counts, pairings, decoding or copied contents fail verification.
    OSError
        If source reading or temporary copying fails.
    """
    import cv2
    import numpy as np

    images = collect_files(IMAGE_DIRECTORY, IMAGE_SUFFIXES)
    labels = collect_files(LABEL_DIRECTORY, {".txt"})

    if len(images) != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected exactly {EXPECTED_IMAGES} validation images; found {len(images)}."
        )
    if images.keys() != labels.keys():
        missing_labels = sorted(images.keys() - labels.keys())
        orphan_labels = sorted(labels.keys() - images.keys())
        raise ValueError(
            f"Image-label pairing failed. Missing labels: {missing_labels}; "
            f"labels without images: {orphan_labels}"
        )

    dataset_paths = {}
    for sigma in SIGMAS:
        root = temporary_root / f"sigma_{sigma}"
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir()
        yaml_path = root / "data.yaml"

        # JSON is valid YAML. The compatibility train entry points to the same
        # validation-only copy; only model.val(split="val") is ever called.
        configuration = {
            "path": root.resolve().as_posix(),
            "train": "images",
            "val": "images",
            "nc": len(names),
            "names": names,
        }
        yaml_path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")
        dataset_paths[sigma] = yaml_path

    manifest = []
    annotation_counts = Counter()

    for index, key in enumerate(sorted(images)):
        source_image = images[key]
        source_label = labels[key]
        label_bytes, counts = read_label(source_label, len(names))

        try:
            encoded_source = np.fromfile(source_image, dtype=np.uint8)
            if encoded_source.size == 0:
                raise ValueError("File is empty.")
            original = cv2.imdecode(encoded_source, cv2.IMREAD_COLOR)
        except (OSError, ValueError, cv2.error) as exc:
            raise ValueError(f"Could not decode {source_image}: {exc}") from exc

        if original is None or original.ndim != 3 or original.shape[2] != 3:
            raise ValueError(f"Could not decode a BGR image: {source_image}")
        if min(original.shape[:2]) < 2:
            raise ValueError(f"Image dimensions are too small: {source_image}")

        copied_stem = f"image_{index:04d}"
        for sigma in SIGMAS:
            if sigma == 0:
                pixels = original
            else:
                kernel_size = 7 if sigma == 1 else 13
                pixels = cv2.GaussianBlur(
                    original,
                    (kernel_size, kernel_size),
                    sigmaX=float(sigma),
                    sigmaY=float(sigma),
                    borderType=cv2.BORDER_REFLECT_101,
                )

            root = dataset_paths[sigma].parent
            image_path = root / "images" / f"{copied_stem}.png"
            label_path = root / "labels" / f"{copied_stem}.txt"

            success, encoded_png = cv2.imencode(
                ".png", pixels, [cv2.IMWRITE_PNG_COMPRESSION, 3]
            )
            if not success:
                raise ValueError(f"PNG encoding failed: {source_image}, sigma={sigma}")
            encoded_png.tofile(image_path)
            label_path.write_bytes(label_bytes)

            # Verify the actual saved PNG is lossless and the label is unchanged.
            decoded_copy = cv2.imdecode(
                np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if decoded_copy is None or not np.array_equal(decoded_copy, pixels):
                raise ValueError(f"Saved PNG verification failed: {image_path}")
            if label_path.read_bytes() != label_bytes:
                raise ValueError(f"Copied label verification failed: {label_path}")

        annotation_counts.update(counts)
        manifest.append({
            "copied_stem": copied_stem,
            "source_image": source_image.relative_to(IMAGE_DIRECTORY).as_posix(),
            "source_label": source_label.relative_to(LABEL_DIRECTORY).as_posix(),
            "width": int(original.shape[1]),
            "height": int(original.shape[0]),
        })

    expected_stems = {record["copied_stem"] for record in manifest}
    for sigma, yaml_path in dataset_paths.items():
        root = yaml_path.parent
        image_stems = {path.stem for path in (root / "images").glob("*.png")}
        label_stems = {path.stem for path in (root / "labels").glob("*.txt")}
        if image_stems != expected_stems or label_stems != expected_stems:
            raise ValueError(f"Final dataset inventory failed for sigma={sigma}.")
        if len(image_stems) != EXPECTED_IMAGES:
            raise ValueError(f"Incomplete validation dataset for sigma={sigma}.")

    print(
        f"Verified all {EXPECTED_IMAGES} decoded images and their labels "
        "in every condition before evaluation."
    )
    return dataset_paths, manifest, annotation_counts


def extract_metrics(metrics: Any, sigma: int) -> dict[str, Any]:
    """Extract actual aggregate detection metrics from Ultralytics.

    Parameters
    ----------
    metrics : object
        Result returned by YOLO.val.
    sigma : int
        Evaluated Gaussian blur sigma.

    Returns
    -------
    dict
        CSV-compatible metric values.

    Raises
    ------
    ValueError
        If required metrics or inference timing are unavailable or invalid.
    """
    try:
        row = {
            "blur_sigma": sigma,
            "n_validation_images": EXPECTED_IMAGES,
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "inference_ms_per_image": float(metrics.speed["inference"]),
        }
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Ultralytics did not return the required metrics.") from exc

    for name in ("precision", "recall", "mAP50", "mAP50_95"):
        if not math.isfinite(row[name]) or not 0 <= row[name] <= 1:
            raise ValueError(f"Invalid Ultralytics metric {name}: {row[name]}")

    timing = row["inference_ms_per_image"]
    if not math.isfinite(timing) or timing < 0:
        raise ValueError(f"Invalid Ultralytics inference timing: {timing}")

    return row


def print_table(rows: list[dict[str, Any]]) -> None:
    """Print measured results and baseline-relative mAP50 changes.

    Parameters
    ----------
    rows : list[dict]
        Completed condition results.

    Returns
    -------
    None
    """
    print("\nSigma  Images  Precision  Recall   mAP50  mAP50-95  Infer ms/img  mAP50 change")
    for row in rows:
        change = row["relative_mAP50_change_percent_vs_sigma0"]
        change_text = "undefined" if change == "" else f"{change:+.3f}%"
        print(
            f"{row['blur_sigma']:5d}  "
            f"{row['n_validation_images']:6d}  "
            f"{row['precision']:9.4f}  "
            f"{row['recall']:6.4f}  "
            f"{row['mAP50']:6.4f}  "
            f"{row['mAP50_95']:8.4f}  "
            f"{row['inference_ms_per_image']:12.3f}  {change_text}"
        )


def run_experiment() -> Path:
    """Run three validation conditions in a uniquely named experiment.

    Returns
    -------
    Path
        Persistent experiment directory.

    Raises
    ------
    RuntimeError
        If CUDA is unavailable or evaluation fails.
    ValueError
        If the checkpoint, data or returned metrics are invalid.
    OSError
        If required files cannot be read or outputs cannot be written.
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

    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    cv2.setRNGSeed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    checkpoint_digest = sha256_file(CHECKPOINT)
    model = YOLO(str(CHECKPOINT), task="detect")
    if model.task != "detect":
        raise ValueError("The checkpoint must be an object-detection model.")

    mapping = model.names
    if isinstance(mapping, dict):
        if set(mapping) != set(range(5)):
            raise ValueError("Expected checkpoint class IDs 0 through 4.")
        names = [str(mapping[index]) for index in range(5)]
    else:
        names = list(mapping)
        if len(names) != 5:
            raise ValueError("Expected exactly five checkpoint classes.")

    wrinkle_ids = [
        index for index, name in enumerate(names)
        if name.strip().casefold() == "wrinkle"
    ]
    if len(wrinkle_ids) != 1:
        raise ValueError("Expected exactly one checkpoint class named 'wrinkle'.")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_directory = Path(
        tempfile.mkdtemp(prefix=f"blur-{timestamp}-", dir=RESULTS_ROOT)
    ).resolve()
    csv_path = run_directory / "results.csv"
    metadata_path = run_directory / "experiment.json"

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Experiment directory: {run_directory}")

    settings = {
        "imgsz": 640,
        "batch": 4,
        "device": 0,
        "workers": 0,
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
        "half": False,
        "augment": False,
        "rect": True,
        "agnostic_nms": False,
        "single_cls": False,
        "seed": 42,
        "deterministic": True,
        "split": "val",
        "plots": True,
        "save_json": False,
        "save_txt": False,
        "verbose": False,
    }

    rows = []
    metadata: dict[str, Any] = {
        "status": "preparing",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": checkpoint_digest,
        "validation_images": str(IMAGE_DIRECTORY),
        "validation_labels": str(LABEL_DIRECTORY),
        "classes": names,
        "settings": settings,
        "blur": {
            "sigmas_pixels": list(SIGMAS),
            "kernel_sizes": {"0": None, "1": [7, 7], "2": [13, 13]},
            "border": "BORDER_REFLECT_101",
            "processing": "OpenCV uint8 BGR; each blur starts from original",
            "output_encoding": "lossless PNG for every condition",
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        },
        "limitations": [
            "No wrinkle annotations are expected in this validation set.",
            "Rare-class detection evidence is limited.",
            "Precision and recall use Ultralytics' aggregate reporting convention.",
            "Inference timing is Ultralytics forward-pass time per image, "
            "not total pipeline latency.",
            "Single-run timing can vary with GPU warm-up, load and temperature.",
            "Sigma-0 PNG evaluation is the baseline for this experiment; "
            "historical validation metrics are not substituted.",
        ],
        "completed_results": rows,
    }

    try:
        # Every dataset and cache is temporary; validation plots are persistent.
        with tempfile.TemporaryDirectory(
            prefix="_validation_copies_", dir=run_directory
        ) as temporary:
            dataset_paths, manifest, counts = prepare_datasets(
                Path(temporary), names
            )

            wrinkle_count = counts[wrinkle_ids[0]]
            if wrinkle_count != 0:
                raise ValueError(
                    f"Expected no wrinkle annotations, but found {wrinkle_count}. "
                    "Resolve this dataset discrepancy before evaluation."
                )

            metadata["source_manifest"] = manifest
            metadata["annotation_counts"] = {
                name: counts[index] for index, name in enumerate(names)
            }
            metadata["verified_images_per_condition"] = len(manifest)
            metadata["status"] = "evaluating"
            metadata_path.write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )

            print("Validation annotation counts:")
            for index, name in enumerate(names):
                print(f"  {name}: {counts[index]}")
            print(
                "No wrinkle annotations: wrinkle performance cannot be established. "
                "Rare classes have limited evidence."
            )

            with csv_path.open("x", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()

            baseline_map50 = None
            for sigma in SIGMAS:
                print(f"\nEvaluating sigma={sigma} on {EXPECTED_IMAGES} images...")
                condition_name = f"sigma_{sigma}"
                expected_output = run_directory / condition_name
                if expected_output.exists():
                    raise FileExistsError(f"Condition output already exists: {expected_output}")

                metrics = model.val(
                    data=str(dataset_paths[sigma]),
                    project=str(run_directory),
                    name=condition_name,
                    exist_ok=False,
                    **settings,
                )
                row = extract_metrics(metrics, sigma)

                if sigma == 0:
                    baseline_map50 = row["mAP50"]
                row["relative_mAP50_change_percent_vs_sigma0"] = (
                    100.0 * (row["mAP50"] - baseline_map50) / baseline_map50
                    if baseline_map50 is not None and baseline_map50 > 0
                    else ""
                )
                rows.append(row)

                with csv_path.open("a", newline="", encoding="utf-8") as handle:
                    csv.DictWriter(handle, fieldnames=CSV_FIELDS).writerow(row)

                actual_output = Path(
                    getattr(metrics, "save_dir", expected_output)
                ).resolve()
                metadata.setdefault("validation_output_directories", {})[
                    str(sigma)
                ] = str(actual_output)
                metadata_path.write_text(
                    json.dumps(metadata, indent=2), encoding="utf-8"
                )

        if sha256_file(CHECKPOINT) != checkpoint_digest:
            raise RuntimeError("Checkpoint contents changed during the experiment.")

        metadata["status"] = "complete"
        metadata["checkpoint_sha256_unchanged"] = True
        metadata["temporary_datasets_removed"] = True
        metadata_path.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    except BaseException as exc:
        metadata["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        try:
            metadata_path.write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
        except OSError:
            pass
        print(f"\nExperiment incomplete. Available outputs: {run_directory}")
        if rows:
            print_table(rows)
        raise

    print_table(rows)
    print(f"\nResults CSV: {csv_path}")
    print(f"Experiment settings and manifest: {metadata_path}")
    print("Validation plots are retained in the sigma_0, sigma_1 and sigma_2 folders.")
    print("Relative changes use this experiment's measured sigma-0 baseline.")
    return run_directory


def main() -> int:
    """Run the blur experiment and report common failures.

    Returns
    -------
    int
        Zero on success, one on failure, or 130 after interruption.
    """
    try:
        run_experiment()
        return 0
    except KeyboardInterrupt:
        print("\nExperiment interrupted.", file=sys.stderr)
        return 130
    except ImportError as exc:
        print(f"Missing or incompatible dependency: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Experiment failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        if "out of memory" in str(exc).lower():
            print(
                "CUDA memory was exhausted. Close other GPU applications "
                "before retrying with the required fixed settings.",
                file=sys.stderr,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())