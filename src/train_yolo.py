"""Train YOLOv8n using isolated copies of the training and validation data.

Original datasets are read only. Ultralytics may create label caches, so training
uses private dataset copies inside each uniquely named experiment directory.
The test split is never copied or evaluated.

Requires Python 3.12, PyTorch, Ultralytics, and PyYAML.
"""

from __future__ import annotations

import argparse
import multiprocessing
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CLASSES = [
    "loose-meat",
    "packaging-error",
    "twisted-meat",
    "unsealed",
    "wrinkle",
]


def repository_path(value: str | Path) -> Path:
    """Resolve a path relative to the repository root.

    Parameters
    ----------
    value : str or Path
        Absolute path or repository-relative path.

    Returns
    -------
    Path
        Resolved absolute path.
    """
    path = Path(value).expanduser()
    return (REPOSITORY_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def read_dataset(yaml_path: Path) -> dict[str, Path]:
    """Read and validate the training and validation directory configuration.

    Parameters
    ----------
    yaml_path : Path
        Existing dataset YAML. Relative dataset roots are interpreted relative
        to this YAML's directory.

    Returns
    -------
    dict[str, Path]
        Source image and label directories for train and val.

    Raises
    ------
    ValueError
        If the YAML, class mapping, or directory layout is unsupported.
    FileNotFoundError
        If the YAML or a required directory is missing.
    """
    import yaml

    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {yaml_path}")

    try:
        configuration = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid dataset YAML: {exc}") from exc

    if not isinstance(configuration, dict):
        raise ValueError("Dataset YAML must contain a mapping.")

    names = configuration.get("names")
    if isinstance(names, dict):
        try:
            names = [names[index] for index in range(5)]
        except KeyError as exc:
            raise ValueError("Class mapping must contain integer IDs 0 through 4.") from exc

    if names != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected class names in this order: {EXPECTED_CLASSES}; found {names!r}"
        )
    if configuration.get("nc", 5) != 5:
        raise ValueError("Dataset must contain exactly five classes.")

    root_value = configuration.get("path")
    if root_value is None:
        dataset_root = yaml_path.parent
    elif isinstance(root_value, str):
        dataset_root = Path(root_value).expanduser()
        if not dataset_root.is_absolute():
            dataset_root = yaml_path.parent / dataset_root
    else:
        raise ValueError("Dataset 'path' must be a directory string.")

    dataset_root = dataset_root.resolve()
    directories = {}

    for split in ("train", "val"):
        value = configuration.get(split)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"'{split}' must specify one image directory.")

        images = Path(value)
        if not images.is_absolute():
            images = dataset_root / images
        images = images.resolve()

        if images.name.lower() != "images":
            raise ValueError(
                f"Expected {split} to use a split/images directory: {images}"
            )
        labels = images.parent / "labels"

        for kind, directory in (("images", images), ("labels", labels)):
            if not directory.is_dir():
                raise FileNotFoundError(f"Missing {split} {kind} directory: {directory}")
            directories[f"{split}_{kind}"] = directory

    train_root = directories["train_images"].parent
    validation_root = directories["val_images"].parent
    if train_root.is_relative_to(validation_root) or validation_root.is_relative_to(train_root):
        raise ValueError("Training and validation must use separate split directories.")

    # Inspect only the configured path string, never test images or annotations.
    test_value = configuration.get("test")
    if isinstance(test_value, str) and test_value.strip():
        test_path = Path(test_value)
        if not test_path.is_absolute():
            test_path = dataset_root / test_path
        test_path = test_path.resolve()

        for split in ("train", "val"):
            images = directories[f"{split}_images"]
            if images.is_relative_to(test_path) or test_path.is_relative_to(images):
                raise ValueError(f"Configured {split} images overlap the test path.")

    return directories


def prepare_dataset(
    run_directory: Path,
    directories: dict[str, Path],
) -> Path:
    """Copy train/validation data and create a YAML without a test entry.

    Parameters
    ----------
    run_directory : Path
        Unique experiment directory.
    directories : dict[str, Path]
        Validated source image and label directories.

    Returns
    -------
    Path
        Private dataset YAML used for training.

    Raises
    ------
    OSError
        If copying or YAML creation fails.
    """
    import yaml

    snapshot = run_directory / "dataset_snapshot"
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            shutil.copytree(
                directories[f"{split}_{kind}"],
                snapshot / split / kind,
            )

    configuration = {
        "path": snapshot.resolve().as_posix(),
        "train": "train/images",
        "val": "val/images",
        "nc": 5,
        "names": EXPECTED_CLASSES,
    }
    destination = snapshot / "data.yaml"
    destination.write_text(
        yaml.safe_dump(configuration, sort_keys=False),
        encoding="utf-8",
    )
    return destination


def print_outputs(run_directory: Path) -> None:
    """Report actual paths and whether expected artifacts currently exist.

    Parameters
    ----------
    run_directory : Path
        Actual Ultralytics experiment directory.

    Returns
    -------
    None
    """
    print(f"\nExperiment directory: {run_directory}")
    for relative in (
        "weights/best.pt",
        "weights/last.pt",
        "results.csv",
        "results.png",
        "args.yaml",
        "dataset_snapshot/data.yaml",
    ):
        path = run_directory / relative
        status = "exists" if path.is_file() else "not present"
        print(f"  [{status}] {path}")

    plots = sorted(run_directory.glob("*.png")) + sorted(run_directory.glob("*.jpg"))
    if plots:
        print("Generated plots:")
        for path in plots:
            print(f"  {path}")


def run_training(args: argparse.Namespace) -> Path:
    """Validate CUDA, prepare an isolated experiment, and train YOLOv8n.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed mode, data path and optional resume checkpoint.

    Returns
    -------
    Path
        Actual experiment directory.

    Raises
    ------
    ValueError
        If configuration or checkpoint settings are invalid.
    RuntimeError
        If CUDA is unavailable or training fails.
    OSError
        If dataset or checkpoint files cannot be accessed.
    """
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(
            "CUDA device 0 is unavailable. Check the NVIDIA driver and "
            "CUDA-enabled PyTorch installation. CPU fallback is disabled."
        )

    print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
    dataset_yaml = repository_path(args.data)
    directories = read_dataset(dataset_yaml)
    epochs = 1 if args.mode == "smoke" else 25

    checkpoint = None
    if args.resume:
        if args.mode != "main":
            raise ValueError("--resume is supported only with --mode main.")
        checkpoint = repository_path(args.resume)
        if not checkpoint.is_file() or checkpoint.name.lower() != "last.pt":
            raise ValueError("--resume must point to an existing last.pt checkpoint.")

        # Load only trusted checkpoints produced by your own training.
        model = YOLO(str(checkpoint))
        state = model.ckpt or {}
        saved_epoch = state.get("epoch", -1)
        if not isinstance(saved_epoch, int) or saved_epoch < 0:
            raise ValueError(
                "Checkpoint is finalised or lacks resumable epoch state. "
                "Use an interrupted-training last.pt checkpoint."
            )
        if saved_epoch + 1 >= epochs:
            raise ValueError("Checkpoint has already reached the 25-epoch maximum.")
        if state.get("optimizer") is None:
            raise ValueError("Checkpoint lacks optimizer state required for resuming.")
        if not (checkpoint.parent / "best.pt").is_file():
            raise FileNotFoundError(
                "Resuming into a new experiment requires best.pt beside last.pt "
                "to preserve the previous validation-selected best checkpoint."
            )
        print(f"Resuming after completed epoch {saved_epoch + 1}; maximum total: {epochs}")
    else:
        model = YOLO("yolov8n.pt")

    project = (REPOSITORY_ROOT / "runs" / "detect").resolve()
    protected_roots = [dataset_yaml.parent]
    protected_roots.extend(directory.parent for directory in directories.values())
    if any(project.is_relative_to(root) for root in protected_roots):
        raise ValueError("Experiment output must be outside the source dataset.")

    project.mkdir(parents=True, exist_ok=True)
    label = f"{args.mode}-resume-" if checkpoint else f"{args.mode}-"
    run_directory = Path(tempfile.mkdtemp(prefix=label, dir=project)).resolve()
    print(f"New experiment: {run_directory}", flush=True)

    try:
        private_yaml = prepare_dataset(run_directory, directories)

        if checkpoint:
            weights_directory = run_directory / "weights"
            weights_directory.mkdir()
            shutil.copy2(checkpoint.parent / "best.pt", weights_directory / "best.pt")

            previous_metrics = checkpoint.parent.parent / "results.csv"
            if previous_metrics.is_file():
                shutil.copy2(previous_metrics, run_directory / "results.csv")

        class IsolatedDetectionTrainer(DetectionTrainer):
            """Keep resumed training in this experiment with its private dataset."""

            def check_resume(self, overrides: dict[str, Any]) -> None:
                """Restore checkpoint settings, then enforce experiment isolation.

                Parameters
                ----------
                overrides : dict
                    Ultralytics training overrides.

                Returns
                -------
                None

                Raises
                ------
                Exception
                    Propagates Ultralytics checkpoint-loading errors.
                """
                super().check_resume(overrides)

                # Ultralytics restores saved arguments during resume. Reapply
                # these settings before its trainer creates output directories
                # or loads data, preserving the old experiment unchanged.
                self.args.save_dir = str(run_directory)
                self.args.project = str(project)
                self.args.name = run_directory.name
                self.args.exist_ok = True
                self.args.data = str(private_yaml)
                self.args.epochs = epochs
                self.args.patience = 7
                self.args.imgsz = 640
                self.args.batch = 4
                self.args.device = 0
                self.args.workers = 0
                self.args.amp = True
                self.args.seed = 42
                self.args.deterministic = True
                self.args.val = True
                self.args.split = "val"
                self.args.save = True
                self.args.plots = True
                self.args.cache = False

        model.train(
            trainer=IsolatedDetectionTrainer,
            data=str(private_yaml),
            epochs=epochs,
            patience=7,
            imgsz=640,
            batch=4,
            device=0,
            workers=0,
            amp=True,
            seed=42,
            deterministic=True,
            val=True,
            split="val",
            save=True,
            plots=True,
            cache=False,
            project=str(project),
            name=run_directory.name,
            exist_ok=True,  # Directory was uniquely reserved above.
            resume=str(checkpoint) if checkpoint else False,
        )

        actual_directory = Path(model.trainer.save_dir).resolve()
        print_outputs(actual_directory)
        return actual_directory

    except Exception:
        print_outputs(run_directory)
        raise


def main() -> int:
    """Parse arguments and run exactly one requested training mode.

    Returns
    -------
    int
        Zero on success, one on failure, or 130 after keyboard interruption.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "main"),
        required=True,
        help="Run one smoke-test epoch or up to 25 main-training epochs.",
    )
    parser.add_argument(
        "--data",
        default="data/cleaned/v1/data.yaml",
        help="Dataset YAML, absolute or relative to the repository root.",
    )
    parser.add_argument(
        "--resume",
        metavar="LAST_PT",
        help="Trusted interrupted main-training last.pt; writes a new experiment.",
    )
    args = parser.parse_args()

    try:
        run_training(args)
        return 0
    except KeyboardInterrupt:
        print(
            "\nTraining interrupted. Resume from last.pt if a complete "
            "epoch checkpoint was saved.",
            file=sys.stderr,
        )
        return 130
    except ImportError as exc:
        print(
            f"Missing or incompatible dependency: {exc}. Use the Python "
            "environment containing PyTorch, Ultralytics and PyYAML.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        message = str(exc)
        print(f"\nTraining failed ({type(exc).__name__}): {message}", file=sys.stderr)
        if "out of memory" in message.lower():
            print(
                "GPU memory was exhausted. Close other GPU applications "
                "before retrying; this script keeps the requested batch size 4.",
                file=sys.stderr,
            )
        else:
            print(
                "Check dataset paths, checkpoint compatibility, available disk "
                "space, and internet access for the initial pretrained download.",
                file=sys.stderr,
            )
        return 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())