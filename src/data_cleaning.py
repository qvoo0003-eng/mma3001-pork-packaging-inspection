"""Create a cleaned YOLO dataset using test > validation > train priority."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from .data_audit import (
    CLASS_NAMES,
    IMAGE_SUFFIXES,
    SPLITS,
    audit_zip,
    extract_source_id,
    file_sha256,
)

EXPECTED_CLEANED_COUNTS = {"train": 3313, "valid": 119, "test": 80}


def plan_cleaning(
    audit: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Plan deterministic source-ID removals from an audited inventory.

    Parameters
    ----------
    audit : dict
        Successful report returned by audit_zip.

    Returns
    -------
    tuple[dict, list]
        Retained records by split and removal records with reasons.

    Raises
    ------
    ValueError
        If the audit contains fatal errors.
    """
    if not audit.get("ok_to_clean"):
        raise ValueError("Cleaning requires an audit with no fatal errors")

    inventory = audit["inventory"]
    test_ids = {
        row["source_id"]
        for row in inventory["test"]
        if row["source_id"] is not None
    }
    validation_ids = {
        row["source_id"]
        for row in inventory["valid"]
        if row["source_id"] is not None
    }

    retained = {split: [] for split in SPLITS}
    removed = []

    for split in SPLITS:
        for row in sorted(inventory[split], key=lambda item: item["image"]):
            source_id = row["source_id"]
            reason = None
            if source_id is not None:
                if split == "valid" and source_id in test_ids:
                    reason = "source ID also occurs in test; test has priority"
                elif split == "train" and source_id in test_ids:
                    reason = "source ID also occurs in test; test has priority"
                elif split == "train" and source_id in validation_ids:
                    reason = (
                        "source ID also occurs in validation; "
                        "validation has priority"
                    )

            if reason is None:
                retained[split].append(dict(row))
            else:
                removed.append({"split": split, **row, "reason": reason})

    return retained, removed


def copy_member(
    archive: zipfile.ZipFile,
    member_name: str,
    destination: Path,
) -> str:
    """Copy one unique ZIP member and verify the copied bytes.

    Parameters
    ----------
    archive : zipfile.ZipFile
        Read-only source archive.
    member_name : str
        Unique archive member name.
    destination : Path
        New destination file inside the staging directory.

    Returns
    -------
    str
        SHA-256 digest of the copied data.

    Raises
    ------
    OSError
        If the file cannot be copied.
    RuntimeError
        If destination contents differ from the bytes read from the ZIP.
    zipfile.BadZipFile
        If member CRC verification fails.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()

    with archive.open(member_name, "r") as source:
        with destination.open("xb") as target:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                target.write(chunk)

    expected_digest = digest.hexdigest()
    if file_sha256(destination) != expected_digest:
        raise RuntimeError(f"Copied file failed verification: {destination}")
    return expected_digest


def verify_cleaned(
    directory: Path,
    retained: dict[str, list[dict[str, Any]]],
) -> dict[str, int]:
    """Verify retained files, exact image-label pairing and source separation.

    Parameters
    ----------
    directory : Path
        Staging or cleaned dataset root.
    retained : dict
        Expected retained image records by split.

    Returns
    -------
    dict[str, int]
        Verified image counts by split.

    Raises
    ------
    RuntimeError
        If any expected file, pairing or source-separation check fails.
    OSError
        If the output cannot be inspected.
    """
    source_sets: dict[str, set[str]] = {}
    counts = {}

    for split in SPLITS:
        image_root = directory / split / "images"
        label_root = directory / split / "labels"
        actual_images = {
            path.relative_to(image_root).as_posix()
            for path in image_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        actual_labels = {
            path.relative_to(label_root).as_posix()
            for path in label_root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".txt"
        }
        expected_images = {
            row["relative_image"] for row in retained[split]
        }
        expected_labels = {
            row["relative_label"] for row in retained[split]
        }

        if actual_images != expected_images:
            raise RuntimeError(f"{split}: retained image inventory mismatch")
        if actual_labels != expected_labels:
            raise RuntimeError(f"{split}: retained label inventory mismatch")

        image_stems = {
            Path(relative).with_suffix("").as_posix()
            for relative in actual_images
        }
        label_stems = {
            Path(relative).with_suffix("").as_posix()
            for relative in actual_labels
        }
        if image_stems != label_stems:
            raise RuntimeError(f"{split}: image-label pairing mismatch")

        source_sets[split] = {
            source_id
            for relative in actual_images
            if (source_id := extract_source_id(relative)) is not None
        }
        counts[split] = len(actual_images)

    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            overlap = source_sets[left] & source_sets[right]
            if overlap:
                raise RuntimeError(
                    f"Recognised source IDs still overlap: {left}/{right}: "
                    f"{sorted(overlap, key=int)}"
                )

    return counts


def clean_dataset(
    zip_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Audit and copy a cleaned dataset while preserving the original ZIP.

    Parameters
    ----------
    zip_path : str or Path
        Original dataset ZIP.
    output_directory : str or Path
        New dataset directory. Existing directories are never overwritten.

    Returns
    -------
    dict
        Cleaning log containing removals, verified counts, copied-file hashes,
        source overlaps, and discrepancies against reference expectations.

    Raises
    ------
    FileExistsError
        If the output path already exists.
    ValueError
        If the output could contain the source ZIP or the audit is fatal.
    RuntimeError
        If copying, test preservation, or output verification fails.
    OSError
        If input/output operations fail.
    zipfile.BadZipFile
        If the ZIP or a copied member is corrupt.
    """
    archive_path = Path(zip_path).resolve()
    requested_output = Path(output_directory)

    # Check the requested path before resolution to reject existing symlinks too.
    if requested_output.exists() or requested_output.is_symlink():
        raise FileExistsError(
            f"Output already exists: {requested_output}. "
            "Choose a new directory or review the existing output manually."
        )

    output = requested_output.resolve()
    if archive_path.is_relative_to(output):
        raise ValueError("Output directory must not contain the original ZIP")

    audit = audit_zip(archive_path)
    retained, removed = plan_cleaning(audit)

    if retained["test"] != sorted(
        audit["inventory"]["test"], key=lambda item: item["image"]
    ):
        raise RuntimeError("Cleaning plan did not preserve every test record")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent)
    )

    try:
        copied_files = []
        with zipfile.ZipFile(archive_path, "r") as archive:
            for split in SPLITS:
                (staging / split / "images").mkdir(parents=True)
                (staging / split / "labels").mkdir(parents=True)

                for row in retained[split]:
                    for kind, member_key, relative_key in (
                        ("images", "image", "relative_image"),
                        ("labels", "label", "relative_label"),
                    ):
                        relative = (
                            Path(split) / kind / row[relative_key]
                        )
                        destination = staging / relative
                        digest = copy_member(
                            archive, row[member_key], destination
                        )
                        copied_files.append(
                            {
                                "source": row[member_key],
                                "destination": relative.as_posix(),
                                "sha256": digest,
                            }
                        )

        counts = verify_cleaned(staging, retained)

        # Every retained file, including every test image and label, was
        # checked byte-for-byte through its SHA-256 digest during copying.
        if counts["test"] != audit["splits"]["test"]["images"]:
            raise RuntimeError("Test image count changed during cleaning")

        if file_sha256(archive_path) != audit["archive_sha256"]:
            raise RuntimeError("Original ZIP changed during cleaning")

        discrepancies = [
            (
                f"{split}: measured {counts[split]} cleaned images; "
                f"expected reference count is {EXPECTED_CLEANED_COUNTS[split]}"
            )
            for split in SPLITS
            if counts[split] != EXPECTED_CLEANED_COUNTS[split]
        ]

        # JSON string syntax is valid YAML scalar syntax and safely quotes
        # spaces and special characters in paths and class names.
        yaml_lines = [
            f"path: {json.dumps(output.as_posix())}",
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            f"nc: {len(CLASS_NAMES)}",
            "names:",
        ]
        yaml_lines.extend(
            f"  {index}: {json.dumps(name)}"
            for index, name in enumerate(CLASS_NAMES)
        )
        (staging / "data.yaml").write_text(
            "\n".join(yaml_lines) + "\n", encoding="utf-8"
        )

        cleaning_log = {
            "schema_version": 1,
            "source_archive": str(archive_path),
            "source_archive_sha256": audit["archive_sha256"],
            "output_directory": str(output),
            "priority": ["test", "valid", "train"],
            "raw_image_counts": {
                split: audit["splits"][split]["images"] for split in SPLITS
            },
            "retained_image_counts": counts,
            "removed_image_counts": {
                split: sum(row["split"] == split for row in removed)
                for split in SPLITS
            },
            "expected_cleaned_image_counts": EXPECTED_CLEANED_COUNTS.copy(),
            "expectation_discrepancies": discrepancies,
            "source_overlaps_before": audit["source_overlaps"],
            "recognised_source_overlaps_after": {
                "train__valid": [],
                "train__test": [],
                "valid__test": [],
            },
            "removed_files": removed,
            "copied_files": copied_files,
            "audit_warnings": audit["warnings"],
            "verification": {
                "image_label_pairing": True,
                "all_copied_file_hashes_checked": True,
                "all_test_images_and_labels_preserved": True,
                "original_zip_hash_unchanged": True,
                "recognised_source_ids_separated": True,
            },
            "limitations": [
                "Unrecognised source IDs are retained and cannot be screened.",
                "Filename screening does not detect all visual or temporal "
                "near-duplicates.",
                "Image decoding was not validated.",
                "Original YAML metadata is not copied; class order follows "
                "the supplied project specification.",
                "Generated data.yaml contains an absolute dataset path; "
                "regenerate or update it if moving the dataset.",
            ],
        }
        (staging / "cleaning_log.json").write_text(
            json.dumps(cleaning_log, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # Refuse overwriting if the destination appeared during processing.
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Output appeared during cleaning: {output}")

        staging.rename(output)
        return cleaning_log

    finally:
        # Only remove this function's newly created staging directory.
        if staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    """Run the cleaning command-line interface.

    Returns
    -------
    int
        Zero on verified success, or one on failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path("data/raw/Pork Rasher Error-Packaging-.v4i.yolov8.zip"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/cleaned")
    )
    args = parser.parse_args()

    try:
        log = clean_dataset(args.zip, args.output)
        print(f"Cleaned dataset: {log['output_directory']}")
        for split in SPLITS:
            print(
                f"{split}: retained={log['retained_image_counts'][split]}, "
                f"removed={log['removed_image_counts'][split]}"
            )
        for discrepancy in log["expectation_discrepancies"]:
            print(f"WARNING: {discrepancy}")
        print("Verification passed: pairing, source separation, copied hashes,")
        print("test preservation, and unchanged original ZIP.")
        print(f"Cleaning log: {args.output / 'cleaning_log.json'}")
        return 0
    except (
        OSError,
        ValueError,
        RuntimeError,
        zipfile.BadZipFile,
    ) as exc:
        print(f"Cleaning failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())