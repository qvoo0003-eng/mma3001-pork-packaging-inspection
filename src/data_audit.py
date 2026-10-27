"""Audit a YOLO detection dataset ZIP without extracting or modifying it."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

SPLITS = ("train", "valid", "test")
CLASS_NAMES = (
    "loose-meat",
    "packaging-error",
    "twisted-meat",
    "unsealed",
    "wrinkle",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SOURCE_PATTERN = re.compile(
    r"frame_with_red_(\d+)_jpg\.rf\.[A-Za-z0-9]+"
    r"\.(?:jpg|jpeg|png|bmp|webp|tif|tiff)",
    re.IGNORECASE,
)
EXPECTED_COUNTS = {"train": 3349, "valid": 120, "test": 80}
BOUNDARY_TOLERANCE = 1e-6


def extract_source_id(filename: str) -> str | None:
    """Extract a canonical numeric source ID from a recognised image name.

    Parameters
    ----------
    filename : str
        Archive path or image filename.

    Returns
    -------
    str or None
        Source ID without leading zeros, or None for an unmatched filename.
    """
    match = SOURCE_PATTERN.fullmatch(PurePosixPath(filename).name)
    return str(int(match.group(1))) if match else None


def validate_annotation(text: str) -> tuple[list[int], list[str]]:
    """Validate YOLO detection rows and return valid class IDs and errors.

    Parameters
    ----------
    text : str
        Decoded annotation content. Blank lines and empty files are allowed.

    Returns
    -------
    tuple[list[int], list[str]]
        Class IDs for valid rows and line-numbered errors. Invalid rows are
        excluded from annotation counts.
    """
    class_ids: list[int] = []
    errors: list[str] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue

        prefix = f"line {line_number}"
        if len(fields) != 5:
            errors.append(f"{prefix}: expected 5 fields, found {len(fields)}")
            continue

        if not re.fullmatch(r"[0-9]+", fields[0]):
            errors.append(f"{prefix}: class ID must be an integer")
            continue

        class_id = int(fields[0])
        if not 0 <= class_id < len(CLASS_NAMES):
            errors.append(f"{prefix}: class ID {class_id} is outside 0..4")
            continue

        try:
            x_center, y_center, width, height = map(float, fields[1:])
        except ValueError:
            errors.append(f"{prefix}: coordinates must be numeric")
            continue

        coordinates = (x_center, y_center, width, height)
        if not all(math.isfinite(value) for value in coordinates):
            errors.append(f"{prefix}: coordinates must be finite")
            continue

        if not all(0.0 <= value <= 1.0 for value in coordinates):
            errors.append(f"{prefix}: coordinates must be normalised to [0, 1]")
            continue

        if width <= 0.0 or height <= 0.0:
            errors.append(f"{prefix}: width and height must be positive")
            continue

        edges = (
            x_center - width / 2.0,
            y_center - height / 2.0,
            x_center + width / 2.0,
            y_center + height / 2.0,
        )
        if (
            min(edges) < -BOUNDARY_TOLERANCE
            or max(edges) > 1.0 + BOUNDARY_TOLERANCE
        ):
            errors.append(f"{prefix}: bounding box extends outside the image")
            continue

        class_ids.append(class_id)

    return class_ids, errors


def file_sha256(path: Path) -> str:
    """Calculate a file's SHA-256 digest using bounded memory.

    Parameters
    ----------
    path : Path
        File to read.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest.

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


def unsafe_member_reason(info: zipfile.ZipInfo) -> str | None:
    """Identify archive paths or member types unsuitable for copying.

    Parameters
    ----------
    info : zipfile.ZipInfo
        Archive member metadata.

    Returns
    -------
    str or None
        Rejection reason, or None when the member is acceptable.
    """
    name = info.filename
    parts = name.rstrip("/").split("/")
    if (
        not name
        or name.startswith("/")
        or "\\" in name
        or any(part in {"", ".", ".."} for part in parts)
        or any(":" in part for part in parts)
    ):
        return "unsafe or non-portable archive path"

    mode = info.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        return "archive member is not a regular file or directory"

    if info.flag_bits & 1:
        return "encrypted archive member is unsupported"

    return None


def identify_member(
    name: str,
) -> tuple[str, str, str, str] | None:
    """Recognise a dataset member's root, split, kind and relative path.

    Parameters
    ----------
    name : str
        POSIX-style archive member name.

    Returns
    -------
    tuple[str, str, str, str] or None
        Dataset root, split, 'images' or 'labels', and relative file path.
        None means that the path does not match the supported layout.
    """
    parts = PurePosixPath(name).parts
    candidates = []
    for index in range(len(parts) - 2):
        if parts[index] in SPLITS and parts[index + 1] in {"images", "labels"}:
            candidates.append(
                (
                    "/".join(parts[:index]),
                    parts[index],
                    parts[index + 1],
                    "/".join(parts[index + 2:]),
                )
            )
    return candidates[0] if len(candidates) == 1 else None


def find_source_overlaps(
    inventory: dict[str, list[dict[str, Any]]],
) -> dict[str, list[str]]:
    """Find recognised source IDs shared between dataset splits.

    Parameters
    ----------
    inventory : dict
        Image records grouped by split, each containing a source_id field.

    Returns
    -------
    dict[str, list[str]]
        Sorted source IDs for each pair of splits.
    """
    source_sets = {
        split: {
            record["source_id"]
            for record in inventory[split]
            if record["source_id"] is not None
        }
        for split in SPLITS
    }
    return {
        f"{left}__{right}": sorted(
            source_sets[left] & source_sets[right], key=int
        )
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1:]
    }


def audit_zip(zip_path: str | Path) -> dict[str, Any]:
    """Audit ZIP layout, image-label pairing, annotations and source overlap.

    Parameters
    ----------
    zip_path : str or Path
        Original dataset ZIP. It is opened only for reading.

    Returns
    -------
    dict
        JSON-compatible report. Dataset problems appear in errors or warnings.
        The inventory contains one record per image.

    Raises
    ------
    OSError
        If the ZIP cannot be read.
    zipfile.BadZipFile
        If the archive cannot be opened.
    """
    archive_path = Path(zip_path).resolve()
    original_digest = file_sha256(archive_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "archive": str(archive_path),
        "archive_sha256": original_digest,
        "classes": list(CLASS_NAMES),
        "expected_raw_images": EXPECTED_COUNTS.copy(),
        "errors": [],
        "warnings": [],
        "duplicate_entries": [],
        "inventory": {split: [] for split in SPLITS},
        "splits": {},
    }
    errors = report["errors"]
    warnings = report["warnings"]
    images: dict[str, dict[str, list[str]]] = {s: {} for s in SPLITS}
    labels: dict[str, dict[str, list[str]]] = {s: {} for s in SPLITS}
    label_details: dict[str, dict[str, Any]] = {}
    roots: set[str] = set()

    with zipfile.ZipFile(archive_path, "r") as archive:
        all_infos = archive.infolist()
        report["archive_structure"] = {
            "entry_count": len(all_infos),
            "file_entry_count": sum(not info.is_dir() for info in all_infos),
            "top_level_entries": sorted(
                {info.filename.split("/")[0] for info in all_infos}
            ),
        }

        grouped: dict[str, list[zipfile.ZipInfo]] = {}
        safe_infos = []
        for info in all_infos:
            grouped.setdefault(info.filename, []).append(info)
            reason = unsafe_member_reason(info)
            if reason:
                errors.append(f"{info.filename!r}: {reason}")
            elif not info.is_dir():
                safe_infos.append(info)

        for name, entries in sorted(grouped.items()):
            if len(entries) < 2:
                continue

            duplicate = {"path": name, "count": len(entries)}
            report["duplicate_entries"].append(duplicate)

            if all(info.is_dir() for info in entries):
                warnings.append(f"Repeated directory entry: {name}")
            elif (
                PurePosixPath(name).name == "data.yaml"
                and all(unsafe_member_reason(info) is None for info in entries)
                and all(not info.is_dir() for info in entries)
            ):
                try:
                    payloads = [archive.read(info) for info in entries]
                    identical = all(data == payloads[0] for data in payloads)
                    duplicate["identical_content"] = identical
                    if identical:
                        warnings.append(
                            f"Repeated identical data.yaml: {name} "
                            f"({len(entries)} entries)"
                        )
                    else:
                        errors.append(f"Conflicting repeated data.yaml: {name}")
                except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                    errors.append(f"Cannot read repeated {name}: {exc}")
            else:
                errors.append(f"Ambiguous repeated archive member: {name}")

        report["yaml_entries"] = sorted(
            info.filename
            for info in safe_infos
            if PurePosixPath(info.filename).name == "data.yaml"
        )
        if not report["yaml_entries"]:
            warnings.append(
                "No data.yaml found; class order follows the supplied project "
                "specification."
            )
        report["class_mapping_basis"] = (
            "Fixed class order supplied by the user; original YAML is not parsed."
        )

        for info in safe_infos:
            name = info.filename
            suffix = PurePosixPath(name).suffix.lower()
            identified = identify_member(name)
            if identified is None:
                if suffix in IMAGE_SUFFIXES:
                    errors.append(f"Image outside supported split layout: {name}")
                continue

            root, split, kind, relative = identified
            roots.add(root)

            if kind == "images":
                if suffix not in IMAGE_SUFFIXES:
                    warnings.append(f"Unsupported file in images directory: {name}")
                    continue
                key = PurePosixPath(relative).with_suffix("").as_posix()
                images[split].setdefault(key, []).append(name)
            else:
                if suffix != ".txt":
                    warnings.append(f"Non-TXT file in labels directory: {name}")
                    continue
                key = PurePosixPath(relative).with_suffix("").as_posix()
                labels[split].setdefault(key, []).append(name)

                # Repeated members are already fatal and are not read ambiguously.
                if len(grouped[name]) != 1:
                    continue
                try:
                    content = archive.read(info).decode("utf-8-sig")
                    class_ids, row_errors = validate_annotation(content)
                    label_details[name] = {
                        "empty": not content.strip(),
                        "class_ids": class_ids,
                        "errors": row_errors,
                    }
                    errors.extend(f"{name}: {message}" for message in row_errors)
                except (
                    UnicodeError,
                    OSError,
                    RuntimeError,
                    zipfile.BadZipFile,
                ) as exc:
                    errors.append(f"Cannot read annotation {name}: {exc}")

        report["dataset_roots"] = sorted(roots)
        if len(roots) != 1:
            errors.append(
                f"Expected exactly one dataset root; found {sorted(roots)!r}"
            )

        for split in SPLITS:
            image_map = images[split]
            label_map = labels[split]
            missing_labels = []
            missing_images = []
            ambiguous_keys = []
            matched_pairs = 0
            unknown_sources = []
            class_counts = Counter()
            empty_labels = []

            if not image_map:
                errors.append(f"{split}: no supported images found")

            for key in sorted(image_map.keys() | label_map.keys()):
                image_names = image_map.get(key, [])
                label_names = label_map.get(key, [])
                if len(image_names) > 1 or len(label_names) > 1:
                    ambiguous_keys.append(key)
                    errors.append(f"{split}: ambiguous image-label stem {key!r}")

                if image_names and not label_names:
                    missing_labels.extend(image_names)
                if label_names and not image_names:
                    missing_images.extend(label_names)
                if len(image_names) == len(label_names) == 1:
                    matched_pairs += 1

                for image_name in image_names:
                    source_id = extract_source_id(image_name)
                    if source_id is None:
                        unknown_sources.append(image_name)
                    report["inventory"][split].append(
                        {
                            "image": image_name,
                            "label": label_names[0] if len(label_names) == 1 else None,
                            "relative_image": identify_member(image_name)[3],
                            "relative_label": (
                                identify_member(label_names[0])[3]
                                if len(label_names) == 1
                                else None
                            ),
                            "source_id": source_id,
                        }
                    )

            for name_list in label_map.values():
                for name in name_list:
                    detail = label_details.get(name)
                    if detail is not None:
                        class_counts.update(detail["class_ids"])
                        if detail["empty"]:
                            empty_labels.append(name)

            if missing_labels:
                errors.append(f"{split}: {len(missing_labels)} images lack labels")
            if missing_images:
                errors.append(f"{split}: {len(missing_images)} labels lack images")
            if unknown_sources:
                warnings.append(
                    f"{split}: {len(unknown_sources)} images have unrecognised "
                    "source IDs; leakage screening cannot cover these files"
                )

            image_count = sum(map(len, image_map.values()))
            if image_count != EXPECTED_COUNTS[split]:
                warnings.append(
                    f"{split}: measured {image_count} images; "
                    f"expected reference count is {EXPECTED_COUNTS[split]}"
                )

            report["splits"][split] = {
                "images": image_count,
                "labels": sum(map(len, label_map.values())),
                "matched_pairs": matched_pairs,
                "empty_labels": sorted(empty_labels),
                "missing_labels": sorted(missing_labels),
                "missing_images": sorted(missing_images),
                "ambiguous_stems": ambiguous_keys,
                "unrecognised_source_images": sorted(unknown_sources),
                "valid_annotations_by_class": {
                    name: class_counts[index]
                    for index, name in enumerate(CLASS_NAMES)
                },
            }

    report["source_overlaps"] = find_source_overlaps(report["inventory"])
    for pair, source_ids in report["source_overlaps"].items():
        if source_ids:
            warnings.append(
                f"Source-ID overlap {pair}: {', '.join(source_ids)}"
            )

    if file_sha256(archive_path) != original_digest:
        errors.append("Original ZIP changed while the audit was running")

    report["fatal_error_count"] = len(errors)
    report["warning_count"] = len(warnings)
    report["ok_to_clean"] = not errors
    return report


def format_summary(report: dict[str, Any]) -> str:
    """Format a readable summary of a dataset audit.

    Parameters
    ----------
    report : dict
        Report returned by audit_zip.

    Returns
    -------
    str
        Summary including counts, class totals, warnings and fatal errors.
    """
    lines = [
        f"Archive: {report['archive']}",
        f"SHA-256: {report['archive_sha256']}",
        f"Archive entries: {report['archive_structure']['entry_count']}",
        f"Dataset roots: {report['dataset_roots']}",
        "",
        "Split    Images  Labels  Matched  Empty labels",
    ]
    for split in SPLITS:
        counts = report["splits"][split]
        lines.append(
            f"{split:<8} {counts['images']:>6} {counts['labels']:>7} "
            f"{counts['matched_pairs']:>8} {len(counts['empty_labels']):>13}"
        )
        lines.append(
            "  Valid annotation counts: "
            + ", ".join(
                f"{name}={count}"
                for name, count in counts["valid_annotations_by_class"].items()
            )
        )

    lines.append("\nSource-ID overlaps:")
    for pair, identifiers in report["source_overlaps"].items():
        lines.append(f"  {pair}: {identifiers}")

    lines.append(f"\nWarnings ({len(report['warnings'])}):")
    lines.extend(f"  WARNING: {message}" for message in report["warnings"])
    lines.append(f"\nFatal errors ({len(report['errors'])}):")
    lines.extend(f"  ERROR: {message}" for message in report["errors"])
    lines.append(f"\nOK to clean: {report['ok_to_clean']}")
    lines.append(
        "Scope: filenames and annotations are checked; image decoding and "
        "visual/temporal near-duplicate detection are not performed."
    )
    return "\n".join(lines)


def save_report(
    report: dict[str, Any], destination: str | Path
) -> None:
    """Save an audit report without overwriting the source ZIP.

    Parameters
    ----------
    report : dict
        Audit report containing the source archive path.
    destination : str or Path
        JSON output file.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the destination resolves to the original ZIP.
    OSError
        If the report cannot be written.
    """
    target = Path(destination).resolve()
    if target == Path(report["archive"]).resolve():
        raise ValueError("Report destination must not overwrite the original ZIP")
    if target.exists() and target.samefile(Path(report["archive"])):
        raise ValueError("Report destination refers to the original ZIP")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    """Run the audit command-line interface.

    Returns
    -------
    int
        Zero for no fatal dataset errors, one for fatal audit findings, or two
        for an input/output failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path("data/raw/Pork Rasher Error-Packaging-.v4i.yolov8.zip"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("results/data_audit.json")
    )
    args = parser.parse_args()

    try:
        report = audit_zip(args.zip)
        save_report(report, args.report)
        print(format_summary(report))
        print(f"\nJSON report: {args.report}")
        return 0 if report["ok_to_clean"] else 1
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"Audit failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
