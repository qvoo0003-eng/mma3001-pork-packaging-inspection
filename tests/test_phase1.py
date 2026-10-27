"""Synthetic tests for Phase 1 dataset auditing and cleaning."""

from __future__ import annotations

import json
import warnings
import zipfile
from pathlib import Path

import pytest

from src.data_audit import (
    SPLITS,
    audit_zip,
    extract_source_id,
    file_sha256,
    save_report,
    validate_annotation,
)
from src.data_cleaning import clean_dataset, plan_cleaning

VALID_LABEL = "1 0.5 0.5 0.2 0.4\n"


def image_name(source_id: int, variant: str = "abc123") -> str:
    """Return a recognised synthetic dataset filename."""
    return f"frame_with_red_{source_id}_jpg.rf.{variant}.jpg"


def make_zip(
    path: Path,
    records: list[tuple[str, str, str | None]],
    prefix: str = "",
    extra_entries: list[tuple[str, bytes]] | None = None,
) -> Path:
    """Write a synthetic dataset ZIP with optional missing labels and extras."""
    root = f"{prefix.rstrip('/')}/" if prefix else ""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{root}data.yaml",
            "names: [loose-meat, packaging-error, twisted-meat, unsealed, wrinkle]\n",
        )
        for split, filename, annotation in records:
            archive.writestr(
                f"{root}{split}/images/{filename}",
                f"synthetic-image:{split}:{filename}".encode("utf-8"),
            )
            if annotation is not None:
                label_name = Path(filename).with_suffix(".txt").as_posix()
                archive.writestr(
                    f"{root}{split}/labels/{label_name}", annotation
                )
        for name, content in extra_entries or []:
            archive.writestr(name, content)
    return path


def basic_records() -> list[tuple[str, str, str | None]]:
    """Return one independent image-label pair per split."""
    return [
        ("train", image_name(1), VALID_LABEL),
        ("valid", image_name(2), VALID_LABEL),
        ("test", image_name(3), ""),
    ]


def test_counts_annotations_and_empty_labels(tmp_path: Path) -> None:
    """Count paired images, valid boxes and legitimate negative samples."""
    archive = make_zip(tmp_path / "dataset.zip", basic_records())
    report = audit_zip(archive)

    assert report["ok_to_clean"], report["errors"]
    for split in SPLITS:
        assert report["splits"][split]["images"] == 1
        assert report["splits"][split]["labels"] == 1
        assert report["splits"][split]["matched_pairs"] == 1

    assert report["splits"]["train"]["valid_annotations_by_class"][
        "packaging-error"
    ] == 1
    assert len(report["splits"]["test"]["empty_labels"]) == 1
    assert not any(report["source_overlaps"].values())


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", []),
        (" \n\t\n", []),
        ("0 0.5 0.5 1 1", [0]),
        ("4 0.5 0.5 0.2 0.2\n", [4]),
        ("0 0.5 0.5 0.2 0.2\n\n3 0.5 0.5 0.1 0.1", [0, 3]),
        # Tiny excursions caused by rounded centre/width are tolerated.
        ("1 0.0999999 0.5 0.2 0.2", [1]),
    ],
)
def test_valid_annotations(text: str, expected: list[int]) -> None:
    """Accept valid rows, blank lines and documented rounding tolerance."""
    class_ids, errors = validate_annotation(text)
    assert class_ids == expected
    assert errors == []


@pytest.mark.parametrize(
    "text",
    [
        "1 0.5 0.5 0.2",
        "1 0.5 0.5 0.2 0.2 extra",
        "1.0 0.5 0.5 0.2 0.2",
        "-1 0.5 0.5 0.2 0.2",
        "5 0.5 0.5 0.2 0.2",
        "1 nan 0.5 0.2 0.2",
        "1 0.5 inf 0.2 0.2",
        "1 x 0.5 0.2 0.2",
        "1 1.1 0.5 0.2 0.2",
        "1 0.5 0.5 0 0.2",
        "1 0.5 0.5 -0.2 0.2",
        "1 0.05 0.5 0.2 0.2",
    ],
)
def test_invalid_annotations(text: str) -> None:
    """Reject malformed classes, coordinates and out-of-image boxes."""
    class_ids, errors = validate_annotation(text)
    assert class_ids == []
    assert len(errors) == 1
    assert errors[0].startswith("line 1:")


def test_mixed_annotation_counts_only_valid_rows() -> None:
    """Keep valid-row counts while reporting a later malformed row."""
    class_ids, errors = validate_annotation(
        VALID_LABEL + "9 0.5 0.5 0.2 0.2\n"
    )
    assert class_ids == [1]
    assert len(errors) == 1
    assert errors[0].startswith("line 2:")


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("frame_with_red_2223_jpg.rf.abc123.jpg", "2223"),
        ("train/images/frame_with_red_0007_jpg.rf.ABC.jpg", "7"),
        ("frame_with_red_8_jpg.rf.hash.PNG", "8"),
        ("unrelated.jpg", None),
        ("prefix_frame_with_red_8_jpg.rf.hash.jpg", None),
        ("frame_with_red_x_jpg.rf.hash.jpg", None),
    ],
)
def test_source_id_extraction(filename: str, expected: str | None) -> None:
    """Recognise the specified filename pattern without partial matches."""
    assert extract_source_id(filename) == expected


def test_missing_pairs_are_fatal(tmp_path: Path) -> None:
    """Report both missing labels and orphan labels."""
    records = basic_records()
    records[0] = ("train", image_name(1), None)
    archive = make_zip(
        tmp_path / "missing.zip",
        records,
        extra_entries=[("train/labels/orphan.txt", b"")],
    )
    report = audit_zip(archive)
    assert not report["ok_to_clean"]
    assert len(report["splits"]["train"]["missing_labels"]) == 1
    assert report["splits"]["train"]["missing_images"] == [
        "train/labels/orphan.txt"
    ]
    with pytest.raises(ValueError, match="no fatal errors"):
        plan_cleaning(report)


def test_bad_label_blocks_cleaning(tmp_path: Path) -> None:
    """Prevent a malformed annotation from entering the cleaned dataset."""
    records = basic_records()
    records[0] = ("train", image_name(1), "8 0.5 0.5 0.2 0.2")
    archive = make_zip(tmp_path / "invalid.zip", records)
    output = tmp_path / "cleaned"

    with pytest.raises(ValueError, match="no fatal errors"):
        clean_dataset(archive, output)
    assert not output.exists()


def test_overlap_priority_and_preservation(tmp_path: Path) -> None:
    """Apply test > valid > train and preserve source and retained bytes."""
    records = [
        ("train", image_name(10, "train10"), VALID_LABEL),
        ("train", image_name(20, "train20"), VALID_LABEL),
        ("train", image_name(30, "train30"), VALID_LABEL),
        ("train", image_name(40, "train40"), ""),
        ("valid", image_name(10, "valid10"), VALID_LABEL),
        ("valid", image_name(20, "valid20"), ""),
        ("valid", image_name(50, "valid50"), VALID_LABEL),
        ("test", image_name(10, "test10"), VALID_LABEL),
        ("test", image_name(60, "test60"), ""),
    ]
    archive = make_zip(tmp_path / "overlap.zip", records)
    original_bytes = archive.read_bytes()
    original_digest = file_sha256(archive)
    audit = audit_zip(archive)

    assert audit["source_overlaps"] == {
        "train__valid": ["10", "20"],
        "train__test": ["10"],
        "valid__test": ["10"],
    }
    retained, removed = plan_cleaning(audit)
    assert {
        split: [row["source_id"] for row in retained[split]]
        for split in SPLITS
    } == {
        "train": ["30", "40"],
        "valid": ["20", "50"],
        "test": ["10", "60"],
    }
    assert len(removed) == 3
    assert all(row["split"] != "test" for row in removed)
    assert (retained, removed) == plan_cleaning(audit)

    output = tmp_path / "cleaned"
    log = clean_dataset(archive, output)
    assert log["retained_image_counts"] == {
        "train": 2, "valid": 2, "test": 2
    }
    assert log["removed_image_counts"] == {
        "train": 2, "valid": 1, "test": 0
    }
    assert all(log["verification"].values())
    assert log["expectation_discrepancies"]
    assert archive.read_bytes() == original_bytes
    assert file_sha256(archive) == original_digest

    with zipfile.ZipFile(archive) as source:
        for row in log["copied_files"]:
            copied = output / row["destination"]
            assert copied.read_bytes() == source.read(row["source"])
            assert file_sha256(copied) == row["sha256"]

        for row in audit["inventory"]["test"]:
            assert (
                output / "test/images" / row["relative_image"]
            ).read_bytes() == source.read(row["image"])
            assert (
                output / "test/labels" / row["relative_label"]
            ).read_bytes() == source.read(row["label"])

    empty_label = (
        output / "train/labels"
        / Path(image_name(40, "train40")).with_suffix(".txt")
    )
    assert empty_label.read_bytes() == b""

    for split in SPLITS:
        image_stems = {
            path.stem for path in (output / split / "images").iterdir()
        }
        label_stems = {
            path.stem for path in (output / split / "labels").iterdir()
        }
        assert image_stems == label_stems

    yaml_text = (output / "data.yaml").read_text(encoding="utf-8")
    assert "val: valid/images" in yaml_text
    assert "nc: 5" in yaml_text
    assert '4: "wrinkle"' in yaml_text
    assert f"path: {json.dumps(output.resolve().as_posix())}" in yaml_text

    saved_log = json.loads(
        (output / "cleaning_log.json").read_text(encoding="utf-8")
    )
    assert saved_log == log


def test_wrapper_and_nested_paths(tmp_path: Path) -> None:
    """Support an enclosing dataset directory and nested image folders."""
    records = [
        ("train", f"camera1/{image_name(1)}", VALID_LABEL),
        ("valid", f"camera2/{image_name(2)}", ""),
        ("test", f"camera3/{image_name(3)}", VALID_LABEL),
    ]
    archive = make_zip(
        tmp_path / "wrapped.zip", records, prefix="dataset"
    )
    report = audit_zip(archive)
    assert report["ok_to_clean"], report["errors"]
    assert report["dataset_roots"] == ["dataset"]

    output = tmp_path / "cleaned"
    clean_dataset(archive, output)
    assert (output / "train/images/camera1" / image_name(1)).is_file()


def test_unknown_sources_warn_but_are_retained(tmp_path: Path) -> None:
    """Retain unmatched filenames while clearly limiting overlap screening."""
    records = basic_records()
    records[0] = ("train", "unknown.jpg", "")
    archive = make_zip(tmp_path / "unknown.zip", records)
    report = audit_zip(archive)

    assert report["ok_to_clean"], report["errors"]
    assert report["splits"]["train"]["unrecognised_source_images"]
    assert any("unrecognised" in message for message in report["warnings"])

    output = tmp_path / "cleaned"
    clean_dataset(archive, output)
    assert (output / "train/images/unknown.jpg").is_file()
    assert (output / "train/labels/unknown.txt").read_bytes() == b""


def test_identical_repeated_yaml_is_warning(tmp_path: Path) -> None:
    """Allow identical repeated YAML while reporting every duplicate entry."""
    archive = make_zip(tmp_path / "duplicate-yaml.zip", basic_records())
    with zipfile.ZipFile(archive) as source:
        original_yaml = source.read("data.yaml")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(archive, "a") as target:
            target.writestr("data.yaml", original_yaml)
            target.writestr("data.yaml", original_yaml)

    report = audit_zip(archive)
    assert report["ok_to_clean"], report["errors"]
    assert report["duplicate_entries"] == [
        {"path": "data.yaml", "count": 3, "identical_content": True}
    ]


def test_conflicting_repeated_yaml_is_fatal(tmp_path: Path) -> None:
    """Reject conflicting metadata rather than selecting an arbitrary entry."""
    archive = make_zip(tmp_path / "conflict.zip", basic_records())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(archive, "a") as target:
            target.writestr("data.yaml", "names: [different]\n")
    report = audit_zip(archive)
    assert not report["ok_to_clean"]
    assert any("Conflicting repeated" in error for error in report["errors"])


def test_duplicate_image_entry_is_fatal(tmp_path: Path) -> None:
    """Reject repeated image entries even if their contents match."""
    archive = make_zip(tmp_path / "duplicate-image.zip", basic_records())
    member = f"train/images/{image_name(1)}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(archive, "a") as target:
            target.writestr(member, b"duplicate")
    report = audit_zip(archive)
    assert not report["ok_to_clean"]
    assert any("Ambiguous repeated" in error for error in report["errors"])


def test_same_stem_multiple_images_is_fatal(tmp_path: Path) -> None:
    """Reject two image formats competing for one label."""
    filename = image_name(1)
    archive = make_zip(
        tmp_path / "stem.zip",
        basic_records(),
        extra_entries=[
            (
                f"train/images/{Path(filename).with_suffix('.png').as_posix()}",
                b"another image",
            )
        ],
    )
    report = audit_zip(archive)
    assert not report["ok_to_clean"]
    assert report["splits"]["train"]["ambiguous_stems"]


def test_unsafe_archive_path_is_fatal(tmp_path: Path) -> None:
    """Reject traversal paths before copying any data."""
    archive = make_zip(
        tmp_path / "unsafe.zip",
        basic_records(),
        extra_entries=[("../escape.txt", b"unsafe")],
    )
    report = audit_zip(archive)
    assert not report["ok_to_clean"]
    assert any("unsafe" in error for error in report["errors"])


def test_missing_split_is_fatal(tmp_path: Path) -> None:
    """Report an absent required split."""
    archive = make_zip(
        tmp_path / "missing-test.zip", basic_records()[:2]
    )
    report = audit_zip(archive)
    assert not report["ok_to_clean"]
    assert "test: no supported images found" in report["errors"]


def test_existing_output_is_not_modified(tmp_path: Path) -> None:
    """Refuse an existing output directory without deleting its contents."""
    archive = make_zip(tmp_path / "dataset.zip", basic_records())
    output = tmp_path / "cleaned"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        clean_dataset(archive, output)
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_report_cannot_overwrite_original_zip(tmp_path: Path) -> None:
    """Protect the raw ZIP from an accidental report destination."""
    archive = make_zip(tmp_path / "dataset.zip", basic_records())
    original = archive.read_bytes()
    report = audit_zip(archive)

    with pytest.raises(ValueError, match="original ZIP"):
        save_report(report, archive)
    assert archive.read_bytes() == original


def test_audit_json_round_trip(tmp_path: Path) -> None:
    """Ensure the audit report can be saved and loaded without information loss."""
    archive = make_zip(tmp_path / "dataset.zip", basic_records())
    report = audit_zip(archive)
    destination = tmp_path / "results/audit.json"

    save_report(report, destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == report