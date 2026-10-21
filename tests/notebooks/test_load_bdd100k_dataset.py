import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

from notebooks import load_bdd100k_dataset as bdd


def _create_archives(
    images_path: Path,
    labels_path: Path,
    *,
    missing_split: str | None = None,
    missing_label: str | None = None,
) -> None:
    with ZipFile(images_path, "w") as images_zip, ZipFile(labels_path, "w") as labels_zip:
        for split in bdd.SPLITS:
            if split == missing_split:
                continue

            images_zip.writestr(f"100k/{split}/frame-{split}.jpg", b"image")
            if split != missing_label:
                labels_zip.writestr(f"100k/{split}/frame-{split}.json", b"{}")


def test_copy_sampled_pairs_copies_matching_files(tmp_path: Path) -> None:
    images_path = tmp_path / "images.zip"
    labels_path = tmp_path / "labels.zip"
    output_dir = tmp_path / "output"
    _create_archives(images_path, labels_path)

    with ZipFile(images_path) as images_zip, ZipFile(labels_path) as labels_zip:
        copied = bdd.copy_sampled_pairs(images_zip, labels_zip, output_dir, seed=42)

    assert copied == {"train": 1, "val": 1, "test": 1}
    for split in bdd.SPLITS:
        assert (output_dir / "images" / split / f"frame-{split}.jpg").read_bytes() == b"image"
        assert (output_dir / "labels" / split / f"frame-{split}.json").read_bytes() == b"{}"


def test_copy_sampled_pairs_rejects_missing_labels_before_copying(
    tmp_path: Path,
) -> None:
    images_path = tmp_path / "images.zip"
    labels_path = tmp_path / "labels.zip"
    output_dir = tmp_path / "output"
    _create_archives(images_path, labels_path, missing_label="train")

    with ZipFile(images_path) as images_zip, ZipFile(labels_path) as labels_zip:
        with pytest.raises(ValueError, match="Image/label files do not pair in train"):
            bdd.copy_sampled_pairs(images_zip, labels_zip, output_dir, seed=42)

    assert not output_dir.exists()


def test_copy_sampled_pairs_rejects_archives_missing_a_split(
    tmp_path: Path,
) -> None:
    images_path = tmp_path / "images.zip"
    labels_path = tmp_path / "labels.zip"
    output_dir = tmp_path / "output"
    _create_archives(images_path, labels_path, missing_split="val")

    with ZipFile(images_path) as images_zip, ZipFile(labels_path) as labels_zip:
        with pytest.raises(ValueError, match="No image or label files found for the val split"):
            bdd.copy_sampled_pairs(images_zip, labels_zip, output_dir, seed=42)

    assert not output_dir.exists()


def test_main_rejects_non_zip_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    images_path = tmp_path / "images.zip"
    labels_path = tmp_path / "labels.zip"
    images_path.write_text("not a zip", encoding="utf-8")
    labels_path.write_text("not a zip", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "load_bdd100k_dataset.py",
            "--images-zip",
            str(images_path),
            "--labels-zip",
            str(labels_path),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        bdd.main()


def test_main_reports_unpaired_archives_as_cli_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    images_path = tmp_path / "images.zip"
    labels_path = tmp_path / "labels.zip"
    _create_archives(images_path, labels_path, missing_label="train")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "load_bdd100k_dataset.py",
            "--images-zip",
            str(images_path),
            "--labels-zip",
            str(labels_path),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        bdd.main()

    assert "Image/label files do not pair in train" in capsys.readouterr().err
