"""Create a paired BDD100K subset by sampling image/label ZIP archives.

CLI:
    python notebooks\\load_bdd100k_dataset.py ^
        --images-zip workspace\\bdd100k_images_100k.zip ^
        --labels-zip workspace\\bdd100k_labels.zip

Samples up to 1,400 train, 200 val, and 400 test pairs, selected
deterministically with ``RANDOM_SEED``. Files are written to
``data\\processed\\BDD100k\\images`` and ``data\\processed\\BDD100k\\labels``,
with a subfolder for each split. Existing files with matching paths are
overwritten; unrelated files are left untouched.
"""

import argparse
import random
import shutil
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile, is_zipfile

ROOT = Path(__file__).resolve().parents[1]

TOTAL_SIZE = 2_000

SPLITS = ["train", "val", "test"]

SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.1,
    "test": 0.2,
}

SPLIT_SIZES = {split: int(TOTAL_SIZE * ratio) for split, ratio in SPLIT_RATIOS.items()}

RANDOM_SEED = 42


def get_split_files(archive: ZipFile, split: str) -> dict[str, str]:
    """Return ZIP members for one split, indexed by their filename stem.

    Args:
        archive: Open ZIP archive containing a ``100k`` directory.
        split: Dataset split name, one of ``train``, ``val``, or ``test``.

    Returns:
        A mapping from file stem to its ZIP member path.

    Raises:
        ValueError: If a split contains duplicate file stems.
    """
    files: dict[str, str] = {}
    for member in archive.namelist():
        if member.endswith("/"):
            continue

        parts = PurePosixPath(member).parts
        try:
            root_index = parts.index("100k")
        except ValueError:
            continue

        if len(parts) <= root_index + 2 or parts[root_index + 1] != split:
            continue

        stem = PurePosixPath(parts[-1]).stem
        if stem in files:
            raise ValueError(
                f"Duplicate filename stem {stem!r} in {split} split of "
                f"{archive.filename}"
            )
        files[stem] = member

    return files


def copy_sampled_pairs(
    images_zip: ZipFile,
    labels_zip: ZipFile,
    output_dir: Path,
    seed: int,
) -> dict[str, int]:
    """Sample and copy paired image/label files for every split.

    Each split is sampled independently up to its ``SPLIT_SIZES`` cap.
    Pairing uses filename stems, so a label such as ``frame.json`` matches
    ``frame.jpg``. Split subdirectories are preserved under both output trees.

    Args:
        images_zip: Open ZIP archive containing BDD100K images.
        labels_zip: Open ZIP archive containing per-image BDD100K labels.
        output_dir: Destination dataset directory.
        seed: Random seed used for repeatable subset selection.

    Returns:
        A mapping of split names to numbers of copied image/label pairs.

    Raises:
        ValueError: If a split is absent or image and label stems do not match.
    """
    image_members = {split: get_split_files(images_zip, split) for split in SPLITS}
    label_members = {split: get_split_files(labels_zip, split) for split in SPLITS}

    for split in SPLITS:
        if not image_members[split] and not label_members[split]:
            raise ValueError(
                f"No image or label files found for the {split} split in the "
                "ZIP archives."
            )

        image_stems = set(image_members[split])
        label_stems = set(label_members[split])
        if image_stems != label_stems:
            missing_images = sorted(label_stems - image_stems)[:5]
            missing_labels = sorted(image_stems - label_stems)[:5]
            raise ValueError(
                f"Image/label files do not pair in {split}: "
                f"{len(label_stems - image_stems)} labels lack images "
                f"(examples: {missing_images}); "
                f"{len(image_stems - label_stems)} images lack labels "
                f"(examples: {missing_labels})"
            )

    rng = random.Random(seed)
    copied_counts: dict[str, int] = {}

    for split in SPLITS:
        stems = sorted(image_members[split])
        sample_size = min(SPLIT_SIZES[split], len(stems))
        selected_stems = rng.sample(stems, sample_size)

        for stem in selected_stems:
            image_member = image_members[split][stem]
            label_member = label_members[split][stem]
            image_name = PurePosixPath(image_member).name
            label_name = PurePosixPath(label_member).name

            image_destination = output_dir / "images" / split / image_name
            label_destination = output_dir / "labels" / split / label_name
            image_destination.parent.mkdir(parents=True, exist_ok=True)
            label_destination.parent.mkdir(parents=True, exist_ok=True)

            with (
                images_zip.open(image_member) as source,
                image_destination.open("wb") as target,
            ):
                shutil.copyfileobj(source, target)
            with (
                labels_zip.open(label_member) as source,
                label_destination.open("wb") as target,
            ):
                shutil.copyfileobj(source, target)

        copied_counts[split] = sample_size
        print(
            f"{split}: copied {sample_size} of {len(stems)} available "
            f"image/label pairs (cap {SPLIT_SIZES[split]})."
        )

    return copied_counts


def main() -> None:
    """Parse the ZIP paths and create the sampled BDD100K dataset.

    Args:
        None.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(
        description=("Randomly sample paired BDD100K image and label files from ZIPs.")
    )
    parser.add_argument(
        "--images-zip",
        type=Path,
        required=True,
        help="Path to the BDD100K images ZIP archive.",
    )
    parser.add_argument(
        "--labels-zip",
        type=Path,
        required=True,
        help="Path to the BDD100K labels ZIP archive.",
    )
    args = parser.parse_args()

    for archive_path, option in (
        (args.images_zip, "--images-zip"),
        (args.labels_zip, "--labels-zip"),
    ):
        if not archive_path.is_file():
            parser.error(f"{option} file not found: {archive_path}")
        if not is_zipfile(archive_path):
            parser.error(f"{option} is not a valid ZIP archive: {archive_path}")

    output_dir = ROOT / "data" / "processed" / "BDD100k"
    try:
        with (
            ZipFile(args.images_zip) as images_zip,
            ZipFile(args.labels_zip) as labels_zip,
        ):
            copied_counts = copy_sampled_pairs(
                images_zip,
                labels_zip,
                output_dir,
                RANDOM_SEED,
            )
    except BadZipFile as exc:
        parser.error(f"Unable to read an input ZIP archive: {exc}")
    except ValueError as exc:
        parser.error(str(exc))

    print(f"BDD100K subset created at: {output_dir}")
    print(f"Total image/label pairs copied: {sum(copied_counts.values())}")


if __name__ == "__main__":
    main()
