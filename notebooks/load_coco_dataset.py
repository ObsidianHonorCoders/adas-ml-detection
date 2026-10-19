"""Download ADAS-focused COCO 2017 train, validation, and test subsets."""

from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz

RANDOM_SEED = 42

ADAS_TARGETS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "truck",
    "bus",
]

SPLITS = ["train", "validation", "test"]

TOTAL_SIZE = 2_000

SPLIT_RATIOS = {
    "train": 0.7,
    "validation": 0.1,
    "test": 0.2,
}

SPLIT_SIZES = {split: int(TOTAL_SIZE * ratio) for split, ratio in SPLIT_RATIOS.items()}


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    processed_dir = project_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Configure where FiftyOne stores downloaded zoo dataset files
    fo.config.dataset_zoo_dir = str(processed_dir)

    datasets = {}

    for split in SPLITS:
        sample_size = SPLIT_SIZES[split]

        print(f"\nLoading COCO 2017 split: {split}")
        print(f"Target sample size: {sample_size}")

        kwargs = {
            "split": split,
            "label_types": ["detections"],
            "max_samples": sample_size,
            "shuffle": True,
            "seed": RANDOM_SEED,
            "dataset_name": f"adas_coco_{split}_{sample_size}",
            "persistent": True,
        }

        if split != "test":
            kwargs["classes"] = ADAS_TARGETS
            kwargs["only_matching"] = True

        dataset = foz.load_zoo_dataset("coco-2017", **kwargs)
        datasets[split] = dataset

        print(f"{split}: {dataset.count()} images")
        print(f"FiftyOne dataset name: {dataset.name}")

    print("\nFinished loading all splits.")


if __name__ == "__main__":
    main()
