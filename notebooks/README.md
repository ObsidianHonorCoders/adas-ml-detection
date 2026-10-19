# Dataset loading scripts

This folder contains scripts for preparing small COCO and BDD100K subsets for
experimentation. The scripts use different input workflows:

- COCO is downloaded by FiftyOne; no local dataset files or paths are needed.
- BDD100K must be downloaded from the official site first; the script takes
  the image and label ZIP paths as command-line arguments.

## COCO 2017

### Inputs and preconditions

No input paths are required. Run the script from the repository root with the
project Python environment and an internet connection:

```bat
python notebooks\load_coco_dataset.py
```

The script downloads COCO 2017 subsets through FiftyOne Dataset Zoo, so
FiftyOne must be installed in the active Python environment. The script
selects detections for `person`, `bicycle`, `car`, `motorcycle`, `truck`, and
`bus` in the train and validation splits. The test split is loaded without a
class filter. A fixed random seed makes partial selection repeatable.

The current total-size setting is 2,000 images, split using 70/10/20 ratios:

| Split | Requested maximum |
|---|---:|
| train | 1,400 |
| validation | 200 |
| test | 400 |

The actual number loaded can be lower if fewer samples are available or match
the requested selection. COCO test ground-truth annotations are not publicly
released, so the test `labels.json` may not contain object annotations.

### Output

FiftyOne's dataset zoo directory is configured to `data/processed`. Each split
is stored under `data/processed/coco-2017`:

```text
data/processed/coco-2017/
├── train/
│   ├── data/
│   │   └── <image files>
│   └── labels.json
├── validation/
│   ├── data/
│   │   └── <image files>
│   └── labels.json
└── test/
    ├── data/
    │   └── <image files>
    └── labels.json
```

Each `labels.json` contains COCO-style dataset metadata and annotations
associated with that split. For train and validation, these include the
detections loaded for the target classes. The test split has no public
ground-truth detection annotations.

## BDD100K

### Inputs and preconditions

Download the BDD100K images and labels archives before running the script.
The official download page is [bdd-data.berkeley.edu/download.html](http://bdd-data.berkeley.edu/download.html).
The script does not download the archives itself.

Provide the two local ZIP paths:

```bat
python notebooks\load_bdd100k_dataset.py ^
  --images-zip "C:\path\to\bdd100k_images_100k.zip" ^
  --labels-zip "C:\path\to\bdd100k_labels.zip"
```

Both ZIP archives must contain files beneath `100k/train`, `100k/val`, and
`100k/test`. Image and label files must pair by filename stem within the same
split; for example, `100k/train/frame-id.jpg` pairs with
`100k/train/frame-id.json`. The script checks that all image and label stems
match before copying.

The current configuration uses a total subset size of 2,000 with 70/10/20
split ratios. These are per-split maximums; the script copies fewer only if
fewer paired files are available:

| Split | Maximum pairs |
|---|---:|
| train | 1,400 |
| val | 200 |
| test | 400 |

Selection uses the fixed seed `42`, so the same archives and settings produce
the same selected filenames.

### Output

The script creates `data/processed/BDD100k` with paired image and label files:

```text
data/processed/BDD100k/
├── images/
│   ├── train/
│   │   └── <image files>
│   ├── val/
│   │   └── <image files>
│   └── test/
│       └── <image files>
└── labels/
    ├── train/
    │   └── <one JSON label file per image>
    ├── val/
    │   └── <one JSON label file per image>
    └── test/
        └── <one JSON label file per image>
```

An image and its label retain the same filename stem in their respective
split directories. Existing output files with the same paths are overwritten;
unrelated files already in the output tree are left untouched.
