# COCO and BDD100K datasets

This note summarizes the detection labels relevant to this ADAS project and
defines a common six-class mapping for comparing or combining annotations.
Dataset category IDs, BDD100K label strings, and YOLO class IDs are different
things; do not treat them as interchangeable.

## COCO

COCO (Common Objects in Context) is a general-purpose dataset for object
detection, segmentation, and image captioning. Its 2017 detection dataset
provides images and bounding-box annotations across 80 object categories.
COCO is useful as a general-object reference, but it is not specifically a
driving dataset.

COCO annotation `category_id` values are official, sparse IDs. The 80
detection categories are:

| COCO `category_id` | Category | COCO `category_id` | Category |
|---:|---|---:|---|
| 1 | person | 2 | bicycle |
| 3 | car | 4 | motorcycle |
| 5 | airplane | 6 | bus |
| 7 | train | 8 | truck |
| 9 | boat | 10 | traffic light |
| 11 | fire hydrant | 13 | stop sign |
| 14 | parking meter | 15 | bench |
| 16 | bird | 17 | cat |
| 18 | dog | 19 | horse |
| 20 | sheep | 21 | cow |
| 22 | elephant | 23 | bear |
| 24 | zebra | 25 | giraffe |
| 27 | backpack | 28 | umbrella |
| 31 | handbag | 32 | tie |
| 33 | suitcase | 34 | frisbee |
| 35 | skis | 36 | snowboard |
| 37 | sports ball | 38 | kite |
| 39 | baseball bat | 40 | baseball glove |
| 41 | skateboard | 42 | surfboard |
| 43 | tennis racket | 44 | bottle |
| 46 | wine glass | 47 | cup |
| 48 | fork | 49 | knife |
| 50 | spoon | 51 | bowl |
| 52 | banana | 53 | apple |
| 54 | sandwich | 55 | orange |
| 56 | broccoli | 57 | carrot |
| 58 | hot dog | 59 | pizza |
| 60 | donut | 61 | cake |
| 62 | chair | 63 | couch |
| 64 | potted plant | 65 | bed |
| 67 | dining table | 70 | toilet |
| 72 | tv | 73 | laptop |
| 74 | mouse | 75 | remote |
| 76 | keyboard | 77 | cell phone |
| 78 | microwave | 79 | oven |
| 80 | toaster | 81 | sink |
| 82 | refrigerator | 84 | book |
| 85 | clock | 86 | vase |
| 87 | scissors | 88 | teddy bear |
| 89 | hair drier | 90 | toothbrush |

### Loading an ADAS-focused COCO sample

Run [`notebooks/load_coco_dataset.py`](../notebooks/load_coco_dataset.py) to
load up to 2,000 randomly selected COCO 2017 validation images containing one
or more of the six ADAS target classes above. The script stores the downloaded
COCO subset under `data/processed/coco-2017` and exposes only the matching
detection labels in its FiftyOne dataset. Change `SAMPLE_SIZE` in the script
to request a different maximum. The returned sample count may be lower if
fewer images meet the class filter.

This download remains in COCO/FiftyOne dataset format; it is not automatically
converted to the YOLO directory structure used for model training.

## BDD100K

BDD100K is a large, geographically and environmentally diverse driving
dataset with video and image annotations for several tasks, including object
detection, multi-object tracking, semantic/instance segmentation, lane
detection, and drivable-area segmentation. The annotation vocabulary depends
on the task: the 40-class panoptic taxonomy is not the same thing as the
10-category object-detection taxonomy. BDD100K's semantic-segmentation
benchmark has its own 19 evaluated classes.

### The 40 panoptic region and object classes

BDD100K documents 40 panoptic classes: 30 "stuff" classes (IDs 1–30) for
regions and scene surfaces, and 10 "thing" classes (IDs 31–40) for
instance-level objects. The label schema also reserves ID 0 for
`unlabeled`/void. Thus the table shows the 40 named panoptic classes plus
that non-target void label. Many classes describe scene regions or
ambiguous/ignored areas rather than independently detectable or trackable
instances. These are panoptic label IDs, not detection category IDs, the
separate 19-class semantic-segmentation IDs, or YOLO IDs.

| Label ID(s) | Panoptic label(s) | What they represent |
|---:|---|---|
| 0 | `unlabeled` | Unannotated or void pixels; ignored in evaluation. |
| 1–4 | `dynamic`, `ego vehicle`, `ground`, `static` | Catch-all, camera/host vehicle, ground, and static/void regions. These are not target object classes. |
| 5–8 | `parking`, `rail track`, `road`, `sidewalk` | Flat surfaces and travel areas. |
| 9–15 | `bridge`, `building`, `fence`, `garage`, `guard rail`, `tunnel`, `wall` | Built structures and roadside infrastructure. |
| 16–27 | `banner`, `billboard`, `lane divider`, `parking sign`, `pole`, `polegroup`, `street light`, `traffic cone`, `traffic device`, `traffic light`, `traffic sign`, `traffic sign frame` | Roadside objects and traffic-control infrastructure. Several labels are ignored or grouped in segmentation evaluation; they are not all detection categories. |
| 28–30 | `terrain`, `vegetation`, `sky` | Natural/background regions. |
| 31–32 | `person`, `rider` | Human and human-on-vehicle instances. |
| 33–40 | `bicycle`, `bus`, `car`, `caravan`, `motorcycle`, `trailer`, `train`, `truck` | Vehicle instances. `caravan` and `trailer` are ignored in the standard semantic evaluation mapping. |

This taxonomy is designed for dense scene understanding: it can distinguish
road, sidewalk, buildings, signs, vehicles, and people at the pixel level.
That is a different goal from producing a compact list of bounding-box
categories for detection or assigning identities to moving objects over
video.

### Detection categories and tracking distinction

BDD100K object detection evaluates 10 road-scene categories. Its official
detection `category_id` values are 1-based. The example YOLO IDs below are
the zero-based conversion order shown in this repository's README; in this
ordering they equal the BDD100K detection IDs minus one. Always convert
explicitly rather than assuming dataset IDs and model IDs are interchangeable.

| BDD100K detection `category_id` | Example YOLO ID | Detection category | Panoptic label ID, if applicable |
|---:|---:|---|---:|
| 1 | 0 | `pedestrian` | 31 (`person`) |
| 2 | 1 | `rider` | 32 (`rider`) |
| 3 | 2 | `car` | 35 |
| 4 | 3 | `truck` | 40 |
| 5 | 4 | `bus` | 34 |
| 6 | 5 | `train` | 39 |
| 7 | 6 | `motorcycle` | 37 |
| 8 | 7 | `bicycle` | 33 |
| 9 | 8 | `traffic light` | 25 |
| 10 | 9 | `traffic sign` | 26 |

The 40-to-10 relationship is **not a literal collapse of all 40 labels into
10**. They are different task vocabularies. Panoptic/semantic annotation
classifies all pixels, including background regions, structures, and small
infrastructure; object detection selects a compact set of bounding-box object
categories. Region labels such as `road`, `sidewalk`, `sky`, and `building`
have no individual object instance to detect or track. Void/catch-all labels
such as `unlabeled`, `dynamic`, and `static` do not provide a reliable target
identity, and many fine-grained infrastructure labels are outside the
detection vocabulary. Some object concepts correspond across tasks (for
example, panoptic `person` and detection `pedestrian`), but they have
different annotation formats and IDs. Use the detection boxes/instances for
detection or tracking; do not infer object boxes from semantic pixels without
a separately defined conversion.

Also distinguish detection from **multi-object tracking (MOT)**: BDD100K
tracking evaluates only the first eight detection categories (`pedestrian`,
`rider`, `car`, `truck`, `bus`, `train`, `motorcycle`, and `bicycle`), for
which instance identity across video frames is meaningful. `traffic light`
and `traffic sign` are the remaining two detection categories, but are not
included in the official box-tracking or segmentation-tracking category set.
Therefore, call the 10 labels "detection categories" and the eight labels
"tracking classes"; a project may define a different tracking scope.

For this project, the six-class ADAS crosswalk below is a further filter of
the detection vocabulary: it excludes `rider`, `train`, `traffic light`, and
`traffic sign`, and maps `pedestrian` to the target `person`. That choice
keeps only the requested targets; it does not describe all BDD100K labels.

## Six-class ADAS crosswalk

The following common IDs are a **proposed target mapping** for a six-class
ADAS model. These contiguous IDs are not the source IDs of either dataset.
Use the same mapping for every dataset being converted to this model.

| Common YOLO ID | ADAS target | COCO `category_id` | BDD100K source category | BDD100K example YOLO ID |
|---:|---|---:|---|---:|
| 0 | person | 1 (`person`) | `pedestrian` | 0 |
| 1 | bicycle | 2 (`bicycle`) | `bicycle` | 7 |
| 2 | car | 3 (`car`) | `car` | 2 |
| 3 | motorcycle | 4 (`motorcycle`) | `motorcycle` | 6 |
| 4 | truck | 8 (`truck`) | `truck` | 3 |
| 5 | bus | 6 (`bus`) | `bus` | 4 |

### Mapping notes

- COCO IDs in the crosswalk are official `category_id` values. Convert them
  to the common IDs above during annotation conversion; do not use them as
  YOLO class IDs directly.
- BDD100K `pedestrian` is the closest match for the target `person`. BDD100K
  `rider` is a separate object category and should not be silently merged into
  `person`; decide explicitly whether the project needs to retain or exclude
  riders.
- BDD100K's `bicycle` and `motorcycle` labels identify the vehicles. They are
  separate from the `rider` label.
- The current [`configs/dataset.yaml`](../configs/dataset.yaml) defines a
  different five-class mapping (`car`, `truck`, `bus`, `pedestrian`,
  `cyclist`). It does not implement the six-class mapping in this document.
  Update the dataset configuration and conversion code together before
  training with these six targets.
- Class definitions and annotation conventions can differ between datasets,
  even when their names appear equivalent. Validate boxes, ignore/crowd
  annotations, and split handling when converting.

## Sources

- [COCO dataset and downloads](https://cocodataset.org/#download)
- [COCO image explorer](https://cocodataset.org/#explore)
- [BDD100K official website](https://bdd-data.berkeley.edu/)
- [BDD100K official repository](https://github.com/bdd100k/bdd100k)
- [BDD100K label format and task class IDs](https://github.com/bdd100k/bdd100k/blob/master/doc/source/format.rst)
- [BDD100K semantic/panoptic label definitions](https://github.com/bdd100k/bdd100k/blob/master/bdd100k/label/label.py)
- [BDD100K dataset paper](https://arxiv.org/abs/1805.04687)
