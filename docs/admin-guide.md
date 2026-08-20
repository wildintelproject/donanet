# Administrator Guide
This guide describes how to maintain the DonaNet repository and publish a new project release, including updated pretrained model weights.
The administrator guide is intended for repository maintainers. Regular users should follow the Installation Guide and User Guide.

## Project Layout

```
Recommended repository layout:

donanet/
├── donanet.py                  ← CLI entry point
├── pyproject.toml              ← project metadata and dependencies
├── uv.lock                     ← locked dependency versions
├── README.md
├── mkdocs.yml                  ← documentation site configuration
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── data.yaml               ← YOLO dataset configuration
│   └── annotations.csv          ← tabular annotation file used for evaluation statistics
├── weights/
│   └── donanet_weights.pt       ← optional local pretrained weights file
├── output/
│   ├── training/
│   │   └── YYYYMMDD/
│   │       ├── weights/
│   │       │   ├── best.pt
│   │       │   └── last.pt
│   │       ├── results.csv
│   │       ├── confusion_matrix.png
│   │       └── ...
│   └── testing/
│       └── predictions_YYYYMMDD_HHMM/
│           ├── predictions/
│           ├── statistics/
│           └── ...
└── docs/
    ├── index.md
    ├── installation-guide.md
    ├── user-guide.md
    ├── admin-guide.md
    ├── img/
    └── stylesheets/
```
Because the pretrained model weights are large files, they are distributed through GitHub Releases instead of being stored directly in the repository.
After downloading the released weights file, place it in the `weights/` directory and rename it to `donanet_weights.pt` so that it can be used with the standard DonaNet testing command.

---

## Dataset Configuration

DonaNet expects a YOLO-format dataset with images and labels split into train, val and test partitions.

The required dataset configuration file is:

`dataset/data.yaml`
This file is not automatically generated.

Example structure:
```yaml
path: dataset
train: images/train
val: images/val
test: images/test

nc: <number_of_classes>
names:
  0: <class_0_name>
  1: <class_1_name>
  2: <class_2_name>
```
The `data.yaml` file is used by Ultralytics YOLO during training and testing.

---
## Annotation Files

YOLO training uses compact .txt label files stored under:
```text
dataset/labels/train/
dataset/labels/val/
dataset/labels/test/
```

Each YOLO label file contains one row per object:
```text
<class_id> <x_center> <y_center> <width> <height>
```
All bounding box values are normalized relative to image width and height.
For evaluation, `annotations.csv` must contain at least the following columns:

```text
file_name, category, bbox_x_center, bbox_y_center, bbox_width, bbox_height
```

Additional columns such as `label`, `path`, `group`, contributor or source information can also be included.

The YOLO `.txt` label files use the compact YOLO format, while annotations.csv stores the same annotation information in tabular form for evaluation and summary statistics.

---

## Training a New Model

To train a new DonaNet model, run:
```bash
python donanet.py train
```
Training outputs are saved under:
```text
output/training/YYYYMMDD/
```
The most important trained weight files are:
```text
output/training/YYYYMMDD/weights/best.pt
output/training/YYYYMMDD/weights/last.pt
```
Use `best.pt` as the candidate pretrained model for a new release, unless there is a specific reason to use another checkpoint.

## Testing Candidate Weights

Before publishing a new release, test the candidate weights on the test partition:

```bash
python donanet.py test --weights output/training/YYYYMMDD/weights/best.pt --conf 0.25
```

Testing outputs are saved under:
```text
output/testing/predictions_YYYYMMDD_HHMM/
```
Before publishing the weights, check the generated evaluation outputs, including available statistics, prediction files and visual outputs.
If the test images do not have a matching `annotations.csv` file with ground-truth annotations, DonaNet can still run inference, but evaluation statistics cannot be generated.

---

## Preparing Pretrained Weights for Release

After testing, copy the selected checkpoint to the release weights name:

```bash
mkdir -p weights
cp output/training/YYYYMMDD/weights/best.pt weights/donanet_weights.pt
```

The public pretrained weights file should be named:
```text
donanet_weights.pt
```
This name is used in the user documentation and should stay stable across releases.
Do not publish untested weights.

---

## Release Checklist

Before creating a new release, verify that:

- the code runs successfully,
- the documentation is up to date,
- `python donanet.py train` works,
- `python donanet.py test --weights output/training/YYYYMMDD/weights/best.pt --conf 0.25` works,
- the selected weights file is renamed to `donanet_weights.pt`,
- the release notes describe what changed,
- the release includes the pretrained weights file as a release asset.

---
## Creating a New GitHub Release

Use GitHub Releases to publish a new DonaNet version.

Recommended version format:

```text
vMAJOR.MINOR.PATCH
```

Examples:

```text
v1.0.0
v1.1.0
v1.1.1
```

Suggested meaning:

| Version type | Use when |
|---|---|
| `MAJOR` | major project changes or incompatible changes |
| `MINOR` | new functionality, new trained weights or relevant documentation updates |
| `PATCH` | bug fixes, small documentation fixes or minor corrections |

To publish a release manually:

1. Go to the DonaNet GitHub repository.
2. Open the Releases page.
3. Click Draft a new release.
4. Create a new tag, for example `v1.0.0`.
5. Add a release title, for example `DonaNet v1.0.0`.
6. Write release notes.
7. Attach the file `donanet_weights.pt` as a release asset.
8. Publish the release.

---

## Recommended Release Notes Template

Use the following structure for release notes:

```text
## DonaNet vX.Y.Z

### Added

- ...

### Changed

- ...

### Fixed

- ...

### Model weights

- Updated pretrained weights: donanet_weights.pt
- Base model: YOLOv8x6
- Dataset: DonaDataset
- Training output used: output/training/YYYYMMDD/weights/best.pt

### Notes
The pretrained weights can be used with:
python donanet.py test --weights weights/donanet_weights.pt --conf 0.25
```

---

## Downloading Release Weights

After publishing a release, users should download the released weights file from the latest GitHub Release and place it in:
```text
weights/donanet_weights.pt
```

Then they can test the pretrained model with:
```bash
python donanet.py test --weights weights/donanet_weights.pt --conf 0.25
```

---

## Adding New Commands

`donanet.py` uses a single `typer.Typer` application object named `app`.
Add new sub-commands with:

```python
@app.command()
def my_command(
    option: str = typer.Option("default", help="My option"),
):
    """Short description shown in --help."""
    ...
```

---

## Building Documentation

```bash
uv run mkdocs build     # static site → site/
uv run mkdocs serve     # live-reload dev server at http://127.0.0.1:8000
```
