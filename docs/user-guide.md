# User Guide
This guide explains how to:
- use DonaNet for testing the provided pretrained weights donanet_weights.pt 
- fine-tune yolov8x6.pt on DonaDataset
- obtain available outputs and statistics

DonaNet is designed around a prepared YOLO-format dataset available at: https://github.com/wildintelproject/donadataset

The repository can be used in two main ways:

1. Use the provided DonaNet weights to test the pretrained model on the test partition of the DonaNet dataset.
Because the pretrained model weights are large files, they are distributed through GitHub Releases instead of being stored directly in the repository.
After downloading the released weights file, place it in the `weights/` directory and rename it to `donanet_weights.pt` so that it can be used with the standard DonaNet testing command.
2. Train a new YOLOv8x6 model on the DonaNet dataset, or on another YOLO-format dataset placed in the dataset/ directory.

## Workflow Overview

```
dataset/
   │
   ├── images/
   │   ├── train/
   │   ├── val/
   │   └── test/
   │
   ├── labels/
   │   ├── train/
   │   ├── val/
   │   └── test/
   │
   ├── data.yaml
   └── annotations.csv
        │
        ▼
   train or test
        │
        ├── output/training/   ← new training outputs, plots, metrics and weights 
        ├── output/testing/    ← testing outputs for newly trained weights
        └── run/               ← testing outputs for the provided DonaNet weights
```

---

## Dataset

DonaNet expects the dataset to follow the YOLO detection format.
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── data.yaml
└── annotations.csv
```

Each image must have a corresponding .txt label file with the same base filename.

Example:

`dataset/images/train/image_001.jpg`
`dataset/labels/train/image_001.txt`

Each label file follows the YOLO bounding box normalized format:

`<class_id> <x_center> <y_center> <width> <height>`

All bounding box values are normalized to [0, 1] relative to the image dimensions.

The `<class_id>` value corresponds to the class index defined in `dataset/data.yaml`.

The YOLO `.txt` label files use the compact YOLO format, while `annotations.csv` stores the same annotation information in a tabular metadata format used for evaluation.

The `annotations.csv` file contains the original annotation metadata, including image name, class label, bounding box coordinates, image path, category and contributor/source information.

For evaluation, `annotations.csv` must contain at least the following columns: `file_name`, `category`, `bbox_x_center`, `bbox_y_center`, `bbox_width` and `bbox_height`. Additional columns such as `label`, `path`, `group`, contributor or source information can also be included.

If `annotations.csv` is missing or does not follow the expected format, the test command can still generate `predictions.csv`, but it cannot generate the full evaluation statistics in `metrics_summary.xlsx`.


### Preparation of the Dataset
The DonaNet repository expects a prepared YOLO-format dataset.

If you want to create a new dataset from raw camera-trap images, the images first need to be detected, labelled, checked and converted into the YOLO dataset structure used by DonaNet.

The basic workflow is:
```
raw images
   │
   ▼
detect animals and create bounding boxes
   │
   ▼
review and correct bounding boxes and labels
   │
   ▼
group images into sequences
   │
   ▼
split sequences into train / val / test
   │
   ▼
create YOLO .txt label files
   │
   ▼
final dataset/
```

### Bounding boxes 
Each object in an image must have a bounding box.
Bounding boxes can be obtained by:
- running predictions with an existing detector;
- drawing the bounding boxes manually.

For YOLO training, each bounding box must be stored in the corresponding YOLO `.txt` label file using the normalized YOLO format:
`<class_id> <x_center> <y_center> <width> <height>`

Where:
- `class_id` is the numerical class index defined in `dataset/data.yaml`;
- `x_center` is the normalized horizontal coordinate of the bounding-box centre;
- `y_center` is the normalized vertical coordinate of the bounding-box centre;
- `width` is the normalized width of the bounding box;
- `height` is the normalized height of the bounding box.

The `annotations.csv` file stores the same annotation information in a tabular metadata format. In `annotations.csv`, the relevant columns include:
- `label` — the numerical class value corresponding to the species in `dataset/data.yaml`;
- `category` — the scientific name of the species;
- `bbox_x_center` — the normalized horizontal coordinate of the bounding-box centre;
- `bbox_y_center` — the normalized vertical coordinate of the bounding-box centre;
- `bbox_width` — the normalized width of the bounding box;
- `bbox_height` — the normalized height of the bounding box.

All coordinate values must be normalized to `[0, 1]` relative to the image dimensions.


### Labels

Every detected object must be assigned to one class.

The class names must be consistent across:

`dataset/data.yaml`
`annotations.csv`
YOLO `.txt` label files

The class index used in the `.txt` label files must correspond to the class order defined in `data.yaml`.


### Sequence-based splitting

Before sorting images into train, val and test, the images should be grouped by time and place.
The recommended split is:

80% train
10% validation
10% test

The split should be done by sequence, not by individual image.
For this project, a sequence can be treated as a group of consecutive camera-trap images separated from the next sequence by approximately 1.5 minutes or more.

The final `annotations.csv` should contain the information needed to trace the original image, class label, bounding box.

After this preprocessing, the dataset is ready to be used by DonaNet for training and testing.

## Train

Train DonaNet using the standard training wrapper:

```bash
python donanet.py train
```

The command trains a YOLO-based detection model using the prepared DonaNet dataset stored in:

```text
dataset/
```

The training uses the `train` and `val` dataset partitions:

```text
dataset/images/train/
dataset/images/val/
dataset/labels/train/
dataset/labels/val/
```

The wrapper runs Ultralytics YOLO in detection mode using the standard DonaNet training configuration:

| Setting                 | Value               |
| ----------------------- | ------------------- |
| Base model              | `yolov8x6.pt`       |
| Dataset configuration   | `dataset/data.yaml` |
| Epochs                  | `1000`              |
| Early stopping patience | `100`               |
| Batch size              | `16`                |
| Output directory        | `output/training/`  |

The training run name is generated automatically from the current date in `YYYYMMDD` format, unless a custom name is provided with `--name`.

After training finishes, outputs are saved under:

```text
output/training/YYYYMMDD/
```

The main trained weights are saved as:

```text
output/training/YYYYMMDD/weights/best.pt
output/training/YYYYMMDD/weights/last.pt
```

Ultralytics YOLO also generates standard training statistics, logs and plots inside the same training run directory.

Typical training outputs include:

```text
output/training/YYYYMMDD/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.csv
├── results.png
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── labels.jpg
├── labels_correlogram.jpg
├── train_batch*.jpg
├── val_batch*.jpg
└── additional training logs and plots
```

Replace `YYYYMMDD` with the training run folder created during training.

---

## Test the provided DonaNet weights
Use this option if you want to test the pretrained DonaNet model provided with this repository.

The expected weights file is:

`weights/donanet_weights.pt`

Run:
```bash
python donanet.py test --weights weights/donanet_weights.pt --conf 0.25
```

The same action can also be run with the dedicated shortcut command:

`python donanet.py test-pretrained --conf 0.25`

When the provided pretrained DonaNet weights are used, testing outputs are saved under:

`run/predictions_YYYYMMDD_HHMM/`

The main outputs are:

`run/predictions_YYYYMMDD_HHMM/predictions.csv`
`run/predictions_YYYYMMDD_HHMM/metrics_summary.xlsx`

The `predictions.csv` file contains one row per detected object.

The `metrics_summary.xlsx` file is generated only when `dataset/annotations.csv` exists and follows the expected format. If `annotations.csv` is missing or incorrectly formatted, DonaNet displays a warning and skips the statistics generation step.

## Test newly trained weights
After training, test the newly trained model by pointing the test command to the generated best.pt file:

```bash
python donanet.py test --weights output/training/YYYYMMDD/weights/best.pt --conf 0.25
```
Replace YYYYMMDD with the training run folder created during training.

When newly trained weights are used, testing outputs are saved under:

`output/testing/predictions_YYYYMMDD_HHMM/`

The main outputs are:

`output/testing/predictions_YYYYMMDD_HHMM/predictions.csv`
`output/testing/predictions_YYYYMMDD_HHMM/metrics_summary.xlsx`

The `predictions.csv` file contains one row per detected object.

The `metrics_summary.xlsx` file is generated by the DonaNet evaluation logic implemented in `donanet.py`.

If `dataset/annotations.csv` does not exist or does not follow the expected format, DonaNet still saves the predictions, but the statistics file cannot be generated.

In that case, check the expected annotation format in the DonaNet dataset repository:

https://github.com/wildintelproject/donadataset

## Output files
### `predictions.csv`

The predictions.csv file contains one row per detected object.

Typical columns include:

```text
file_name
label
category
bbox_x_center
bbox_y_center
bbox_width
bbox_height
confidence
path
```

The bounding-box coordinates in predictions.csv are normalized to [0, 1].

### `metrics_summary.xlsx`

The `metrics_summary.xlsx` workbook contains evaluation statistics generated from the comparison between predictions and ground-truth annotations.

The evaluation requires:

`dataset/annotations.csv`

or another valid ground-truth CSV file passed with:

`--gt-csv`

The workbook may include per-class and summary statistics such as:


- ground-truth counts;
- prediction counts;
- TP, TN, FP and FN counts;
- precision;
- recall;
- accuracy;
- F1-score;
- MCC;
- AP50;
- mAP50-95;
- empty-image metrics;
- warnings.

Evaluation is performed on the test split. Predictions are matched to ground-truth boxes using Intersection over Union.

---

## info

Show available weights and a dataset summary in one go:

```bash
python donanet.py info
```
