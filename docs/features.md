# Features


## Pretrained YOLO network for Doñana

DonaNet provides a ready-to-use YOLO model trained on camera-trap imagery from
[Doñana National Park](https://www.miteco.gob.es/es/red-parques-nacionales/nuestros-parques/donana/).

The pretrained weights are distributed through GitHub Releases because the weights file is large.
After downloading the released weights file, place it in the `weights/` directory and rename it to:

`donanet_weights.pt`

The expected path is:

`weights/donanet_weights.pt`

No training is required to test the pretrained model.

- Detects and classifies mammal species present in Doñana National Park.
- Adapted to camera-trap imagery from Doñana National Park.
- Based on [Ultralytics YOLO](https://docs.ultralytics.com/).
- Can generate predictions and evaluation statistics when the required `annotations.csv` file is available.

---

## Training on YOLO-format datasets

The included `donanet.py` CLI allows training a new YOLOv8x6 model on the DonaNet dataset or on another compatible YOLO-format dataset.

The dataset must follow the expected YOLO directory structure:

`dataset/images/train/`
`dataset/images/val/`
`dataset/images/test/`
`dataset/labels/train/`
`dataset/labels/val/`
`dataset/labels/test/`
`dataset/data.yaml`

- Uses `dataset/data.yaml` as the dataset configuration file.
- Uses pretrained `yolov8x6.pt` weights as the starting model.
- Saves training outputs under `output/training/YYYYMMDD/`.
- Saves the main trained weights as `best.pt` and `last.pt`.
- Allows newly trained weights to be tested with the same `test` command.

---

## Testing and evaluation

DonaNet can run inference or evaluation using either the pretrained DonaNet weights or newly trained weights.

Testing the pretrained weights:

`python donanet.py test --weights weights/donanet_weights.pt --conf 0.25`

Outputs are saved under:

`run/predictions_YYYYMMDD_HHMM/`

Testing newly trained weights:

`python donanet.py test --weights output/training/YYYYMMDD/weights/best.pt --conf 0.25`

Outputs are saved under:

`output/testing/predictions_YYYYMMDD_HHMM/`

The main outputs are:

`predictions.csv`
`metrics_summary.xlsx`

The `metrics_summary.xlsx` file is generated only when `dataset/annotations.csv` exists and follows the expected format.

---

## Interactive CLI

All functionality is exposed through a command-line interface built with
[Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/).

Available commands include:

- `train` — train DonaNet using the standard YOLOv8x6 configuration.
- `test` — run inference or evaluation with a selected weights file.
- `info` — show available weights and dataset summary.
- `list-datasets` — show dataset partitions and image counts.

Example commands:

`python donanet.py test --weights weights/donanet_weights.pt --conf 0.25`

`python donanet.py train`

`python donanet.py test --weights output/training/YYYYMMDD/weights/best.pt --conf 0.25`

`python donanet.py info`

`python donanet.py list-datasets`
