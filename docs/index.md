# DonaNet Documentation

![WildINTEL](img/wildIntel_logo.webp){ style="display: block; margin: 0 auto;" }

Welcome to the DonaNet documentation.

**DonaNet** is a neural network designed to detect and classify the mammals that inhabit
[Doñana National Park](https://www.miteco.gob.es/es/red-parques-nacionales/nuestros-parques/donana/) (**SW**  Spain). **located in the Mediterranean biogeographical region**. **It has been developed within the framework of the WildINTEL project, funded by Biodiversa+ under the 2022-2023 BiodivMon joint call. It has been co-funded by the European Commission (GA No. 101052342) and the following funding organisations: Agencia Estatal de Investigación (Spain, PCI2023-145963-2, PCI2024-153489), National Science Centre (Poland, UMO-2023/05/Y/NZ8/00104), the Research Council of Norway (Norway, NFR350962) and the German Research Foundation (Germany).** 

This project provides the **pre-trained weights** of a [YOLO](https://docs.ultralytics.com/) network
specifically adapted to Doñana, as well as the **`donanet.py`** CLI, which allows users to train
or test YOLO-based mammal detection models.

---

## Documentation Map

- [DonaNet Model](donanet-model.md)
- [Installation Guide](installation-guide.md)
- [User Guide](user-guide.md)
- [Administrator Guide](admin-guide.md)
- [About](about.md)

---

## Core Concepts

- DonaNet is based on the Ultralytics YOLO object-detection framework.
- The pretrained DonaNet weights are expected at `weights/donanet_weights.pt`.
- The dataset follows the YOLO directory structure with separate `images/` and `labels/` folders.
- Dataset partitions are stored under `dataset/images/train`, `dataset/images/val`, `dataset/images/test`,
  `dataset/labels/train`, `dataset/labels/val` and `dataset/labels/test`.
- Labels follow the YOLO format: `<class_id> <x_center> <y_center> <width> <height>`.
- Bounding box coordinates are normalised to the image dimensions.
- Training outputs are saved under `output/training/YYYYMMDD/`.
- Testing pretrained weights saves predictions under `run/predictions_YYYYMMDD_HHMM/`.
- Testing newly trained weights saves predictions under `output/testing/predictions_YYYYMMDD_HHMM/`.
- Evaluation statistics are generated only when `dataset/annotations.csv` is available and follows the expected format.
