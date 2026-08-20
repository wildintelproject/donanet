# DonaNet — Model Overview

**DonaNet** is a one-stage YOLO neural network trained to detect and classify the mammals and
other wildlife present in camera-trap images from
[Doñana National Park](https://www.miteco.gob.es/es/red-parques-nacionales/nuestros-parques/donana/)
(Andalusia, Spain). In a single forward pass it simultaneously localises each animal in the image
and assigns it a species-level label, making it practical for large-scale automated annotation of
camera-trap image streams.

The model was developed in the context of the
[WildINTEL project](https://wildintel.eu/), funded by Biodiversa+ (Joint Research Call 2022–2023).

---

## Architecture

DonaNet is based on the **YOLOv8x6** architecture from
[Ultralytics](https://docs.ultralytics.com/), initialised from COCO pretrained weights and
fine-tuned end-to-end on the Doñana camera-trap dataset. Unlike two-stage pipelines that run a
detector and a classifier sequentially, DonaNet performs detection and species classification
jointly in a single inference step.

| Property               | Value                        |
|------------------------|------------------------------|
| Base architecture      | YOLOv8x6                     |
| Pretrained starting point | COCO                      |
| Task                   | Detection + classification   |
| Output classes         | 16 (see table below)         |
| Input resolution       | 832 × 832 px                 |
| Parameters             | ≈ 88 M                       |
| Computational cost     | ≈ 230.4 GFLOPs               |

---

## Detected Classes

The model outputs 16 classes: 14 mammal species, birds (as a single class), and one label for humans and vehicles.
Empty images — those containing no annotated object — were used as negative training examples but
are not a separate output class.

| Class label            | Common name (EN)       | Notes                                      |
|------------------------|------------------------|--------------------------------------------|
| *Lynx pardinus*        | Iberian lynx           | Endangered; flagship species of Doñana     |
| *Sus scrofa*           | Wild boar              |                                            |
| *Cervus elaphus*       | Red deer               |                                            |
| *Dama dama*            | Fallow deer            |                                            |
| *Equus* sp.            | Horse / donkey         | Semi-free-ranging horses of Doñana         |
| *Felis catus*          | Feral cat              |                                            |
| *Genetta genetta*      | Common genet           |                                            |
| *Bos taurus*           | Cattle                 | Free-ranging cattle                        |
| *Meles meles*          | European badger        |                                            |
| *Vulpes vulpes*        | Red fox                |                                            |
| *Canis familiaris*     | Dog                    |                                            |
| *Oryctolagus cuniculus*| European rabbit        |                                            |
| *Lepus granatensis*    | Iberian hare           |                                            |
| *Herpestes ichneumon*  | Egyptian mongoose      |                                            |
| Ave                    | Bird (any species)     | All bird species collapsed into one class  |
| Homo sapiens           | Human and vehicle      |                                            |


---

## Training Dataset

All experiments were based on a single camera-trap dataset collected in Doñana National Park.

| Property                    | Value                                    |
|-----------------------------|------------------------------------------|
| Unique images               | 51,874                                   |
| Total bounding-box annotations | 60,100                                |
| Classes                     | 16                                      |
| Negative examples           | Empty images, not used as an output class|
| Annotation format           | YOLO normalised bounding boxes           |

### Class distribution

The table below shows the number of images and bounding boxes per category across the full
dataset and each split.

| Category               | Total images / bbxs | Train images / bbxs | Val images / bbxs | Test images / bbxs |
|------------------------|---------------------|---------------------|-------------------|--------------------|
| Ave                    | 1,704 / 2,576       | 1,317 / 2,033       | 195 / 301         | 192 / 242          |
| *Bos taurus*           | 3,001 / 6,002       | 2,480 / 5,157       | 221 / 388         | 300 / 457          |
| *Canis familiaris*     | 1,591 / 1,844       | 1,288 / 1,487       | 153 / 187         | 150 / 170          |
| *Cervus elaphus*       | 3,034 / 4,528       | 2,522 / 3,773       | 207 / 276         | 305 / 479          |
| *Dama dama*            | 2,998 / 3,862       | 2,476 / 3,115       | 295 / 411         | 227 / 336          |
| *Oryctolagus cuniculus*| 3,399 / 3,620       | 1,873 / 2,011       | 752 / 801         | 774 / 808          |
| Empty                  | 7,935 / —           | 5,877 / —           | 1,022 / —         | 1,036 / —          |
| *Equus* sp.            | 3,108 / 7,360       | 2,502 / 5,885       | 303 / 602         | 303 / 873          |
| *Felis catus*          | 3,418 / 3,455       | 1,417 / 1,444       | 1,001 / 1,006     | 1,000 / 1,005      |
| *Genetta genetta*      | 1,745 / 1,799       | 1,428 / 1,474       | 165 / 167         | 152 / 158          |
| *Herpestes ichneumon*  | 2,864 / 3,631       | 2,322 / 2,951       | 260 / 326         | 282 / 354          |
| Homo sapiens           | 2,991 / 4,471       | 2,229 / 3,221       | 368 / 609         | 394 / 641          |
| *Lepus granatensis*    | 1,879 / 1,961       | 1,657 / 1,731       | 120 / 126         | 102 / 104          |
| *Lynx pardinus*        | 2,872 / 4,006       | 2,307 / 3,227       | 265 / 359         | 300 / 420          |
| *Meles meles*          | 3,688 / 3,797       | 3,081 / 3,180       | 350 / 357         | 257 / 260          |
| *Sus scrofa*           | 3,035 / 4,120       | 2,498 / 3,373       | 288 / 360         | 249 / 387          |
| *Vulpes vulpes*        | 2,925 / 3,068       | 2,452 / 2,592       | 275 / 277         | 198 / 199          |
| **Total**              | **52,187 / 60,100** | **39,726 / 46,654** | **6,240 / 6,553** | **6,221 / 6,893**  |

The dataset contains 51,874 unique images. The total number of images (52,187) summed across categories is higher because a single image may contain individuals from multiple categories and is therefore counted once for each relevant label.

### Dataset split strategy

To prevent information leakage caused by near-identical images appearing in both training and
test sets (a common problem with camera-trap sequences), the dataset was partitioned by
**camera location and temporal sequence** rather than by random image splitting. A temporal
sequence was defined as a set of consecutive images captured with less than 90 seconds between
them. All images from the same camera location and temporal sequence were assigned exclusively to
a single subset. The resulting split ratio was approximately **80 : 10 : 10**
(train : val : test), though exact counts vary due to the spatial and temporal constraints of
camera-trap sequences.

---

## Training Specifications

DonaNet (YOLO-1stage) was trained as an end-to-end one-stage model using the
[Ultralytics](https://docs.ultralytics.com/) training pipeline.

| Hyperparameter         | Value                              |
|------------------------|------------------------------------|
| Framework              | PyTorch (via Ultralytics)          |
| Input resolution       | 832 × 832 px                       |
| Batch size             | 16                                 |
| Initial learning rate  | lr0 = 1 × 10⁻⁴                     |
| Final learning rate    | lrf = 0.01                         |
| Max epochs             | 1,000                              |
| Early stopping patience| 100 consecutive epochs             |
| Final training lengths | 581 / 374 / 714 epochs             |

The model was trained on the Doñana dataset to perform localisation and class assignment
simultaneously within a single architecture over the harmonised 16-class output label space.
Empty images were used as negative training examples but not as a prediction class.

---

## Evaluation

Performance was evaluated on the held-out test split using **Precision**, **Recall**, and **F1
score** as the primary metrics. Both
micro-averaged (instance-weighted) and macro-averaged (class-weighted) variants were reported to
account for class imbalance.

For object detection, bounding-box predictions were matched to ground-truth annotations using
**Intersection over Union (IoU)** at thresholds of 0.5, 0.6, 0.7, 0.8, and 0.9. A prediction
was considered correct only when it both localised the object sufficiently well (IoU ≥ threshold)
and assigned the correct class label.

Empty images (no annotated objects) were included in the test set as negative examples; any
detection produced in an empty image was counted as a false positive.

### Results

On the held-out test split, DonaNet achieved:

| Metric         | Micro value | Macro value |
|----------------|-------------|-------------|
| Recall         | 92.4 %      | 92.1 %      |
| Precision      | 90.7 %      | 89.0 %      |
| F1 - score     | 91.6 %      | 90.4 %      |


These results were obtained on the same Doñana test split used throughout the study. For
comparison, the best two-stage pipeline (DeepFaune detector + ViT-Large classifier) achieved
92.1%/91.6% F1-score on the same task, at the cost of a substantially higher computational
footprint and a more complex inference workflow.

---

## Context and Reference

DonaNet was developed and evaluated as part of a broader study comparing one-stage and two-stage
annotation workflows for camera-trap images in Doñana National Park. The study is described in
full in the following manuscript:

> **Automatic Annotation of Camera-Trap Images in Doñana National Park.**  
> Departamento de Ciencias Integradas, Facultad de Ciencias Experimentales,  
> Universidad de Huelva — Campus El Carmen, Avda. Fuerzas Armadas, s/n, 21071 Huelva, Spain.  
> *(Revised integrated draft, 2025)*

The study demonstrated that, for the Doñana dataset, 
the one-stage end-to-end YOLO approach achieves annotation performance 
similar to the optimised two-stage detection-and-classification workflow, 
while providing a simpler inference pipeline for biodiversity monitoring.