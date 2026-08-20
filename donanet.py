#!/usr/bin/env python3
"""DonaNet — WildINTEL YOLO training & inference CLI."""

from __future__ import annotations

# import random
# import shutil
# import sys
# from pathlib import Path
# from typing import Annotated, Optional
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
# from rich.text import Text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ROOT = Path(__file__).parent.resolve()
# DATASET_DIR = Path(ROOT, "dataset")
# WEIGHTS_DIR = Path(ROOT, "weights")
# RUNS_DIR = Path(ROOT, "runs")
# DATASET_YAML = ROOT / "dataset.yaml"

# IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# PARTITIONS = ("train", "val", "test")
ROOT = Path(__file__).parent.resolve()

DATASET_DIR = ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
DATASET_YAML = DATASET_DIR / "data.yaml"

WEIGHTS_DIR = ROOT / "weights"
PRETRAINED_WEIGHTS = WEIGHTS_DIR / "donanet_weights.pt"
RUN_DIR = ROOT / "run"

OUTPUT_DIR = ROOT / "output"
TRAIN_OUTPUT_DIR = OUTPUT_DIR / "training"
TEST_OUTPUT_DIR = OUTPUT_DIR / "testing"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

PARTITIONS = ("train", "val", "test")

console = Console()

# ---------------------------------------------------------------------------
# Typer app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="donanet",
    help="DonaNet — WildINTEL YOLO training & inference CLI",
    add_completion=True,
    rich_markup_mode="rich",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# def _ensure_dirs() -> None:
#     """Create the standard dataset / weights / runs directories if missing."""
#     for partition in PARTITIONS:
#         for sub in ("images", "labels"):
#             (DATASET_DIR / partition / sub).mkdir(parents=True, exist_ok=True)
#     WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
#     RUNS_DIR.mkdir(parents=True, exist_ok=True)
def _ensure_dirs() -> None:
    """Create the standard dataset and output directories if missing."""
    for partition in PARTITIONS:
        (IMAGES_DIR / partition).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / partition).mkdir(parents=True, exist_ok=True)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# def _count_partition(partition: str) -> tuple[int, int]:
#     """Return (image_count, label_count) for a partition."""
#     img_dir = DATASET_DIR / partition / "images"
#     lbl_dir = DATASET_DIR / partition / "labels"
#     imgs = [f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS] if img_dir.exists() else []
#     lbls = [f for f in lbl_dir.iterdir() if f.suffix == ".txt"] if lbl_dir.exists() else []
#     return len(imgs), len(lbls)
def _count_partition(partition: str) -> tuple[int, int]:
    """Return (image_count, label_count) for a dataset partition."""
    img_dir = IMAGES_DIR / partition
    lbl_dir = LABELS_DIR / partition

    imgs = [f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS] if img_dir.exists() else []
    lbls = [f for f in lbl_dir.iterdir() if f.suffix.lower() == ".txt"] if lbl_dir.exists() else []

    return len(imgs), len(lbls)

# def _write_dataset_yaml(class_names: list[str]) -> None:
#     """Write (or overwrite) dataset.yaml at the project root."""
#     data = {
#         "path": str(ROOT),
#         "train": "dataset/train/images",
#         "val": "dataset/val/images",
#         "test": "dataset/test/images",
#         "nc": len(class_names),
#         "names": {i: name for i, name in enumerate(class_names)},
#     }
#     with DATASET_YAML.open("w") as fh:
#         yaml.dump(data, fh, default_flow_style=False, allow_unicode=True)
#     rprint(f"[green]✔[/green] dataset.yaml written → [bold]{DATASET_YAML}[/bold]")
def _write_dataset_yaml(class_names: list[str]) -> None:
    """Write or overwrite dataset/data.yaml."""
    data = {
        "path": str(DATASET_DIR),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
    }

    with DATASET_YAML.open("w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, allow_unicode=True)

    rprint(f"[green]✔[/green] data.yaml written → [bold]{DATASET_YAML}[/bold]")


# def _copy_weights(run_dir: Path, run_name: str) -> None:
#     """Copy best.pt / last.pt from a Ultralytics run dir into weights/<name>/."""
#     dest = WEIGHTS_DIR / run_name
#     dest.mkdir(parents=True, exist_ok=True)
#     copied = False
#     for fname in ("best.pt", "last.pt"):
#         src = run_dir / "weights" / fname
#         if src.exists():
#             shutil.copy2(src, dest / fname)
#             rprint(f"[green]✔[/green] {fname} → [bold]{dest / fname}[/bold]")
#             copied = True
#     if not copied:
#         rprint(f"[yellow]⚠[/yellow] No weight files found in {run_dir / 'weights'}")
def run_prediction(
    weights: Path,
    source: Path,
    output_csv: Path,
    conf: float,
    imgsz: int,
    device: Optional[str] = None,
) -> Path:
    """Run YOLO prediction and save detections to a CSV file."""

    try:
        import pandas as pd
        from ultralytics import YOLO
    except ImportError:
        rprint("[red]✗[/red] Required packages are missing. Run: pip install ultralytics pandas")
        raise typer.Exit(code=1)

    rprint(f"[cyan]Loading YOLO model from:[/cyan] {weights}")
    model = YOLO(str(weights))
    class_names = model.names

    if source.is_file():
        image_paths = [source] if source.suffix.lower() in IMAGE_EXTS else []
    else:
        image_paths = sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )

    rprint(f"[cyan]Found {len(image_paths)} images.[/cyan]")

    if not image_paths:
        rprint(f"[yellow]⚠[/yellow] No images found in {source}")
        raise typer.Exit(code=0)

    output_rows = []

    for img_path in image_paths:
        predict_kwargs = {
            "source": str(img_path),
            "imgsz": imgsz,
            "conf": conf,
            "save": False,
            "verbose": False,
        }

        if device is not None:
            predict_kwargs["device"] = device

        results = model.predict(**predict_kwargs)

        if len(results) == 0:
            continue

        result = results[0]
        height, width = result.orig_shape
        file_name = img_path.name

        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            box_width = x2 - x1
            box_height = y2 - y1
            x_center = x1 + box_width / 2
            y_center = y1 + box_height / 2

            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            output_rows.append(
                {
                    "file_name": file_name,
                    "label": class_id,
                    "category": class_names[class_id],
                    "bbox_x_center": x_center / width,
                    "bbox_y_center": y_center / height,
                    "bbox_width": box_width / width,
                    "bbox_height": box_height / height,
                    "confidence": confidence,
                    "path": str(img_path),
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(output_rows)
    df.to_csv(output_csv, index=False)

    rprint(f"[green]✔[/green] Saved {len(df)} detections to [bold]{output_csv}[/bold]")

    return output_csv

def run_evaluation(
    gt_csv: Path,
    prediction_csv: Path,
    output_xlsx: Path,
    split: str,
    model_run: str,
) -> Path:
    """Evaluate prediction CSV against ground truth and save metrics workbook.

    Rules:
    - All non-Empty labels are evaluated separately.
    - Empty is treated as image-level background.
    - Oryctolagus cuniculus and Lepus granatensis are NOT merged.
    - Matching is greedy best-IoU matching per image.
    """

    import math
    from dataclasses import dataclass
    from typing import Any

    import numpy as np
    import pandas as pd

    DEFAULT_IOUS_MAIN = [0.5, 0.6, 0.7, 0.8, 0.9]
    DEFAULT_IOUS_MAP = [round(x / 100.0, 2) for x in range(50, 100, 5)]
    BACKGROUND_TOKEN = "__background__"

    GT_REQUIRED = {
        "file_name",
        "category",
        "bbox_x_center",
        "bbox_y_center",
        "bbox_width",
        "bbox_height",
    }

    PRED_REQUIRED = {
        "file_name",
        "category",
        "bbox_x_center",
        "bbox_y_center",
        "bbox_width",
        "bbox_height",
    }

    @dataclass
    class Box:
        file_name: str
        category: str
        xc: float
        yc: float
        w: float
        h: float
        confidence: float = 1.0
        path: str = ""

    @dataclass
    class FinalPred:
        file_name: str
        pred_label: str
        gt_label: str
        confidence: float

    def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = (
            df.columns.astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
        )
        return df

    def normalize_file_name(x: Any) -> str:
        s = "" if pd.isna(x) else str(x).strip()
        s = Path(s).name
        s = Path(s).stem
        if s.endswith(".txt"):
            s = Path(s).stem
        return s

    def normalize_category(x: Any) -> str:
        if pd.isna(x):
            return ""
        return str(x).strip()

    def is_empty_category(cat: str) -> bool:
        return normalize_category(cat).lower() == "empty"

    def safe_div(num: float, den: float) -> float:
        return float(num) / float(den) if den else np.nan

    def precision_from_counts(tp: int, fp: int) -> float:
        return safe_div(tp, tp + fp)

    def recall_from_counts(tp: int, fn: int) -> float:
        return safe_div(tp, tp + fn)

    def accuracy_from_counts(tp: int, tn: int, fp: int, fn: int) -> float:
        return safe_div(tp + tn, tp + tn + fp + fn)

    def f1_from_counts(tp: int, fp: int, fn: int) -> float:
        p = precision_from_counts(tp, fp)
        r = recall_from_counts(tp, fn)

        if np.isnan(p) or np.isnan(r) or (p + r) == 0:
            return np.nan

        return 2 * p * r / (p + r)

    def mcc_from_counts(tp: int, tn: int, fp: int, fn: int) -> float:
        den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)

        if den <= 0:
            return np.nan

        return ((tp * tn) - (fp * fn)) / math.sqrt(den)

    def xywh_to_xyxy(box: Box) -> tuple[float, float, float, float]:
        x1 = box.xc - box.w / 2.0
        y1 = box.yc - box.h / 2.0
        x2 = box.xc + box.w / 2.0
        y2 = box.yc + box.h / 2.0
        return x1, y1, x2, y2

    def iou(box_a: Box, box_b: Box) -> float:
        ax1, ay1, ax2, ay2 = xywh_to_xyxy(box_a)
        bx1, by1, bx2, by2 = xywh_to_xyxy(box_b)

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union = area_a + area_b - inter_area

        if union <= 0:
            return 0.0

        return inter_area / union

    def ensure_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
        missing = sorted(required - set(df.columns))

        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    def nanmean_or_nan(values: list[float]) -> float:
        vals = [v for v in values if not pd.isna(v)]

        if not vals:
            return np.nan

        return float(np.mean(vals))

    def read_gt_csv(path: Path, split_filter: str | None = None) -> pd.DataFrame:
        gt = pd.read_csv(path, encoding="utf-8-sig")
        gt = clean_columns(gt)
        ensure_columns(gt, GT_REQUIRED, "Ground-truth CSV")

        if split_filter is not None and "group" in gt.columns:
            gt = gt[gt["group"].astype(str).str.lower() == split_filter.lower()].copy()

        gt["file_name"] = gt["file_name"].map(normalize_file_name)
        gt["category"] = gt["category"].map(normalize_category)

        for col in ["bbox_x_center", "bbox_y_center", "bbox_width", "bbox_height"]:
            gt[col] = pd.to_numeric(gt[col], errors="coerce")

        return gt

    def read_prediction_csv(path: Path) -> pd.DataFrame:
        pred = pd.read_csv(path, encoding="utf-8-sig")
        pred = clean_columns(pred)
        ensure_columns(pred, PRED_REQUIRED, "Prediction CSV")

        pred["file_name"] = pred["file_name"].map(normalize_file_name)
        pred["category"] = pred["category"].map(normalize_category)

        for col in ["bbox_x_center", "bbox_y_center", "bbox_width", "bbox_height"]:
            pred[col] = pd.to_numeric(pred[col], errors="coerce")

        if "confidence" not in pred.columns:
            pred["confidence"] = 1.0

        pred["confidence"] = pd.to_numeric(pred["confidence"], errors="coerce").fillna(0.0)

        if "path" not in pred.columns:
            pred["path"] = ""

        return pred

    def get_eval_labels(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> list[str]:
        gt_labels = gt_df["category"].dropna().map(normalize_category).tolist()
        pred_labels = pred_df["category"].dropna().map(normalize_category).tolist()

        labels = sorted(
            {
                label
                for label in gt_labels + pred_labels
                if label and not is_empty_category(label)
            }
        )

        return labels

    def is_valid_final_label(cat: str, labels: list[str]) -> bool:
        cat = normalize_category(cat)
        return cat in labels

    def gt_df_to_boxes_by_file(gt_df: pd.DataFrame) -> dict[str, list[Box]]:
        out: dict[str, list[Box]] = {}

        for _, row in gt_df.reset_index(drop=True).iterrows():
            file_name = normalize_file_name(row["file_name"])
            category = normalize_category(row["category"])

            coords = [
                row["bbox_x_center"],
                row["bbox_y_center"],
                row["bbox_width"],
                row["bbox_height"],
            ]

            if file_name == "" or any(pd.isna(v) for v in coords):
                continue

            box = Box(
                file_name=file_name,
                category=category,
                xc=float(row["bbox_x_center"]),
                yc=float(row["bbox_y_center"]),
                w=float(row["bbox_width"]),
                h=float(row["bbox_height"]),
                confidence=1.0,
                path=str(row.get("path", "")),
            )

            out.setdefault(file_name, []).append(box)

        return out

    def pred_df_to_boxes_by_file(pred_df: pd.DataFrame, labels: list[str]) -> dict[str, list[Box]]:
        out: dict[str, list[Box]] = {}

        pred_df = pred_df[pred_df["category"].map(lambda x: is_valid_final_label(x, labels))].copy()

        for _, row in pred_df.reset_index(drop=True).iterrows():
            file_name = normalize_file_name(row["file_name"])
            category = normalize_category(row["category"])

            coords = [
                row["bbox_x_center"],
                row["bbox_y_center"],
                row["bbox_width"],
                row["bbox_height"],
            ]

            if file_name == "" or any(pd.isna(v) for v in coords):
                continue

            box = Box(
                file_name=file_name,
                category=category,
                xc=float(row["bbox_x_center"]),
                yc=float(row["bbox_y_center"]),
                w=float(row["bbox_width"]),
                h=float(row["bbox_height"]),
                confidence=float(row["confidence"]),
                path=str(row.get("path", "")),
            )

            out.setdefault(file_name, []).append(box)

        return out

    def greedy_global_pairs(pred_boxes: list[Box], gt_boxes: list[Box]) -> list[tuple[int, int, float]]:
        pairs: list[tuple[float, int, int]] = []

        for pred_idx, pred_box in enumerate(pred_boxes):
            for gt_idx, gt_box in enumerate(gt_boxes):
                overlap = iou(pred_box, gt_box)

                if overlap > 0:
                    pairs.append((overlap, pred_idx, gt_idx))

        pairs.sort(key=lambda x: x[0], reverse=True)

        used_predictions = set()
        used_gt = set()
        matched_pairs: list[tuple[int, int, float]] = []

        for overlap, pred_idx, gt_idx in pairs:
            if pred_idx in used_predictions or gt_idx in used_gt:
                continue

            used_predictions.add(pred_idx)
            used_gt.add(gt_idx)
            matched_pairs.append((pred_idx, gt_idx, overlap))

        return matched_pairs

    def evaluate_for_iou_threshold(
        gt_by_file: dict[str, list[Box]],
        pred_by_file: dict[str, list[Box]],
        all_images: list[str],
        labels: list[str],
        threshold: float,
    ) -> tuple[
        dict[str, dict[str, int]],
        dict[str, int],
        dict[str, int],
        list[FinalPred],
        list[dict[str, Any]],
    ]:
        all_events: list[tuple[str, str]] = []
        final_predictions: list[FinalPred] = []
        warnings_list: list[dict[str, Any]] = []

        gt_counts_by_label = {label: 0 for label in labels}
        pred_counts_by_label = {label: 0 for label in labels}

        for image in all_images:
            gt_boxes = gt_by_file.get(image, [])
            pred_boxes = pred_by_file.get(image, [])

            gt_eval_boxes = [
                gt_box
                for gt_box in gt_boxes
                if is_valid_final_label(gt_box.category, labels)
            ]

            gt_nonempty_boxes = [
                gt_box
                for gt_box in gt_boxes
                if not is_empty_category(gt_box.category)
            ]

            for gt_box in gt_eval_boxes:
                gt_counts_by_label[gt_box.category] += 1

            pairs = greedy_global_pairs(pred_boxes, gt_boxes)

            matched_predictions = set()
            matched_gt = set()

            for pred_idx, gt_idx, overlap in pairs:
                if overlap < threshold:
                    continue

                matched_predictions.add(pred_idx)
                matched_gt.add(gt_idx)

                pred_box = pred_boxes[pred_idx]
                gt_box = gt_boxes[gt_idx]

                pred_label = normalize_category(pred_box.category)
                gt_label = normalize_category(gt_box.category)

                if is_empty_category(gt_label):
                    all_events.append((pred_label, BACKGROUND_TOKEN))
                    pred_counts_by_label[pred_label] += 1

                    warnings_list.append(
                        {
                            "file_name": image,
                            "type": "matched_prediction_to_empty_gt_row",
                            "details": (
                                f"pred_category={pred_label}; "
                                f"confidence={float(pred_box.confidence):.6f}; "
                                f"iou={overlap:.4f}"
                            ),
                        }
                    )
                    continue

                if not is_valid_final_label(gt_label, labels):
                    all_events.append((pred_label, BACKGROUND_TOKEN))
                    pred_counts_by_label[pred_label] += 1

                    warnings_list.append(
                        {
                            "file_name": image,
                            "type": "matched_prediction_to_non_eval_gt",
                            "details": (
                                f"pred_category={pred_label}; "
                                f"gt_category={gt_label}; "
                                f"confidence={float(pred_box.confidence):.6f}; "
                                f"iou={overlap:.4f}"
                            ),
                        }
                    )
                    continue

                if not is_valid_final_label(pred_label, labels):
                    all_events.append((BACKGROUND_TOKEN, gt_label))

                    warnings_list.append(
                        {
                            "file_name": image,
                            "type": "invalid_prediction_label",
                            "details": (
                                f"pred_category={pred_label}; "
                                f"gt_category={gt_label}; "
                                f"confidence={float(pred_box.confidence):.6f}; "
                                f"iou={overlap:.4f}"
                            ),
                        }
                    )
                    continue

                pred_counts_by_label[pred_label] += 1

                final_predictions.append(
                    FinalPred(
                        file_name=image,
                        pred_label=pred_label,
                        gt_label=gt_label,
                        confidence=float(pred_box.confidence),
                    )
                )

                all_events.append((pred_label, gt_label))

            image_has_any_gt = len(gt_boxes) > 0
            image_has_nonempty_gt = len(gt_nonempty_boxes) > 0
            image_has_eval_gt = len(gt_eval_boxes) > 0

            for pred_idx, pred_box in enumerate(pred_boxes):
                if pred_idx in matched_predictions:
                    continue

                pred_label = normalize_category(pred_box.category)

                if not is_valid_final_label(pred_label, labels):
                    warnings_list.append(
                        {
                            "file_name": image,
                            "type": "unmatched_invalid_prediction_label_ignored",
                            "details": (
                                f"pred_category={pred_label}; "
                                f"confidence={float(pred_box.confidence):.6f}"
                            ),
                        }
                    )
                    continue

                pred_counts_by_label[pred_label] += 1
                all_events.append((pred_label, BACKGROUND_TOKEN))

                final_predictions.append(
                    FinalPred(
                        file_name=image,
                        pred_label=pred_label,
                        gt_label=BACKGROUND_TOKEN,
                        confidence=float(pred_box.confidence),
                    )
                )

                if not image_has_any_gt or not image_has_nonempty_gt:
                    warning_type = "fp_on_empty_image"
                elif image_has_eval_gt:
                    warning_type = "fp_wrong_localization_on_nonempty_image"
                else:
                    warning_type = "fp_on_non_eval_nonempty_image"

                warnings_list.append(
                    {
                        "file_name": image,
                        "type": warning_type,
                        "details": (
                            f"pred_category={pred_label}; "
                            f"confidence={float(pred_box.confidence):.6f}"
                        ),
                    }
                )

            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue

                gt_label = normalize_category(gt_box.category)

                if is_valid_final_label(gt_label, labels):
                    all_events.append((BACKGROUND_TOKEN, gt_label))

        total_events = len(all_events)

        counts_by_label = {
            label: {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
            for label in labels
        }

        for label in labels:
            tp = sum(
                1
                for pred_label, gt_label in all_events
                if pred_label == label and gt_label == label
            )
            fp = sum(
                1
                for pred_label, gt_label in all_events
                if pred_label == label and gt_label != label
            )
            fn = sum(
                1
                for pred_label, gt_label in all_events
                if pred_label != label and gt_label == label
            )
            tn = total_events - tp - fp - fn

            counts_by_label[label] = {
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
            }

        return (
            counts_by_label,
            gt_counts_by_label,
            pred_counts_by_label,
            final_predictions,
            warnings_list,
        )

    def counts_to_metric_row(
        counts_by_label: dict[str, dict[str, int]],
        labels: list[str],
        gt_counts_by_label: dict[str, int],
        pred_counts_by_label: dict[str, int],
    ) -> dict[str, dict[str, float]]:
        output: dict[str, dict[str, float]] = {}

        for label in labels:
            tp = counts_by_label[label]["TP"]
            tn = counts_by_label[label]["TN"]
            fp = counts_by_label[label]["FP"]
            fn = counts_by_label[label]["FN"]

            output[label] = {
                "GT_count": int(gt_counts_by_label.get(label, 0)),
                "Pred_count": int(pred_counts_by_label.get(label, 0)),
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "precision": precision_from_counts(tp, fp),
                "recall": recall_from_counts(tp, fn),
                "accuracy": accuracy_from_counts(tp, tn, fp, fn),
                "f1": f1_from_counts(tp, fp, fn),
                "mcc": mcc_from_counts(tp, tn, fp, fn),
            }

        return output

    def micro_counts(
        counts_by_label: dict[str, dict[str, int]],
        labels: list[str],
    ) -> dict[str, float]:
        tp = sum(counts_by_label[label]["TP"] for label in labels)
        fp = sum(counts_by_label[label]["FP"] for label in labels)
        fn = sum(counts_by_label[label]["FN"] for label in labels)

        return {
            "TP": tp,
            "TN": np.nan,
            "FP": fp,
            "FN": fn,
            "precision": precision_from_counts(tp, fp),
            "recall": recall_from_counts(tp, fn),
            "accuracy": np.nan,
            "f1": f1_from_counts(tp, fp, fn),
            "mcc": np.nan,
        }

    def macro_from_label_metrics(
        label_metrics: dict[str, dict[str, float]],
        labels: list[str],
        fields: list[str],
    ) -> dict[str, float]:
        output = {}

        for field in fields:
            values = [
                label_metrics[label][field]
                for label in labels
                if not pd.isna(label_metrics[label][field])
            ]

            output[field] = float(np.mean(values)) if values else np.nan

        return output

    def make_metric_table(
        rows: list[dict[str, Any]],
        value_key: str,
        labels: list[str],
        include_micro: bool = True,
        include_macro: bool = True,
    ) -> pd.DataFrame:
        data = []

        for row in rows:
            record = {
                "model_run": row["model_run"],
                "absolute_csv_path": row["absolute_csv_path"],
                "iou": row["iou"],
            }

            for label in labels:
                record[label] = row["per_label"].get(label, {}).get(value_key, np.nan)

            if include_micro:
                record["__micro"] = row.get("micro", {}).get(value_key, np.nan)

            if include_macro:
                record["__macro"] = row.get("macro", {}).get(value_key, np.nan)

            data.append(record)

        df = pd.DataFrame(data)

        if df.empty:
            return df

        return df.sort_values(["model_run", "iou"]).reset_index(drop=True)

    def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
        if recall.size == 0 or precision.size == 0:
            return np.nan

        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        for idx in range(len(mpre) - 1, 0, -1):
            mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])

        changed = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1])

        return float(ap)

    def compute_ap_for_class_from_final_predictions(
        gt_by_file: dict[str, list[Box]],
        final_predictions: list[FinalPred],
        target_class: str,
    ) -> float:
        gt_class = {
            image: [
                gt_box
                for gt_box in boxes
                if normalize_category(gt_box.category) == target_class
            ]
            for image, boxes in gt_by_file.items()
        }

        n_gt = sum(len(values) for values in gt_class.values())

        if n_gt == 0:
            return np.nan

        predictions = []

        for final_prediction in final_predictions:
            if final_prediction.pred_label != target_class:
                continue

            predictions.append(
                (
                    final_prediction.confidence,
                    final_prediction.gt_label,
                )
            )

        predictions.sort(key=lambda x: x[0], reverse=True)

        if len(predictions) == 0:
            return 0.0

        tp = np.zeros(len(predictions), dtype=float)
        fp = np.zeros(len(predictions), dtype=float)

        for idx, (_confidence, gt_label) in enumerate(predictions):
            if gt_label == target_class:
                tp[idx] = 1.0
            else:
                fp[idx] = 1.0

        cumulative_tp = np.cumsum(tp)
        cumulative_fp = np.cumsum(fp)

        recall = cumulative_tp / max(n_gt, 1)
        precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)

        return compute_ap(recall, precision)

    def make_ap_table(
        rows: list[dict[str, Any]],
        labels: list[str],
        value_key: str,
    ) -> pd.DataFrame:
        data = []

        for row in rows:
            record = {
                "model_run": row["model_run"],
                "absolute_csv_path": row["absolute_csv_path"],
            }

            for label in labels:
                record[label] = row.get("per_label", {}).get(label, np.nan)

            record["__macro"] = row.get(value_key, np.nan)

            data.append(record)

        df = pd.DataFrame(data)

        if df.empty:
            return df

        return df.sort_values(["model_run"]).reset_index(drop=True)

    def compute_empty_image_metrics(
        gt_df: pd.DataFrame,
        pred_by_file: dict[str, list[Box]],
        all_images: list[str],
        labels: list[str],
    ) -> pd.DataFrame:
        gt_nonempty_by_file: dict[str, bool] = {}

        for image in all_images:
            sub = gt_df[gt_df["file_name"] == image]
            has_valid_gt = sub["category"].map(lambda x: is_valid_final_label(x, labels)).any()
            gt_nonempty_by_file[image] = bool(has_valid_gt)

        tp = tn = fp = fn = 0

        for image in all_images:
            gt_empty = not gt_nonempty_by_file.get(image, False)
            pred_empty = len(pred_by_file.get(image, [])) == 0

            if pred_empty and gt_empty:
                tp += 1
            elif pred_empty and not gt_empty:
                fp += 1
            elif not pred_empty and gt_empty:
                fn += 1
            else:
                tn += 1

        return pd.DataFrame(
            [
                {
                    "positive_class": "Empty",
                    "TP_empty": tp,
                    "TN_empty": tn,
                    "FP_empty": fp,
                    "FN_empty": fn,
                    "precision_empty": precision_from_counts(tp, fp),
                    "recall_empty": recall_from_counts(tp, fn),
                    "accuracy_empty": accuracy_from_counts(tp, tn, fp, fn),
                    "f1_empty": f1_from_counts(tp, fp, fn),
                    "mcc_empty": mcc_from_counts(tp, tn, fp, fn),
                }
            ]
        )

    # ------------------------------------------------------------------
    # Main evaluation body
    # ------------------------------------------------------------------

    split_filter = split if str(split).strip() else None

    gt_df = read_gt_csv(gt_csv, split_filter=split_filter)
    pred_df = read_prediction_csv(prediction_csv)

    labels = get_eval_labels(gt_df, pred_df)

    if not labels:
        raise ValueError("No non-Empty labels found for evaluation.")

    gt_by_file = gt_df_to_boxes_by_file(gt_df)
    pred_by_file = pred_df_to_boxes_by_file(pred_df, labels)

    all_images = sorted(
        set(gt_df["file_name"].dropna().astype(str).map(normalize_file_name).tolist())
    )

    gt_counts_global = (
        gt_df.loc[
            gt_df["category"].map(lambda x: is_valid_final_label(x, labels)),
            "category",
        ]
        .map(normalize_category)
        .value_counts()
        .reindex(labels, fill_value=0)
        .to_dict()
    )

    metric_rows: list[dict[str, Any]] = []
    warnings_list: list[dict[str, Any]] = []

    for threshold in DEFAULT_IOUS_MAIN:
        (
            counts_by_label,
            gt_counts_threshold,
            pred_counts_threshold,
            final_predictions,
            threshold_warnings,
        ) = evaluate_for_iou_threshold(
            gt_by_file=gt_by_file,
            pred_by_file=pred_by_file,
            all_images=all_images,
            labels=labels,
            threshold=threshold,
        )

        for warning in threshold_warnings:
            warnings_list.append(
                {
                    "model_run": model_run,
                    "absolute_csv_path": str(prediction_csv.resolve()),
                    "iou": threshold,
                    **warning,
                }
            )

        label_metrics = counts_to_metric_row(
            counts_by_label=counts_by_label,
            labels=labels,
            gt_counts_by_label=gt_counts_threshold,
            pred_counts_by_label=pred_counts_threshold,
        )

        micro = micro_counts(counts_by_label, labels)

        macro = macro_from_label_metrics(
            label_metrics=label_metrics,
            labels=labels,
            fields=["precision", "recall", "accuracy", "f1", "mcc"],
        )

        metric_rows.append(
            {
                "model_run": model_run,
                "absolute_csv_path": str(prediction_csv.resolve()),
                "iou": threshold,
                "per_label": label_metrics,
                "micro": micro,
                "macro": macro,
            }
        )

    # AP50
    _, _, _, final_predictions_05, _ = evaluate_for_iou_threshold(
        gt_by_file=gt_by_file,
        pred_by_file=pred_by_file,
        all_images=all_images,
        labels=labels,
        threshold=0.50,
    )

    ap50_per_label = {
        label: compute_ap_for_class_from_final_predictions(
            gt_by_file=gt_by_file,
            final_predictions=final_predictions_05,
            target_class=label,
        )
        for label in labels
    }

    ap50_rows = [
        {
            "model_run": model_run,
            "absolute_csv_path": str(prediction_csv.resolve()),
            "per_label": ap50_per_label,
            "mAP50": nanmean_or_nan(list(ap50_per_label.values())),
        }
    ]

    # mAP50-95
    map5095_per_label = {}

    for label in labels:
        values = []

        for threshold in DEFAULT_IOUS_MAP:
            _, _, _, final_predictions_t, _ = evaluate_for_iou_threshold(
                gt_by_file=gt_by_file,
                pred_by_file=pred_by_file,
                all_images=all_images,
                labels=labels,
                threshold=threshold,
            )

            values.append(
                compute_ap_for_class_from_final_predictions(
                    gt_by_file=gt_by_file,
                    final_predictions=final_predictions_t,
                    target_class=label,
                )
            )

        map5095_per_label[label] = nanmean_or_nan(values)

    map5095_rows = [
        {
            "model_run": model_run,
            "absolute_csv_path": str(prediction_csv.resolve()),
            "per_label": map5095_per_label,
            "mAP5095": nanmean_or_nan(list(map5095_per_label.values())),
        }
    ]

    metric_names = [
        "GT_count",
        "Pred_count",
        "TP",
        "TN",
        "FP",
        "FN",
        "precision",
        "recall",
        "accuracy",
        "f1",
        "mcc",
    ]

    tables: dict[str, pd.DataFrame] = {}

    for metric_name in metric_names:
        tables[f"combined_{metric_name}"] = make_metric_table(
            rows=metric_rows,
            value_key=metric_name,
            labels=labels,
            include_micro=(metric_name != "Pred_count"),
            include_macro=metric_name in {"precision", "recall", "accuracy", "f1", "mcc"},
        )

    tables["GT_counts"] = pd.DataFrame(
        [
            {
                "absolute_gt_csv": str(gt_csv.resolve()),
                "split_filter": split_filter if split_filter is not None else "<none>",
                **{label: int(gt_counts_global.get(label, 0)) for label in labels},
                "__total_gt_boxes": int(sum(gt_counts_global.values())),
            }
        ]
    )

    tables["prediction_file"] = pd.DataFrame(
        [
            {
                "model_run": model_run,
                "absolute_csv_path": str(prediction_csv.resolve()),
            }
        ]
    )

    tables["empty_image_metrics"] = compute_empty_image_metrics(
        gt_df=gt_df,
        pred_by_file=pred_by_file,
        all_images=all_images,
        labels=labels,
    )

    tables["warnings"] = (
        pd.DataFrame(warnings_list)
        if warnings_list
        else pd.DataFrame(
            columns=[
                "model_run",
                "absolute_csv_path",
                "iou",
                "file_name",
                "type",
                "details",
            ]
        )
    )

    tables["AP50_detector_conf"] = make_ap_table(
        rows=ap50_rows,
        labels=labels,
        value_key="mAP50",
    )

    tables["mAP50_95_detector_conf"] = make_ap_table(
        rows=map5095_rows,
        labels=labels,
        value_key="mAP5095",
    )

    tables["metadata"] = pd.DataFrame(
        [
            ["model_run", model_run],
            ["absolute_gt_csv", str(gt_csv.resolve())],
            ["absolute_prediction_csv", str(prediction_csv.resolve())],
            ["absolute_out_xlsx", str(output_xlsx.resolve())],
            ["split_filter", split_filter if split_filter is not None else "<none>"],
            ["labels", ", ".join(labels)],
            ["empty_rule", "Empty is treated as image-level background."],
            ["lagomorph_merge", "Disabled. Oryctolagus cuniculus and Lepus granatensis are evaluated separately."],
            ["matching_rule", "Greedy global IoU matching per image."],
            ["main_ious", ", ".join(map(str, DEFAULT_IOUS_MAIN))],
            ["map_ious", ", ".join(map(str, DEFAULT_IOUS_MAP))],
        ],
        columns=["parameter", "value"],
    )

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            table.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    rprint(f"[green]✔[/green] Evaluation metrics written to [bold]{output_xlsx}[/bold]")

    return output_xlsx

def infer_model_run_from_weights(weights: Path) -> str:
    """Infer model run name from a weights path."""
    if weights.resolve() == PRETRAINED_WEIGHTS.resolve():
        return PRETRAINED_WEIGHTS.stem

    if weights.parent.name == "weights":
        return weights.parent.parent.name

    return weights.stem


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


# @app.command()
# def prepare_dataset(
#     source: Annotated[
#         Path,
#         typer.Option("--source", "-s", help="Directory with raw images (and label .txt files beside them).", show_default=False),
#     ],
#     train_ratio: Annotated[
#         float,
#         typer.Option("--train", "-tr", help="Fraction of images for training.", min=0.0, max=1.0),
#     ] = 0.7,
#     val_ratio: Annotated[
#         float,
#         typer.Option("--val", "-v", help="Fraction of images for validation.", min=0.0, max=1.0),
#     ] = 0.2,
#     test_ratio: Annotated[
#         float,
#         typer.Option("--test", "-te", help="Fraction of images for testing.", min=0.0, max=1.0),
#     ] = 0.1,
#     seed: Annotated[int, typer.Option("--seed", help="Random seed for reproducibility.")] = 42,
#     move: Annotated[bool, typer.Option("--move", help="Move files instead of copying.")] = False,
# ) -> None:
#     """Split raw images + labels into train / val / test partitions."""

#     # --- validate ratios ---
#     total = round(train_ratio + val_ratio + test_ratio, 6)
#     if abs(total - 1.0) > 1e-4:
#         rprint(f"[red]✗[/red] Ratios must sum to 1.0 — got {total:.4f}")
#         raise typer.Exit(code=1)

#     if not source.exists() or not source.is_dir():
#         rprint(f"[red]✗[/red] Source directory not found: {source}")
#         raise typer.Exit(code=1)

#     images = sorted(f for f in source.iterdir() if f.suffix.lower() in IMAGE_EXTS)
#     if not images:
#         rprint(f"[yellow]⚠[/yellow] No images found in {source}")
#         raise typer.Exit(code=0)

#     random.seed(seed)
#     random.shuffle(images)

#     n = len(images)
#     n_train = int(n * train_ratio)
#     n_val = int(n * val_ratio)
#     # the rest go to test (avoids rounding drift)
#     splits: dict[str, list[Path]] = {
#         "train": images[:n_train],
#         "val": images[n_train : n_train + n_val],
#         "test": images[n_train + n_val :],
#     }

#     _ensure_dirs()
#     op = shutil.move if move else shutil.copy2
#     op_label = "Moved" if move else "Copied"

#     rprint(Panel.fit(f"[bold]Splitting {n} images[/bold]  seed={seed}", title="prepare-dataset"))

#     for partition, files in splits.items():
#         img_dst = DATASET_DIR / partition / "images"
#         lbl_dst = DATASET_DIR / partition / "labels"
#         count_img = count_lbl = 0
#         for img_path in files:
#             shutil.copy2(img_path, img_dst / img_path.name) if not move else shutil.move(str(img_path), img_dst / img_path.name)
#             count_img += 1
#             label_src = img_path.with_suffix(".txt")
#             if label_src.exists():
#                 shutil.copy2(label_src, lbl_dst / label_src.name) if not move else shutil.move(str(label_src), lbl_dst / label_src.name)
#                 count_lbl += 1
#         rprint(
#             f"  [cyan]{partition:5s}[/cyan]  {op_label} {count_img} images"
#             + (f", {count_lbl} labels" if count_lbl else " (no labels found)")
#         )

#     rprint("[green]✔[/green] Dataset split complete.")


# @app.command()
# def train(
#     model: Annotated[
#         str,
#         typer.Option("--model", "-m", help="Base YOLO model (e.g. yolov8n.pt, yolov8s.pt) or path to a .pt file."),
#     ] = "yolov8n.pt",
#     epochs: Annotated[int, typer.Option("--epochs", "-e", help="Number of training epochs.", min=1)] = 100,
#     imgsz: Annotated[int, typer.Option("--imgsz", help="Input image size (pixels).", min=32)] = 640,
#     batch: Annotated[int, typer.Option("--batch", "-b", help="Batch size. -1 = auto.")] = 16,
#     name: Annotated[str, typer.Option("--name", "-n", help="Run name. Weights stored in weights/<name>/")] = "run",
#     device: Annotated[
#         Optional[str],
#         typer.Option("--device", "-d", help="Device: cpu, 0, 0,1, … (default: auto)"),
#     ] = None,
#     patience: Annotated[int, typer.Option("--patience", help="Early-stopping patience (epochs).", min=1)] = 50,
#     resume: Annotated[bool, typer.Option("--resume", help="Resume from the last checkpoint.")] = False,
#     class_names: Annotated[
#         Optional[str],
#         typer.Option("--classes", help='Comma-separated class names, e.g. "fox,deer,boar". Required if dataset.yaml is missing.'),
#     ] = None,
# ) -> None:
#     """Train a YOLO model on the dataset and save weights to weights/<name>/."""

#     try:
#         from ultralytics import YOLO
#     except ImportError:
#         rprint("[red]✗[/red] ultralytics is not installed. Run: pip install ultralytics")
#         raise typer.Exit(code=1)

#     # --- ensure dataset.yaml exists ---
#     if not DATASET_YAML.exists():
#         if class_names is None:
#             rprint(
#                 "[yellow]⚠[/yellow] dataset.yaml not found.\n"
#                 "       Pass [bold]--classes[/bold] to generate it automatically.\n"
#                 "       Example: [italic]--classes fox,deer,boar[/italic]"
#             )
#             raise typer.Exit(code=1)
#         _write_dataset_yaml([c.strip() for c in class_names.split(",")])

#     # --- sanity check: are there training images? ---
#     n_train, _ = _count_partition("train")
#     if n_train == 0:
#         rprint("[yellow]⚠[/yellow] No images found in dataset/train/images. Run [bold]prepare-dataset[/bold] first.")
#         raise typer.Exit(code=1)

#     rprint(
#         Panel.fit(
#             f"[bold]model:[/bold] {model}  [bold]epochs:[/bold] {epochs}  "
#             f"[bold]batch:[/bold] {batch}  [bold]imgsz:[/bold] {imgsz}  "
#             f"[bold]name:[/bold] {name}",
#             title="[green]train[/green]",
#         )
#     )

#     yolo = YOLO(model)

#     extra: dict = {}
#     if device is not None:
#         extra["device"] = device

#     results = yolo.train(
#         data=str(DATASET_YAML),
#         epochs=epochs,
#         imgsz=imgsz,
#         batch=batch,
#         name=name,
#         project=str(RUNS_DIR),
#         patience=patience,
#         resume=resume,
#         **extra,
#     )

#     # --- copy weights ---
#     run_dir = RUNS_DIR / name
#     if run_dir.exists():
#         _copy_weights(run_dir, name)
#     else:
#         # Ultralytics may append a numeric suffix on collisions
#         candidates = sorted(RUNS_DIR.glob(f"{name}*/"))
#         if candidates:
#             _copy_weights(candidates[-1], name)

#     rprint(f"[green]✔[/green] Training complete. Results in [bold]{RUNS_DIR / name}[/bold]")
@app.command()
def train(
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Base YOLO model or path to a .pt file."),
    ] = "yolov8x6.pt",
    epochs: Annotated[
        int,
        typer.Option("--epochs", "-e", help="Number of training epochs.", min=1),
    ] = 1000,
    imgsz: Annotated[
        int,
        typer.Option("--imgsz", help="Input image size in pixels.", min=32),
    ] = 640,
    batch: Annotated[
        int,
        typer.Option("--batch", "-b", help="Batch size. -1 = auto."),
    ] = 16,
    name: Annotated[
        Optional[str],
        typer.Option("--name", "-n", help="Run name. Default: current date in YYYYMMDD format."),
    ] = None,
    device: Annotated[
        Optional[str],
        typer.Option("--device", "-d", help="Device: cpu, 0, 0,1, … default: auto."),
    ] = None,
    patience: Annotated[
        int,
        typer.Option("--patience", help="Early-stopping patience in epochs.", min=1),
    ] = 100,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Resume from the last checkpoint."),
    ] = False,
    class_names: Annotated[
        Optional[str],
        typer.Option(
            "--classes",
            help='Comma-separated class names, e.g. "fox,deer,boar". Required if dataset/data.yaml is missing.',
        ),
    ] = None,
) -> None:
    """Train DonaNet and save outputs under output/training/YYYYMMDD/."""

    try:
        from ultralytics import YOLO
    except ImportError:
        rprint("[red]✗[/red] ultralytics is not installed. Run: pip install ultralytics")
        raise typer.Exit(code=1)

    if name is None:
        name = datetime.now().strftime("%Y%m%d")

    # --- ensure dataset/data.yaml exists ---
    if not DATASET_YAML.exists():
        if class_names is None:
            rprint(
                "[yellow]⚠[/yellow] dataset/data.yaml not found.\n"
                "       Pass [bold]--classes[/bold] to generate it automatically.\n"
                "       Example: [italic]--classes fox,deer,boar[/italic]"
            )
            raise typer.Exit(code=1)

        _write_dataset_yaml([c.strip() for c in class_names.split(",")])

    # --- sanity check: are there training images? ---
    n_train, _ = _count_partition("train")
    if n_train == 0:
        rprint("[yellow]⚠[/yellow] No images found in dataset/images/train.")
        raise typer.Exit(code=1)

    run_dir = TRAIN_OUTPUT_DIR / name

    rprint(
        Panel.fit(
            f"[bold]model:[/bold] {model}  [bold]epochs:[/bold] {epochs}  "
            f"[bold]batch:[/bold] {batch}  [bold]imgsz:[/bold] {imgsz}  "
            f"[bold]name:[/bold] {name}\n"
            f"[bold]output:[/bold] {run_dir}",
            title="[green]train[/green]",
        )
    )

    yolo = YOLO(model)

    extra: dict = {}
    if device is not None:
        extra["device"] = device

    yolo.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=name,
        project=str(TRAIN_OUTPUT_DIR),
        patience=patience,
        resume=resume,
        **extra,
    )

    rprint(f"[green]✔[/green] Training complete. Results in [bold]{run_dir}[/bold]")
    rprint(f"[green]✔[/green] Main weights: [bold]{run_dir / 'weights' / 'best.pt'}[/bold]")

# @app.command()
# def test(
#     weights: Annotated[
#         Path,
#         typer.Option("--weights", "-w", help="Path to the .pt weight file.", show_default=False),
#     ],
#     source: Annotated[
#         Optional[Path],
#         typer.Option("--source", "-s", help="Images / directory to run inference on (default: dataset/test/images)."),
#     ] = None,
#     conf: Annotated[float, typer.Option("--conf", help="Confidence threshold.", min=0.0, max=1.0)] = 0.25,
#     iou: Annotated[float, typer.Option("--iou", help="IoU threshold for NMS.", min=0.0, max=1.0)] = 0.45,
#     imgsz: Annotated[int, typer.Option("--imgsz", help="Input image size.", min=32)] = 640,
#     device: Annotated[Optional[str], typer.Option("--device", "-d", help="Device: cpu, 0, …")] = None,
#     save_images: Annotated[bool, typer.Option("--save-images", help="Save annotated output images.")] = False,
#     save_txt: Annotated[bool, typer.Option("--save-txt", help="Save predictions as YOLO .txt files.")] = False,
# ) -> None:
#     """Run inference or evaluation on the test partition (or a custom source)."""

#     try:
#         from ultralytics import YOLO
#     except ImportError:
#         rprint("[red]✗[/red] ultralytics is not installed. Run: pip install ultralytics")
#         raise typer.Exit(code=1)

#     if not weights.exists():
#         rprint(f"[red]✗[/red] Weights file not found: {weights}")
#         raise typer.Exit(code=1)

#     if source is None:
#         source = DATASET_DIR / "test" / "images"

#     if not source.exists():
#         rprint(f"[red]✗[/red] Source not found: {source}")
#         raise typer.Exit(code=1)

#     rprint(
#         Panel.fit(
#             f"[bold]weights:[/bold] {weights}\n"
#             f"[bold]source:[/bold]  {source}\n"
#             f"[bold]conf:[/bold]    {conf}   [bold]iou:[/bold] {iou}",
#             title="[cyan]test[/cyan]",
#         )
#     )

#     yolo = YOLO(str(weights))

#     extra: dict = {}
#     if device is not None:
#         extra["device"] = device

#     yolo.predict(
#         source=str(source),
#         conf=conf,
#         iou=iou,
#         imgsz=imgsz,
#         save=save_images,
#         save_txt=save_txt,
#         project=str(RUNS_DIR),
#         name="predict",
#         **extra,
#     )

#     rprint(f"[green]✔[/green] Inference complete.  Results in [bold]{RUNS_DIR / 'predict'}[/bold]")
@app.command()
def test(
    weights: Annotated[
        Path,
        typer.Option("--weights", "-w", help="Path to the .pt weight file.", show_default=False),
    ],
    source: Annotated[
        Optional[Path],
        typer.Option("--source", "-s", help="Images or directory to run inference on. Default: dataset/images/test."),
    ] = None,
    gt_csv: Annotated[
        Path,
        typer.Option("--gt-csv", help="Ground-truth annotations CSV. Default: dataset/annotations.csv."),
    ] = DATASET_DIR / "annotations.csv",
    conf: Annotated[
        float,
        typer.Option("--conf", help="Confidence threshold.", min=0.0, max=1.0),
    ] = 0.25,
    imgsz: Annotated[
        int,
        typer.Option("--imgsz", help="Input image size.", min=32),
    ] = 640,
    device: Annotated[
        Optional[str],
        typer.Option("--device", "-d", help="Device: cpu, 0, … default: auto."),
    ] = None,
) -> None:
    """Run prediction and, when ground-truth annotations are available, evaluation on the test set."""

    if source is None:
        source = IMAGES_DIR / "test"

    if not weights.exists():
        rprint(f"[red]✗[/red] Weights file not found: {weights}")
        raise typer.Exit(code=1)

    if not source.exists():
        rprint(f"[red]✗[/red] Source not found: {source}")
        raise typer.Exit(code=1)

    has_gt_csv = gt_csv.exists()

    if not has_gt_csv:
        rprint(
            f"[yellow]⚠[/yellow] Ground-truth CSV not found: {gt_csv}\n"
            "       Predictions will be saved, but evaluation statistics will be skipped."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if weights.resolve() == PRETRAINED_WEIGHTS.resolve():
        output_dir = RUN_DIR / f"predictions_{timestamp}"
    else:
        output_dir = TEST_OUTPUT_DIR / f"predictions_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_csv = output_dir / "predictions.csv"
    metrics_xlsx = output_dir / "metrics_summary.xlsx"
    model_run = infer_model_run_from_weights(weights)

    rprint(
        Panel.fit(
            f"[bold]weights:[/bold]     {weights}\n"
            f"[bold]model_run:[/bold]   {model_run}\n"
            f"[bold]source:[/bold]      {source}\n"
            f"[bold]gt_csv:[/bold]      {gt_csv}\n"
            f"[bold]conf:[/bold]        {conf}\n"
            f"[bold]imgsz:[/bold]       {imgsz}\n"
            f"[bold]predictions:[/bold] {prediction_csv}\n"
            f"[bold]metrics:[/bold]     {metrics_xlsx}",
            title="[cyan]test[/cyan]",
        )
    )

    prediction_csv = run_prediction(
        weights=weights,
        source=source,
        output_csv=prediction_csv,
        conf=conf,
        imgsz=imgsz,
        device=device,
    )

    if has_gt_csv:
        try:
            metrics_xlsx = run_evaluation(
                gt_csv=gt_csv,
                prediction_csv=prediction_csv,
                output_xlsx=metrics_xlsx,
                split="test",
                model_run=model_run,
            )
        except Exception as exc:
            rprint(
                f"[yellow]⚠[/yellow] Evaluation statistics could not be generated: {exc}\n"
                "       Predictions were saved successfully."
            )
            metrics_xlsx = None
    else:
        metrics_xlsx = None

    rprint("[green]✔[/green] Test complete.")
    rprint(f"[green]✔[/green] Predictions: [bold]{prediction_csv}[/bold]")

    if metrics_xlsx is not None:
        rprint(f"[green]✔[/green] Metrics: [bold]{metrics_xlsx}[/bold]")
    else:
        rprint("[yellow]⚠[/yellow] Metrics were not generated.")

@app.command("test-pretrained")
def test_pretrained(
    source: Annotated[
        Optional[Path],
        typer.Option("--source", "-s", help="Images or directory to run inference on. Default: dataset/images/test."),
    ] = None,
    gt_csv: Annotated[
        Path,
        typer.Option("--gt-csv", help="Ground-truth annotations CSV. Default: dataset/annotations.csv."),
    ] = DATASET_DIR / "annotations.csv",
    conf: Annotated[
        float,
        typer.Option("--conf", help="Confidence threshold.", min=0.0, max=1.0),
    ] = 0.25,
    imgsz: Annotated[
        int,
        typer.Option("--imgsz", help="Input image size.", min=32),
    ] = 640,
    device: Annotated[
        Optional[str],
        typer.Option("--device", "-d", help="Device: cpu, 0, … default: auto."),
    ] = None,
) -> None:
    """Test the pretrained DonaNet weights stored in weights/donanet_weights.pt."""

    test(
        weights=PRETRAINED_WEIGHTS,
        source=source,
        gt_csv=gt_csv,
        conf=conf,
        imgsz=imgsz,
        device=device,
    )

@app.command()
def list_datasets() -> None:
    """Show dataset partitions and their image / label counts."""

    table = Table(title="Dataset partitions", show_header=True, header_style="bold cyan")
    table.add_column("Partition", style="cyan", width=12)
    table.add_column("Images", justify="right")
    table.add_column("Labels", justify="right")
    table.add_column("Paired", justify="right")

    for partition in PARTITIONS:
        n_img, n_lbl = _count_partition(partition)
        paired = min(n_img, n_lbl)
        style = "green" if n_img > 0 else "dim"
        table.add_row(partition, str(n_img), str(n_lbl), str(paired), style=style)

    console.print(table)


# @app.command()
# def info() -> None:
#     """Display available weights and a dataset summary."""

#     rprint(Panel.fit("[bold white]DonaNet[/bold white] — WildINTEL YOLO CLI", subtitle=f"root: {ROOT}"))

#     # --- dataset ---
#     rprint("\n[bold cyan]Dataset[/bold cyan]")
#     list_datasets()

#     # --- dataset.yaml ---
#     if DATASET_YAML.exists():
#         with DATASET_YAML.open() as fh:
#             data = yaml.safe_load(fh)
#         nc = data.get("nc", "?")
#         names = data.get("names", {})
#         rprint(f"\n[bold cyan]dataset.yaml[/bold cyan]  classes={nc}")
#         for idx, cls_name in (names.items() if isinstance(names, dict) else enumerate(names)):
#             rprint(f"  [dim]{idx}[/dim] {cls_name}")
#     else:
#         rprint("\n[yellow]⚠[/yellow] dataset.yaml not found (run [bold]train --classes ...[/bold] to generate it)")

#     # --- weights ---
#     rprint("\n[bold cyan]Weights[/bold cyan]")
#     if WEIGHTS_DIR.exists():
#         weight_files = sorted(WEIGHTS_DIR.rglob("*.pt"))
#         if weight_files:
#             wt = Table(show_header=True, header_style="bold")
#             wt.add_column("File", style="green")
#             wt.add_column("Size", justify="right")
#             for wf in weight_files:
#                 size_mb = wf.stat().st_size / (1024 * 1024)
#                 wt.add_row(str(wf.relative_to(ROOT)), f"{size_mb:.1f} MB")
#             console.print(wt)
#         else:
#             rprint("  [dim]No weights found. Run [bold]train[/bold] first.[/dim]")
#     else:
#         rprint("  [dim]weights/ directory does not exist.[/dim]")
@app.command()
def info() -> None:
    """Display available trained weights and a dataset summary."""

    rprint(Panel.fit("[bold white]DonaNet[/bold white] — WildINTEL YOLO CLI", subtitle=f"root: {ROOT}"))

    # --- dataset ---
    rprint("\n[bold cyan]Dataset[/bold cyan]")
    list_datasets()

    # --- dataset/data.yaml ---
    if DATASET_YAML.exists():
        with DATASET_YAML.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        nc = data.get("nc", "?")
        names = data.get("names", {})

        rprint(f"\n[bold cyan]dataset/data.yaml[/bold cyan]  classes={nc}")

        if isinstance(names, dict):
            items = names.items()
        else:
            items = enumerate(names)

        for idx, cls_name in items:
            rprint(f"  [dim]{idx}[/dim] {cls_name}")
    else:
        rprint("\n[yellow]⚠[/yellow] dataset/data.yaml not found")

    # --- pretrained weights ---
    rprint("\n[bold cyan]Pretrained weights[/bold cyan]")

    if PRETRAINED_WEIGHTS.exists():
        size_mb = PRETRAINED_WEIGHTS.stat().st_size / (1024 * 1024)
        rprint(f"  [green]{PRETRAINED_WEIGHTS.relative_to(ROOT)}[/green]  {size_mb:.1f} MB")
    else:
        rprint(f"  [yellow]Missing:[/yellow] {PRETRAINED_WEIGHTS.relative_to(ROOT)}")

    # --- trained weights ---
    rprint("\n[bold cyan]Trained weights[/bold cyan]")

    weight_files = sorted(TRAIN_OUTPUT_DIR.rglob("*.pt"))

    if weight_files:
        wt = Table(show_header=True, header_style="bold")
        wt.add_column("File", style="green")
        wt.add_column("Size", justify="right")

        for wf in weight_files:
            size_mb = wf.stat().st_size / (1024 * 1024)
            wt.add_row(str(wf.relative_to(ROOT)), f"{size_mb:.1f} MB")

        console.print(wt)
    else:
        rprint("  [dim]No trained weights found. Run [bold]python donanet.py train[/bold] first.[/dim]")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _ensure_dirs()
    app()
