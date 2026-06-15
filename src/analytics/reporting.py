from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


CLASS_ORDER = ["healthy", "sick_but_no_tb", "active_tb", "latent_tb"]


def add_class_name(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def classify(row: pd.Series) -> str:
        if row.get("image_type") == "healthy":
            return "healthy"
        if row.get("image_type") == "sick_but_no_tb":
            return "sick_but_no_tb"
        if row.get("target") == "tb":
            return "latent_tb" if row.get("tb_type") == "latent_tb" else "active_tb"
        return str(row.get("image_type") or "unknown")

    df["class_name"] = df.apply(classify, axis=1)
    return df


def generate_report_plots(
    dataset_csv: str = "tbx11k-simplified/data.csv",
    detection_results_csv: str = "tb_detection/stage2_det/results.csv",
    classification_results_csv: str = "tb_classification/stage1_cls/results.csv",
    output_dir: str = "reports/plots",
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    dataset_path = Path(dataset_csv)
    if dataset_path.exists():
        dataset = add_class_name(pd.read_csv(dataset_path))
        created.extend(_plot_dataset(dataset, output))

    det_path = Path(detection_results_csv)
    if det_path.exists():
        detection = pd.read_csv(det_path)
        created.extend(_plot_detection_metrics(detection, output))

    cls_path = Path(classification_results_csv)
    if cls_path.exists():
        classification = pd.read_csv(cls_path)
        created.extend(_plot_classification_metrics(classification, output))

    return created


def _plot_dataset(df: pd.DataFrame, output: Path) -> list[Path]:
    created = []
    created.append(_bar_plot(df["target"].value_counts(), "Target distribution", "Target", "Images", output / "dataset_target_distribution.png"))
    created.append(_bar_plot(df["image_type"].value_counts(), "Image type distribution", "Image type", "Images", output / "dataset_image_type_distribution.png"))
    class_counts = df["class_name"].value_counts().reindex(CLASS_ORDER).fillna(0)
    created.append(_bar_plot(class_counts, "Four-class distribution", "Class", "Images", output / "dataset_class_distribution.png"))
    bbox_counts = df["bbox"].apply(lambda value: "has_bbox" if pd.notna(value) and value != "none" else "no_bbox").value_counts()
    created.append(_bar_plot(bbox_counts, "Bounding box availability", "BBox status", "Images", output / "dataset_bbox_distribution.png"))
    if "source" in df.columns:
        pivot = df.pivot_table(index="class_name", columns="source", values="fname", aggfunc="count", fill_value=0)
        pivot = pivot.reindex(CLASS_ORDER).fillna(0)
        path = output / "dataset_split_by_class.png"
        ax = pivot.plot(kind="bar", figsize=(10, 5), color=["#2563eb", "#f97316", "#16a34a"])
        ax.set_title("Train/validation split by class")
        ax.set_xlabel("Class")
        ax.set_ylabel("Images")
        ax.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        created.append(path)
    return created


def _plot_detection_metrics(df: pd.DataFrame, output: Path) -> list[Path]:
    created = []
    created.append(_line_plot(df, ["train/box_loss", "val/box_loss"], "Detection box loss", output / "detection_box_loss.png"))
    created.append(_line_plot(df, ["train/cls_loss", "val/cls_loss"], "Detection classification loss", output / "detection_cls_loss.png"))
    created.append(_line_plot(df, ["train/dfl_loss", "val/dfl_loss"], "Detection DFL loss", output / "detection_dfl_loss.png"))
    created.append(_line_plot(df, ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"], "Detection validation metrics", output / "detection_validation_metrics.png"))
    created.append(_line_plot(df, ["lr/pg0", "lr/pg1", "lr/pg2"], "Detection learning rate", output / "detection_learning_rate.png"))
    return created


def _plot_classification_metrics(df: pd.DataFrame, output: Path) -> list[Path]:
    created = []
    created.append(_line_plot(df, ["train/loss", "val/loss"], "Classification loss", output / "classification_loss.png"))
    created.append(_line_plot(df, ["metrics/accuracy_top1", "metrics/accuracy_top5"], "Classification accuracy", output / "classification_accuracy.png"))
    created.append(_line_plot(df, ["lr/pg0", "lr/pg1", "lr/pg2"], "Classification learning rate", output / "classification_learning_rate.png"))
    return created


def _bar_plot(series: pd.Series, title: str, xlabel: str, ylabel: str, path: Path) -> Path:
    ax = series.plot(kind="bar", figsize=(9, 5), color=["#2563eb", "#dc2626", "#f59e0b", "#16a34a"])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def _line_plot(df: pd.DataFrame, columns: Iterable[str], title: str, path: Path) -> Path:
    available = [column for column in columns if column in df.columns]
    if not available:
        return path
    ax = df.plot(x="epoch", y=available, figsize=(10, 5), marker="o", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


if __name__ == "__main__":
    for plot_path in generate_report_plots():
        print(plot_path)
