from pathlib import Path
from typing import Any

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


def dataset_summary(csv_path: Path) -> dict[str, Any]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")
    df = add_class_name(pd.read_csv(csv_path))
    bbox_series = df["bbox"].apply(lambda value: "has_bbox" if pd.notna(value) and value != "none" else "no_bbox")
    return {
        "total_rows": int(len(df)),
        "columns": [str(col) for col in df.columns],
        "target_distribution": _counts(df, "target"),
        "image_type_distribution": _counts(df, "image_type"),
        "class_distribution": {name: int((df["class_name"] == name).sum()) for name in CLASS_ORDER},
        "bbox_distribution": {key: int(value) for key, value in bbox_series.value_counts().to_dict().items()},
        "source_distribution": _counts(df, "source"),
    }


def _counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    return {str(key): int(value) for key, value in df[column].value_counts(dropna=False).to_dict().items()}
