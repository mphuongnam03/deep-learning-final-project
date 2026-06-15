from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.core.config import Settings, get_settings
from backend.app.db.models import TrainingMetric, User
from backend.app.db.session import get_db
from backend.app.routers.deps import get_current_user
from backend.app.schemas import DatasetAnalytics, TrainingMetricRead
from backend.app.services.analytics_service import dataset_summary


router = APIRouter(tags=["analytics"])


@router.get("/analytics/dataset", response_model=DatasetAnalytics)
def get_dataset_analytics(
    current_user: User = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    try:
        return dataset_summary(settings.resolve_path(settings.dataset_csv_path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/training-metrics/import")
def import_training_metrics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    imports = [
        ("stage1_cls", "classification", Path("tb_classification/stage1_cls/results.csv")),
        ("stage2_det", "detection", Path("tb_detection/stage2_det/results.csv")),
    ]
    inserted = 0
    for run_name, model_type, csv_path in imports:
        if not csv_path.exists():
            continue
        db.query(TrainingMetric).filter(
            TrainingMetric.run_name == run_name,
            TrainingMetric.model_type == model_type,
        ).delete()
        frame = pd.read_csv(csv_path)
        for _, row in frame.iterrows():
            db.add(_metric_from_row(run_name, model_type, row))
            inserted += 1
    db.commit()
    return {"inserted": inserted}


@router.get("/training-metrics", response_model=list[TrainingMetricRead])
def list_training_metrics(
    model_type: Optional[str] = Query(default=None),
    limit: int = Query(default=300, ge=1, le=1000),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(TrainingMetric)
    if model_type:
        query = query.filter(TrainingMetric.model_type == model_type)
    query = query.order_by(TrainingMetric.model_type, TrainingMetric.epoch).limit(limit)
    return [TrainingMetricRead(**_metric_dict(item)) for item in query.all()]


def _clean(value):
    if pd.isna(value):
        return None
    return float(value)


def _metric_from_row(run_name: str, model_type: str, row: pd.Series) -> TrainingMetric:
    return TrainingMetric(
        run_name=run_name,
        model_type=model_type,
        epoch=int(row["epoch"]),
        train_box_loss=_clean(row.get("train/box_loss")),
        train_cls_loss=_clean(row.get("train/cls_loss")),
        train_dfl_loss=_clean(row.get("train/dfl_loss")),
        train_loss=_clean(row.get("train/loss")),
        val_box_loss=_clean(row.get("val/box_loss")),
        val_cls_loss=_clean(row.get("val/cls_loss")),
        val_dfl_loss=_clean(row.get("val/dfl_loss")),
        val_loss=_clean(row.get("val/loss")),
        precision=_clean(row.get("metrics/precision(B)")),
        recall=_clean(row.get("metrics/recall(B)")),
        map50=_clean(row.get("metrics/mAP50(B)")),
        map50_95=_clean(row.get("metrics/mAP50-95(B)")),
        accuracy_top1=_clean(row.get("metrics/accuracy_top1")),
        accuracy_top5=_clean(row.get("metrics/accuracy_top5")),
        lr_pg0=_clean(row.get("lr/pg0")),
        lr_pg1=_clean(row.get("lr/pg1")),
        lr_pg2=_clean(row.get("lr/pg2")),
    )


def _metric_dict(metric: TrainingMetric) -> dict:
    return {
        "id": metric.id,
        "run_name": metric.run_name,
        "model_type": metric.model_type,
        "epoch": metric.epoch,
        "train_box_loss": metric.train_box_loss,
        "train_cls_loss": metric.train_cls_loss,
        "train_dfl_loss": metric.train_dfl_loss,
        "train_loss": metric.train_loss,
        "val_box_loss": metric.val_box_loss,
        "val_cls_loss": metric.val_cls_loss,
        "val_dfl_loss": metric.val_dfl_loss,
        "val_loss": metric.val_loss,
        "precision": metric.precision,
        "recall": metric.recall,
        "map50": metric.map50,
        "map50_95": metric.map50_95,
        "accuracy_top1": metric.accuracy_top1,
        "accuracy_top5": metric.accuracy_top5,
        "lr_pg0": metric.lr_pg0,
        "lr_pg1": metric.lr_pg1,
        "lr_pg2": metric.lr_pg2,
    }
