from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str = Field(..., min_length=2, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserRead(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserRead


class ProbabilityRead(BaseModel):
    class_name: str
    probability: float


class BoxRead(BaseModel):
    bbox: list[int]
    det_class: str
    det_conf: float
    roi_class: Optional[str] = None
    roi_conf: Optional[float] = None
    kept: bool
    reason: Optional[str] = None


class PredictionRead(BaseModel):
    id: int
    filename: str
    predicted_class: str
    confidence: float
    conf_threshold: float
    cls_source: str
    raw_detection_count: int
    kept_detection_count: int
    dropped_detection_count: int
    image_width: int
    image_height: int
    processing_time_ms: float
    created_at: datetime
    probabilities: list[ProbabilityRead]
    boxes: list[BoxRead]
    annotated_image_base64: Optional[str] = None


class DatasetAnalytics(BaseModel):
    total_rows: int
    columns: list[str]
    target_distribution: dict[str, int]
    image_type_distribution: dict[str, int]
    class_distribution: dict[str, int]
    bbox_distribution: dict[str, int]
    source_distribution: dict[str, int]


class TrainingMetricRead(BaseModel):
    id: int
    run_name: str
    model_type: str
    epoch: int
    train_box_loss: Optional[float] = None
    train_cls_loss: Optional[float] = None
    train_dfl_loss: Optional[float] = None
    train_loss: Optional[float] = None
    val_box_loss: Optional[float] = None
    val_cls_loss: Optional[float] = None
    val_dfl_loss: Optional[float] = None
    val_loss: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    map50: Optional[float] = None
    map50_95: Optional[float] = None
    accuracy_top1: Optional[float] = None
    accuracy_top5: Optional[float] = None
    lr_pg0: Optional[float] = None
    lr_pg1: Optional[float] = None
    lr_pg2: Optional[float] = None
