from datetime import date, datetime
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


class PatientSummary(BaseModel):
    id: int
    patient_code: str
    full_name: str
    gender: Optional[str] = None
    date_of_birth: Optional[date] = None


class XrayStudySummary(BaseModel):
    id: int
    patient_id: int
    original_filename: str
    study_status: str
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    created_at: datetime


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
    patient: Optional[PatientSummary] = None
    xray_study: Optional[XrayStudySummary] = None
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


class MedicalReportContent(BaseModel):
    clinical_summary: str
    imaging_findings: list[str]
    ai_interpretation: str
    risk_level: str
    recommendations: list[str]
    patient_advice: list[str]
    red_flags: list[str]
    limitations: list[str]
    next_steps: list[str]
    disclaimer: str


class MedicalReportRead(BaseModel):
    id: int
    prediction_id: int
    user_id: int
    status: str
    language: str
    model_name: str
    report: Optional[MedicalReportContent] = None
    report_html: Optional[str] = None
    patient: Optional[PatientSummary] = None
    safety_disclaimer: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class PatientBase(BaseModel):
    patient_code: Optional[str] = Field(default=None, max_length=64)
    full_name: str = Field(..., min_length=2, max_length=255)
    gender: Optional[str] = Field(default=None, max_length=32)
    date_of_birth: Optional[date] = None
    phone: Optional[str] = Field(default=None, max_length=64)
    address: Optional[str] = None
    national_id: Optional[str] = Field(default=None, max_length=64)
    insurance_id: Optional[str] = Field(default=None, max_length=64)
    medical_history: Optional[str] = None
    allergy_history: Optional[str] = None
    current_symptoms: Optional[str] = None
    notes: Optional[str] = None


class PatientCreate(PatientBase):
    pass


class PatientUpdate(BaseModel):
    patient_code: Optional[str] = Field(default=None, max_length=64)
    full_name: Optional[str] = Field(default=None, min_length=2, max_length=255)
    gender: Optional[str] = Field(default=None, max_length=32)
    date_of_birth: Optional[date] = None
    phone: Optional[str] = Field(default=None, max_length=64)
    address: Optional[str] = None
    national_id: Optional[str] = Field(default=None, max_length=64)
    insurance_id: Optional[str] = Field(default=None, max_length=64)
    medical_history: Optional[str] = None
    allergy_history: Optional[str] = None
    current_symptoms: Optional[str] = None
    notes: Optional[str] = None
    is_active: Optional[bool] = None


class PatientRead(PatientBase):
    id: int
    patient_code: str
    created_by_user_id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime


class XrayStudyRead(BaseModel):
    id: int
    patient_id: int
    uploaded_by_user_id: int
    prediction_id: Optional[int] = None
    original_filename: str
    stored_image_path: str
    annotated_image_path: Optional[str] = None
    study_status: str
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    created_at: datetime
    prediction: Optional[PredictionRead] = None
