from datetime import datetime

from sqlalchemy import Boolean, Column, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from backend.app.db.session import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)
    password_salt = Column(String(64), nullable=False)
    role = Column(String(32), nullable=False, default="student")
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")
    medical_reports = relationship("MedicalReport", back_populates="user", cascade="all, delete-orphan")
    patients = relationship("Patient", back_populates="created_by", cascade="all, delete-orphan")
    xray_studies = relationship("XrayStudy", back_populates="uploaded_by")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    patient_code = Column(String(64), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=False, index=True)
    gender = Column(String(32), nullable=True)
    date_of_birth = Column(Date, nullable=True)
    phone = Column(String(64), nullable=True, index=True)
    address = Column(Text, nullable=True)
    national_id = Column(String(64), nullable=True)
    insurance_id = Column(String(64), nullable=True)
    medical_history = Column(Text, nullable=True)
    allergy_history = Column(Text, nullable=True)
    current_symptoms = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    created_by = relationship("User", back_populates="patients")
    predictions = relationship("Prediction", back_populates="patient")
    xray_studies = relationship("XrayStudy", back_populates="patient", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True, index=True)
    xray_study_id = Column(Integer, ForeignKey("xray_studies.id"), nullable=True, index=True)
    filename = Column(String(512), nullable=False)
    predicted_class = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    conf_threshold = Column(Float, nullable=False)
    cls_source = Column(String(64), nullable=False)
    raw_detection_count = Column(Integer, nullable=False, default=0)
    kept_detection_count = Column(Integer, nullable=False, default=0)
    dropped_detection_count = Column(Integer, nullable=False, default=0)
    image_width = Column(Integer, nullable=False)
    image_height = Column(Integer, nullable=False)
    processing_time_ms = Column(Float, nullable=False)
    annotated_image_base64 = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="predictions")
    patient = relationship("Patient", back_populates="predictions")
    xray_study = relationship("XrayStudy", foreign_keys=[xray_study_id])
    probabilities = relationship(
        "PredictionProbability",
        back_populates="prediction",
        cascade="all, delete-orphan",
    )
    boxes = relationship(
        "PredictionBox",
        back_populates="prediction",
        cascade="all, delete-orphan",
    )
    medical_reports = relationship(
        "MedicalReport",
        back_populates="prediction",
        cascade="all, delete-orphan",
    )


class XrayStudy(Base):
    __tablename__ = "xray_studies"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    uploaded_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=True, index=True)
    original_filename = Column(String(512), nullable=False)
    stored_image_path = Column(String(1024), nullable=False)
    annotated_image_path = Column(String(1024), nullable=True)
    study_status = Column(String(32), nullable=False, default="uploaded", index=True)
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    patient = relationship("Patient", back_populates="xray_studies")
    uploaded_by = relationship("User", back_populates="xray_studies")
    prediction = relationship("Prediction", foreign_keys=[prediction_id], post_update=True)


class PredictionProbability(Base):
    __tablename__ = "prediction_probabilities"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False, index=True)
    class_name = Column(String(64), nullable=False)
    probability = Column(Float, nullable=False)

    prediction = relationship("Prediction", back_populates="probabilities")


class PredictionBox(Base):
    __tablename__ = "prediction_boxes"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False, index=True)
    x1 = Column(Integer, nullable=False)
    y1 = Column(Integer, nullable=False)
    x2 = Column(Integer, nullable=False)
    y2 = Column(Integer, nullable=False)
    det_class = Column(String(64), nullable=False)
    det_conf = Column(Float, nullable=False)
    roi_class = Column(String(64), nullable=True)
    roi_conf = Column(Float, nullable=True)
    kept = Column(Boolean, nullable=False, default=True)
    reason = Column(String(128), nullable=True)

    prediction = relationship("Prediction", back_populates="boxes")


class TrainingMetric(Base):
    __tablename__ = "training_metrics"

    id = Column(Integer, primary_key=True, index=True)
    run_name = Column(String(255), nullable=False, index=True)
    model_type = Column(String(64), nullable=False, index=True)
    epoch = Column(Integer, nullable=False)
    train_box_loss = Column(Float, nullable=True)
    train_cls_loss = Column(Float, nullable=True)
    train_dfl_loss = Column(Float, nullable=True)
    train_loss = Column(Float, nullable=True)
    val_box_loss = Column(Float, nullable=True)
    val_cls_loss = Column(Float, nullable=True)
    val_dfl_loss = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    map50 = Column(Float, nullable=True)
    map50_95 = Column(Float, nullable=True)
    accuracy_top1 = Column(Float, nullable=True)
    accuracy_top5 = Column(Float, nullable=True)
    lr_pg0 = Column(Float, nullable=True)
    lr_pg1 = Column(Float, nullable=True)
    lr_pg2 = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class MedicalReport(Base):
    __tablename__ = "medical_reports"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    status = Column(String(32), nullable=False, default="pending")
    language = Column(String(16), nullable=False, default="vi")
    model_name = Column(String(128), nullable=False)
    report_json = Column(Text, nullable=True)
    report_html = Column(Text, nullable=True)
    safety_disclaimer = Column(Text, nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    prediction = relationship("Prediction", back_populates="medical_reports")
    user = relationship("User", back_populates="medical_reports")
