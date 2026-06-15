from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
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


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
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
