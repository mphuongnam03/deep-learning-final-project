from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session, joinedload

from backend.app.core.config import Settings, get_settings
from backend.app.db.models import Prediction, PredictionBox, PredictionProbability, User
from backend.app.db.session import get_db
from backend.app.routers.deps import get_current_user
from backend.app.schemas import BoxRead, PatientSummary, PredictionRead, ProbabilityRead, XrayStudySummary
from backend.app.services.prediction_service import ModelService, decode_upload_image, predict_image


router = APIRouter(tags=["predictions"])
model_service = None


def configure_model_service(service: ModelService) -> None:
    global model_service
    model_service = service


@router.post("/predict", response_model=PredictionRead)
async def predict(
    file: UploadFile = File(...),
    conf_threshold: float = Query(default=0.25, ge=0.05, le=0.95),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    if model_service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model service is not configured")
    models = model_service.load()
    image = await decode_upload_image(file)
    payload = predict_image(image, file.filename or "uploaded_image.jpg", conf_threshold, models, settings)
    prediction = _save_prediction(db, current_user.id, payload)
    return _prediction_read(prediction)


@router.get("/predictions", response_model=list[PredictionRead])
def list_predictions(
    limit: int = Query(default=20, ge=1, le=100),
    patient_id: Optional[int] = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = (
        db.query(Prediction)
        .options(
            joinedload(Prediction.probabilities),
            joinedload(Prediction.boxes),
            joinedload(Prediction.patient),
            joinedload(Prediction.xray_study),
        )
        .order_by(Prediction.created_at.desc())
    )
    if current_user.role != "admin":
        query = query.filter(Prediction.user_id == current_user.id)
    if patient_id is not None:
        query = query.filter(Prediction.patient_id == patient_id)
    query = query.limit(limit)
    return [_prediction_read(item, include_image=False) for item in query.all()]


@router.get("/predictions/{prediction_id}", response_model=PredictionRead)
def get_prediction(
    prediction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    prediction = (
        db.query(Prediction)
        .options(
            joinedload(Prediction.probabilities),
            joinedload(Prediction.boxes),
            joinedload(Prediction.patient),
            joinedload(Prediction.xray_study),
        )
        .filter(Prediction.id == prediction_id)
        .first()
    )
    if not prediction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    if current_user.role != "admin" and prediction.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You cannot access this prediction")
    return _prediction_read(prediction)


def _save_prediction(
    db: Session,
    user_id: int,
    payload: dict,
    patient_id: Optional[int] = None,
    xray_study_id: Optional[int] = None,
) -> Prediction:
    prediction = Prediction(
        user_id=user_id,
        patient_id=patient_id,
        xray_study_id=xray_study_id,
        filename=payload["filename"],
        predicted_class=payload["predicted_class"],
        confidence=payload["confidence"],
        conf_threshold=payload["conf_threshold"],
        cls_source=payload["cls_source"],
        raw_detection_count=payload["raw_detection_count"],
        kept_detection_count=payload["kept_detection_count"],
        dropped_detection_count=payload["dropped_detection_count"],
        image_width=payload["image_width"],
        image_height=payload["image_height"],
        processing_time_ms=payload["processing_time_ms"],
        annotated_image_base64=payload["annotated_image_base64"],
    )
    prediction.probabilities = [
        PredictionProbability(class_name=item["class_name"], probability=item["probability"])
        for item in payload["probabilities"]
    ]
    prediction.boxes = [
        PredictionBox(
            x1=item["bbox"][0],
            y1=item["bbox"][1],
            x2=item["bbox"][2],
            y2=item["bbox"][3],
            det_class=item["det_class"],
            det_conf=item["det_conf"],
            roi_class=item["roi_class"],
            roi_conf=item["roi_conf"],
            kept=item["kept"],
            reason=item["reason"],
        )
        for item in payload["boxes"]
    ]
    db.add(prediction)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    db.refresh(prediction)
    return (
        db.query(Prediction)
        .options(
            joinedload(Prediction.probabilities),
            joinedload(Prediction.boxes),
            joinedload(Prediction.patient),
            joinedload(Prediction.xray_study),
        )
        .filter(Prediction.id == prediction.id)
        .one()
    )


def _prediction_read(prediction: Prediction, include_image: bool = True) -> PredictionRead:
    patient = None
    if prediction.patient:
        patient = PatientSummary(
            id=prediction.patient.id,
            patient_code=prediction.patient.patient_code,
            full_name=prediction.patient.full_name,
            gender=prediction.patient.gender,
            date_of_birth=prediction.patient.date_of_birth,
        )
    xray_study = None
    if prediction.xray_study:
        xray_study = XrayStudySummary(
            id=prediction.xray_study.id,
            patient_id=prediction.xray_study.patient_id,
            original_filename=prediction.xray_study.original_filename,
            study_status=prediction.xray_study.study_status,
            image_width=prediction.xray_study.image_width,
            image_height=prediction.xray_study.image_height,
            created_at=prediction.xray_study.created_at,
        )
    return PredictionRead(
        id=prediction.id,
        filename=prediction.filename,
        predicted_class=prediction.predicted_class,
        confidence=prediction.confidence,
        conf_threshold=prediction.conf_threshold,
        cls_source=prediction.cls_source,
        raw_detection_count=prediction.raw_detection_count,
        kept_detection_count=prediction.kept_detection_count,
        dropped_detection_count=prediction.dropped_detection_count,
        image_width=prediction.image_width,
        image_height=prediction.image_height,
        processing_time_ms=prediction.processing_time_ms,
        created_at=prediction.created_at,
        patient=patient,
        xray_study=xray_study,
        annotated_image_base64=prediction.annotated_image_base64 if include_image else None,
        probabilities=[
            ProbabilityRead(class_name=item.class_name, probability=item.probability)
            for item in sorted(prediction.probabilities, key=lambda value: value.id)
        ],
        boxes=[
            BoxRead(
                bbox=[box.x1, box.y1, box.x2, box.y2],
                det_class=box.det_class,
                det_conf=box.det_conf,
                roi_class=box.roi_class,
                roi_conf=box.roi_conf,
                kept=box.kept,
                reason=box.reason,
            )
            for box in sorted(prediction.boxes, key=lambda value: value.id)
        ],
    )
