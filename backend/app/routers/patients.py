import base64
import re
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import or_
from sqlalchemy.orm import Session, joinedload

from backend.app.core.config import Settings, get_settings
from backend.app.db.models import Patient, Prediction, User, XrayStudy
from backend.app.db.session import get_db
from backend.app.routers.deps import get_current_user
from backend.app.routers.predictions import _prediction_read, _save_prediction
from backend.app.schemas import PatientCreate, PatientRead, PatientUpdate, XrayStudyRead
from backend.app.services.prediction_service import ModelService, decode_image_bytes, predict_image


router = APIRouter(tags=["patients"])
model_service = None


def configure_model_service(service: ModelService) -> None:
    global model_service
    model_service = service


@router.post("/patients", response_model=PatientRead, status_code=status.HTTP_201_CREATED)
def create_patient(
    payload: PatientCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    patient_code = (payload.patient_code or _generate_patient_code(db)).strip()
    _ensure_unique_patient_code(db, patient_code)
    patient = Patient(
        patient_code=patient_code,
        full_name=payload.full_name.strip(),
        gender=payload.gender,
        date_of_birth=payload.date_of_birth,
        phone=payload.phone,
        address=payload.address,
        national_id=payload.national_id,
        insurance_id=payload.insurance_id,
        medical_history=payload.medical_history,
        allergy_history=payload.allergy_history,
        current_symptoms=payload.current_symptoms,
        notes=payload.notes,
        created_by_user_id=current_user.id,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return _patient_read(patient)


@router.get("/patients", response_model=list[PatientRead])
def list_patients(
    search: Optional[str] = Query(default=None),
    limit: int = Query(default=30, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    include_inactive: bool = Query(default=False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(Patient).order_by(Patient.created_at.desc())
    if current_user.role != "admin":
        query = query.filter(Patient.created_by_user_id == current_user.id)
    if not include_inactive:
        query = query.filter(Patient.is_active.is_(True))
    if search:
        keyword = f"%{search.strip()}%"
        query = query.filter(
            or_(
                Patient.patient_code.ilike(keyword),
                Patient.full_name.ilike(keyword),
                Patient.phone.ilike(keyword),
            )
        )
    patients = query.offset(offset).limit(limit).all()
    return [_patient_read(item) for item in patients]


@router.get("/patients/{patient_id}", response_model=PatientRead)
def get_patient(
    patient_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return _patient_read(_get_patient_for_user(db, patient_id, current_user))


@router.put("/patients/{patient_id}", response_model=PatientRead)
def update_patient(
    patient_id: int,
    payload: PatientUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    patient = _get_patient_for_user(db, patient_id, current_user)
    updates = payload.model_dump(exclude_unset=True)
    if "patient_code" in updates and updates["patient_code"]:
        updates["patient_code"] = updates["patient_code"].strip()
        if updates["patient_code"] != patient.patient_code:
            _ensure_unique_patient_code(db, updates["patient_code"])
    for key, value in updates.items():
        if key == "full_name" and value:
            value = value.strip()
        setattr(patient, key, value)
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return _patient_read(patient)


@router.delete("/patients/{patient_id}", response_model=PatientRead)
def deactivate_patient(
    patient_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    patient = _get_patient_for_user(db, patient_id, current_user)
    patient.is_active = False
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return _patient_read(patient)


@router.post("/patients/{patient_id}/xray-studies", response_model=XrayStudyRead, status_code=status.HTTP_201_CREATED)
async def upload_xray_study(
    patient_id: int,
    file: UploadFile = File(...),
    conf_threshold: float = Query(default=0.25, ge=0.05, le=0.95),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    if model_service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model service is not configured")
    patient = _get_patient_for_user(db, patient_id, current_user)
    if not patient.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Patient is inactive")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file must be an image")

    raw = await file.read()
    image = decode_image_bytes(raw)
    filename = _safe_filename(file.filename or "xray.jpg")
    original_path = _study_upload_dir(settings, patient.id) / f"{uuid.uuid4().hex}_{filename}"
    original_path.parent.mkdir(parents=True, exist_ok=True)
    original_path.write_bytes(raw)

    study = XrayStudy(
        patient_id=patient.id,
        uploaded_by_user_id=current_user.id,
        original_filename=filename,
        stored_image_path=str(original_path),
        study_status="uploaded",
        image_width=int(image.shape[1]),
        image_height=int(image.shape[0]),
    )
    db.add(study)
    db.commit()
    db.refresh(study)

    try:
        models = model_service.load()
        payload = predict_image(image, filename, conf_threshold, models, settings)
        prediction = _save_prediction(
            db,
            current_user.id,
            payload,
            patient_id=patient.id,
            xray_study_id=study.id,
        )
        annotated_path = original_path.with_name(f"{original_path.stem}_annotated.jpg")
        if payload.get("annotated_image_base64"):
            annotated_path.write_bytes(base64.b64decode(payload["annotated_image_base64"]))
        study.prediction_id = prediction.id
        study.annotated_image_path = str(annotated_path)
        study.study_status = "diagnosed"
        db.add(study)
        db.commit()
        db.refresh(study)
    except HTTPException:
        db.rollback()
        failed_study = db.query(XrayStudy).filter(XrayStudy.id == study.id).first()
        if failed_study:
            failed_study.study_status = "failed"
            db.add(failed_study)
            db.commit()
        raise
    except Exception as exc:
        db.rollback()
        failed_study = db.query(XrayStudy).filter(XrayStudy.id == study.id).first()
        if failed_study:
            failed_study.study_status = "failed"
            db.add(failed_study)
            db.commit()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return _study_read(_load_study(db, study.id), include_prediction_image=True)


@router.get("/patients/{patient_id}/xray-studies", response_model=list[XrayStudyRead])
def list_patient_xray_studies(
    patient_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    patient = _get_patient_for_user(db, patient_id, current_user)
    studies = (
        db.query(XrayStudy)
        .options(
            joinedload(XrayStudy.prediction).joinedload(Prediction.probabilities),
            joinedload(XrayStudy.prediction).joinedload(Prediction.boxes),
            joinedload(XrayStudy.prediction).joinedload(Prediction.patient),
            joinedload(XrayStudy.prediction).joinedload(Prediction.xray_study),
        )
        .filter(XrayStudy.patient_id == patient.id)
        .order_by(XrayStudy.created_at.desc())
        .all()
    )
    return [_study_read(item, include_prediction_image=False) for item in studies]


@router.get("/xray-studies/{study_id}", response_model=XrayStudyRead)
def get_xray_study(
    study_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    study = _get_study_for_user(db, study_id, current_user)
    return _study_read(study)


@router.get("/xray-studies/{study_id}/image")
def get_xray_image(
    study_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    study = _get_study_for_user(db, study_id, current_user)
    return _file_response(study.stored_image_path)


@router.get("/xray-studies/{study_id}/annotated-image")
def get_annotated_xray_image(
    study_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    study = _get_study_for_user(db, study_id, current_user)
    if not study.annotated_image_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Annotated image not found")
    return _file_response(study.annotated_image_path)


def _get_patient_for_user(db: Session, patient_id: int, user: User) -> Patient:
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")
    if user.role != "admin" and patient.created_by_user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You cannot access this patient")
    return patient


def _get_study_for_user(db: Session, study_id: int, user: User) -> XrayStudy:
    study = _load_study(db, study_id)
    if not study:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="X-ray study not found")
    if user.role != "admin" and study.uploaded_by_user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You cannot access this X-ray study")
    return study


def _load_study(db: Session, study_id: int) -> Optional[XrayStudy]:
    return (
        db.query(XrayStudy)
        .options(
            joinedload(XrayStudy.prediction).joinedload(Prediction.probabilities),
            joinedload(XrayStudy.prediction).joinedload(Prediction.boxes),
            joinedload(XrayStudy.prediction).joinedload(Prediction.patient),
            joinedload(XrayStudy.prediction).joinedload(Prediction.xray_study),
        )
        .filter(XrayStudy.id == study_id)
        .first()
    )


def _patient_read(patient: Patient) -> PatientRead:
    return PatientRead(
        id=patient.id,
        patient_code=patient.patient_code,
        full_name=patient.full_name,
        gender=patient.gender,
        date_of_birth=patient.date_of_birth,
        phone=patient.phone,
        address=patient.address,
        national_id=patient.national_id,
        insurance_id=patient.insurance_id,
        medical_history=patient.medical_history,
        allergy_history=patient.allergy_history,
        current_symptoms=patient.current_symptoms,
        notes=patient.notes,
        created_by_user_id=patient.created_by_user_id,
        is_active=patient.is_active,
        created_at=patient.created_at,
        updated_at=patient.updated_at,
    )


def _study_read(study: XrayStudy, include_prediction_image: bool = True) -> XrayStudyRead:
    return XrayStudyRead(
        id=study.id,
        patient_id=study.patient_id,
        uploaded_by_user_id=study.uploaded_by_user_id,
        prediction_id=study.prediction_id,
        original_filename=study.original_filename,
        stored_image_path=study.stored_image_path,
        annotated_image_path=study.annotated_image_path,
        study_status=study.study_status,
        image_width=study.image_width,
        image_height=study.image_height,
        created_at=study.created_at,
        prediction=_prediction_read(study.prediction, include_image=include_prediction_image) if study.prediction else None,
    )


def _generate_patient_code(db: Session) -> str:
    return f"PT-{db.query(Patient).count() + 1:06d}"


def _ensure_unique_patient_code(db: Session, patient_code: str) -> None:
    if db.query(Patient).filter(Patient.patient_code == patient_code).first():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Patient code already exists")


def _safe_filename(filename: str) -> str:
    stem = Path(filename).name.strip() or "xray.jpg"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)


def _study_upload_dir(settings: Settings, patient_id: int) -> Path:
    return settings.resolve_path(settings.uploads_dir) / "patients" / str(patient_id) / "studies"


def _file_response(path_value: str) -> FileResponse:
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file not found")
    return FileResponse(path)
