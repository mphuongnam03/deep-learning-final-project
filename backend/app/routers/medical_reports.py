from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.app.core.config import Settings, get_settings
from backend.app.db.models import User
from backend.app.db.session import get_db
from backend.app.routers.deps import get_current_user
from backend.app.schemas import MedicalReportRead
from backend.app.services.medical_report_service import (
    create_medical_report,
    get_latest_prediction_report,
    get_report_for_user,
    report_to_response,
)


router = APIRouter(tags=["medical-reports"])


@router.post("/predictions/{prediction_id}/medical-report", response_model=MedicalReportRead)
def generate_medical_report(
    prediction_id: int,
    force: bool = Query(default=False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    report = create_medical_report(
        db=db,
        prediction_id=prediction_id,
        user=current_user,
        settings=settings,
        force=force,
    )
    return report_to_response(report)


@router.get("/predictions/{prediction_id}/medical-report", response_model=MedicalReportRead)
def get_prediction_medical_report(
    prediction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return report_to_response(get_latest_prediction_report(db, prediction_id, current_user))


@router.get("/medical-reports/{report_id}", response_model=MedicalReportRead)
def get_medical_report(
    report_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return report_to_response(get_report_for_user(db, report_id, current_user))
