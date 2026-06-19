import html
import json
from datetime import datetime
from typing import Any

from fastapi import HTTPException, status
from pydantic import ValidationError
from sqlalchemy.orm import Session, joinedload

from backend.app.core.config import Settings
from backend.app.db.models import MedicalReport, Prediction, User
from backend.app.schemas import MedicalReportContent, PatientSummary


SAFETY_DISCLAIMER = (
    "Báo cáo này được tạo bởi hệ thống AI cho mục đích hỗ trợ học thuật và tham khảo. "
    "Kết quả không thay thế chẩn đoán, tư vấn hoặc điều trị của bác sĩ. "
    "Người bệnh cần được bác sĩ chuyên khoa hô hấp hoặc chẩn đoán hình ảnh đánh giá và xác nhận."
)


def get_prediction_for_user(db: Session, prediction_id: int, user: User) -> Prediction:
    prediction = (
        db.query(Prediction)
        .options(joinedload(Prediction.probabilities), joinedload(Prediction.boxes), joinedload(Prediction.patient))
        .filter(Prediction.id == prediction_id)
        .first()
    )
    if not prediction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    if user.role != "admin" and prediction.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You cannot access this prediction")
    return prediction


def get_report_for_user(db: Session, report_id: int, user: User) -> MedicalReport:
    report = db.query(MedicalReport).filter(MedicalReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Medical report not found")
    if user.role != "admin" and report.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You cannot access this report")
    return report


def get_latest_prediction_report(db: Session, prediction_id: int, user: User) -> MedicalReport:
    prediction = get_prediction_for_user(db, prediction_id, user)
    report = (
        db.query(MedicalReport)
        .filter(MedicalReport.prediction_id == prediction.id)
        .order_by(MedicalReport.created_at.desc())
        .first()
    )
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Medical report not found")
    return report


def create_medical_report(
    db: Session,
    prediction_id: int,
    user: User,
    settings: Settings,
    force: bool = False,
) -> MedicalReport:
    if not settings.enable_medical_reports:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Medical reports are disabled")

    prediction = get_prediction_for_user(db, prediction_id, user)
    existing = (
        db.query(MedicalReport)
        .filter(MedicalReport.prediction_id == prediction.id, MedicalReport.status == "completed")
        .order_by(MedicalReport.created_at.desc())
        .first()
    )
    if existing and not force:
        return existing

    report = MedicalReport(
        prediction_id=prediction.id,
        user_id=prediction.user_id,
        status="pending",
        language="vi",
        model_name=settings.gemini_model,
        safety_disclaimer=SAFETY_DISCLAIMER,
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    try:
        if not settings.gemini_api_key:
            raise RuntimeError("Missing GEMINI_API_KEY")
        content = _call_gemini(prediction, settings)
        report.report_json = content.model_dump_json(ensure_ascii=False)
        report.report_html = render_report_html(content)
        report.status = "completed"
        report.error_message = None
    except Exception as exc:
        report.status = "failed"
        report.error_message = str(exc)
        report.report_json = None
        report.report_html = None
    finally:
        report.updated_at = datetime.utcnow()
        db.add(report)
        db.commit()
        db.refresh(report)

    if report.status == "failed":
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=report.error_message)
    return report


def report_to_response(report: MedicalReport) -> dict[str, Any]:
    content = None
    if report.report_json:
        try:
            content = MedicalReportContent.model_validate_json(report.report_json)
        except ValidationError:
            content = None
    patient = None
    if report.prediction and report.prediction.patient:
        patient = PatientSummary(
            id=report.prediction.patient.id,
            patient_code=report.prediction.patient.patient_code,
            full_name=report.prediction.patient.full_name,
            gender=report.prediction.patient.gender,
            date_of_birth=report.prediction.patient.date_of_birth,
        )
    return {
        "id": report.id,
        "prediction_id": report.prediction_id,
        "user_id": report.user_id,
        "status": report.status,
        "language": report.language,
        "model_name": report.model_name,
        "report": content,
        "report_html": report.report_html,
        "patient": patient,
        "safety_disclaimer": report.safety_disclaimer,
        "error_message": report.error_message,
        "created_at": report.created_at,
        "updated_at": report.updated_at,
    }


def _call_gemini(prediction: Prediction, settings: Settings) -> MedicalReportContent:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("google-genai is required to generate medical reports") from exc

    client = genai.Client(api_key=settings.gemini_api_key)
    prompt = _build_prompt(prediction)
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MedicalReportContent,
            temperature=0.2,
            system_instruction=(
                "Bạn là trợ lý soạn thảo báo cáo y khoa dựa trên kết quả AI đọc X-quang ngực. "
                "Bạn không phải bác sĩ điều trị. Không kê đơn, không đưa liều thuốc, không thay thế chẩn đoán lâm sàng. "
                "Luôn yêu cầu bác sĩ chuyên khoa xác nhận và trả lời bằng tiếng Việt."
            ),
        ),
    )
    if getattr(response, "parsed", None) is not None:
        return response.parsed
    text = getattr(response, "text", "")
    if not text:
        raise RuntimeError("Gemini returned an empty report")
    return MedicalReportContent.model_validate_json(text)


def _build_prompt(prediction: Prediction) -> str:
    patient_context = None
    if prediction.patient:
        patient_context = {
            "gender": prediction.patient.gender,
            "date_of_birth": prediction.patient.date_of_birth.isoformat() if prediction.patient.date_of_birth else None,
            "medical_history": prediction.patient.medical_history,
            "allergy_history": prediction.patient.allergy_history,
            "current_symptoms": prediction.patient.current_symptoms,
            "notes": prediction.patient.notes,
        }
    payload = {
        "prediction_id": prediction.id,
        "filename": prediction.filename,
        "patient_context": patient_context,
        "image_size": {"width": prediction.image_width, "height": prediction.image_height},
        "predicted_class": prediction.predicted_class,
        "confidence": prediction.confidence,
        "classification_source": prediction.cls_source,
        "raw_detection_count": prediction.raw_detection_count,
        "kept_detection_count": prediction.kept_detection_count,
        "dropped_detection_count": prediction.dropped_detection_count,
        "probabilities": [
            {"class_name": item.class_name, "probability": item.probability}
            for item in sorted(prediction.probabilities, key=lambda value: value.id)
        ],
        "boxes": [
            {
                "bbox": [box.x1, box.y1, box.x2, box.y2],
                "det_class": box.det_class,
                "det_conf": box.det_conf,
                "roi_class": box.roi_class,
                "roi_conf": box.roi_conf,
                "kept": box.kept,
                "reason": box.reason,
            }
            for box in sorted(prediction.boxes, key=lambda value: value.id)
        ],
        "processing_time_ms": prediction.processing_time_ms,
    }
    return f"""
Hãy tạo báo cáo y khoa có cấu trúc dựa trên kết quả AI phân tích ảnh X-quang ngực sau.

Dữ liệu đầu vào:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Yêu cầu nội dung:
- Chỉ dựa trên dữ liệu AI được cung cấp; không suy đoán thông tin bệnh nhân ngoài dữ liệu.
- Nếu kết quả là active_tb hoặc latent_tb, nhấn mạnh cần khám chuyên khoa và xét nghiệm xác nhận lao.
- Nếu kết quả là healthy hoặc sick_but_no_tb, nêu rõ kết quả AI không loại trừ hoàn toàn bệnh nếu có triệu chứng.
- Khuyến nghị chỉ ở mức định hướng an toàn: đi khám, xét nghiệm xác nhận, theo dõi triệu chứng, phòng lây khi nghi lao.
- Không kê thuốc, không nêu liều, không đưa phác đồ điều trị cá nhân hóa.
- Trả về đúng schema JSON được yêu cầu.
"""


def render_report_html(content: MedicalReportContent) -> str:
    sections = [
        ("Tóm tắt lâm sàng", [content.clinical_summary]),
        ("Nhận định hình ảnh", content.imaging_findings),
        ("Diễn giải AI", [content.ai_interpretation]),
        ("Mức độ nguy cơ", [content.risk_level]),
        ("Khuyến nghị tiếp theo", content.recommendations),
        ("Lời khuyên cho bệnh nhân", content.patient_advice),
        ("Dấu hiệu cần đi khám gấp", content.red_flags),
        ("Giới hạn của AI", content.limitations),
        ("Các bước nên thực hiện", content.next_steps),
        ("Lưu ý an toàn", [content.disclaimer]),
    ]
    html_sections = []
    for title, items in sections:
        html_sections.append(
            "<section class=\"medical-report-section\">"
            f"<h3>{html.escape(title)}</h3>"
            f"{_render_items(items)}"
            "</section>"
        )
    return "<article class=\"medical-report\">" + "".join(html_sections) + "</article>"


def _render_items(items: list[str]) -> str:
    if len(items) == 1:
        return f"<p>{html.escape(items[0])}</p>"
    return "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in items) + "</ul>"
