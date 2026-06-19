# TB AI Backend API Guide for Flutter

Tài liệu này dành cho Flutter app tích hợp với backend FastAPI của hệ thống TB AI Diagnosis.

## 1. Base URL

Local machine:

```text
http://localhost:8000/api
```

Thiết bị mobile cùng Wi-Fi/LAN:

```text
http://<LOCAL_IP>:8000/api
```

Ví dụ:

```text
http://192.168.1.5:8000/api
```

Swagger/OpenAPI:

```text
http://<LOCAL_IP>:8000/docs
http://<LOCAL_IP>:8000/openapi.json
```

Health check không có prefix `/api`:

```http
GET /health
```

## 2. Authentication

Backend dùng Bearer JWT. Sau khi `register` hoặc `login`, Flutter lưu `access_token` và gửi trong mọi API cần đăng nhập.

Header:

```http
Authorization: Bearer <access_token>
```

### Register

```http
POST /api/auth/register
Content-Type: application/json
```

Request:

```json
{
  "email": "student@example.com",
  "full_name": "Nguyen Van A",
  "password": "Password123"
}
```

Response `201`:

```json
{
  "access_token": "jwt_token_here",
  "token_type": "bearer",
  "user": {
    "id": 1,
    "email": "student@example.com",
    "full_name": "Nguyen Van A",
    "role": "admin",
    "is_active": true,
    "created_at": "2026-06-15T12:00:00"
  }
}
```

Ghi chú:
- User đầu tiên trong database là `admin`.
- Các user sau mặc định là `student`.

### Login

```http
POST /api/auth/login
Content-Type: application/json
```

Request:

```json
{
  "email": "student@example.com",
  "password": "Password123"
}
```

Response giống `register`.

### Current User

```http
GET /api/auth/me
Authorization: Bearer <access_token>
```

### Logout

```http
POST /api/auth/logout
```

Backend không giữ session state. Flutter chỉ cần xóa token local.

## 3. Prediction API

### Upload X-ray Image And Predict

```http
POST /api/predict?conf_threshold=0.25
Authorization: Bearer <access_token>
Content-Type: multipart/form-data
```

Multipart field:

```text
file: image file
```

Query:

| Name | Type | Default | Range | Description |
| --- | --- | --- | --- | --- |
| `conf_threshold` | float | `0.25` | `0.05` to `0.95` | Detection confidence threshold |

Response:

```json
{
  "id": 12,
  "filename": "xray.png",
  "predicted_class": "active_tb",
  "confidence": 0.921,
  "conf_threshold": 0.25,
  "cls_source": "roi_filtered_avg",
  "raw_detection_count": 2,
  "kept_detection_count": 1,
  "dropped_detection_count": 1,
  "image_width": 512,
  "image_height": 512,
  "processing_time_ms": 843.5,
  "created_at": "2026-06-15T12:00:00",
  "probabilities": [
    { "class_name": "active_tb", "probability": 0.921 },
    { "class_name": "healthy", "probability": 0.021 },
    { "class_name": "latent_tb", "probability": 0.044 },
    { "class_name": "sick_but_no_tb", "probability": 0.014 }
  ],
  "boxes": [
    {
      "bbox": [120, 80, 260, 220],
      "det_class": "active_tb",
      "det_conf": 0.81,
      "roi_class": "active_tb",
      "roi_conf": 0.92,
      "kept": true,
      "reason": null
    }
  ],
  "annotated_image_base64": "/9j/4AAQSkZJRgABAQ..."
}
```

Flutter hiển thị ảnh annotated:

```dart
Image.memory(base64Decode(prediction.annotatedImageBase64))
```

Nếu lấy danh sách lịch sử, `annotated_image_base64` thường là `null` để giảm payload.

### List Prediction History

```http
GET /api/predictions?limit=20
Authorization: Bearer <access_token>
```

Response:

```json
[
  {
    "id": 12,
    "filename": "xray.png",
    "predicted_class": "active_tb",
    "confidence": 0.921,
    "conf_threshold": 0.25,
    "cls_source": "roi_filtered_avg",
    "raw_detection_count": 2,
    "kept_detection_count": 1,
    "dropped_detection_count": 1,
    "image_width": 512,
    "image_height": 512,
    "processing_time_ms": 843.5,
    "created_at": "2026-06-15T12:00:00",
    "probabilities": [],
    "boxes": [],
    "annotated_image_base64": null
  }
]
```

Admin thấy tất cả predictions. Student chỉ thấy predictions của chính mình.

### Get Prediction Detail

```http
GET /api/predictions/{prediction_id}
Authorization: Bearer <access_token>
```

Trả về `PredictionRead` đầy đủ, bao gồm `annotated_image_base64`.

## 3A. Patient Workflow API

Recommended clinical workflow:

1. Doctor logs in.
2. Doctor creates/selects a patient.
3. Doctor uploads a chest X-ray to `/patients/{patient_id}/xray-studies`.
4. Backend stores original image, runs AI diagnosis, stores prediction and returns annotated result.

New users after the first admin are created with role `doctor`. Existing `student` users can still use the doctor workflow.

### Create Patient

```http
POST /api/patients
Authorization: Bearer <access_token>
Content-Type: application/json
```

Request:

```json
{
  "patient_code": "PT-000001",
  "full_name": "Nguyen Van A",
  "gender": "male",
  "date_of_birth": "1990-01-20",
  "phone": "0900000000",
  "address": "Ho Chi Minh City",
  "national_id": "optional",
  "insurance_id": "optional",
  "medical_history": "No known chronic disease",
  "allergy_history": "No known allergy",
  "current_symptoms": "Cough and fever",
  "notes": "Initial screening"
}
```

If `patient_code` is null or omitted, backend generates one.

### List Patients

```http
GET /api/patients?search=nguyen&limit=30&offset=0
Authorization: Bearer <access_token>
```

Admin sees all patients. Non-admin users see only patients they created.

### Get/Update/Deactivate Patient

```http
GET /api/patients/{patient_id}
PUT /api/patients/{patient_id}
DELETE /api/patients/{patient_id}
Authorization: Bearer <access_token>
```

`DELETE` is soft delete: backend sets `is_active=false`.

### Upload Patient X-ray And Diagnose

```http
POST /api/patients/{patient_id}/xray-studies?conf_threshold=0.25
Authorization: Bearer <access_token>
Content-Type: multipart/form-data
```

Multipart field:

```text
file: image file
```

Response:

```json
{
  "id": 5,
  "patient_id": 1,
  "uploaded_by_user_id": 2,
  "prediction_id": 12,
  "original_filename": "xray.jpg",
  "stored_image_path": "backend/uploads/patients/1/studies/...",
  "annotated_image_path": "backend/uploads/patients/1/studies/..._annotated.jpg",
  "study_status": "diagnosed",
  "image_width": 512,
  "image_height": 512,
  "created_at": "2026-06-15T12:00:00",
  "prediction": {
    "id": 12,
    "predicted_class": "active_tb",
    "confidence": 0.921,
    "patient": {
      "id": 1,
      "patient_code": "PT-000001",
      "full_name": "Nguyen Van A"
    },
    "annotated_image_base64": "/9j/4AAQSkZJRgABAQ..."
  }
}
```

### List Patient X-ray Studies

```http
GET /api/patients/{patient_id}/xray-studies
Authorization: Bearer <access_token>
```

### Get X-ray Study And Images

```http
GET /api/xray-studies/{study_id}
GET /api/xray-studies/{study_id}/image
GET /api/xray-studies/{study_id}/annotated-image
Authorization: Bearer <access_token>
```

For image endpoints, Flutter can display the response bytes directly:

```dart
final response = await dio.get(
  '/xray-studies/$studyId/annotated-image',
  options: Options(responseType: ResponseType.bytes),
);
Image.memory(Uint8List.fromList(response.data));
```

### Filter Prediction History By Patient

```http
GET /api/predictions?patient_id=1&limit=20
Authorization: Bearer <access_token>
```

Old quick upload endpoint `/api/predict` still works, but those predictions are not linked to a patient.

## 4. Medical Report API

Báo cáo y khoa được tạo on-demand bằng Gemini API. Backend chỉ dựa trên prediction đã lưu, không yêu cầu nhập thông tin bệnh nhân.

### Generate Medical Report

```http
POST /api/predictions/{prediction_id}/medical-report
Authorization: Bearer <access_token>
```

Optional query:

```text
force=true
```

`force=true` tạo lại report mới thay vì dùng report completed cũ.

Response:

```json
{
  "id": 3,
  "prediction_id": 12,
  "user_id": 1,
  "status": "completed",
  "language": "vi",
  "model_name": "gemini-3.5-flash",
  "report": {
    "clinical_summary": "Kết quả AI gợi ý...",
    "imaging_findings": [
      "Có vùng nghi tổn thương được mô hình phát hiện..."
    ],
    "ai_interpretation": "Phân tích AI nghi ngờ active_tb...",
    "risk_level": "Cần đánh giá y khoa sớm",
    "recommendations": [
      "Khám chuyên khoa hô hấp hoặc chẩn đoán hình ảnh.",
      "Làm xét nghiệm xác nhận lao nếu có chỉ định."
    ],
    "patient_advice": [
      "Không tự dùng thuốc kháng lao.",
      "Theo dõi ho, sốt, sụt cân, khó thở."
    ],
    "red_flags": [
      "Khó thở tăng nhanh.",
      "Ho ra máu."
    ],
    "limitations": [
      "AI không thay thế bác sĩ.",
      "X-quang đơn độc không đủ để kết luận chẩn đoán."
    ],
    "next_steps": [
      "Mang kết quả đến cơ sở y tế để được đánh giá."
    ],
    "disclaimer": "Báo cáo chỉ có mục đích hỗ trợ..."
  },
  "report_html": "<article class=\"medical-report\">...</article>",
  "safety_disclaimer": "Báo cáo này được tạo bởi hệ thống AI...",
  "error_message": null,
  "created_at": "2026-06-15T12:00:00",
  "updated_at": "2026-06-15T12:00:02"
}
```

Flutter nên ưu tiên render `report` thành native widgets. `report_html` có thể dùng nếu app có WebView/HTML renderer.

### Get Latest Medical Report For Prediction

```http
GET /api/predictions/{prediction_id}/medical-report
Authorization: Bearer <access_token>
```

Response giống `Generate Medical Report`.

Nếu chưa có report:

```json
{
  "detail": "Medical report not found"
}
```

### Get Medical Report By ID

```http
GET /api/medical-reports/{report_id}
Authorization: Bearer <access_token>
```

## 5. Analytics APIs

### Dataset Analytics

```http
GET /api/analytics/dataset
Authorization: Bearer <access_token>
```

Response:

```json
{
  "total_rows": 8563,
  "columns": ["fname", "image_width", "image_height", "source", "bbox", "target", "tb_type", "image_type", "class_name"],
  "target_distribution": {
    "no_tb": 7400,
    "tb": 1163
  },
  "image_type_distribution": {
    "healthy": 3800,
    "sick_but_no_tb": 3600,
    "active_tb": 924,
    "latent_tb": 239
  },
  "class_distribution": {
    "healthy": 3800,
    "sick_but_no_tb": 3600,
    "active_tb": 924,
    "latent_tb": 239
  },
  "bbox_distribution": {
    "no_bbox": 7400,
    "has_bbox": 1163
  },
  "source_distribution": {
    "train": 6849,
    "val": 1714
  }
}
```

### Training Metrics

```http
GET /api/training-metrics?model_type=detection&limit=300
Authorization: Bearer <access_token>
```

Query:

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `model_type` | string | no | `classification` or `detection` |
| `limit` | int | no | 1 to 1000, default 300 |

### Import Training Metrics

Admin only:

```http
POST /api/training-metrics/import
Authorization: Bearer <access_token>
```

Response:

```json
{
  "inserted": 200
}
```

## 6. Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok",
  "database": {
    "connected": true,
    "error": null
  },
  "models": {
    "loaded": true,
    "classification_classes": ["active_tb", "healthy", "latent_tb", "sick_but_no_tb"],
    "detection_classes": ["active_tb", "latent_tb"]
  },
  "dataset_csv": "D:\\Workspace\\deep-learning-final-project\\tbx11k-simplified\\data.csv"
}
```

`status = degraded` nghĩa là backend vẫn phản hồi nhưng database hoặc model có vấn đề.

## 7. Error Format

FastAPI thường trả lỗi dạng:

```json
{
  "detail": "Invalid email or password"
}
```

Validation error:

```json
{
  "detail": [
    {
      "loc": ["body", "password"],
      "msg": "String should have at least 8 characters",
      "type": "string_too_short"
    }
  ]
}
```

Common HTTP status:

| Status | Meaning |
| --- | --- |
| `400` | File upload không hợp lệ hoặc request sai |
| `401` | Thiếu/sai/expired token |
| `403` | Không có quyền truy cập resource |
| `404` | Không tìm thấy prediction/report |
| `409` | Email đã tồn tại |
| `422` | Validation error |
| `502` | Gemini/report generation failed |
| `503` | Model service/report feature disabled |

## 8. Flutter Integration Notes

Recommended packages:

```yaml
dependencies:
  dio: ^5.0.0
  shared_preferences: ^2.0.0
```

Login request with Dio:

```dart
final dio = Dio(BaseOptions(baseUrl: 'http://192.168.1.5:8000/api'));

final response = await dio.post('/auth/login', data: {
  'email': email,
  'password': password,
});

final token = response.data['access_token'];
```

Authenticated request:

```dart
dio.options.headers['Authorization'] = 'Bearer $token';
final me = await dio.get('/auth/me');
```

Image upload:

```dart
final formData = FormData.fromMap({
  'file': await MultipartFile.fromFile(imagePath, filename: 'xray.jpg'),
});

final response = await dio.post(
  '/predict',
  queryParameters: {'conf_threshold': 0.25},
  data: formData,
);
```

Render base64 image:

```dart
final bytes = base64Decode(response.data['annotated_image_base64']);
Image.memory(bytes);
```

Generate report:

```dart
final report = await dio.post('/predictions/$predictionId/medical-report');
```

Recommended app screens:

- Login/Register
- Upload X-ray and Prediction Result
- Prediction History
- Prediction Detail
- Medical Report Detail
- Dataset/Training Analytics dashboard

## 9. Safety Notes For UI

Always show this warning near diagnosis and report screens:

```text
Kết quả AI chỉ phục vụ mục đích hỗ trợ và học thuật, không thay thế chẩn đoán hoặc điều trị của bác sĩ.
```

For TB-positive classes (`active_tb`, `latent_tb`), UI should guide the patient to seek specialist evaluation and confirmatory testing. Do not present medication dosage or treatment regimen in the mobile app.
