# TB AI FastAPI Backend

## Run PostgreSQL

```powershell
docker compose up -d postgres
```

## Install backend dependencies

```powershell
pip install -r requirements.txt
```

## Start API

```powershell
uvicorn backend.app.main:app --reload
```

API base URL: `http://localhost:8000/api`

Useful endpoints:

- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `POST /api/predict`
- `GET /api/predictions`
- `GET /api/analytics/dataset`
- `POST /api/training-metrics/import` (admin only)
- `GET /api/training-metrics`
- `POST /api/predictions/{prediction_id}/medical-report`
- `GET /api/predictions/{prediction_id}/medical-report`
- `GET /api/medical-reports/{report_id}`

Passwords are hashed with PBKDF2-HMAC-SHA256 and per-user salts. Plain SHA, SHA1, and MD5 are not used for password storage.

The first registered account is created as `admin`; later accounts default to `student`.

Medical reports use Gemini API when `GEMINI_API_KEY` is configured in `.env`. The report prompt forbids medication dosages, prescriptions, and replacing physician review.
