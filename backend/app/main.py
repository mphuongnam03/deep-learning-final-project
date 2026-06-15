from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from backend.app.core.config import get_settings
from backend.app.db.session import SessionLocal, init_db
from backend.app.routers import analytics, auth, predictions
from backend.app.services.prediction_service import ModelService


settings = get_settings()
model_service = ModelService(settings)

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictions.configure_model_service(model_service)
app.include_router(auth.router, prefix=settings.api_prefix)
app.include_router(predictions.router, prefix=settings.api_prefix)
app.include_router(analytics.router, prefix=settings.api_prefix)


@app.on_event("startup")
def startup() -> None:
    try:
        init_db()
    except Exception as exc:
        print(f"Database initialization skipped: {exc}")


@app.get("/health")
def health():
    db_ok = False
    db_error = None
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception as exc:
        db_error = str(exc)
    finally:
        try:
            db.close()
        except Exception:
            pass
    return {
        "status": "ok" if db_ok else "degraded",
        "database": {"connected": db_ok, "error": db_error},
        "models": model_service.status(),
        "dataset_csv": str(settings.resolve_path(settings.dataset_csv_path)),
    }
