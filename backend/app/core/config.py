from functools import lru_cache
import os
from dataclasses import dataclass
from pathlib import Path


try:
    from dotenv import load_dotenv

    load_dotenv()
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
except Exception:
    pass


@dataclass
class Settings:
    app_name: str = "TB AI Diagnosis API"
    api_prefix: str = "/api"
    cors_origins: str = "*"

    database_url: str = "postgresql+psycopg2://tb_user:tb_password@localhost:5432/tb_ai_db"
    jwt_secret_key: str = "change-this-secret-key"
    jwt_expire_minutes: int = 60 * 8

    dataset_csv_path: str = "tbx11k-simplified/data.csv"
    dataset_images_dir: str = "tbx11k-simplified/images"
    dataset_test_dir: str = "tbx11k-simplified/test"
    classification_model_path: str = "tb_classification/stage1_cls/weights/best.pt"
    detection_model_path: str = "tb_detection/stage2_det/weights/best.pt"
    non_tb_drop_threshold: float = 0.80
    roi_padding_ratio: float = 0.20
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"
    enable_medical_reports: bool = True
    uploads_dir: str = "backend/uploads"

    @property
    def repo_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.repo_root / path

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("APP_NAME", Settings.app_name),
        api_prefix=os.getenv("API_PREFIX", Settings.api_prefix),
        cors_origins=os.getenv("CORS_ORIGINS", Settings.cors_origins),
        database_url=os.getenv("DATABASE_URL", Settings.database_url),
        jwt_secret_key=os.getenv("JWT_SECRET_KEY", Settings.jwt_secret_key),
        jwt_expire_minutes=int(os.getenv("JWT_EXPIRE_MINUTES", str(Settings.jwt_expire_minutes))),
        dataset_csv_path=os.getenv("DATASET_CSV_PATH", Settings.dataset_csv_path),
        dataset_images_dir=os.getenv("DATASET_IMAGES_DIR", Settings.dataset_images_dir),
        dataset_test_dir=os.getenv("DATASET_TEST_DIR", Settings.dataset_test_dir),
        classification_model_path=os.getenv("CLASSIFICATION_MODEL_PATH", Settings.classification_model_path),
        detection_model_path=os.getenv("DETECTION_MODEL_PATH", Settings.detection_model_path),
        non_tb_drop_threshold=float(os.getenv("NON_TB_DROP_THRESHOLD", str(Settings.non_tb_drop_threshold))),
        roi_padding_ratio=float(os.getenv("ROI_PADDING_RATIO", str(Settings.roi_padding_ratio))),
        gemini_api_key=os.getenv("GEMINI_API_KEY", Settings.gemini_api_key),
        gemini_model=os.getenv("GEMINI_MODEL", Settings.gemini_model),
        enable_medical_reports=os.getenv("ENABLE_MEDICAL_REPORTS", "true").lower() in {"1", "true", "yes", "on"},
        uploads_dir=os.getenv("UPLOADS_DIR", Settings.uploads_dir),
    )
