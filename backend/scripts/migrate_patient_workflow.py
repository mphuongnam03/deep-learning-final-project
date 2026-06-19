import sys
from pathlib import Path

from sqlalchemy import inspect, text

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.app.db.models import Base
from backend.app.db.session import engine


def main() -> None:
    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)
    prediction_columns = {column["name"] for column in inspector.get_columns("predictions")}

    statements = []
    if "patient_id" not in prediction_columns:
        statements.append("ALTER TABLE predictions ADD COLUMN patient_id INTEGER REFERENCES patients(id)")
    if "xray_study_id" not in prediction_columns:
        statements.append("ALTER TABLE predictions ADD COLUMN xray_study_id INTEGER REFERENCES xray_studies(id)")

    statements.extend(
        [
            "CREATE INDEX IF NOT EXISTS ix_predictions_patient_id ON predictions(patient_id)",
            "CREATE INDEX IF NOT EXISTS ix_predictions_xray_study_id ON predictions(xray_study_id)",
        ]
    )

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))

    print("Patient workflow migration completed")


if __name__ == "__main__":
    main()
