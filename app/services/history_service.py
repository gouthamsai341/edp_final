"""
app/services/history_service.py

Reads and writes prediction history records via SQLAlchemy.
"""

from __future__ import annotations

from app.extensions import db
from app.models_db import Prediction


def save_prediction_record(image_filename: str, plant, disease, confidence, status: str) -> Prediction:
    record = Prediction(
        image_filename=image_filename,
        plant=plant,
        disease=disease,
        confidence=confidence,
        status=status,
    )
    db.session.add(record)
    db.session.commit()
    return record


def get_all_predictions(limit: int = 200) -> list[Prediction]:
    return (
        Prediction.query.order_by(Prediction.created_at.desc())
        .limit(limit)
        .all()
    )


def delete_prediction(prediction_id: int) -> bool:
    record = Prediction.query.get(prediction_id)
    if record is None:
        return False
    db.session.delete(record)
    db.session.commit()
    return True


def clear_all_predictions() -> int:
    count = Prediction.query.delete()
    db.session.commit()
    return count
