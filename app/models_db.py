"""
app/models_db.py

SQLAlchemy model for the `predictions` table (prediction history).
"""

from __future__ import annotations

from datetime import datetime

from app.extensions import db


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    image_filename = db.Column(db.String(255), nullable=False)
    plant = db.Column(db.String(100), nullable=True)
    disease = db.Column(db.String(150), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    status = db.Column(db.String(20), nullable=False)  # success | uncertain | error
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "image_filename": self.image_filename,
            "plant": self.plant,
            "disease": self.disease,
            "confidence": self.confidence,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
        }
