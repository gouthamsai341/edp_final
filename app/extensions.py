"""
app/extensions.py

Holds shared extension instances (SQLAlchemy) so they can be imported
by both app.py (for init_app) and models_db.py / services (for use)
without circular imports.
"""

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
