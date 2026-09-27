"""
run.py

Entry point for running PlantCare AI locally:

    python run.py

Reads host/port/debug from config.yaml (app/config.py -> AppConfig).
Debug mode is disabled by default (see Security requirements) - set
flask.debug: true in config.yaml only for local development.
"""

from app.app import create_app
from app.config import AppConfig

app = create_app()

if __name__ == "__main__":
    app.run(host=AppConfig.HOST, port=AppConfig.PORT, debug=AppConfig.DEBUG)
