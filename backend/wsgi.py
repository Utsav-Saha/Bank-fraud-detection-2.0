"""WSGI entrypoint for production deployments (e.g., Gunicorn)."""

import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import create_app

env_name = os.getenv("FLASK_ENV", "production")
app = create_app(env_name)

if __name__ == "__main__":
    app.run()
