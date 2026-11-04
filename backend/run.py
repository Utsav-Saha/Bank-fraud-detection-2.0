"""Local development runner for SecureBank AI Backend."""

import os
import sys
from pathlib import Path

# Ensure backend directory is in sys.path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import create_app

if __name__ == "__main__":
    env_name = os.getenv("FLASK_ENV", "development")
    app = create_app(env_name)
    host = os.getenv("FLASK_RUN_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_RUN_PORT", "5000"))
    debug = env_name == "development"
    print(f" * Starting SecureBank AI Fraud Detection API on http://{host}:{port} (Environment: {env_name})")
    app.run(host=host, port=port, debug=debug)
