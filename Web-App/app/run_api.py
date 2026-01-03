"""Convenience script to start the FastAPI inference service."""
# http://localhost:8000/docs
import os
from pathlib import Path

from uvicorn import run


if __name__ == "__main__":
    app_dir = Path(__file__).resolve().parent
    os.chdir(app_dir)
    run("api_service:app", host="0.0.0.0", port=8000, reload=False)
