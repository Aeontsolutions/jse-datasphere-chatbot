"""Wrapper that loads fastapi_app/.env, aliases CHATBOT_API_KEY → GOOGLE_API_KEY,
and starts uvicorn on :8000.

The chatbot's centralized config expects GOOGLE_API_KEY but the project's .env
only defines CHATBOT_API_KEY. Other code paths (financial_utils.py) already
fall back across these names; this script lets the centralized config boot
without modifying the .env file or leaking values through the shell.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / "fastapi_app" / ".env"

if not ENV_PATH.exists():
    print(f"ERROR: missing {ENV_PATH}", file=sys.stderr)
    sys.exit(2)

load_dotenv(ENV_PATH)

if "CHATBOT_API_KEY" in os.environ and not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["CHATBOT_API_KEY"]

# Ensure the fastapi_app dir is on sys.path so `from app.main import app` works
sys.path.insert(0, str(REPO_ROOT / "fastapi_app"))

import uvicorn  # noqa: E402

uvicorn.run(
    "app.main:app",
    host="127.0.0.1",
    port=8000,
    log_level="info",
)
