"""Wrapper that loads fastapi_app/.env and invokes evals.cli.

This avoids needing to leak env-var values onto the command line or
into shell history. The chatbot's `.env` only defines `CHATBOT_API_KEY`,
so we alias it to `GOOGLE_API_KEY` (which the eval suite expects).
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

from evals.cli import main  # noqa: E402

sys.exit(main())
