"""Wrapper that loads fastapi_app/.env and invokes evals.cli.

This avoids needing to leak env-var values onto the command line or
into shell history.

The eval suite reads the Gemini key from `GOOGLE_API_KEY` (see
`gemini_api_key_env` in evals/config/default.yaml). A chatbot `.env` may
name that key any of several ways, so we resolve it using the same alias
chain the app itself accepts (`fastapi_app/app/config.py`,
`financial_utils.py`, `dspy_modules.py`) and export the winner as
`GOOGLE_API_KEY`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / "fastapi_app" / ".env"

# Keep in sync with AliasChoices in fastapi_app/app/config.py.
API_KEY_ALIASES = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "CHATBOT_API_KEY")

if not ENV_PATH.exists():
    print(f"ERROR: missing {ENV_PATH}", file=sys.stderr)
    sys.exit(2)

load_dotenv(ENV_PATH)

if not os.environ.get("GOOGLE_API_KEY"):
    for alias in API_KEY_ALIASES:
        if os.environ.get(alias):
            os.environ["GOOGLE_API_KEY"] = os.environ[alias]
            break
    else:
        print(
            "ERROR: no Gemini API key found. Set one of "
            f"{', '.join(API_KEY_ALIASES)} in {ENV_PATH}.",
            file=sys.stderr,
        )
        sys.exit(2)

from evals.cli import main  # noqa: E402

sys.exit(main())
