"""Commit provenance for the running build.

`.dockerignore` excludes `.git/`, so a container cannot derive its own commit.
CI writes the deploying commit to `app/BUILD_SHA` immediately before
`copilot svc deploy`, and the Dockerfile's `COPY . .` bakes it into the image.
Locally the file is absent and the commit reads "unknown".

The release pipeline reads this back over `GET /version` to prove dev is
serving the commit a release tag points at.
"""

from __future__ import annotations

from pathlib import Path

BUILD_SHA_PATH = Path(__file__).resolve().parent / "BUILD_SHA"
UNKNOWN = "unknown"


def read_build_sha(path: Path | None = None) -> str:
    """Return the commit this build was made from, or "unknown" if unstamped."""
    target = BUILD_SHA_PATH if path is None else path
    try:
        sha = target.read_text(encoding="utf-8").strip()
    except OSError:
        return UNKNOWN
    return sha or UNKNOWN
