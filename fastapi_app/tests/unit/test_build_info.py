"""Tests for build provenance reading.

CI writes the deploying commit to app/BUILD_SHA before the image is built.
Locally the file is absent, which must degrade to "unknown" rather than
raising -- every local run and every unit test hits that path.
"""

from app.build_info import UNKNOWN, read_build_sha


def test_reads_commit_from_file(tmp_path):
    sha_file = tmp_path / "BUILD_SHA"
    sha_file.write_text("abc123def456", encoding="utf-8")
    assert read_build_sha(sha_file) == "abc123def456"


def test_strips_trailing_newline(tmp_path):
    # `echo "$GITHUB_SHA" > BUILD_SHA` always leaves a trailing newline.
    sha_file = tmp_path / "BUILD_SHA"
    sha_file.write_text("abc123def456\n", encoding="utf-8")
    assert read_build_sha(sha_file) == "abc123def456"


def test_missing_file_is_unknown(tmp_path):
    assert read_build_sha(tmp_path / "does_not_exist") == UNKNOWN


def test_empty_file_is_unknown(tmp_path):
    sha_file = tmp_path / "BUILD_SHA"
    sha_file.write_text("   \n", encoding="utf-8")
    assert read_build_sha(sha_file) == UNKNOWN
