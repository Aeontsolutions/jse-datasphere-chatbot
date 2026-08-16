"""Stable, non-forgeable identifiers for the S3 documents we cite.

A cited document is exposed to clients as an opaque `document_id`, never as
an S3 path. The id is a truncated SHA-256 of the path and is resolved by a
dict lookup built from metadata.json, so a client cannot craft an id that
reaches an object we did not index. Presigned URLs are minted per request
and never cached, so a cached response can never hand out a dead link.
"""

import hashlib
from typing import Any, Dict, Optional, Tuple

from app.logging_config import get_logger

logger = get_logger(__name__)

DOCUMENT_ID_LENGTH = 16
PRESIGNED_URL_TTL_SECONDS = 300


def make_document_id(s3_path: str) -> str:
    """Derive a stable, one-way id for an S3 path."""
    return hashlib.sha256(s3_path.encode("utf-8")).hexdigest()[:DOCUMENT_ID_LENGTH]


def parse_s3_path(s3_path: str) -> Tuple[str, str]:
    """Split an ``s3://bucket/key`` path into ``(bucket, key)``."""
    if not s3_path or not s3_path.startswith("s3://"):
        raise ValueError(f"Not an s3:// path: {s3_path!r}")
    remainder = s3_path[len("s3://") :]
    bucket, _, key = remainder.partition("/")
    if not bucket or not key:
        raise ValueError(f"S3 path is missing a bucket or key: {s3_path!r}")
    return bucket, key


def build_document_index(metadata: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Map document_id -> resolution info for every document in S3 metadata.

    Metadata is shaped ``{company_name: [document, ...]}``. It is loaded from
    S3 at startup, so a malformed shape is tolerated rather than fatal.
    """
    index: Dict[str, Dict[str, Any]] = {}
    if not metadata or not isinstance(metadata, dict):
        return index

    for company, documents in metadata.items():
        if not isinstance(documents, list):
            continue
        for doc in documents:
            if not isinstance(doc, dict):
                continue
            s3_path = doc.get("document_link")
            if not s3_path:
                continue
            index[make_document_id(s3_path)] = {
                "s3_path": s3_path,
                "filename": doc.get("filename") or s3_path.rsplit("/", 1)[-1],
                "company": company,
                "year": doc.get("year") or doc.get("period"),
            }

    logger.info(f"Document index built: {len(index)} documents")
    return index


def presign_document(
    s3_client: Any, s3_path: str, expires_in: int = PRESIGNED_URL_TTL_SECONDS
) -> str:
    """Mint a short-lived presigned GET URL for a document."""
    bucket, key = parse_s3_path(s3_path)
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_in,
    )
