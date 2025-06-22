#!/usr/bin/env python3
"""upload_summaries_to_chroma.py – Bulk-upload summary text files from S3 into a Chroma vector store.

This script scans a given S3 prefix (defaults to ``summaries/`` in the
``jse-renamed-docs`` bucket), downloads every ``.txt`` file, infers useful
metadata from the file name, and pushes the documents into a Chroma instance
via its REST ``/chroma/update`` endpoint.

Metadata attached to each document
----------------------------------
company_name   – Normalised company short name (fuzzy-matched via companies.json)
year           – First 4-digit year (1900-2100) appearing in the filename or
                 "Unknown" if none found.
file_type      – "financial" if the word *financial* appears in the filename,
                 otherwise "non-financial".
filename       – Original file name.

The REST payload sent to Chroma looks like:
```
{
  "documents": ["..."],
  "metadatas": [{...}],
  "ids": ["..."]
}
```

Usage
-----
```bash
python upload_summaries_to_chroma.py \
    --bucket jse-renamed-docs \
    --prefix summaries/ \
    --batch-size 50 \
    --update-url http://localhost:8000/chroma/update
```

Environment variables
---------------------
AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY  – S3 credentials (usual AWS SDK env).
CHROMA_UPDATE_URL                         – Default REST endpoint if the
                                            ``--update-url`` flag is omitted.
```
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import List
import time

import boto3
import requests
from rapidfuzz import fuzz
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Helpers for company name normalisation (using companies.json at repo root)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent
_COMPANIES_PATH = _REPO_ROOT / "companies.json"
if not _COMPANIES_PATH.exists():
    logging.error("Required companies.json not found at %s", _COMPANIES_PATH)
    sys.exit(1)

with _COMPANIES_PATH.open() as f:
    _COMPANIES: List[dict] = __import__("json").load(f)

# Build fuzzy lookup: every alias → canonical short_name
_LOOKUP: dict[str, str] = {}
for c in _COMPANIES:
    for name in (c.get("security_name"), c.get("short_name"), c.get("ticker_symbol")):
        if name:
            _LOOKUP[name.lower()] = c["short_name"]

def _norm_company(term: str) -> str | None:
    """Return canonical company short name if *term* fuzzy-matches an alias."""
    term_l = term.lower()
    for alias, short in _LOOKUP.items():
        if fuzz.partial_ratio(term_l, alias) > 80:
            return short
    return None

# ---------------------------------------------------------------------------
# Core S3 → Chroma loader
# ---------------------------------------------------------------------------

def _collect_summaries(bucket: str, prefix: str) -> tuple[list[str], list[str], list[dict]]:
    """Download all `.txt` files under *prefix* in *bucket*.

    Returns (ids, documents, metadatas).
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.lower().endswith(".txt"):
                continue

            obj_resp = s3.get_object(Bucket=bucket, Key=key)
            body = obj_resp["Body"].read().decode("utf-8", errors="replace")
            filename = Path(key).name

            # --- metadata inference ---
            company_guess = _norm_company(filename.split("-")[0])
            # guarantee every metadata value is a plain string (Chroma rejects null)
            company_name = company_guess if company_guess is not None else "Unknown"
            year_match = re.search(r"\d{4}", filename)
            if year_match and 1900 <= int(year_match.group()) <= 2100:
                year = year_match.group()
            else:
                year = "Unknown"

            metas.append({
                "company_name": company_name,
                "year": year,
                "file_type": "financial" if "financial" in filename.lower() else "non-financial",
                "filename": filename,
            })

            ids.append(filename)
            docs.append(body)

    return ids, docs, metas


def _post_to_chroma(update_url: str, ids: list[str], docs: list[str], metas: list[dict], batch_size: int = 100, retries: int = 5):
    """Send documents to Chroma in *batch_size* chunks."""
    if not docs:
        logging.warning("No documents found – nothing to upload.")
        return

    logging.info("Uploading %d documents to %s", len(docs), update_url)
    for start in range(0, len(docs), batch_size):
        end = start + batch_size
        payload = {
            "ids": ids[start:end],
            "documents": docs[start:end],
            "metadatas": metas[start:end],
        }
        for attempt in range(1, retries + 1):
            try:
                resp = requests.post(update_url, json=payload, timeout=120)
                resp.raise_for_status()
                logging.info("Batch %d–%d OK → %s", start + 1, min(end, len(docs)), resp.json())
                break
            except Exception as exc:
                logging.exception("Failed to upload batch %d–%d (attempt %d/%d): %s", start + 1, end, attempt, retries, exc)
                if attempt < retries:
                    wait = 2 ** attempt
                    logging.info("Retrying in %d seconds...", wait)
                    time.sleep(wait)
                else:
                    raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Upload summary .txt files from S3 to Chroma.")
    parser.add_argument("--bucket", default="jse-renamed-docs", help="S3 bucket name")
    parser.add_argument("--prefix", default="summaries/", help="S3 prefix containing .txt summaries")
    parser.add_argument("--batch-size", type=int, default=100, help="Docs per POST request")
    parser.add_argument("--resume-from", type=int, default=0, help="Skip first N documents (useful to resume after a failed batch)")
    parser.add_argument("--retries", type=int, default=5, help="Max retries per batch on 5xx/connection errors")
    parser.add_argument("--update-url", help="Override CHROMA_UPDATE_URL env var")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    update_url = args.update_url or os.getenv("CHROMA_UPDATE_URL", "http://localhost:8000/chroma/update")

    ids, docs, metas = _collect_summaries(args.bucket, args.prefix)

    if args.resume_from:
        logging.info("Resuming upload: skipping first %d documents", args.resume_from)
        ids = ids[args.resume_from:]
        docs = docs[args.resume_from:]
        metas = metas[args.resume_from:]

    _post_to_chroma(update_url, ids, docs, metas, batch_size=args.batch_size, retries=args.retries)


if __name__ == "__main__":
    main() 