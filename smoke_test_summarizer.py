"""smoke_test_summarizer.py  –  Async PDF→summary pipeline

Key features
============
1.  Download PDFs from S3 (optionally filter by ticker symbol).
2.  Summarise each file with Gemini (`summarize_and_save`).
3.  Write `.txt` summaries locally *and* (optional) re-upload them to S3.
4.  Batch-push the new summaries into a Chroma vector store.
5.  Checkpoint file (`processed.json`) ensures each S3 object is processed once.

Required env-vars
-----------------
AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY  – S3 access
SUMMARIZER_API_KEY                        – Gemini client key
CHROMA_UPDATE_URL                         – REST endpoint (defaults to localhost)

Installation
------------
```bash
pip install aioboto3 tenacity google-genai boto3 python-dotenv rapidfuzz requests
```

CLI usage
---------
```bash
# basic smoke test – first 3 PDFs in bucket
python smoke_test_summarizer.py

# process specific symbols (all PDFs)
python smoke_test_summarizer.py --symbols 138SL,NCBFG --concurrency 5

# limit to 10 PDFs per run and skip uploading summaries back to S3
python smoke_test_summarizer.py --num-files 10 --no-upload-summaries

# store summaries in a separate bucket
python smoke_test_summarizer.py --symbols EDUF --summary-bucket my-summary-bucket
```

Argument overview
-----------------
--bucket            Target bucket for source PDFs (default `jse-renamed-docs`).
--symbols           Comma-separated tickers to restrict the crawl.
--num-files         Max PDFs to process (per symbol or overall if no symbols).
--concurrency       Max concurrent Gemini calls (default 3).
--upload-summaries / --no-upload-summaries
                    Toggle S3 upload of generated `.txt` (default on).
--summary-bucket    Bucket to store summaries (defaults to `--bucket`).

All other behaviour (exponential back-off, checkpointing, vector-DB upload)
requires no additional flags.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pathlib
import re
from typing import List, Optional

import aioboto3
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from google import genai
from google.genai import types  # type: ignore

from dotenv import load_dotenv
load_dotenv()

# ----------------------------------------------------------------------------
# 0.  Prompts (verbatim from notebook) – shorten if you wish
# ----------------------------------------------------------------------------
system_prompt = """
  You are a seasoned financial analyst reviewing a newly submitted financial document.
  Your task is to summarize the submission clearly and concisely for stakeholders such as investors, executives, or credit officers.

  Analyze the document and produce a structured summary using the format below.
  If data is not available, say "Not disclosed." Use clean markdown formatting.

  ---
  ### 📄 1. Document Overview
  - **Document Type**:  
  - **Company Name & Ticker (if public)**:  
  - **Reporting Period**:  
  - **Date of Submission / Release**:  
  - **Auditor (if applicable)**:  
  - **Currency**:  
  ---
  ### 📊 2. Key Financial Metrics
  - **Revenue**:  
  - **Operating Profit / EBIT**:  
  - **Net Income**:  
  - **EPS (Basic & Diluted)**:  
  - **Free Cash Flow**:  
  - **Key Ratios**:
    - Gross Margin  
    - Operating Margin  
    - Net Margin  
    - ROE / ROA  
    - Debt-to-Equity  
    - Current Ratio  
  ---
  ### 🔍 3. Performance Highlights
  - Revenue and margin drivers  
  - Cost trends (COGS, SG&A, R&D)  
  - Operational efficiency comments  
  ---
  ### 🧾 4. Balance Sheet Snapshot
  - **Cash & Equivalents**  
  - **Total Assets**  
  - **Total Liabilities**  
  - **Shareholder Equity**  
  - Notable changes in structure or working capital
  ---
  ### 💵 5. Cash Flow Overview
  - **Operating Activities**:  
  - **Investing Activities**:  
  - **Financing Activities**:  
  - Major capital movements  
  ---
  ### 📈 6. Forward Guidance / Outlook
  - Management guidance (if available)  
  - Risks or opportunities  
  ---
  ### ⚠️ 7. Analyst Notes & Red Flags
  - Auditor or regulatory concerns  
  - Related party or insider issues  
  - Liquidity warnings or covenant risks  
  ---
  ### 🧩 8. Additional Context
  - Strategic initiatives (e.g., M&A, restructuring)  
  - ESG or sustainability disclosures  
  - Industry comparison if relevant
"""

non_findoc_sys_prompt = """
    You are a corporate analyst reviewing a non-financial document submitted by a company.  
    Your job is to extract and summarize the most relevant qualitative insights that provide context for investors, executives, or decision-makers.  
    These documents may include investor presentations, management letters, ESG disclosures, strategic plans, or earnings call transcripts.

    Analyze the document and generate a structured summary using the format below.  
    If any information is not provided, state "Not disclosed." Use clean markdown formatting.

    ---
    ### 📄 0. Document Overview
    - **Document Type**:  
    - **Company Name & Ticker (if public)**:  
    - **Date of Submission / Release**:  
    - **Author / Division**:  
    - **Purpose of Document**:  
    ---
    ### 🏢 1. Company Overview
    - **Company Name**:  
    - **Headquarters / Region**:  
    - **CEO / Key Executives**:  
    - **Board of Directors (if disclosed)**:  
    - **Business Segments / Focus Areas**:  
    ---
    ### 🧭 2. Strategic Themes & Objectives
    - **Primary goals or initiatives discussed**:  
    - **Target markets, segments, or geographies**:  
    - **Key operational or structural changes (e.g., M&A, partnerships)**:  
    ---
    ### 📊 3. Business Drivers & Risks
    - **Growth strategies or opportunity areas**:  
    - **Competitive positioning / market commentary**:  
    - **Risks, headwinds, or concerns raised**:  
    ---
    ### 🌿 4. ESG & Governance (if applicable)
    - **Environmental goals or initiatives**:  
    - **Social impact or workforce developments**:  
    - **Governance or compliance updates**:  
    ---
    ### 🔮 5. Forward Outlook & Implications
    - **Management tone or sentiment**:  
    - **Implications for upcoming financial performance**:  
    - **Signals for strategic or operational shifts**:  
    ---
    ### 📌 6. Analyst Notes & Takeaways
    - Alignment with prior financial results or guidance  
    - Notable changes in strategy, tone, or risk profile  
    - Items to monitor in future disclosures  
    ---
"""

# ----------------------------------------------------------------------------
# 1.  Summariser helpers (copied from notebook, unmodified)
# ----------------------------------------------------------------------------

def summarize_document(document_path: str, system_prompt: str) -> str:
    """Summarise a document using Gemini and return the plain-text summary."""
    client = genai.Client(api_key=os.getenv("SUMMARIZER_API_KEY"))

    # Upload the PDF to Gemini File API
    file = client.files.upload(file=document_path)

    response = client.models.generate_content(
        model="gemini-2.5-pro-preview-03-25",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0,
        ),
        contents=[file],
    )
    return response.text


def write_summary_to_file(summary: str, original_file_path: str) -> str:
    """Write summary next to the PDF (under ./summaries) and return the txt path."""
    base_dir = pathlib.Path(original_file_path).parent
    summaries_dir = base_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    out_path = summaries_dir / (pathlib.Path(original_file_path).stem + ".txt")
    out_path.write_text(summary)
    return str(out_path)


def summarize_and_save(document_path: str, system_prompt: str) -> str:
    summary = summarize_document(document_path, system_prompt)
    return write_summary_to_file(summary, document_path)

# ----------------------------------------------------------------------------
# 2.  Tiny helper to fetch *some* S3 keys – synchronous boto3 is fine here
# ----------------------------------------------------------------------------
import boto3
from botocore.exceptions import ClientError, NoCredentialsError


def get_first_n_pdfs(bucket: str, n: int = 0) -> List[str]:
    """Return at most *n* PDF object URIs from the bucket. If *n* is 0 (or <1) return **all** PDFs."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    page_iter = paginator.paginate(Bucket=bucket, Prefix="organized/")

    found: List[str] = []
    for page in page_iter:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".pdf"):
                found.append(f"s3://{bucket}/{key}")
                if n and n > 0 and len(found) >= n:
                    return found
    return found

# ----------------------------------------------------------------------------
# 3.  Async pipeline
# ----------------------------------------------------------------------------
_S3_RE = re.compile(r"s3://([^/]+)/(.+)")
DEST_DIR = pathlib.Path("pdfs_tmp")
DEST_DIR.mkdir(exist_ok=True)


async def download_pdf(s3_client, s3_uri: str) -> pathlib.Path:
    bucket, key = _S3_RE.match(s3_uri).groups()  # type: ignore
    local_path = DEST_DIR / pathlib.Path(key).name
    await s3_client.download_file(Bucket=bucket, Key=key, Filename=str(local_path))
    return local_path


async def _summarize_with_retry(path: pathlib.Path, prompt: str) -> str:
    loop = asyncio.get_running_loop()
    async for attempt in AsyncRetrying(
        wait=wait_exponential(multiplier=2, min=6, max=90),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    ):
        with attempt:
            return await loop.run_in_executor(None, summarize_and_save, str(path), prompt)


def choose_prompt(filename: str) -> str:
    return system_prompt if "financial" in filename.lower() else non_findoc_sys_prompt


# -------------------------------------------------------------
# Checkpoint file to remember processed S3 URIs
# -------------------------------------------------------------
PROCESSED_PATH = pathlib.Path("processed.json")


def _load_processed() -> set[str]:
    if PROCESSED_PATH.exists():
        try:
            import json

            return set(json.loads(PROCESSED_PATH.read_text()))
        except Exception:
            logging.warning("Could not parse %s – starting with empty set", PROCESSED_PATH)
    return set()


def _save_processed(processed: set[str]):
    import json

    PROCESSED_PATH.write_text(json.dumps(sorted(processed)))


async def process_uri(
    s3_client,
    sem: asyncio.Semaphore,
    uri: str,
    processed: set[str],
    new_paths: list[pathlib.Path],
    completed_uris: list[str],
    *,
    upload_bucket: Optional[str] = None,
    upload_prefix: str = "summaries/",
):
    if uri in processed:
        logging.info("Skipping already-processed file: %s", uri)
        return

    async with sem:
        try:
            logging.info("Downloading %s", uri)
            local_path = await download_pdf(s3_client, uri)
            prompt = choose_prompt(local_path.name)
            logging.info("Summarising %s", local_path.name)
            out_path_str = await _summarize_with_retry(local_path, prompt)
            out_path = pathlib.Path(out_path_str)
            logging.info("✅  Completed → %s", out_path)

            # bookkeeping (vector-store & processed tracking deferred)
            new_paths.append(out_path)

            # optional upload of summary back to S3
            if upload_bucket:
                key = f"{upload_prefix}{out_path.name}"
                await s3_client.upload_file(
                    Filename=str(out_path),
                    Bucket=upload_bucket,
                    Key=key,
                )
                logging.info("☁️  Uploaded summary to s3://%s/%s", upload_bucket, key)

            # record uri as successfully summarised **after** optional S3 upload
            completed_uris.append(uri)
        except Exception as exc:
            logging.exception("❌  Failed on %s: %s", uri, exc)


def _list_pdfs_for_symbols(bucket: str, symbols: list[str]) -> List[str]:
    """Return all PDF object URIs for the given ticker symbols."""
    s3 = boto3.client("s3")
    uris: List[str] = []
    for sym in symbols:
        prefix = f"organized/{sym}/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.lower().endswith(".pdf"):
                    uris.append(f"s3://{bucket}/{key}")
    return uris


async def async_main(bucket: str, num_files: int, max_conc: int, symbols: Optional[List[str]], upload_summaries: bool, summary_bucket: Optional[str]):
    if symbols:
        uris = _list_pdfs_for_symbols(bucket, symbols)
        if num_files:
            uris = uris[:num_files]
    else:
        uris = get_first_n_pdfs(bucket, num_files)
    if not uris:
        logging.error("No PDFs found in bucket %s", bucket)
        return

    processed = _load_processed()

    # filter uris down to those not processed unless user explicitly wants all via num_files param
    unprocessed_uris = [u for u in uris if u not in processed]
    if not unprocessed_uris:
        logging.info("All of the first %d PDFs have already been processed.", len(uris))
        return

    logging.info("Running smoke test on %d PDFs (skipping %d already processed)", len(unprocessed_uris), len(uris) - len(unprocessed_uris))
    sem = asyncio.Semaphore(max_conc)
    session = aioboto3.Session()

    # keep track of new summaries
    new_summary_paths: list[pathlib.Path] = []
    completed_uris: list[str] = []

    async with session.client("s3") as s3:
        tasks = [
            asyncio.create_task(
                process_uri(
                    s3,
                    sem,
                    u,
                    processed,
                    new_summary_paths,
                    completed_uris,
                    upload_bucket=summary_bucket if upload_summaries else None,
                )
            )
            for u in unprocessed_uris
        ]
        await asyncio.gather(*tasks)

    # -------------------------------------------------------------
    # 🚀  After all summaries are written, push them to the vector DB
    # -------------------------------------------------------------
    vector_upload_success = True
    if new_summary_paths:
        try:
            _upload_summaries_to_vectordb(new_summary_paths)
        except Exception as exc:
            vector_upload_success = False
            logging.exception("Vector-DB upload failed: %s", exc)
    else:
        logging.info("No new summaries to upload – skipping vector-store call.")

    # -------------------------------------------------------------
    # 📝  Persist checkpoint only if ALL downstream steps succeeded
    # -------------------------------------------------------------
    if vector_upload_success and completed_uris:
        processed.update(completed_uris)
        _save_processed(processed)


# -------------------------------------------------------------
# 5.  Vector-store upload helper (Chroma REST API)
# -------------------------------------------------------------

import json, requests, re
from rapidfuzz import fuzz


def _upload_summaries_to_vectordb(paths: list[pathlib.Path]):
    """Upload provided summary Paths to the Chroma REST endpoint."""

    # 1) build fuzzy lookup for company names
    with open("companies.json", "r") as f:
        companies = json.load(f)

    lookup = {}
    for c in companies:
        for n in (c["security_name"], c["short_name"], c["ticker_symbol"]):
            lookup[n.lower()] = c["short_name"]

    def _norm_company(q: str):
        for name in lookup:
            if fuzz.partial_ratio(q.lower(), name) > 80:
                return lookup[name]
        return None

    # 2) collect docs + metadata
    ids, docs, metas = [], [], []
    for path in paths:
        ids.append(path.name)
        docs.append(path.read_text())

        fname = path.name
        company_name = _norm_company(fname.split("-")[0])
        year_match = re.search(r"\d{4}", fname)
        year = year_match.group() if year_match and 1900 <= int(year_match.group()) <= 2100 else "Unknown"

        metas.append({
            "company_name": company_name,
            "year": year,
            "file_type": "financial" if "financial" in fname.lower() else "non-financial",
            "filename": fname,
        })

    if not docs:
        logging.info("No summaries to upload – skipping vector upload.")
        return

    payload = {"documents": docs, "metadatas": metas, "ids": ids}
    update_url = os.getenv("CHROMA_UPDATE_URL", "http://localhost:8000/chroma/update")

    logging.info("Uploading %d summaries to %s", len(docs), update_url)
    resp = requests.post(update_url, json=payload, timeout=120)
    resp.raise_for_status()
    logging.info("Vector-store response: %s", resp.json())

# ----------------------------------------------------------------------------
# 4.  CLI entry-point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Asynchronous smoke-test for Gemini summariser.")
    parser.add_argument("--bucket", default="jse-renamed-docs", help="S3 bucket name")
    parser.add_argument("--num-files", type=int, default=0, help="Max PDFs to process (0 = no limit)")
    parser.add_argument("--concurrency", type=int, default=3, help="Max concurrent Gemini calls")
    parser.add_argument("--symbols", type=str, help="Comma-separated list of ticker symbols to process")
    parser.add_argument("--update-url", help="Override CHROMA_UPDATE_URL for vector DB upload")

    # Boolean flag: --upload-summaries / --no-upload-summaries (Python ≥3.9)
    try:
        from argparse import BooleanOptionalAction  # type: ignore

        parser.add_argument(
            "--upload-summaries",
            action=BooleanOptionalAction,
            default=True,
            help="Toggle uploading .txt summaries back to S3 (default: true)",
        )
    except ImportError:
        # Fallback for older Python: use two explicit flags
        parser.add_argument("--upload-summaries", dest="upload_summaries", action="store_true", help="Upload .txt summaries to S3 (default)")
        parser.add_argument("--no-upload-summaries", dest="upload_summaries", action="store_false", help="Do not upload .txt summaries to S3")
        parser.set_defaults(upload_summaries=True)

    parser.add_argument("--summary-bucket", help="Bucket to store summaries (defaults to --bucket)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    symbol_list = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    asyncio.run(
        async_main(
            args.bucket,
            args.num_files,
            args.concurrency,
            symbol_list,
            args.upload_summaries,
            args.summary_bucket or args.bucket,
        )
    )
    if args.update_url:
        os.environ["CHROMA_UPDATE_URL"] = args.update_url 