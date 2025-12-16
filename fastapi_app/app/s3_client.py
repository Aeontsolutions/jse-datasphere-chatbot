"""
S3 client initialization and document download operations.

This module provides both synchronous and asynchronous S3 client operations
for downloading and extracting text from PDF documents stored in S3.
"""

import logging
import tempfile
import asyncio
import time
import os
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import boto3
import aioboto3
from botocore.exceptions import ClientError
from fastapi import HTTPException

from app.config import get_config, S3DownloadConfig
from app.pdf_utils import extract_text_from_pdf, extract_text_from_pdf_bytes

logger = logging.getLogger(__name__)


# =============================================================================
# SYNCHRONOUS S3 CLIENT
# =============================================================================


def init_s3_client():
    """Initialize and return an S3 client using environment variables"""
    config = get_config()

    aws_access_key_id = config.aws.access_key_id
    aws_secret_access_key = config.aws.secret_access_key
    aws_region = config.aws.region

    if not all([aws_access_key_id, aws_secret_access_key, aws_region]):
        logger.error("AWS credentials not found in environment variables")
        raise ValueError("AWS credentials not found in environment variables")

    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region,
        )
        logger.info("Successfully connected to AWS S3")
        return s3_client
    except Exception as e:
        logger.error(f"Error connecting to AWS S3: {str(e)}")
        raise


def download_and_extract_from_s3(s3_client, s3_path):
    """Download a PDF from S3 and extract its text"""
    bucket_name = None
    key = None

    try:
        # Parse S3 path to get bucket and key
        # s3://jse-renamed-docs/organized/... format
        if not s3_path.startswith("s3://"):
            logger.error(
                "s3_invalid_path_format",
                extra={"s3_path": s3_path, "error": "Path does not start with s3://"},
            )
            raise HTTPException(status_code=400, detail="Invalid S3 path format")

        path_without_prefix = s3_path[5:]  # Remove "s3://"
        bucket_name = path_without_prefix.split("/")[0]
        key = "/".join(path_without_prefix.split("/")[1:])

        # Log the attempt
        logger.info(
            "s3_download_start", extra={"bucket": bucket_name, "key": key, "s3_path": s3_path}
        )

        # Create a temporary file to store the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            # Download the file from S3
            s3_client.download_fileobj(bucket_name, key, tmp_file)
            tmp_file_path = tmp_file.name

        # Extract text from the downloaded PDF
        with open(tmp_file_path, "rb") as pdf_file:
            text = extract_text_from_pdf(pdf_file)

        # Clean up the temporary file
        os.unlink(tmp_file_path)

        logger.info(
            "s3_download_success",
            extra={"bucket": bucket_name, "key": key, "text_length": len(text) if text else 0},
        )

        return text

    except HTTPException:
        raise
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        logger.error(
            "s3_download_failed",
            extra={
                "s3_path": s3_path,
                "bucket": bucket_name,
                "key": key,
                "error": str(e),
                "error_code": error_code,
            },
        )

        # Check for NoSuchKey error
        if error_code == "NoSuchKey":
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {os.path.basename(key) if key else 'unknown'}",
            )

        raise HTTPException(
            status_code=503,
            detail=f"Failed to download document: {os.path.basename(key) if key else 'unknown'}",
        )
    except Exception as e:
        logger.error(
            "s3_unexpected_error",
            extra={
                "s3_path": s3_path,
                "bucket": bucket_name,
                "key": key,
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred while processing your request"
        )


# =============================================================================
# ASYNC S3 CLIENT
# =============================================================================


class DownloadResult:
    """Result of a download operation"""

    def __init__(
        self,
        success: bool,
        content: Optional[str] = None,
        error: Optional[str] = None,
        file_path: Optional[str] = None,
        download_time: float = 0.0,
        retry_count: int = 0,
    ):
        self.success = success
        self.content = content
        self.error = error
        self.file_path = file_path
        self.download_time = download_time
        self.retry_count = retry_count


async def init_async_s3_client():
    """Initialize and return an async S3 client using environment variables"""
    config = get_config()

    aws_access_key_id = config.aws.access_key_id
    aws_secret_access_key = config.aws.secret_access_key
    aws_region = config.aws.region

    if not all([aws_access_key_id, aws_secret_access_key, aws_region]):
        logger.error("AWS credentials not found in environment variables")
        raise ValueError("AWS credentials not found in environment variables")

    try:
        session = aioboto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region,
        )

        logger.info("Successfully created async AWS S3 session")
        return session
    except Exception as e:
        logger.error(f"Error creating async AWS S3 session: {str(e)}")
        raise


async def download_and_extract_from_s3_async(
    s3_path: str,
    config: Optional[S3DownloadConfig] = None,
    progress_callback: Optional[callable] = None,
) -> DownloadResult:
    """
    Asynchronously download a PDF from S3 and extract its text with robust error handling.

    Args:
        s3_path: S3 path in format s3://bucket/key
        config: Download configuration options
        progress_callback: Optional callback for progress updates

    Returns:
        DownloadResult with success status, content, and metadata
    """
    if config is None:
        config = get_config().s3_download

    start_time = time.time()
    retry_count = 0

    try:
        # Parse S3 path
        if not s3_path.startswith("s3://"):
            error_msg = f"Invalid S3 path format: {s3_path}"
            logger.error(
                "s3_async_invalid_path",
                extra={"s3_path": s3_path, "error": "Path does not start with s3://"},
            )
            return DownloadResult(success=False, error=error_msg)

        path_without_prefix = s3_path[5:]  # Remove "s3://"
        bucket_name = path_without_prefix.split("/")[0]
        key = "/".join(path_without_prefix.split("/")[1:])

        logger.info(f"Starting async download: Bucket='{bucket_name}', Key='{key}'")

        if progress_callback:
            await progress_callback(
                "download_start", f"Starting download of {os.path.basename(key)}"
            )

        # Retry loop with exponential backoff
        for attempt in range(config.max_retries):
            try:
                retry_count = attempt
                if attempt > 0:
                    delay = min(config.retry_delay * (2 ** (attempt - 1)), config.max_retry_delay)
                    logger.info(
                        f"Retrying download after {delay:.1f}s (attempt {attempt + 1}/{config.max_retries})"
                    )
                    await asyncio.sleep(delay)

                # Initialize async S3 session
                session = await init_async_s3_client()

                async with session.client("s3") as s3_client:
                    # Download file with timeout
                    download_task = asyncio.create_task(
                        _download_s3_object_async(s3_client, bucket_name, key, config)
                    )

                    try:
                        file_content = await asyncio.wait_for(download_task, timeout=config.timeout)
                    except asyncio.TimeoutError:
                        raise Exception(f"Download timeout after {config.timeout}s")

                    if progress_callback:
                        await progress_callback(
                            "download_complete", f"Downloaded {len(file_content)} bytes"
                        )

                    # Extract text from PDF in a thread pool to avoid blocking
                    if progress_callback:
                        await progress_callback("text_extraction", "Extracting text from PDF")

                    with ThreadPoolExecutor(max_workers=1) as executor:
                        loop = asyncio.get_event_loop()
                        text = await loop.run_in_executor(
                            executor, extract_text_from_pdf_bytes, file_content
                        )

                    if text is None:
                        raise Exception("Failed to extract text from PDF")

                    download_time = time.time() - start_time

                    if progress_callback:
                        await progress_callback(
                            "extraction_complete", f"Extracted {len(text)} characters"
                        )

                    logger.info(
                        f"Successfully downloaded and extracted: {s3_path} ({download_time:.2f}s, {retry_count} retries)"
                    )

                    return DownloadResult(
                        success=True,
                        content=text,
                        download_time=download_time,
                        retry_count=retry_count,
                    )

            except Exception as e:
                error_msg = f"Download attempt {attempt + 1} failed: {str(e)}"
                logger.warning(
                    "s3_async_download_attempt_failed",
                    extra={
                        "s3_path": s3_path,
                        "bucket": bucket_name,
                        "key": key,
                        "attempt": attempt + 1,
                        "max_retries": config.max_retries,
                        "error": str(e),
                    },
                )

                # Fail fast for NoSuchKey errors - retrying won't help
                if "NoSuchKey" in str(e) or "The specified key does not exist" in str(e):
                    download_time = time.time() - start_time
                    final_error = f"File does not exist in S3: {s3_path} - {str(e)}"
                    logger.error(
                        "s3_async_file_not_found",
                        extra={
                            "s3_path": s3_path,
                            "bucket": bucket_name,
                            "key": key,
                            "error": str(e),
                            "download_time": download_time,
                        },
                    )

                    if progress_callback:
                        await progress_callback("download_failed", final_error)

                    return DownloadResult(
                        success=False,
                        error=final_error,
                        download_time=download_time,
                        retry_count=retry_count,
                    )

                if attempt == config.max_retries - 1:
                    # Final attempt failed
                    download_time = time.time() - start_time
                    final_error = f"Failed to download {s3_path} after {config.max_retries} attempts: {str(e)}"
                    logger.error(
                        "s3_async_download_failed_all_retries",
                        extra={
                            "s3_path": s3_path,
                            "bucket": bucket_name,
                            "key": key,
                            "error": str(e),
                            "max_retries": config.max_retries,
                            "download_time": download_time,
                        },
                    )

                    if progress_callback:
                        await progress_callback("download_failed", final_error)

                    return DownloadResult(
                        success=False,
                        error=final_error,
                        download_time=download_time,
                        retry_count=retry_count,
                    )
                # Continue to next retry

    except Exception as e:
        download_time = time.time() - start_time
        error_msg = f"Unexpected error downloading {s3_path}: {str(e)}"
        logger.error(
            "s3_async_unexpected_error",
            extra={
                "s3_path": s3_path,
                "error": str(e),
                "error_type": type(e).__name__,
                "download_time": download_time,
            },
        )

        if progress_callback:
            await progress_callback("download_error", error_msg)

        return DownloadResult(
            success=False, error=error_msg, download_time=download_time, retry_count=retry_count
        )


async def _download_s3_object_async(
    s3_client, bucket_name: str, key: str, config: S3DownloadConfig
) -> bytes:
    """Download S3 object and return its content as bytes"""
    try:
        # Use get_object to stream the content
        response = await s3_client.get_object(Bucket=bucket_name, Key=key)

        # Read the content in chunks to handle large files
        content = b""
        chunk_count = 0

        async for chunk in response["Body"]:
            content += chunk
            chunk_count += 1

            # Optional: Add progress tracking for large files
            if chunk_count % 100 == 0:
                logger.debug(f"Downloaded {len(content)} bytes so far...")

        return content

    except Exception as e:
        logger.error(
            "s3_async_object_download_failed",
            extra={
                "bucket": bucket_name,
                "key": key,
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        raise
