"""
Metadata loading and parsing operations.

This module provides functions for downloading and parsing metadata files
from S3, supporting both synchronous and asynchronous operations.
"""

import json
import time
import asyncio
from typing import Dict, Optional

from botocore.exceptions import ClientError
from fastapi import HTTPException

from app.config import get_config, S3DownloadConfig
from app.s3_client import init_async_s3_client, _download_s3_object_async, DownloadResult
from app.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# SYNCHRONOUS METADATA OPERATIONS
# =============================================================================


def download_metadata_from_s3(s3_client, bucket_name, key="metadata.json"):
    """Download metadata JSON file from S3"""
    try:
        # Download the metadata file from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        metadata_content = response["Body"].read().decode("utf-8")

        logger.info(
            "metadata_download_success",
            bucket=bucket_name,
            key=key,
            size=len(metadata_content),
        )

        return metadata_content
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        logger.error(
            "metadata_download_failed",
            bucket=bucket_name,
            key=key,
            error=str(e),
            error_code=error_code,
        )

        if error_code == "NoSuchKey":
            raise HTTPException(status_code=404, detail="Metadata file not found")

        raise HTTPException(status_code=503, detail="Failed to download metadata from storage")
    except Exception as e:
        logger.error(
            "metadata_unexpected_error",
            bucket=bucket_name,
            key=key,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred while loading metadata"
        )


def parse_metadata_file(metadata_content):
    """Parse metadata JSON content"""
    try:
        # Parse the metadata JSON
        metadata = json.loads(metadata_content)

        logger.info(
            "metadata_parse_success",
            extra={"record_count": len(metadata) if isinstance(metadata, (list, dict)) else 0},
        )

        return metadata
    except json.JSONDecodeError as e:
        logger.error(
            "metadata_parse_failed",
            extra={
                "error": str(e),
                "content_length": len(metadata_content) if metadata_content else 0,
            },
        )
        raise HTTPException(status_code=500, detail="Failed to parse metadata file")


def load_metadata_from_s3(s3_client):
    """Load metadata from S3 bucket specified in environment variables"""
    try:
        config = get_config()
        bucket_name = config.aws.s3_bucket

        if not bucket_name:
            logger.error(
                "metadata_bucket_not_configured",
                extra={"error": "DOCUMENT_METADATA_S3_BUCKET not found in environment variables"},
            )
            raise HTTPException(status_code=503, detail="Metadata storage not configured")

        # Get metadata key from config
        metadata_key = config.metadata_key

        # Download metadata from S3
        metadata_content = download_metadata_from_s3(s3_client, bucket_name, metadata_key)

        # Parse the downloaded metadata
        return parse_metadata_file(metadata_content)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "metadata_load_unexpected_error",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        raise HTTPException(status_code=500, detail="Failed to load metadata")


# =============================================================================
# ASYNC METADATA OPERATIONS
# =============================================================================


async def download_metadata_from_s3_async(
    bucket_name: str,
    key: str = "metadata.json",
    config: Optional[S3DownloadConfig] = None,
    progress_callback: Optional[callable] = None,
) -> DownloadResult:
    """
    Asynchronously download metadata JSON file from S3 with robust error handling.

    Args:
        bucket_name: S3 bucket name
        key: S3 object key
        config: Download configuration
        progress_callback: Optional callback for progress updates

    Returns:
        DownloadResult with success status and metadata content
    """
    if config is None:
        config = get_config().s3_download

    start_time = time.time()
    retry_count = 0

    try:
        logger.info(f"Starting async metadata download: Bucket='{bucket_name}', Key='{key}'")

        if progress_callback:
            await progress_callback("metadata_download_start", f"Downloading metadata: {key}")

        # Retry loop with exponential backoff
        for attempt in range(config.max_retries):
            try:
                retry_count = attempt
                if attempt > 0:
                    delay = min(config.retry_delay * (2 ** (attempt - 1)), config.max_retry_delay)
                    logger.info(
                        f"Retrying metadata download after {delay:.1f}s (attempt {attempt + 1}/{config.max_retries})"
                    )
                    await asyncio.sleep(delay)

                # Initialize async S3 session
                session = await init_async_s3_client()

                async with session.client("s3") as s3_client:
                    # Download with timeout
                    download_task = asyncio.create_task(
                        _download_s3_object_async(s3_client, bucket_name, key, config)
                    )

                    try:
                        file_content = await asyncio.wait_for(download_task, timeout=config.timeout)
                    except asyncio.TimeoutError:
                        raise Exception(f"Metadata download timeout after {config.timeout}s")

                    # Decode the JSON content
                    metadata_content = file_content.decode("utf-8")

                    # Validate JSON format
                    try:
                        json.loads(metadata_content)
                    except json.JSONDecodeError as e:
                        raise Exception(f"Invalid JSON metadata: {str(e)}")

                    download_time = time.time() - start_time

                    if progress_callback:
                        await progress_callback(
                            "metadata_download_complete",
                            f"Downloaded metadata ({len(metadata_content)} bytes)",
                        )

                    logger.info(
                        f"Successfully downloaded metadata: {bucket_name}/{key} ({download_time:.2f}s)"
                    )

                    return DownloadResult(
                        success=True,
                        content=metadata_content,
                        download_time=download_time,
                        retry_count=retry_count,
                    )

            except Exception as e:
                error_msg = f"Metadata download attempt {attempt + 1} failed: {str(e)}"
                logger.warning(
                    "metadata_async_download_attempt_failed",
                    extra={
                        "bucket": bucket_name,
                        "key": key,
                        "attempt": attempt + 1,
                        "max_retries": config.max_retries,
                        "error": str(e),
                    },
                )

                if attempt == config.max_retries - 1:
                    # Final attempt failed
                    download_time = time.time() - start_time
                    final_error = f"Failed to download metadata {bucket_name}/{key} after {config.max_retries} attempts: {str(e)}"
                    logger.error(
                        "metadata_async_download_failed_all_retries",
                        extra={
                            "bucket": bucket_name,
                            "key": key,
                            "error": str(e),
                            "max_retries": config.max_retries,
                            "download_time": download_time,
                        },
                    )

                    if progress_callback:
                        await progress_callback("metadata_download_failed", final_error)

                    return DownloadResult(
                        success=False,
                        error=final_error,
                        download_time=download_time,
                        retry_count=retry_count,
                    )

    except Exception as e:
        download_time = time.time() - start_time
        error_msg = f"Unexpected error downloading metadata {bucket_name}/{key}: {str(e)}"
        logger.error(
            "metadata_async_unexpected_error",
            extra={
                "bucket": bucket_name,
                "key": key,
                "error": str(e),
                "error_type": type(e).__name__,
                "download_time": download_time,
            },
        )

        if progress_callback:
            await progress_callback("metadata_download_error", error_msg)

        return DownloadResult(
            success=False, error=error_msg, download_time=download_time, retry_count=retry_count
        )


async def load_metadata_from_s3_async(
    config: Optional[S3DownloadConfig] = None, progress_callback: Optional[callable] = None
) -> Optional[Dict]:
    """
    Asynchronously load metadata from S3 bucket specified in environment variables.

    Args:
        config: Download configuration
        progress_callback: Optional callback for progress updates

    Returns:
        Parsed metadata dictionary or None if failed
    """
    try:
        app_config = get_config()
        bucket_name = app_config.aws.s3_bucket

        if not bucket_name:
            error_msg = "DOCUMENT_METADATA_S3_BUCKET not found in environment variables"
            logger.error("metadata_async_bucket_not_configured", error=error_msg)
            if progress_callback:
                await progress_callback("metadata_error", error_msg)
            return None

        # Get metadata key from config
        metadata_key = app_config.metadata_key

        # Download metadata from S3 asynchronously
        result = await download_metadata_from_s3_async(
            bucket_name, metadata_key, config, progress_callback
        )

        if not result.success:
            logger.error(f"Failed to download metadata: {result.error}")
            return None

        # Parse the downloaded metadata
        try:
            metadata = parse_metadata_file(result.content)
            if progress_callback:
                await progress_callback(
                    "metadata_parsed",
                    f"Parsed metadata for {len(metadata) if metadata else 0} companies",
                )
            return metadata
        except Exception as e:
            error_msg = f"Failed to parse metadata: {str(e)}"
            logger.error(
                "metadata_async_parse_failed",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            if progress_callback:
                await progress_callback("metadata_parse_error", error_msg)
            return None

    except Exception as e:
        error_msg = f"Error loading metadata from S3: {str(e)}"
        logger.error(
            "metadata_async_load_unexpected_error",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        if progress_callback:
            await progress_callback("metadata_load_error", error_msg)
        return None
