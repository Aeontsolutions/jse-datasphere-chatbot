"""
PDF text extraction utilities.

This module provides functions for extracting text from PDF files,
supporting both file-based and byte-based inputs.
"""

from io import BytesIO
from typing import Optional

import pypdf
from fastapi import HTTPException

from app.logging_config import get_logger

logger = get_logger(__name__)


def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file"""
    try:
        pdf_reader = pypdf.PdfReader(pdf_file)
        text = ""
        page_count = len(pdf_reader.pages)

        for page_num in range(page_count):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            text += page_text + "\n\n"

        logger.info(
            "pdf_extraction_success",
            page_count=page_count,
            text_length=len(text),
        )

        return text

    except pypdf.errors.PdfReadError as e:
        logger.error("pdf_read_error", error=str(e), error_type="PdfReadError")
        raise HTTPException(
            status_code=400, detail="Failed to read PDF file - file may be corrupted or invalid"
        )
    except Exception as e:
        logger.error(
            "pdf_extraction_unexpected_error",
            error=str(e),
            error_type=type(e).__name__,
        )
        raise HTTPException(status_code=500, detail="Failed to extract text from PDF")


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> Optional[str]:
    """Extract text from PDF bytes (runs in thread pool to avoid blocking)"""
    try:
        pdf_file = BytesIO(pdf_bytes)
        pdf_reader = pypdf.PdfReader(pdf_file)

        text = ""
        page_count = len(pdf_reader.pages)

        for page_num in range(page_count):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            text += page_text + "\n\n"

        logger.info(
            "pdf_bytes_extraction_success",
            extra={
                "page_count": page_count,
                "text_length": len(text),
                "bytes_size": len(pdf_bytes),
            },
        )

        return text

    except pypdf.errors.PdfReadError as e:
        logger.error(
            "pdf_bytes_read_error",
            extra={"error": str(e), "error_type": "PdfReadError", "bytes_size": len(pdf_bytes)},
        )
        raise HTTPException(
            status_code=400, detail="Failed to read PDF data - file may be corrupted or invalid"
        )
    except Exception as e:
        logger.error(
            "pdf_bytes_extraction_unexpected_error",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "bytes_size": len(pdf_bytes) if pdf_bytes else 0,
            },
        )
        raise HTTPException(status_code=500, detail="Failed to extract text from PDF data")
