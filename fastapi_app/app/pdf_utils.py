"""
PDF text extraction utilities.

This module provides functions for extracting text from PDF files,
supporting both file-based and byte-based inputs.
"""

import logging
from io import BytesIO
from typing import Optional

import pypdf

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file"""
    try:
        pdf_reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            text += page_text + "\n\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return None


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> Optional[str]:
    """Extract text from PDF bytes (runs in thread pool to avoid blocking)"""
    try:
        pdf_file = BytesIO(pdf_bytes)
        pdf_reader = pypdf.PdfReader(pdf_file)

        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            text += page_text + "\n\n"

        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF bytes: {str(e)}")
        return None
