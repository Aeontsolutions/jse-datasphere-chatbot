import tempfile
import logging
from typing import Optional, Dict, Any
import boto3
import PyPDF2
from io import BytesIO
from ..config.settings import settings

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region
        )
        
    def extract_text_from_pdf(self, pdf_file: BytesIO) -> Optional[str]:
        """Extract text content from a PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                text += page_text + "\n\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return None
            
    def download_and_extract_from_s3(self, s3_path: str) -> Optional[str]:
        """Download a PDF from S3 and extract its text content."""
        try:
            if not s3_path.startswith("s3://"):
                logger.error(f"Invalid S3 path format: {s3_path}")
                return None
                
            path_without_prefix = s3_path[5:]  # Remove "s3://"
            bucket_name = path_without_prefix.split('/')[0]
            key = '/'.join(path_without_prefix.split('/')[1:])
            
            logger.info(f"Downloading S3 object: Bucket='{bucket_name}', Key='{key}'")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                self.s3_client.download_fileobj(bucket_name, key, tmp_file)
                tmp_file_path = tmp_file.name
                
            with open(tmp_file_path, 'rb') as pdf_file:
                text = self.extract_text_from_pdf(pdf_file)
                
            # Clean up temporary file
            import os
            os.unlink(tmp_file_path)
            
            return text
        except Exception as e:
            logger.error(f"Error downloading/processing PDF from S3: {str(e)}")
            return None
            
    def download_metadata_from_s3(self, bucket_name: str, key: str = "metadata.json") -> Optional[str]:
        """Download metadata file from S3."""
        try:
            response = self.s3_client.get_object(Bucket=bucket_name, Key=key)
            return response['Body'].read().decode('utf-8')
        except Exception as e:
            logger.error(f"Error downloading metadata from S3: {str(e)}")
            return None
            
    def parse_metadata_file(self, metadata_content: str) -> Optional[Dict[str, Any]]:
        """Parse metadata JSON content."""
        try:
            import json
            return json.loads(metadata_content)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing metadata file: {str(e)}")
            return None 