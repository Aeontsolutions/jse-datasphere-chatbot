import os
from dotenv import load_dotenv
from typing import Dict, Any

class Settings:
    def __init__(self):
        load_dotenv()
        
        # AWS Configuration
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_DEFAULT_REGION")
        self.metadata_s3_bucket = os.getenv("DOCUMENT_METADATA_S3_BUCKET")
        
        # Google Cloud Configuration
        self.google_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.google_project_id = self._get_google_project_id()
        
        # Application Settings
        self.max_documents = 3
        self.max_conversation_history = 20
        self.max_feedback_history = 50
        
    def _get_google_project_id(self) -> str:
        """Extract project ID from Google service account credentials."""
        if not self.google_credentials_path or not os.path.exists(self.google_credentials_path):
            return ""
            
        try:
            import json
            with open(self.google_credentials_path, 'r') as f:
                credentials = json.load(f)
                return credentials.get("project_id", "")
        except Exception:
            return ""
            
    def validate(self) -> Dict[str, Any]:
        """Validate all required settings are present."""
        errors = {}
        
        # Validate AWS credentials
        if not all([self.aws_access_key_id, self.aws_secret_access_key, self.aws_region]):
            errors["aws"] = "Missing required AWS credentials"
            
        if not self.metadata_s3_bucket:
            errors["s3_bucket"] = "Missing DOCUMENT_METADATA_S3_BUCKET"
            
        # Validate Google credentials
        if not self.google_credentials_path:
            errors["google_credentials"] = "Missing GOOGLE_APPLICATION_CREDENTIALS"
        elif not os.path.exists(self.google_credentials_path):
            errors["google_credentials"] = f"Credentials file not found at {self.google_credentials_path}"
            
        if not self.google_project_id:
            errors["google_project"] = "Could not determine Google Cloud project ID"
            
        return errors

# Create a singleton instance
settings = Settings() 