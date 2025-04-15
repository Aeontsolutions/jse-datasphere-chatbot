import os
from dotenv import load_dotenv
from vertex_search_utils import populate_vertex_search
from google.oauth2 import service_account
from google.cloud import aiplatform

# Load environment variables
load_dotenv()

# Configuration
PROJECT_ID = "jse-datasphere"  # Make sure this matches your project ID
REGION = "us-central1"
# Update with the actual index ID from your project
INDEX_ID = "JseFindocIndex"  # This is the ID of your Vector Search index
SUMMARIES_DIR = "pdfs/summaries"

# Set Google credentials
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service-account/credentials.json")
credentials = service_account.Credentials.from_service_account_file(credentials_path)

# Initialize Vertex AI
aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    credentials=credentials
)

if __name__ == "__main__":
    print("Starting to populate Vertex AI Search...")
    populate_vertex_search(
        project_id=PROJECT_ID,
        region=REGION,
        index_id=INDEX_ID,
        summaries_dir=SUMMARIES_DIR
    )
    print("Finished populating Vertex AI Search.") 