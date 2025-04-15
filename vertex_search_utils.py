import os
import re
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from typing import Dict, List, Optional
from google.cloud.aiplatform.matching_engine.matching_engine_index_config import DistanceMeasureType
from google.cloud.aiplatform_v1.types.index import IndexDatapoint

def embed_document(text: str, model_name: str = "text-embedding-005") -> Optional[List[float]]:
    """
    Generate embeddings for a document using Vertex AI's text embedding model.
    
    Args:
        text (str): The text content to embed
        model_name (str): The name of the embedding model to use
        
    Returns:
        Optional[List[float]]: The embedding vector, or None if embedding failed
    """
    try:
        # Initialize the model
        model = TextEmbeddingModel.from_pretrained(model_name)
        embeddings = model.get_embeddings([text])
        if embeddings and len(embeddings) > 0:
            return embeddings[0].values
        else:
            print("Error: No embedding vector received.")
            return None
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def upload_to_vertex_search(
    document_id: str,
    content: str,
    metadata: Dict,
    index_id: str,
    project_id: str,
    location: str
) -> bool:
    """
    Upload a document to Vertex AI Search.
    
    Args:
        document_id (str): Unique identifier for the document
        content (str): The text content of the document
        metadata (Dict): Document metadata
        index_id (str): The full name or ID of the index to update
        project_id (str): Google Cloud project ID
        location (str): Region for the index
        
    Returns:
        bool: True if upload was successful, False otherwise
    """
    try:
        # Generate embeddings
        embedding = embed_document(content)
        if not embedding:
            print(f"Failed to generate embeddings for document {document_id}")
            return False
            
        # Initialize the Vertex AI client
        aiplatform.init(project=project_id, location=location)
        
        # Create the index instance
        index = aiplatform.MatchingEngineIndex(index_name=index_id)
        
        # Get metadata values
        company_name = metadata.get("company_name", "unknown")
        year = metadata.get("year", "unknown")
        file_type = metadata.get("file_type", "unknown")
        
        # Create a properly formatted datapoint for the API
        restrictions = [
            IndexDatapoint.Restriction(namespace="company", allow_list=[company_name]),
            IndexDatapoint.Restriction(namespace="year", allow_list=[year]),
            IndexDatapoint.Restriction(namespace="file_type", allow_list=[file_type])
        ]
        
        datapoint = IndexDatapoint(
            id=document_id,
            feature_vector=embedding,
            restricts=restrictions
        )
        
        # Upload to the index
        index.upsert_datapoints(datapoints=[datapoint])
        
        print(f"Successfully uploaded document {document_id}")
        return True
        
    except Exception as e:
        print(f"Error uploading document {document_id}: {e}")
        return False

def populate_vertex_search(
    project_id: str,
    region: str,
    index_id: str,
    summaries_dir: str = "pdfs/summaries"
) -> None:
    """
    Populate the Vertex AI Search index with all document summaries.
    
    Args:
        project_id (str): Google Cloud project ID
        region (str): Region where the index is deployed
        index_id (str): ID of the index
        summaries_dir (str): Directory containing document summaries
    """
    # Set environment variables for the embedding model
    os.environ["PROJECT_ID"] = project_id
    os.environ["REGION"] = region
    
    # Process all summaries
    for file in os.listdir(summaries_dir):
        if file.endswith(".txt"):
            file_path = os.path.join(summaries_dir, file)
            with open(file_path, "r") as f:
                content = f.read()
                
            # Extract metadata from filename
            company_name = file.split("-")[0]
            year_match = re.search(r'\d{4}', file)
            year = year_match.group() if year_match else "Unknown"
            file_type = "financial" if "financial" in file.lower() else "non-financial"
            
            metadata = {
                "company_name": company_name,
                "year": year,
                "file_type": file_type
            }
            
            # Upload to Vertex Search
            upload_to_vertex_search(
                document_id=file.split(".")[0],
                content=content,
                metadata=metadata,
                index_id=index_id,
                project_id=project_id,
                location=region
            ) 