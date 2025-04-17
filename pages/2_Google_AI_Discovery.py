import streamlit as st
import requests
import json
import os
import subprocess
import logging
from typing import Dict, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('google_ai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_ID = "1069196447797"
LOCATION = "global"
COLLECTION = "default_collection"
ENGINE = "jse-docs"
SERVING_CONFIG = "default_search"

def get_access_token() -> str:
    """Get Google Cloud access token using gcloud CLI."""
    try:
        logger.info("Getting access token...")
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True,
            text=True
        )
        token = result.stdout.strip()
        if token:
            logger.info("Successfully obtained access token")
        else:
            logger.error("Access token is empty")
        return token
    except Exception as e:
        logger.error(f"Error getting access token: {str(e)}")
        st.error(f"Error getting access token: {str(e)}")
        return ""

def perform_search(query: str) -> Dict:
    """Perform a search using Google AI Discovery Engine."""
    logger.info(f"Performing search for query: {query}")
    access_token = get_access_token()
    if not access_token:
        logger.error("No access token available")
        return {}
    
    url = f"https://discoveryengine.googleapis.com/v1alpha/projects/{PROJECT_ID}/locations/{LOCATION}/collections/{COLLECTION}/engines/{ENGINE}/servingConfigs/{SERVING_CONFIG}:search"
    logger.info(f"Search URL: {url}")
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    data = {
        "query": query,
        "pageSize": 10,
        "queryExpansionSpec": {"condition": "AUTO"},
        "spellCorrectionSpec": {"mode": "AUTO"},
        "languageCode": "en-GB",
        "contentSearchSpec": {
            "extractiveContentSpec": {"maxExtractiveAnswerCount": 1}
        },
        "userInfo": {"timeZone": "Asia/Tokyo"},
        "session": f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/{COLLECTION}/engines/{ENGINE}/sessions/-"
    }
    
    try:
        logger.info("Sending search request...")
        logger.info(f"Request data: {json.dumps(data, indent=2)}")
        response = requests.post(url, headers=headers, json=data)
        logger.info(f"Search response status: {response.status_code}")
        logger.info(f"Search response headers: {dict(response.headers)}")
        
        # Log the raw response text before parsing
        logger.info(f"Raw response text: {response.text}")
        
        response.raise_for_status()
        result = response.json()
        
        # Log the complete response structure
        logger.info("Complete response structure:")
        logger.info(json.dumps(result, indent=2))
        
        # Log all top-level keys in the response
        logger.info(f"Response keys: {list(result.keys())}")
        
        # Check for specific fields
        if "queryId" not in result:
            logger.error("queryId not found in response")
        if "session" not in result:
            logger.error("session not found in response")
            
        return result
    except Exception as e:
        logger.error(f"Error performing search: {str(e)}")
        if hasattr(e, 'response'):
            logger.error(f"Error response: {e.response.text}")
        st.error(f"Error performing search: {str(e)}")
        return {}

def get_answer(query_id: str, session: str, original_query: str) -> Dict:
    """Get an answer from Google AI Discovery Engine."""
    logger.info(f"Getting answer for query_id: {query_id}, session: {session}, query: {original_query}")
    access_token = get_access_token()
    if not access_token:
        logger.error("No access token available")
        return {}
    
    url = f"https://discoveryengine.googleapis.com/v1alpha/projects/{PROJECT_ID}/locations/{LOCATION}/collections/{COLLECTION}/engines/{ENGINE}/servingConfigs/{SERVING_CONFIG}:answer"
    logger.info(f"Answer URL: {url}")
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    data = {
        "query": {
            "text": original_query,  # Include the original query text
            "queryId": query_id
        },
        "session": session,
        "relatedQuestionsSpec": {
            "enable": True
        },
        "answerGenerationSpec": {
            "ignoreAdversarialQuery": True,
            "ignoreNonAnswerSeekingQuery": False,
            "ignoreLowRelevantContent": True,
            "multimodalSpec": {},
            "includeCitations": True,
            "answerLanguageCode": "en",
            "modelSpec": {
                "modelVersion": "gemini-2.0-flash-001/answer_gen/v1"
            }
        }
    }
    
    try:
        logger.info("Sending answer request...")
        logger.info(f"Request data: {json.dumps(data, indent=2)}")
        response = requests.post(url, headers=headers, json=data)
        logger.info(f"Answer response status: {response.status_code}")
        logger.info(f"Answer response headers: {dict(response.headers)}")
        
        # Log the raw response text before parsing
        logger.info(f"Raw response text: {response.text}")
        
        response.raise_for_status()
        result = response.json()
        logger.info(f"Answer response: {json.dumps(result, indent=2)}")
        return result
    except Exception as e:
        logger.error(f"Error getting answer: {str(e)}")
        if hasattr(e, 'response'):
            logger.error(f"Error response: {e.response.text}")
        st.error(f"Error getting answer: {str(e)}")
        return {}

# Main interface
st.title("Google AI Discovery Engine")

# Display chat messages
for message in st.session_state.google_ai_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("citations"):
            with st.expander("Citations"):
                for citation in message["citations"]:
                    st.write(citation)

# Chat input
if prompt := st.chat_input("Ask a question..."):
    logger.info(f"Received prompt: {prompt}")
    # Display user message
    st.session_state.google_ai_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Perform search
    search_response = perform_search(prompt)
    if search_response:
        # Get queryId and session from sessionInfo
        session_info = search_response.get("sessionInfo", {})
        query_id = session_info.get("queryId")
        session = session_info.get("name")
        logger.info(f"Search response - query_id: {query_id}, session: {session}")
        
        if query_id and session:
            # Get answer
            answer_response = get_answer(query_id, session, prompt)
            if answer_response:
                answer = answer_response.get("answer", {}).get("answerText", "")
                citations = answer_response.get("answer", {}).get("citations", [])
                logger.info(f"Answer: {answer}")
                logger.info(f"Citations: {citations}")
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.write(answer)
                    if citations:
                        with st.expander("Citations"):
                            for citation in citations:
                                st.write(citation)
                
                # Add to conversation history
                st.session_state.google_ai_messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations
                })
            else:
                logger.error("No answer received from get_answer")
        else:
            logger.error("No query_id or session in search response")
    else:
        logger.error("No search response received")

# Add some helpful information
with st.expander("About this Interface"):
    st.markdown("""
    This interface allows you to interact with the Google AI Discovery Engine.
    
    **Features:**
    - Ask questions and get AI-generated answers
    - View citations and sources
    - Maintain conversation history
    
    **How to use:**
    1. Make sure you're authenticated with Google Cloud
    2. Enter your question in the chat input
    3. View the response and expand the "Citations" section to see sources
    """) 