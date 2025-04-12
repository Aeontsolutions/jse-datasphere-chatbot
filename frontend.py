import streamlit as st
import requests
import json
import os
from typing import List, Dict

# Configure the page
st.set_page_config(
    page_title="JSE Document Chat",
    page_icon="📄",
    layout="wide"
)

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for API URL
if "api_url" not in st.session_state:
    st.session_state.api_url = os.getenv("API_URL", "http://localhost:8000")

# Sidebar for settings
with st.sidebar:
    st.title("Settings")
    
    # API URL configuration
    api_url = st.text_input(
        "API URL",
        value=st.session_state.api_url,
        help="URL of the API endpoint"
    )
    if api_url != st.session_state.api_url:
        st.session_state.api_url = api_url
        st.rerun()
    
    # Memory toggle
    memory_enabled = st.checkbox(
        "Enable conversation memory",
        value=True,
        help="When enabled, the AI will remember previous questions and answers"
    )
    
    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.title("JSE Document Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Display context used
        if message.get("documents_used"):
            with st.expander("Documents Used"):
                for doc in message["documents_used"]:
                    st.subheader(doc["filename"])
                    if doc.get("reason"):
                        st.caption(f"Reason: {doc['reason']}")
                    st.text_area(
                        "Content",
                        value=doc["content"],
                        height=200,
                        disabled=True
                    )
        
        # Display document recommendation
        if message.get("document_recommendation"):
            with st.expander("Document Recommendation"):
                st.write(message["document_recommendation"])

# Chat input
if prompt := st.chat_input("Ask a question about JSE documents..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Prepare API request
    try:
        # Get conversation history for the API
        conversation_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages[:-1]  # Exclude current message
        ]
        
        # Make API request
        response = requests.post(
            f"{st.session_state.api_url}/chat",
            json={
                "query": prompt,
                "conversation_history": conversation_history,
                "memory_enabled": memory_enabled
            }
        )
        
        # Check for errors
        response.raise_for_status()
        data = response.json()
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.write(data["answer"])
            
            # Add documents used if available
            if data.get("documents_used"):
                with st.expander("Documents Used"):
                    for doc in data["documents_used"]:
                        st.subheader(doc["filename"])
                        if doc.get("reason"):
                            st.caption(f"Reason: {doc['reason']}")
                        st.text_area(
                            "Content",
                            value=doc["content"],
                            height=200,
                            disabled=True
                        )
            
            # Add document recommendation if available
            if data.get("document_recommendation"):
                with st.expander("Document Recommendation"):
                    st.write(data["document_recommendation"])
        
        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": data["answer"],
            "documents_used": data.get("documents_used"),
            "document_recommendation": data.get("document_recommendation")
        })
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {str(e)}")
        st.info("Make sure the API server is running and the URL is correct.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add some helpful information
with st.expander("About this Chat Interface"):
    st.markdown("""
    This interface allows you to interact with the JSE Document Chat API.
    
    **Features:**
    - Ask questions about JSE documents
    - View the actual documents used for each answer
    - See which documents were recommended
    - Maintain conversation history
    - Configure API settings
    
    **How to use:**
    1. Make sure the API server is running
    2. Enter your question in the chat input
    3. View the response and expand the "Documents Used" section to see the actual documents
    4. Use the sidebar to configure settings or clear the conversation
    """) 