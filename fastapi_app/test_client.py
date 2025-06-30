import streamlit as st
import requests
import json
import time  # For execution-time measurement

# -------------------
# Configuration
# -------------------
REMOTE_PROD_BASE_URL = "http://jse-da-Publi-YNJSNI96davV-367279008.us-east-1.elb.amazonaws.com"
REMOTE_STAGE_BASE_URL = "http://jse-da-Publi-wupYrzJkbRrm-92790362.us-east-1.elb.amazonaws.com"
LOCAL_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="JSE Datasphere Chat", page_icon="💬")
st.title("💬 JSE Datasphere Chat")

# Ensure we have a slot in session state for timing information
if "last_exec_time" not in st.session_state:
    st.session_state.last_exec_time = None

# -------------------
# Sidebar controls
# -------------------
env = st.sidebar.selectbox("Environment", [ "Local", "Remote Prod", "Remote Stage"])
BASE_URL = LOCAL_BASE_URL if env == "Local" else REMOTE_STAGE_BASE_URL if env == "Remote Stage" else REMOTE_PROD_BASE_URL

# Primary mode (chat or vector upload)
mode = st.sidebar.radio(
    "Mode",
    ["Chat", "Streaming Chat", "Add Document", "Query Vector DB", "Query Meta DB"],
    horizontal=True,
)

# -------------------
# Diagnostic sidebar info
# -------------------
if st.session_state.last_exec_time is not None:
    st.sidebar.metric("⏱️ Last Execution (s)", f"{st.session_state.last_exec_time:.2f}")

    # Add a reset button to clear the stopwatch/last execution timer
    if st.sidebar.button("🔄 Reset Timer"):
        st.session_state.last_exec_time = None
        st.rerun()

# -------------------
# Chat Mode
# -------------------
if mode == "Chat":
    # Choose which chat endpoint to hit
    chat_variant = st.sidebar.radio(
        "Chat Endpoint",
        ["Standard", "Fast"],
        horizontal=True,
        help="Standard = original /chat (S3-based); Fast = new /fast_chat (vector-based)",
    )
    endpoint_path = "/fast_chat" if chat_variant == "Fast" else "/chat"

    # Initialize session state (chat history)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display previous messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Prompt user input
    if prompt := st.chat_input("Ask something about your documents..."):
        # Show user message immediately
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Call the chat endpoint
        start_time = time.perf_counter()
        try:
            response = requests.post(
                f"{BASE_URL}{endpoint_path}",
                json={
                    "query": prompt,
                    "conversation_history": st.session_state.chat_history,
                    "auto_load_documents": True,
                    "memory_enabled": True,
                },
                timeout=300,  # increase timeout for potential S3 latency
            )
            response.raise_for_status()
            result = response.json()
            answer = result.get("response", "No response received.")
        except Exception as e:
            answer = f"❌ Error: {e}"
        finally:
            # Record execution time regardless of success/failure
            st.session_state.last_exec_time = time.perf_counter() - start_time

        # Display assistant answer
        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Reset button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

elif mode == "Streaming Chat":
    st.header("💬 Streaming Chat with Real-time Progress")
    st.info("This mode shows real-time progress updates while your request is being processed. Note: Due to Streamlit limitations, the progress updates are simulated rather than true real-time streaming.")
    
    # Endpoint selection for streaming
    stream_endpoint = st.radio(
        "Choose streaming endpoint:",
        ["Traditional Chat (S3)", "Fast Chat (Vector DB)"],
        horizontal=True
    )
    
    endpoint_path = "/chat/stream" if "Traditional" in stream_endpoint else "/fast_chat/stream"
    
    # Chat options
    col1, col2 = st.columns(2)
    with col1:
        auto_load = st.checkbox("Auto-load documents", value=True, key="stream_auto")
    with col2:
        memory_enabled = st.checkbox("Enable memory", value=True, key="stream_memory")
    
    # Chat input
    if "stream_chat_history" not in st.session_state:
        st.session_state.stream_chat_history = []
    
    # Display chat history
    for message in st.session_state.stream_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about JSE documents..."):
        # Show user message immediately
        st.chat_message("user").markdown(prompt)
        st.session_state.stream_chat_history.append({"role": "user", "content": prompt})

        # Create progress indicators
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            details_text = st.empty()
        
        # Call the streaming endpoint (note: this simulates streaming for demo)
        start_time = time.perf_counter()
        
        try:
            # Update progress - Step 1
            progress_bar.progress(10)
            status_text.text("🔍 Preparing search query...")
            time.sleep(0.5)
            
            # Update progress - Step 2  
            progress_bar.progress(30)
            status_text.text("📄 Selecting relevant documents...")
            time.sleep(0.5)
            
            # Update progress - Step 3
            progress_bar.progress(60)
            if "Fast" in stream_endpoint:
                status_text.text("🔍 Searching vector database...")
            else:
                status_text.text("☁️ Loading documents from S3...")
            time.sleep(1.0)
            
            # Update progress - Step 4
            progress_bar.progress(85)
            status_text.text("🤖 Generating AI response...")
            time.sleep(0.5)
            
            # Make the actual API call
            response = requests.post(
                f"{BASE_URL}{endpoint_path.replace('/stream', '')}",  # Use non-streaming endpoint for compatibility
                json={
                    "query": prompt,
                    "conversation_history": st.session_state.stream_chat_history if memory_enabled else None,
                    "auto_load_documents": auto_load,
                    "memory_enabled": memory_enabled,
                },
                timeout=300,
            )
            response.raise_for_status()
            result = response.json()
            
            # Final progress update
            progress_bar.progress(100)
            status_text.text("✅ Response generation complete!")
            time.sleep(0.3)
            
            # Clear progress indicators
            progress_container.empty()
            
            answer = result.get("response", "No response received.")
            
            # Show additional info if available
            if result.get("documents_loaded"):
                details_text.success(f"📚 Loaded {len(result['documents_loaded'])} documents")
            if result.get("document_selection_message"):
                st.info(f"🎯 {result['document_selection_message']}")
                
        except Exception as e:
            progress_container.empty()
            answer = f"❌ Error: {e}"
        finally:
            # Record execution time
            st.session_state.last_exec_time = time.perf_counter() - start_time

        # Display assistant answer
        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.stream_chat_history.append({"role": "assistant", "content": answer})

    # Reset button for streaming chat
    if st.button("🗑️ Clear Streaming Chat History"):
        st.session_state.stream_chat_history = []
        st.rerun()

# -------------------
# Add Document Mode
# -------------------
elif mode == "Add Document":
    st.subheader("📄 Add / Upsert Document to Vector DB")

    # Inputs for the new document
    doc_text = st.text_area("Document Text", height=200)

    metadata_default = \
"""{
  "company_name": "",
  "file_type": "",
  "year": "",
  "source": ""
}"""
    metadata_raw = st.text_area(
        "Metadata JSON (optional – must be valid JSON and align 1-to-1 with documents)",
        value=metadata_default,
        height=150,
    )

    custom_id = st.text_input("Custom ID (optional)")

    if st.button("🚀 Add / Upsert"):
        if not doc_text.strip():
            st.error("Document text cannot be empty.")
            st.stop()

        # Build payload
        payload = {"documents": [doc_text]}

        # Parse metadata JSON if provided
        if metadata_raw.strip():
            try:
                metadata_parsed = json.loads(metadata_raw)
                if not isinstance(metadata_parsed, dict):
                    raise ValueError("Metadata must be a JSON object (dictionary).")
                payload["metadatas"] = [metadata_parsed]
            except Exception as e:
                st.error(f"Invalid metadata JSON: {e}")
                st.stop()

        # Attach custom ID if provided
        if custom_id.strip():
            payload["ids"] = [custom_id.strip()]

        # Call the update endpoint
        start_time = time.perf_counter()
        try:
            res = requests.post(f"{BASE_URL}/chroma/update", json=payload, timeout=60)
            if res.status_code >= 400:
                # Show server-provided error details if available
                try:
                    err_json = res.json()
                    detail = err_json.get("detail", err_json)
                except Exception:
                    detail = res.text
                st.error(f"❌ Server {res.status_code} Error: {detail}")
            else:
                st.success(f"✅ Success: {res.json()}")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Request failed: {e}")
        finally:
            st.session_state.last_exec_time = time.perf_counter() - start_time

# -------------------
# Query Vector DB Mode
# -------------------
elif mode == "Query Vector DB":
    st.subheader("🔎 Query Vector Database")

    # Input fields for the query
    query_text = st.text_input("Query Text", placeholder="e.g. What is Company X's revenue in 2023?")
    n_results = st.number_input(
        "Number of Results",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    where_default = "{}"  # empty filter by default
    where_raw = st.text_area(
        "Optional 'where' JSON filter (leave blank for none)",
        value=where_default,
        height=100,
    )

    if st.button("🔍 Run Query"):
        if not query_text.strip():
            st.error("Query text cannot be empty.")
            st.stop()

        payload = {"query": query_text, "n_results": n_results}

        # Attempt to parse the optional where filter
        if where_raw.strip() and where_raw.strip() != "{}":
            try:
                where_parsed = json.loads(where_raw)
                payload["where"] = where_parsed
            except Exception as parse_err:
                st.error(f"Invalid where JSON: {parse_err}")
                st.stop()

        # Call the Chroma query endpoint
        start_time = time.perf_counter()
        try:
            res = requests.post(f"{BASE_URL}/chroma/query", json=payload, timeout=60)
            res.raise_for_status()
            result = res.json()

            # Pretty-print results
            st.write("### Results:")
            st.json(result)
        except requests.exceptions.RequestException as req_err:
            st.error(f"❌ Request failed: {req_err}")
        finally:
            st.session_state.last_exec_time = time.perf_counter() - start_time
            
# -------------------
# Query Meta DB Mode
# -------------------
elif mode == "Query Meta DB":
    st.subheader("🔎 Query Meta Database")

    # Input fields for the query
    query_text = st.text_input("Query Text", placeholder="e.g. What is Company X's revenue in 2023?")
    n_results = st.number_input(
        "Number of Results",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    if st.button("🔍 Run Query"):
        if not query_text.strip():
            st.error("Query text cannot be empty.")
            st.stop()

        payload = {"query": query_text, "n_results": n_results}

        # Call the Chroma query endpoint
        start_time = time.perf_counter()
        try:
            res = requests.post(f"{BASE_URL}/chroma/meta/query", json=payload, timeout=60)
            res.raise_for_status()
            result = res.json()

            # Pretty-print results
            st.write("### Results:")
            st.json(result)
        except requests.exceptions.RequestException as req_err:
            st.error(f"❌ Request failed: {req_err}")
        finally:
            st.session_state.last_exec_time = time.perf_counter() - start_time
            