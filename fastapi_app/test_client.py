import streamlit as st
import requests
import json

# -------------------
# Configuration
# -------------------
REMOTE_PROD_BASE_URL = "http://jse-da-Publi-YNJSNI96davV-367279008.us-east-1.elb.amazonaws.com"
REMOTE_STAGE_BASE_URL = "http://jse-da-Publi-wupYrzJkbRrm-92790362.us-east-1.elb.amazonaws.com"
LOCAL_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="JSE Datasphere Chat", page_icon="💬")
st.title("💬 JSE Datasphere Chat")

# -------------------
# Sidebar controls
# -------------------
env = st.sidebar.selectbox("Environment", [ "Local", "Remote Prod", "Remote Stage"])
BASE_URL = LOCAL_BASE_URL if env == "Local" else REMOTE_STAGE_BASE_URL if env == "Remote Stage" else REMOTE_PROD_BASE_URL

# Primary mode (chat or vector upload)
mode = st.sidebar.radio(
    "Mode",
    ["Chat", "Add Document", "Query Vector DB"],
    horizontal=True,
)

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

        # Display assistant answer
        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Reset button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
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
        try:
            res = requests.post(f"{BASE_URL}/chroma/query", json=payload, timeout=60)
            res.raise_for_status()
            result = res.json()

            # Pretty-print results
            st.write("### Results:")
            st.json(result)
        except requests.exceptions.RequestException as req_err:
            st.error(f"❌ Request failed: {req_err}")