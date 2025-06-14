import streamlit as st
import requests
import json

# -------------------
# Configuration
# -------------------
REMOTE_BASE_URL = "http://jse-da-Publi-YNJSNI96davV-367279008.us-east-1.elb.amazonaws.com"
LOCAL_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="JSE Datasphere Chat", page_icon="💬")
st.title("💬 JSE Datasphere Chat")

# -------------------
# Sidebar controls
# -------------------
env = st.sidebar.selectbox("Environment", ["Remote", "Local"])
BASE_URL = REMOTE_BASE_URL if env == "Remote" else LOCAL_BASE_URL

mode = st.sidebar.radio("Mode", ["Chat", "Add Document"], horizontal=True)

# -------------------
# Chat Mode
# -------------------
if mode == "Chat":
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
                f"{BASE_URL}/chat",
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
  "doctype": "",
  "year": "",
  "source": ""
+}"""
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