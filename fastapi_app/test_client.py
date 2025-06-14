import streamlit as st
import requests

API_URL = "http://jse-da-Publi-YNJSNI96davV-367279008.us-east-1.elb.amazonaws.com/chat"

st.set_page_config(page_title="Gemini Chat", page_icon="💬")
st.title("💬 Gemini AI Document Chat")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat UI display
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Prompt user input
if prompt := st.chat_input("Ask something about your documents..."):
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Send to API
    try:
        response = requests.post(
            API_URL,
            json={
                "query": prompt,
                "conversation_history": st.session_state.chat_history,
                "auto_load_documents": True,
                "memory_enabled": True,
            },
            timeout=300  # Increase timeout if S3 load is slow
        )
        response.raise_for_status()
        result = response.json()
        answer = result.get("response", "No response received.")
    except Exception as e:
        answer = f"❌ Error: {e}"

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# Reset button
if st.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()