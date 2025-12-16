import streamlit as st
import requests
import json
import time  # For execution-time measurement

# -------------------
# Configuration
# -------------------
REMOTE_PROD_BASE_URL = "http://jse-da-Publi-YNJSNI96davV-367279008.us-east-1.elb.amazonaws.com"
REMOTE_STAGE_BASE_URL = "http://jse-da-Publi-wupYrzJkbRrm-92790362.us-east-1.elb.amazonaws.com"
REMOTE_DEV_BASE_URL = "http://jse-da-Publi-1wgQS45FnwpV-1192199831.us-east-1.elb.amazonaws.com"
LOCAL_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="JSE Datasphere Chat", page_icon="💬")
st.title("💬 JSE Datasphere Chat")

# Ensure we have a slot in session state for timing information
if "last_exec_time" not in st.session_state:
    st.session_state.last_exec_time = None

# -------------------
# Sidebar controls
# -------------------
env = st.sidebar.selectbox("Environment", ["Local", "Remote Prod", "Remote Stage", "Remote Dev"])
BASE_URL = (
    LOCAL_BASE_URL
    if env == "Local"
    else (
        REMOTE_STAGE_BASE_URL
        if env == "Remote Stage"
        else REMOTE_DEV_BASE_URL if env == "Remote Dev" else REMOTE_PROD_BASE_URL
    )
)

# Primary mode (chat or vector upload)
mode = st.sidebar.radio(
    "Mode",
    [
        "Chat",
        "Streaming Chat",
        "Add Document",
        "Query Vector DB",
        "Query Meta DB",
        "Financial Metadata",
    ],
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
        ["Standard", "Fast", "Financial V2"],
        horizontal=True,
        help="Standard = original /chat (S3-based); Fast = new /fast_chat (vector-based); Financial V2 = /fast_chat_v2 (financial data AI)",
    )

    if chat_variant == "Financial V2":
        endpoint_path = "/fast_chat_v2"
    elif chat_variant == "Fast":
        endpoint_path = "/fast_chat"
    else:
        endpoint_path = "/chat"

    # Show example queries for Financial V2 endpoint
    if chat_variant == "Financial V2":
        st.info(
            "🚀 **Financial V2 Endpoint**: AI-powered financial data analysis with natural language queries. Supports conversation memory and provides detailed insights!"
        )
        st.sidebar.markdown("### 💡 Example Financial Queries")
        example_queries = [
            "Show me MDS revenue for 2024",
            "Compare JBG and CPJ profit margins",
            "What is SOS total assets for the last 3 years?",
            "Show me all companies' revenue in 2023",
            "What about 2022?",  # Follow-up example
        ]

        for query in example_queries:
            if st.sidebar.button(f"📊 {query}", key=f"example_{hash(query)}"):
                # Add the example query to chat
                st.session_state.chat_history.append({"role": "user", "content": query})
                st.rerun()

        # Add a financial health check button
        st.sidebar.markdown("---")
        if st.sidebar.button("🏥 Check Financial Data Status"):
            try:
                health_res = requests.get(f"{BASE_URL}/health", timeout=30)
                health_res.raise_for_status()
                health_data = health_res.json()

                financial_status = health_data.get("financial_data", {})
                status = financial_status.get("status", "unknown")
                records = financial_status.get("records", 0)

                if status == "available":
                    st.sidebar.success(f"✅ Financial data ready ({records} records)")
                else:
                    st.sidebar.error(f"❌ Financial data {status}")
            except Exception as e:
                st.sidebar.error(f"❌ Health check failed: {e}")

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
            # Build request payload based on endpoint
            if chat_variant == "Financial V2":
                payload = {
                    "query": prompt,
                    "memory_enabled": True,
                    "conversation_history": st.session_state.chat_history,
                }
            else:
                payload = {
                    "query": prompt,
                    "conversation_history": st.session_state.chat_history,
                    "auto_load_documents": True,
                    "memory_enabled": True,
                }

            response = requests.post(
                f"{BASE_URL}{endpoint_path}",
                json=payload,
                timeout=300,  # increase timeout for potential S3 latency
            )
            response.raise_for_status()
            result = response.json()
            answer = result.get("response", "No response received.")

            # Display additional info for Financial V2 endpoint
            if chat_variant == "Financial V2":
                # Show financial data specific information
                data_found = result.get("data_found", False)
                record_count = result.get("record_count", 0)

                # Display data status
                if data_found:
                    st.success(f"📊 Found {record_count} financial records")
                else:
                    st.warning("📊 No financial data found for this query")

                # Show filters used
                filters_used = result.get("filters_used")
                if filters_used:
                    with st.expander("🔍 Query Filters Applied"):
                        if filters_used.get("companies"):
                            st.write(f"**Companies:** {', '.join(filters_used['companies'])}")
                        if filters_used.get("symbols"):
                            st.write(f"**Symbols:** {', '.join(filters_used['symbols'])}")
                        if filters_used.get("years"):
                            st.write(f"**Years:** {', '.join(filters_used['years'])}")
                        if filters_used.get("items"):
                            st.write(f"**Metrics:** {', '.join(filters_used['items'])}")
                        if filters_used.get("interpretation"):
                            st.write(f"**AI Interpretation:** {filters_used['interpretation']}")

                # Show data preview
                data_preview = result.get("data_preview")
                if data_preview and len(data_preview) > 0:
                    with st.expander(f"📈 Data Preview ({len(data_preview)} records)"):
                        for i, record in enumerate(data_preview[:5]):  # Show first 5 records
                            st.write(
                                f"**{i+1}.** {record.get('company', 'N/A')} ({record.get('symbol', 'N/A')}) - {record.get('year', 'N/A')}"
                            )
                            st.write(
                                f"   {record.get('item', 'N/A')}: {record.get('formatted_value', record.get('item', 'N/A'))}"
                            )
                        if len(data_preview) > 5:
                            st.write(f"... and {len(data_preview) - 5} more records")

                # Show warnings and suggestions
                warnings = result.get("warnings")
                if warnings:
                    for warning in warnings:
                        st.warning(f"⚠️ {warning}")

                suggestions = result.get("suggestions")
                if suggestions:
                    st.info("💡 **Suggestions:**")
                    for suggestion in suggestions:
                        st.info(f"   • {suggestion}")

            else:
                # Show standard chat info for other endpoints
                if result.get("documents_loaded"):
                    st.info(f"📚 Loaded {len(result['documents_loaded'])} documents")
                if result.get("document_selection_message"):
                    st.info(f"🎯 {result['document_selection_message']}")

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
    st.info(
        "This mode shows real-time progress updates while your request is being processed. Note: Due to Streamlit limitations, the progress updates are simulated rather than true real-time streaming."
    )

    # Endpoint selection for streaming
    stream_endpoint = st.radio(
        "Choose streaming endpoint:",
        ["Traditional Chat (S3)", "Fast Chat (Vector DB)", "Financial Chat V2 (AI)"],
        horizontal=True,
    )

    if "Financial" in stream_endpoint:
        endpoint_path = "/fast_chat_v2/stream"  # Now has streaming version
    elif "Traditional" in stream_endpoint:
        endpoint_path = "/chat/stream"
    else:
        endpoint_path = "/fast_chat/stream"

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
            if "Financial" in stream_endpoint:
                status_text.text("🧠 Parsing financial query with AI...")
            elif "Fast" in stream_endpoint:
                status_text.text("🔍 Searching vector database...")
            else:
                status_text.text("☁️ Loading documents from S3...")
            time.sleep(1.0)

            # Update progress - Step 4
            progress_bar.progress(85)
            if "Financial" in stream_endpoint:
                status_text.text("📊 Querying financial database...")
            else:
                status_text.text("🤖 Generating AI response...")
            time.sleep(0.5)

            # Build request payload based on endpoint
            if "Financial" in stream_endpoint:
                payload = {
                    "query": prompt,
                    "memory_enabled": memory_enabled,
                    "conversation_history": (
                        st.session_state.stream_chat_history if memory_enabled else None
                    ),
                }
                api_endpoint = endpoint_path  # Now uses streaming endpoint
            else:
                payload = {
                    "query": prompt,
                    "conversation_history": (
                        st.session_state.stream_chat_history if memory_enabled else None
                    ),
                    "auto_load_documents": auto_load,
                    "memory_enabled": memory_enabled,
                }
                api_endpoint = endpoint_path  # Use streaming endpoint

            # For streaming endpoints, we need to handle the stream differently
            if "/stream" in api_endpoint:
                # Make streaming API call
                response = requests.post(
                    f"{BASE_URL}{api_endpoint}",
                    json=payload,
                    timeout=300,
                    stream=True,
                )
                response.raise_for_status()

                # Process streaming response
                current_event = ""
                buffer = ""

                for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                    if chunk:
                        buffer += chunk
                        lines = buffer.split("\n")

                        # Process complete lines, keep incomplete line in buffer
                        buffer = lines[-1] if not lines[-1].endswith("\n") else ""

                        for line in lines[:-1] if not lines[-1].endswith("\n") else lines:
                            line = line.strip()

                            if line.startswith("event: "):
                                current_event = line[7:]
                            elif line.startswith("data: "):
                                data_str = line[6:]
                                if data_str and data_str != "{}":  # Ignore empty heartbeats
                                    try:
                                        data = json.loads(data_str)
                                        if current_event == "progress":
                                            progress_bar.progress(data.get("progress", 0) / 100)
                                            status_text.text(data.get("message", ""))
                                            if data.get("details"):
                                                details_text.text(f"Details: {data.get('details')}")
                                        elif current_event == "result":
                                            result = data
                                            break
                                        elif current_event == "error":
                                            st.error(f"Error: {data.get('error', 'Unknown error')}")
                                            break
                                    except json.JSONDecodeError:
                                        pass
            else:
                # Make regular API call
                response = requests.post(
                    f"{BASE_URL}{api_endpoint}",
                    json=payload,
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

            # Show additional info based on endpoint type
            if "Financial" in stream_endpoint:
                # Show financial-specific information
                data_found = result.get("data_found", False)
                record_count = result.get("record_count", 0)

                if data_found:
                    details_text.success(f"📊 Found {record_count} financial records")
                else:
                    details_text.warning("📊 No financial data found")

                # Show warnings and suggestions in the UI
                warnings = result.get("warnings")
                if warnings:
                    for warning in warnings:
                        st.warning(f"⚠️ {warning}")

                suggestions = result.get("suggestions")
                if suggestions:
                    with st.expander("💡 Suggestions"):
                        for suggestion in suggestions:
                            st.write(f"• {suggestion}")
            else:
                # Show standard chat info for other endpoints
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

    metadata_default = """{
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
    query_text = st.text_input(
        "Query Text", placeholder="e.g. What is Company X's revenue in 2023?"
    )
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
    query_text = st.text_input(
        "Query Text", placeholder="e.g. What is Company X's revenue in 2023?"
    )
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

# -------------------
# Financial Metadata Mode
# -------------------
elif mode == "Financial Metadata":
    st.subheader("📊 Financial Data Metadata Explorer")
    st.info(
        "Explore the available financial data including companies, symbols, years, and metrics."
    )

    if st.button("🔍 Load Financial Metadata"):
        start_time = time.perf_counter()
        try:
            res = requests.get(f"{BASE_URL}/financial/metadata", timeout=60)
            res.raise_for_status()
            result = res.json()

            if result.get("status") == "success":
                metadata = result.get("metadata", {})

                # Display overview
                st.success("✅ Financial metadata loaded successfully!")

                # Create tabs for different metadata sections
                tab1, tab2, tab3, tab4, tab5 = st.tabs(
                    ["📊 Overview", "🏢 Companies", "📈 Symbols", "📅 Years", "📋 Metrics"]
                )

                with tab1:
                    st.write("### Data Overview")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        companies = metadata.get("companies", [])
                        if isinstance(companies, dict):
                            total_companies = companies.get(
                                "total_count", len(companies.get("sample", []))
                            )
                        else:
                            total_companies = len(companies)
                        st.metric("Total Companies", total_companies)

                    with col2:
                        symbols = metadata.get("symbols", [])
                        if isinstance(symbols, dict):
                            total_symbols = symbols.get(
                                "total_count", len(symbols.get("sample", []))
                            )
                        else:
                            total_symbols = len(symbols)
                        st.metric("Total Symbols", total_symbols)

                    with col3:
                        years = metadata.get("years", [])
                        st.metric("Years Available", len(years) if isinstance(years, list) else 0)

                    with col4:
                        items = metadata.get("standard_items", [])
                        if isinstance(items, dict):
                            total_items = items.get("total_count", len(items.get("sample", [])))
                        else:
                            total_items = len(items)
                        st.metric("Financial Metrics", total_items)

                with tab2:
                    st.write("### Available Companies")
                    companies = metadata.get("companies", [])
                    if isinstance(companies, dict):
                        st.info(companies.get("note", ""))
                        companies_list = companies.get("sample", [])
                    else:
                        companies_list = companies

                    if companies_list:
                        for i, company in enumerate(companies_list, 1):
                            st.write(f"{i}. {company}")
                    else:
                        st.warning("No companies found")

                with tab3:
                    st.write("### Stock Symbols")
                    symbols = metadata.get("symbols", [])
                    if isinstance(symbols, dict):
                        st.info(symbols.get("note", ""))
                        symbols_list = symbols.get("sample", [])
                    else:
                        symbols_list = symbols

                    if symbols_list:
                        # Display symbols in a nice grid
                        cols = st.columns(4)
                        for i, symbol in enumerate(symbols_list):
                            with cols[i % 4]:
                                st.code(symbol)
                    else:
                        st.warning("No symbols found")

                with tab4:
                    st.write("### Available Years")
                    years = metadata.get("years", [])
                    if years:
                        # Display years in a nice grid
                        cols = st.columns(6)
                        for i, year in enumerate(years):
                            with cols[i % 6]:
                                st.code(year)
                    else:
                        st.warning("No years found")

                with tab5:
                    st.write("### Financial Metrics")
                    items = metadata.get("standard_items", [])
                    if isinstance(items, dict):
                        st.info(items.get("note", ""))
                        items_list = items.get("sample", [])
                    else:
                        items_list = items

                    if items_list:
                        for i, item in enumerate(items_list, 1):
                            st.write(f"{i}. {item}")
                    else:
                        st.warning("No financial metrics found")

            else:
                st.error("❌ Failed to load financial metadata")

        except requests.exceptions.RequestException as req_err:
            st.error(f"❌ Request failed: {req_err}")
        except Exception as e:
            st.error(f"❌ Error: {e}")
        finally:
            st.session_state.last_exec_time = time.perf_counter() - start_time
