# Archived Scripts

Manual test tooling that targeted endpoints/routes no longer registered in
`fastapi_app/app/main.py`. Kept for reference, not runnable as-is.

- **test_client.py** - Streamlit manual test UI. Posts to `/chroma/update`,
  `/chroma/query`, `/chroma/meta/query` - none of these routes have ever
  existed in `main.py` (no Chroma vector store is wired into the app).
- **streaming_test_client.html** - Browser-based manual streaming test page.
  Its "Deep Research" mode drove `/chat/stream/v0`'s SSE/async-job flow
  (`polling_url`/`/jobs/{id}`), which was archived on 2026-08-14 along with
  the rest of the async-job infrastructure (see
  `fastapi_app/app/_archive/README.md`). Its "Chat" mode (`/fast_chat_v2`) is
  still live.
