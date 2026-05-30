# _archive

This directory holds code that is **not active** but is preserved for reference.

Files here are intentionally excluded from the running application.

## Contents

| File | Original path | What it is | Status |
|---|---|---|---|
| `agent_orchestrator.py` | `app/agent.py` | `AgentOrchestrator` — a 3-phase agent (clarification → routing → synthesis) with both Google Search grounding *and* JSE financial-DB tool calling via Gemini function calling. | Restored in R10 (ATS-334). `app/agent.py` is the live version. This copy is kept for historical reference. |

## Restoration (R10 — ATS-334)

Completed. `AgentOrchestrator` is live at `app/agent.py` and wired into `/chat/stream`.
Changes made:
- Removed `ARCHIVED` header from module docstring
- Split `model_name` into `flash_model` ("gemini-2.5-flash", phases 1–2) and `pro_model` ("gemini-2.5-pro", phase 3)
- Fixed conversation-history bug (now appends new turn before returning)
- Instantiated in `lifespan` on `app.state.agent_orchestrator`
