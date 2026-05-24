# _archive

This directory holds code that is **not active** but is preserved for reference.

Files here are intentionally excluded from the running application. They are kept
because they document non-obvious capabilities (tool declarations, 3-phase pipeline
patterns, financial DB query logic) that may be revived in a future ticket (see R10).

## Contents

| File | Original path | What it is |
|---|---|---|
| `agent_orchestrator.py` | `app/agent.py` | `AgentOrchestrator` — a 3-phase agent (clarification → routing → synthesis) with both Google Search grounding *and* JSE financial-DB tool calling via Gemini function calling. Was imported in `main.py` but never instantiated; all live endpoints use the simpler `AgentV2` instead. |

## Restoring

To wire `AgentOrchestrator` back up (ticket R10):
1. Move `agent_orchestrator.py` back to `app/agent.py` and remove the `ARCHIVED` header from the module docstring
2. Re-add `from app.agent import AgentOrchestrator` in `main.py`
3. In the `lifespan` context manager in `main.py`, instantiate the orchestrator after `financial_manager` is ready and store it on app state:
   ```python
   app.state.agent_orchestrator = AgentOrchestrator(financial_manager=app.state.financial_manager)
   ```
4. In the `/chat/stream` endpoint, swap out `AgentV2` for `app.state.agent_orchestrator` — note that `AgentOrchestrator.run()` returns a full dict (not a streaming generator), so the endpoint response shape will need to be reconciled with the current streaming contract
