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
1. Move `agent_orchestrator.py` back to `app/agent.py`
2. Re-add `from app.agent import AgentOrchestrator` in `main.py`
3. Instantiate it in the lifespan and route `/chat/stream` through it
