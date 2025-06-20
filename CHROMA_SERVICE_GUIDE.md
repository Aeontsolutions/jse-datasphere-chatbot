# 📚 Chroma Service Guide

This document explains how to check the health of your Chroma vector-store, inspect collections, and debug issues—both **locally** (Docker-Compose) and in the **AWS Copilot** environment.

---
## 1  Service Endpoints

| Environment | Base URL | Notes |
|-------------|----------|-------|
| Local Compose | `http://localhost:8001` | Port `8001` is mapped to container `8000` |
| Copilot (inside VPC) | `http://chroma:8000` | Reachable from other Copilot workloads via **Service Connect** |
| Copilot (laptop) | via `copilot svc proxy` | Creates a local tunnel on port `8000` |

Key REST routes (prefix `/api/v1`):

```text
/heartbeat            → simple liveness ping
/collections          → list collections
/collections/{name}   → details (+ size)
```

---
## 2  Quick Checks

### 2.1 Heartbeat (local)
```bash
curl -s http://localhost:8001/api/v1/heartbeat
```

### 2.2 Heartbeat (Copilot from another service)
```bash
copilot svc exec --name api --env <env> -- \
  curl -s http://chroma:8000/api/v1/heartbeat
```

### 2.3 List collections & sizes (Python)
```python
import chromadb
client = chromadb.HttpClient(host="chroma", port=8000) # change host for local
print("Collections:", client.list_collections())
col = client.get_collection("documents")  # default collection
print("Docs in collection:", col.count())
```
Run interactively:
```bash
copilot svc exec --name api --env <env>
python - <<'PY'
import chromadb
c = chromadb.HttpClient(host="chroma", port=8000)
print("Heartbeat:", c.heartbeat())
print("Collection size:", c.get_collection("documents").count())
PY
```

---
## 3  Uploading & Querying via FastAPI

FastAPI wraps useful routes so you can interact externally through the public ALB:

| Route | Body | Purpose |
|-------|------|---------|
| `POST /chroma/update` | `{documents, metadatas, ids}` | Upsert docs |
| `POST /chroma/query`  | `{query, n_results}`          | Vector search |

Example:
```bash
curl -X POST https://<ALB>/chroma/query \
     -H "Content-Type: application/json" \
     -d '{"query":"hello","n_results":2}'
```

---
## 4  Port-Forward from Laptop

If you prefer local tools without `svc exec`:
```bash
copilot svc proxy --name chroma --env staging --port 8000
# new terminal
curl -s http://localhost:8000/api/v1/collections | jq
```

---
## 5  Seeding Data

Use the provided script to ingest PDF summaries:
```bash
python smoke_test_summarizer.py \
  --bucket jse-renamed-docs \
  --num-files 50 \
  --concurrency 4 \
  --update-url https://<ALB>/chroma/update
```
Or run the script **inside** the VPC (recommended for large batches):
```bash
copilot task run --env staging --app jse-datasphere-chatbot -- \
  python smoke_test_summarizer.py --num-files 500
```

---
## 6  Troubleshooting

1. **`curl` or `python` not found** – Chroma image is distroless; exec into *another* container (e.g. `api`) or use `svc proxy`.
2. **No collection named `summaries`** – default collection is `documents`. Update `get_or_create_collection` or pass a different name in the API payload.
3. **Service unreachable** – ensure `network.connect: true` in both manifests and that the environment has completed deployment.

---
_Last updated: <!-- CURSOR timestamp -->_ 