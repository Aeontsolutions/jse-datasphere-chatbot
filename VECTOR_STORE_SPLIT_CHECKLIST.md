# Vector-Store Split & Deployment Checklist

_A living document tracking work on decoupling ChromaDB from the FastAPI API, adding durability (EFS) and production-grade deployment._

---

## 🎯 Goal

Separate ChromaDB into its own service so data persists across API deployments, runs on EFS in AWS, and is reachable only by trusted backend services (FastAPI RAG, maintenance tasks).

---

## ✅ Completed

- **Local split**
  - [x] Added dedicated `chroma` service to `docker-compose.yml`.
  - [x] Mounted named volume `chroma_data` for local persistence.
  - [x] Updated `api` service to use `CHROMA_HOST=http://chroma:8000`.
- **Code updates**
  - [x] `init_chroma_client` chooses HTTP vs. on-disk client automatically.
  - [x] Query helper supports missing metadata & `file_type`/`doctype` variations with tiered fallback.
- **Validation**
  - [x] `POST /chroma/update` successfully uploads docs.
  - [x] `POST /chroma/query` returns results (no-filter fallback confirmed).
  - [x] Restarted `api` container → vectors persisted (proof split works).
- **Design**
  - [x] Drafted Copilot manifests: `chroma` Backend Service + EFS, API with Service Connect.
  - [x] Outlined data-migration task and security hardening plan.

---

## 🔜 In-progress / Next

| Task | Owner | Status |
|------|-------|--------|
| Create Copilot `chroma` service (`svc init`) |  | ✅ |
| Add EFS volume to `copilot/chroma/manifest.yml` |  | ✅ |
| Deploy `chroma` to **staging** |  | ✅ |
| Update & deploy `api` manifest to **staging** |  | ✅ |
| Copy local `chroma_data` to new EFS (one-off task) |  | ☐ |
| Smoke-test staging (`/health`, `/fast_chat`) |  | ☐ |
| Move secrets to AWS Secrets Manager (`secrets:`) |  | ☐ |
| Add Cognito/JWT auth middleware to FastAPI |  | ☐ |
| Restrict SG: allow TCP 8000 from API SG only |  | ☐ |
| Enable EFS automatic backups & monitoring |  | ☐ |
| Initialise Copilot pipeline (CI/CD) |  | ☐ |
| Promote `chroma` & `api` to **prod** |  | ☐ |

---

## ✨ Future enhancements

- Bulk document ingest endpoint and async processing.
- SSE / WebSocket streaming chat responses.
- OpenTelemetry tracing across services.
- Prometheus metrics exporter & Grafana dashboards.
- Scheduled vector-store re-index after embedding model upgrades.

---

_Last updated: <!-- CURSOR will insert timestamp on save -->_ 