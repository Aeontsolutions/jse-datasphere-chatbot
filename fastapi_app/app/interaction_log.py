"""
Permanent interaction logging to BigQuery.

Every call to /fast_chat_v2 and /chat/stream is logged here (success and
failure) as the permanent record for regression/behavior monitoring, chosen
over relying on Langfuse's own (time-limited) trace retention.

Logging failures must never affect the user-facing request: `.log()` never
raises, and callers are expected to fire it via `asyncio.create_task(...)`
rather than awaiting it inline on the request path.
"""

import asyncio
import json
import uuid
from typing import Any, Dict, Optional

from google.cloud import bigquery
from google.oauth2 import service_account

from app.logging_config import get_logger

logger = get_logger(__name__)

# Schema for the `interactions` table. Kept alongside the writer so the
# table-creation script and the writer never drift apart.
INTERACTIONS_SCHEMA = [
    bigquery.SchemaField("interaction_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("request_id", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("environment", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("endpoint", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("query", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("response", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("conversation_history_json", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("memory_enabled", "BOOLEAN", mode="NULLABLE"),
    bigquery.SchemaField("enable_web_search", "BOOLEAN", mode="NULLABLE"),
    bigquery.SchemaField("enable_financial_data", "BOOLEAN", mode="NULLABLE"),
    bigquery.SchemaField("model", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("input_tokens", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("output_tokens", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("total_tokens", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("thinking_tokens", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("cost_usd", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("phase_costs_json", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("latency_ms", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("cache_hit", "BOOLEAN", mode="REQUIRED"),
    bigquery.SchemaField("data_found", "BOOLEAN", mode="NULLABLE"),
    bigquery.SchemaField("record_count", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("success", "BOOLEAN", mode="REQUIRED"),
    bigquery.SchemaField("error_message", "STRING", mode="NULLABLE"),
    # How generation actually ended. `success` only reports whether the request
    # threw, so a MAX_TOKENS stop reads as a clean success without these.
    bigquery.SchemaField("finish_reason", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("truncated", "BOOLEAN", mode="NULLABLE"),
    # Provenance returned to the caller. `source_count` is denormalised on
    # purpose: "what share of answers cited nothing" is the grounding-failure
    # signal, and it should not require an UNNEST to ask.
    bigquery.SchemaField("source_count", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField(
        "sources",
        "RECORD",
        mode="REPEATED",
        fields=[
            bigquery.SchemaField("type", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("title", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("url", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("domain", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("document_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("company", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("year", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("table", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("record_count", "INTEGER", mode="NULLABLE"),
        ],
    ),
]

# Source fields carried into the log. `detail` and `retrieved_at` are
# deliberately excluded: `detail` is model-generated prose with little
# analytical value, and `retrieved_at` is within milliseconds of the row's
# own `timestamp`.
_LOGGED_SOURCE_FIELDS = (
    "type",
    "title",
    "url",
    "domain",
    "document_id",
    "company",
    "year",
    "table",
    "record_count",
)


def _normalize_sources(sources: Optional[list]) -> list:
    """Flatten Source models (or plain dicts) into BigQuery RECORD rows.

    Both shapes reach this function: the fresh request path hands over
    `Source` objects, while a `/chat/stream` cache hit hands over the raw
    dicts read back from Redis.
    """
    if not sources:
        return []
    rows = []
    for source in sources:
        data = source.model_dump() if hasattr(source, "model_dump") else dict(source)
        rows.append({field: data.get(field) for field in _LOGGED_SOURCE_FIELDS})
    return rows


class InteractionLogger:
    """Fire-and-forget writer for the permanent `interactions` BigQuery table."""

    def __init__(
        self,
        bq_client: Optional[bigquery.Client],
        dataset: Optional[str],
        table: str,
        enabled: bool = True,
        environment: Optional[str] = None,
    ):
        self._client = bq_client
        self._table_ref = f"{dataset}.{table}" if bq_client and dataset else None
        self._enabled = bool(enabled and bq_client is not None and self._table_ref)
        self._environment = environment

    @property
    def enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def build_row(
        *,
        endpoint: str,
        query: str,
        response: Optional[str],
        request_id: Optional[str] = None,
        conversation_history: Optional[list] = None,
        memory_enabled: Optional[bool] = None,
        enable_web_search: Optional[bool] = None,
        enable_financial_data: Optional[bool] = None,
        model: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        thinking_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        finish_reason: Optional[str] = None,
        cost_usd: float = 0.0,
        phase_costs: Optional[Any] = None,
        latency_ms: Optional[float] = None,
        cache_hit: bool = False,
        data_found: Optional[bool] = None,
        record_count: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        timestamp: Optional[str] = None,
        sources: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Build a row dict matching INTERACTIONS_SCHEMA. Caller supplies `timestamp`
        (ISO string) since this module avoids datetime.now() for testability."""
        return {
            "interaction_id": uuid.uuid4().hex,
            "request_id": request_id,
            "endpoint": endpoint,
            "timestamp": timestamp,
            "query": query,
            "response": response,
            "conversation_history_json": (
                json.dumps(conversation_history, default=str) if conversation_history else None
            ),
            "memory_enabled": memory_enabled,
            "enable_web_search": enable_web_search,
            "enable_financial_data": enable_financial_data,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            # Prefer the total the API reported: it includes thinking tokens,
            # which input + output alone silently drops (issue #72).
            "total_tokens": (
                total_tokens
                if total_tokens is not None
                else (input_tokens or 0) + (output_tokens or 0)
            ),
            "finish_reason": finish_reason,
            # None means "the model did not tell us", which is not the same as
            # "not truncated" — keep the three states distinguishable.
            "truncated": (finish_reason == "MAX_TOKENS" if finish_reason else None),
            "cost_usd": cost_usd,
            "phase_costs_json": json.dumps(phase_costs, default=str) if phase_costs else None,
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
            "data_found": data_found,
            "record_count": record_count,
            "success": success,
            "error_message": error_message,
            # None (a failed request) and [] (a request that succeeded but
            # cited nothing) are different facts, so they stay distinguishable.
            "source_count": len(sources) if sources is not None else None,
            "sources": _normalize_sources(sources),
        }

    async def log(self, row: Dict[str, Any]) -> None:
        if not self._enabled:
            return
        row = {**row, "environment": self._environment}
        try:
            errors = await asyncio.to_thread(self._client.insert_rows_json, self._table_ref, [row])
            if errors:
                logger.error("interaction_log_insert_errors", extra={"errors": errors})
        except Exception as exc:
            # Never propagate — a logging failure must never affect the user-facing request.
            logger.error("interaction_log_insert_failed", extra={"error": str(exc)})


def _initialize_bigquery_client(project_id: Optional[str]) -> Optional[bigquery.Client]:
    """Build a BigQuery client, mirroring financial_utils.py's auth modes."""
    import os

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    service_account_info = os.getenv("GCP_SERVICE_ACCOUNT_INFO")
    try:
        if credentials_path and os.path.exists(credentials_path):
            return bigquery.Client.from_service_account_json(credentials_path, project=project_id)
        if service_account_info:
            info = json.loads(service_account_info)
            credentials = service_account.Credentials.from_service_account_info(info)
            return bigquery.Client(credentials=credentials, project=project_id)
        return bigquery.Client(project=project_id)
    except Exception as exc:
        logger.error("interaction_log_bigquery_client_init_failed", extra={"error": str(exc)})
        return None


def build_interaction_logger() -> InteractionLogger:
    """Construct an InteractionLogger from the current app config."""
    from app.config import get_config

    cfg = get_config()
    bq_client = None
    if cfg.bigquery.logging_enabled:
        bq_client = _initialize_bigquery_client(cfg.gcp.project_id)
    return InteractionLogger(
        bq_client=bq_client,
        dataset=cfg.bigquery.resolved_interactions_dataset,
        table=cfg.bigquery.interactions_table,
        enabled=cfg.bigquery.logging_enabled,
        environment=cfg.environment,
    )
