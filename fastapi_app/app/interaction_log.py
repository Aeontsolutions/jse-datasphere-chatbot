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
    bigquery.SchemaField("cost_usd", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("phase_costs_json", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("latency_ms", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("cache_hit", "BOOLEAN", mode="REQUIRED"),
    bigquery.SchemaField("data_found", "BOOLEAN", mode="NULLABLE"),
    bigquery.SchemaField("record_count", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("success", "BOOLEAN", mode="REQUIRED"),
    bigquery.SchemaField("error_message", "STRING", mode="NULLABLE"),
]


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
        cost_usd: float = 0.0,
        phase_costs: Optional[Any] = None,
        latency_ms: Optional[float] = None,
        cache_hit: bool = False,
        data_found: Optional[bool] = None,
        record_count: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        timestamp: Optional[str] = None,
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
            "total_tokens": (input_tokens or 0) + (output_tokens or 0),
            "cost_usd": cost_usd,
            "phase_costs_json": json.dumps(phase_costs, default=str) if phase_costs else None,
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
            "data_found": data_found,
            "record_count": record_count,
            "success": success,
            "error_message": error_message,
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
