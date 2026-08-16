#!/usr/bin/env python3
"""
One-time (idempotent) creation of the BigQuery `interactions` table used for
permanent interaction logging by fastapi_app/app/interaction_log.py.

Requires real GCP credentials (GOOGLE_APPLICATION_CREDENTIALS or ADC) — run
this once per environment/dataset before deploying interaction logging.
"""

import argparse
import sys

from google.api_core.exceptions import NotFound
from google.cloud import bigquery

sys.path.insert(0, "fastapi_app")
from app.interaction_log import INTERACTIONS_SCHEMA  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Create the BigQuery `interactions` table if it doesn't already exist."
    )
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--dataset", required=True, help="BigQuery dataset name")
    parser.add_argument("--table", default="interactions", help="Table name. Default: interactions")
    parser.add_argument("--location", default="US", help="BigQuery location. Default: US")

    args = parser.parse_args()

    client = bigquery.Client(project=args.project, location=args.location)
    table_ref = f"{args.project}.{args.dataset}.{args.table}"

    try:
        existing = client.get_table(table_ref)
    except NotFound:
        table = bigquery.Table(table_ref, schema=INTERACTIONS_SCHEMA)
        client.create_table(table)
        print(f"Created table '{table_ref}'.")
        return

    # The table already exists — reconcile it with INTERACTIONS_SCHEMA rather
    # than no-op'ing, so adding a column to the writer doesn't silently drop
    # that field on every insert into an already-deployed environment.
    present = {field.name for field in existing.schema}
    missing = [field for field in INTERACTIONS_SCHEMA if field.name not in present]

    if not missing:
        print(f"Table '{table_ref}' already exists and is up to date. Nothing to do.")
        return

    # Only additive changes are safe to apply automatically. A new column must
    # be NULLABLE: existing rows have no value for it, and BigQuery rejects
    # adding a REQUIRED field to a populated table.
    required = [field.name for field in missing if field.mode == "REQUIRED"]
    if required:
        raise SystemExit(
            f"Refusing to auto-migrate '{table_ref}': new REQUIRED column(s) "
            f"{required} cannot be added to an existing table. Add them as "
            "NULLABLE, or migrate the table manually."
        )

    existing.schema = list(existing.schema) + missing
    client.update_table(existing, ["schema"])
    print(f"Added column(s) to '{table_ref}': {', '.join(f.name for f in missing)}")


if __name__ == "__main__":
    main()
