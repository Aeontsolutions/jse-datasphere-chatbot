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
        client.get_table(table_ref)
        print(f"Table '{table_ref}' already exists. Nothing to do.")
        return
    except NotFound:
        pass

    table = bigquery.Table(table_ref, schema=INTERACTIONS_SCHEMA)
    client.create_table(table)
    print(f"Created table '{table_ref}'.")


if __name__ == "__main__":
    main()
