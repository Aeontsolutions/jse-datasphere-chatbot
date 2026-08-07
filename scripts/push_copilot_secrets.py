#!/usr/bin/env python3
"""
Push secrets from a .env file to AWS Systems Manager (SSM) Parameter Store
for AWS Copilot services.
"""

import argparse
import os
import sys
import boto3
from dotenv import dotenv_values


def main():
    parser = argparse.ArgumentParser(
        description="Push environment secrets from .env file to AWS SSM Parameter Store for AWS Copilot."
    )
    parser.add_argument(
        "--env",
        default="dev",
        help="Copilot environment name (e.g. dev, test, staging, prod). Default: dev",
    )
    parser.add_argument(
        "--app",
        default="jse-datasphere-chatbot",
        help="Copilot application name. Default: jse-datasphere-chatbot",
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("AWS_PROFILE", "ats-elroy"),
        help="AWS CLI Profile name to use. Default: $AWS_PROFILE or 'ats-elroy'",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        help="AWS Region. Default: us-east-1",
    )
    parser.add_argument(
        "--env-file",
        default="fastapi_app/.env",
        help="Path to the .env file containing secret variables. Default: fastapi_app/.env",
    )

    args = parser.parse_args()

    env_path = os.path.abspath(args.env_file)
    if not os.path.exists(env_path):
        print(f"Error: .env file not found at '{env_path}'", file=sys.stderr)
        sys.exit(1)

    print(f"Reading environment variables from '{env_path}'...")
    env_vars = dotenv_values(env_path)

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    ssm = session.client("ssm")

    secret_keys = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "GOOGLE_API_KEY",
        "GCP_SERVICE_ACCOUNT_INFO",
    ]

    pushed_count = 0
    for key in secret_keys:
        val = env_vars.get(key)
        if not val:
            print(f"⚠️  Skipping {key}: Not found in {env_path}")
            continue

        param_name = f"/copilot/{args.app}/{args.env}/secrets/{key}"
        print(f"Pushing '{key}' -> SSM: {param_name}...")
        try:
            ssm.put_parameter(
                Name=param_name,
                Value=val,
                Type="SecureString",
                Overwrite=True,
            )
            print(f"✅ Successfully pushed {key}")
            pushed_count += 1
        except Exception as e:
            print(f"❌ Failed to push {key}: {e}", file=sys.stderr)

    print(f"\nDone! Pushed {pushed_count} secret(s) to AWS SSM Parameter Store.")


if __name__ == "__main__":
    main()
