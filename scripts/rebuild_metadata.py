#!/usr/bin/env python3
"""
Script to rebuild metadata.json from S3 bucket structure.

This script will:
1. List all documents in the S3 bucket
2. Parse document paths to extract metadata
3. Create a complete metadata.json file
4. Upload the metadata.json back to S3
"""

import boto3
import json
import os
import csv
from typing import Dict, List, Optional
from collections import defaultdict
from botocore.exceptions import ClientError, NoCredentialsError
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MetadataBuilder:
    def __init__(
        self,
        bucket_name: str = "jse-renamed-docs-copy",
        base_prefix: str = "organized/",
        company_mapping_file: str = "symbol_and_company.csv",
    ):
        """
        Initialize the MetadataBuilder.

        Args:
            bucket_name: Name of the S3 bucket
            base_prefix: Base prefix path in the bucket
            company_mapping_file: Path to CSV file with symbol-to-company mapping
        """
        self.bucket_name = bucket_name
        self.base_prefix = base_prefix
        self.company_mapping_file = company_mapping_file
        self.s3_client = None
        self.metadata = defaultdict(list)
        self.symbol_to_company = {}

    def initialize_s3_client(self):
        """Initialize S3 client with error handling."""
        try:
            self.s3_client = boto3.client("s3")
            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Successfully connected to S3 bucket: {self.bucket_name}")
        except NoCredentialsError:
            logger.error(
                "AWS credentials not found. Please configure your credentials."
            )
            raise
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                logger.error(f"Bucket {self.bucket_name} not found")
            else:
                logger.error(f"Error connecting to S3: {e}")
            raise

    def load_company_mapping(self):
        """Load the symbol-to-company mapping from CSV file."""
        try:
            if not os.path.exists(self.company_mapping_file):
                logger.error(
                    f"Company mapping file not found: {self.company_mapping_file}"
                )
                raise FileNotFoundError(
                    f"Company mapping file not found: {self.company_mapping_file}"
                )

            with open(self.company_mapping_file, "r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    symbol = row["Symbol"].strip()
                    company = (
                        row["Company"].strip().lower()
                    )  # Convert to lowercase for consistency
                    self.symbol_to_company[symbol] = company

            logger.info(
                f"Loaded {len(self.symbol_to_company)} company mappings from {self.company_mapping_file}"
            )

        except Exception as e:
            logger.error(f"Error loading company mapping: {e}")
            raise

    def parse_document_path(self, s3_key: str) -> Optional[Dict[str, str]]:
        """
        Parse S3 document path to extract metadata.

        Expected format: organized/{company_code}/{document_type}/{year}/{filename}

        Args:
            s3_key: Full S3 object key

        Returns:
            Dictionary with parsed metadata or None if parsing fails
        """
        try:
            # Remove the base prefix
            if not s3_key.startswith(self.base_prefix):
                return None

            path_parts = s3_key[len(self.base_prefix) :].split("/")

            if len(path_parts) < 4:
                logger.warning(f"Unexpected path structure: {s3_key}")
                return None

            company_code = path_parts[0]
            document_type = path_parts[1].replace(
                "_", " "
            )  # Convert underscores to spaces
            year = path_parts[2]
            filename = path_parts[3]

            # Get company name from mapping
            if company_code in self.symbol_to_company:
                company_name = self.symbol_to_company[company_code]
            else:
                # Fallback: use company code as name (converted to lowercase)
                company_name = company_code.replace("_", " ").lower()
                logger.warning(
                    f"No mapping found for company code '{company_code}', using fallback name: '{company_name}'"
                )

            return {
                "company_name": company_name,
                "company_code": company_code,
                "document_type": document_type,
                "period": year,
                "filename": filename,
                "document_link": f"s3://{self.bucket_name}/{s3_key}",
            }

        except Exception as e:
            logger.warning(f"Error parsing path {s3_key}: {e}")
            return None

    def list_s3_objects(self) -> List[str]:
        """
        List all objects in the S3 bucket with the specified prefix.

        Returns:
            List of S3 object keys
        """
        if not self.s3_client:
            raise RuntimeError("S3 client not initialized")

        objects = []
        paginator = self.s3_client.get_paginator("list_objects_v2")

        try:
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name, Prefix=self.base_prefix
            )

            for page in page_iterator:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        # Only include PDF files
                        if key.endswith(".pdf"):
                            objects.append(key)

            logger.info(f"Found {len(objects)} PDF documents in S3 bucket")
            return objects

        except ClientError as e:
            logger.error(f"Error listing S3 objects: {e}")
            raise

    def build_metadata(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Build the complete metadata dictionary from S3 objects.

        Returns:
            Metadata dictionary grouped by company name
        """
        logger.info("Starting metadata building process...")

        # Get all S3 objects
        s3_objects = self.list_s3_objects()

        # Parse each object and build metadata
        parsed_count = 0
        skipped_count = 0

        for s3_key in s3_objects:
            parsed_data = self.parse_document_path(s3_key)

            if parsed_data:
                company_name = parsed_data["company_name"]

                # Create document metadata
                doc_metadata = {
                    "document_link": parsed_data["document_link"],
                    "filename": parsed_data["filename"],
                    "document_type": parsed_data["document_type"],
                    "period": parsed_data["period"],
                }

                self.metadata[company_name].append(doc_metadata)
                parsed_count += 1
            else:
                skipped_count += 1

        logger.info(f"Successfully parsed {parsed_count} documents")
        logger.info(f"Skipped {skipped_count} documents due to parsing errors")
        logger.info(f"Found {len(self.metadata)} unique companies")

        # Convert defaultdict to regular dict and sort
        result = dict(self.metadata)

        # Sort documents within each company by period and document type
        for company_name in result:
            result[company_name].sort(key=lambda x: (x["period"], x["document_type"]))

        return result

    def save_metadata_locally(
        self, metadata: Dict, filename: str = "metadata_2025-09-14.json"
    ) -> str:
        """
        Save metadata to a local JSON file.

        Args:
            metadata: Metadata dictionary to save
            filename: Local filename to save to

        Returns:
            Path to the saved file
        """
        filepath = os.path.join(os.getcwd(), filename)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"Metadata saved locally to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error saving metadata locally: {e}")
            raise

    def upload_metadata_to_s3(
        self, local_filepath: str, s3_key: str = "metadata_2025-09-14.json"
    ) -> bool:
        """
        Upload the metadata file to S3.

        Args:
            local_filepath: Path to the local metadata file
            s3_key: S3 key where to upload the file

        Returns:
            True if successful, False otherwise
        """
        if not self.s3_client:
            raise RuntimeError("S3 client not initialized")

        try:
            self.s3_client.upload_file(
                local_filepath,
                self.bucket_name,
                s3_key,
                ExtraArgs={"ContentType": "application/json"},
            )

            logger.info(f"Metadata uploaded to S3: s3://{self.bucket_name}/{s3_key}")
            return True

        except Exception as e:
            logger.error(f"Error uploading to S3: {e}")
            return False

    def run(
        self, upload_to_s3: bool = True, local_filename: str = "metadata.json"
    ) -> Dict:
        """
        Run the complete metadata building process.

        Args:
            upload_to_s3: Whether to upload the result to S3
            local_filename: Local filename for the metadata file

        Returns:
            The built metadata dictionary
        """
        try:
            # Load company mapping
            self.load_company_mapping()

            # Initialize S3 client
            self.initialize_s3_client()

            # Build metadata
            metadata = self.build_metadata()

            # Save locally
            local_filepath = self.save_metadata_locally(metadata, local_filename)

            # Upload to S3 if requested
            if upload_to_s3:
                success = self.upload_metadata_to_s3(
                    local_filepath, "metadata_2025-09-14.json"
                )
                if success:
                    logger.info("Metadata building process completed successfully!")
                else:
                    logger.warning("Metadata built locally but S3 upload failed")
            else:
                logger.info("Metadata building process completed (local only)")

            return metadata

        except Exception as e:
            logger.error(f"Error in metadata building process: {e}")
            raise


def main():
    """Main function to run the metadata builder."""
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild metadata.json from S3 bucket")
    parser.add_argument(
        "--bucket", default="jse-renamed-docs-copy", help="S3 bucket name"
    )
    parser.add_argument("--prefix", default="organized/", help="S3 prefix path")
    parser.add_argument(
        "--mapping",
        default="symbol_and_company.csv",
        help="CSV file with symbol-to-company mapping",
    )
    parser.add_argument(
        "--no-upload", action="store_true", help="Don't upload to S3, save locally only"
    )
    parser.add_argument(
        "--output", default="metadata_2025-09-14.json", help="Local output filename"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    # Create metadata builder
    builder = MetadataBuilder(
        bucket_name=args.bucket,
        base_prefix=args.prefix,
        company_mapping_file=args.mapping,
    )

    try:
        if args.dry_run:
            logger.info("DRY RUN MODE - No changes will be made")
            builder.load_company_mapping()
            builder.initialize_s3_client()
            objects = builder.list_s3_objects()
            logger.info(f"Would process {len(objects)} documents")

            # Show sample parsing
            if objects:
                sample_obj = objects[0]
                parsed = builder.parse_document_path(sample_obj)
                logger.info(f"Sample parsing for '{sample_obj}':")
                logger.info(f"  Result: {parsed}")
        else:
            # Run the full process
            metadata = builder.run(
                upload_to_s3=not args.no_upload, local_filename=args.output
            )

            # Print summary
            total_docs = sum(len(docs) for docs in metadata.values())
            logger.info(
                f"Summary: {len(metadata)} companies, {total_docs} total documents"
            )

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Process failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
