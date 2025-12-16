#!/usr/bin/env python3
"""
Simple script to run the metadata rebuild process with sensible defaults.
"""

from rebuild_metadata import MetadataBuilder
import logging


def main():
    """Run the metadata rebuild with user confirmation."""

    print("JSE Datasphere Metadata Rebuild Tool")
    print("=" * 40)
    print()
    print("This script will:")
    print("1. Scan your S3 bucket for all PDF documents")
    print("2. Extract metadata from the document paths")
    print("3. Generate a new metadata.json file")
    print("4. Upload the file back to S3")
    print()

    # Ask for confirmation
    response = input("Do you want to proceed? (y/N): ").strip().lower()
    if response not in ["y", "yes"]:
        print("Operation cancelled.")
        return 0

    print()
    print("Starting metadata rebuild process...")
    print("-" * 40)

    try:
        # Create and run the metadata builder
        builder = MetadataBuilder()
        metadata = builder.run()

        # Print summary
        total_docs = sum(len(docs) for docs in metadata.values())
        print()
        print("✅ Metadata rebuild completed successfully!")
        print(f"📊 Summary: {len(metadata)} companies, {total_docs} total documents")
        print("📁 Local file saved: metadata.json")
        print("☁️  Uploaded to S3: s3://jse-renamed-docs-copy/metadata.json")

        # Show top 5 companies by document count
        print()
        print("Top 5 companies by document count:")
        sorted_companies = sorted(metadata.items(), key=lambda x: len(x[1]), reverse=True)
        for i, (company, docs) in enumerate(sorted_companies[:5]):
            print(f"  {i+1}. {company}: {len(docs)} documents")

        return 0

    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nFor more detailed error information, run:")
        print("python rebuild_metadata.py --dry-run")
        return 1


if __name__ == "__main__":
    # Set up logging to be less verbose for the simple interface
    logging.basicConfig(level=logging.WARNING)
    exit(main())
