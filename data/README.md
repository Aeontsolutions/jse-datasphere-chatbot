# Data Files

This directory contains local data files used for development and testing.

## Files

- **metadata_*.json** - Document metadata files with S3 paths and document info
- **symbol_and_company.csv** - JSE company symbol mappings

## Note

These files are git-ignored and should not be committed to version control. They are typically:
- Generated locally by metadata rebuild scripts
- Downloaded from S3 for local development
- Large files (>2MB) that belong in S3 or other object storage

## Production Data

Production metadata is stored in S3:
- Bucket: `jse-renamed-docs-copy`
- Key: `metadata_2025-11-26.json` (or latest version)
