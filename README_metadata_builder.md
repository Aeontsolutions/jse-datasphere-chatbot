# Metadata Builder for JSE Documents

This script rebuilds the `metadata.json` file by scanning your S3 bucket and extracting metadata from the document paths.

## Files

- `rebuild_metadata.py` - Main script to rebuild metadata from S3
- `test_metadata_builder.py` - Test script to validate parsing logic
- `README_metadata_builder.md` - This documentation

## Prerequisites

1. **AWS Credentials**: Ensure your AWS credentials are configured
   - Via AWS CLI: `aws configure`
   - Via environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
   - Via IAM role (if running on EC2)

2. **Required Permissions**: Your AWS credentials need:
   - `s3:ListBucket` on the bucket
   - `s3:GetObject` on the bucket contents
   - `s3:PutObject` to upload the metadata file

3. **Python Dependencies**: All dependencies are already in your `requirements.txt`

## Usage

### Basic Usage
```bash
# Rebuild metadata and upload to S3
python rebuild_metadata.py

# Rebuild metadata locally only (no S3 upload)
python rebuild_metadata.py --no-upload

# Use a different output filename
python rebuild_metadata.py --output my_metadata.json
```

### Advanced Usage
```bash
# Use a different bucket
python rebuild_metadata.py --bucket my-other-bucket

# Use a different S3 prefix
python rebuild_metadata.py --prefix documents/organized/

# Dry run to see what would be processed
python rebuild_metadata.py --dry-run
```

### Testing
```bash
# Test the parsing logic
python test_metadata_builder.py
```

## How It Works

1. **S3 Scanning**: Lists all PDF files in the S3 bucket under the specified prefix
2. **Path Parsing**: Extracts metadata from the S3 path structure:
   ```
   organized/{company_code}/{document_type}/{year}/{filename}
   ```
3. **Company Name Extraction**: Parses the filename to extract the full company name
4. **Metadata Generation**: Creates the JSON structure grouped by company name
5. **Output**: Saves locally and optionally uploads to S3

## Expected S3 Structure

The script expects your S3 bucket to follow this structure:
```
s3://bucket-name/organized/
├── COMPANY_CODE_1/
│   ├── audited_financial_statements/
│   │   ├── 2015/
│   │   │   └── company_name-COMPANY_CODE_1-audited_financial_statements:date.pdf
│   │   └── 2016/
│   └── unaudited_financial_statements/
├── COMPANY_CODE_2/
│   └── ...
```

## Output Format

The generated `metadata.json` follows this structure:
```json
{
  "company name": [
    {
      "document_link": "s3://bucket/path/to/document.pdf",
      "filename": "document.pdf",
      "document_type": "audited financial statements",
      "period": "2015"
    }
  ]
}
```

## Error Handling

- **AWS Errors**: The script handles missing credentials, bucket access issues, etc.
- **Parsing Errors**: Documents that don't follow the expected naming pattern are logged and skipped
- **Upload Failures**: If S3 upload fails, the local file is still saved

## Logging

The script provides detailed logging to help you monitor progress:
- Connection status
- Number of documents found and processed
- Parsing errors and warnings
- Upload status

## Customization

You can modify the `MetadataBuilder` class to:
- Handle different S3 path structures
- Extract additional metadata fields
- Apply different sorting or filtering logic
- Support different file types beyond PDFs
