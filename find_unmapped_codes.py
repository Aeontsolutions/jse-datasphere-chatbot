#!/usr/bin/env python3
"""
Script to find company codes that are in S3 but not in the mapping CSV.
"""

import csv
from rebuild_metadata import MetadataBuilder

def main():
    # Load existing mapping
    builder = MetadataBuilder()
    builder.load_company_mapping()
    mapped_codes = set(builder.symbol_to_company.keys())
    
    # Get all company codes from S3
    builder.initialize_s3_client()
    s3_objects = builder.list_s3_objects()
    
    s3_codes = set()
    for s3_key in s3_objects:
        if s3_key.startswith("organized/"):
            path_parts = s3_key[len("organized/"):].split('/')
            if len(path_parts) >= 4:
                company_code = path_parts[0]
                s3_codes.add(company_code)
    
    # Find unmapped codes
    unmapped_codes = s3_codes - mapped_codes
    
    print(f"Total company codes in S3: {len(s3_codes)}")
    print(f"Mapped company codes: {len(mapped_codes)}")
    print(f"Unmapped company codes: {len(unmapped_codes)}")
    
    if unmapped_codes:
        print("\nUnmapped company codes:")
        for code in sorted(unmapped_codes):
            print(f"  {code}")
        
        # Create a CSV with the unmapped codes for manual completion
        with open('unmapped_codes.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Company', 'Symbol'])
            for code in sorted(unmapped_codes):
                writer.writerow(['[FILL IN COMPANY NAME]', code])
        
        print(f"\nCreated 'unmapped_codes.csv' with {len(unmapped_codes)} codes to be mapped manually.")
    else:
        print("\nAll company codes are mapped!")

if __name__ == "__main__":
    main()
