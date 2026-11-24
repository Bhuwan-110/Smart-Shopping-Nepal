"""
SmartShopping Nepal - Main Application Entry Point
Phase 1: Data Processing Pipeline & Price Comparison Engine
"""

import os
import sys
import json
import pandas as pd

# Add current directory to Python module search path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.pipeline import DataProcessingPipeline


def run_main():
    print("=" * 70)
    print("      SMARTSHOPPING NEPAL - STRUCTURED DATA PROCESSING PIPELINE     ")
    print("=" * 70)

    pipeline = DataProcessingPipeline(CURRENT_DIR)
    print("\n[Step 1/4] Executing Data Ingestion, Cleaning & Validation Pipeline...")
    quality_report = pipeline.execute(inject_demo_listings=True)

    print("\n[Step 2/4] Pipeline Data Quality & Validation Summary:")
    print(f"      - Total Listings Processed : {quality_report['total_records_processed']}")
    print(f"      - Unique Products (Keys)   : {quality_report['unique_product_keys']}")
    print(f"      - Data Tiers Breakdown     : {quality_report['record_type_breakdown']}")
    print(f"      - Validated URLs           : {quality_report['validation_metrics']['valid_urls_count']}")
    print(f"      - Validated Prices         : {quality_report['validation_metrics']['valid_prices_count']}")
    print(f"      - Registered Platforms     : {quality_report['validation_metrics']['registered_platforms_count']}")
    print(f"      - Duplicate Listings Found : {quality_report['duplicate_audit']['duplicate_count']}")
    print(f"      - Multi-Store Products     : {quality_report['duplicate_audit']['multi_store_products_count']}")

    print("\n[Step 3/4] Multi-Store Price Comparison & Savings Summary:")
    summary_path = os.path.join(CURRENT_DIR, "data", "processed", "price_comparison_summary.csv")
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        print("\n" + "-" * 70)
        print(summary_df.to_string(index=False))
        print("-" * 70)

    print("\n[Step 4/4] Generated Processed Artifacts in data/processed/:")
    for artifact in quality_report["artifacts_generated"]:
        rel_path = os.path.relpath(artifact, CURRENT_DIR)
        size_kb = round(os.path.getsize(artifact) / 1024, 2)
        print(f"      ✓ {rel_path} ({size_kb} KB)")

    print("\n" + "=" * 70)
    print("            PHASE 1 DATA PROCESSING COMPLETED SUCCESSFULLY         ")
    print("=" * 70)


if __name__ == "__main__":
    run_main()
