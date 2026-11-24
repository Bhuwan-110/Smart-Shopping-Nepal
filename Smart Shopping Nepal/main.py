"""
SmartShopping Nepal - Main Application Entry Point
Phases 1 & 2: Product Data Collection & Structured Processing Pipeline
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
from src.collectors.manager import CollectionManager


def run_main():
    print("=" * 75)
    print("   SMARTSHOPPING NEPAL - AUTHORIZED COLLECTION & COMPARISON PIPELINE    ")
    print("=" * 75)

    # 1. Execute Authorized Data Collection (Phase 2)
    print("\n[Step 1/5] Initializing Authorized Platform Connectors...")
    collector = CollectionManager(CURRENT_DIR)
    collector.load_authorized_connectors()
    print(f"      Loaded {len(collector.connectors)} authorized connectors:")
    for conn in collector.connectors:
        print(f"      - {conn.platform_name} (Source: {conn.data_source}, Approved: {conn.is_approved})")

    print("\n[Step 2/5] Running Authorized Product Data Collection...")
    collection_res = collector.run_collection()
    audit_log = collection_res["audit_log"]
    print(f"      - Records Attempted : {audit_log['total_records_attempted']}")
    print(f"      - Records Collected : {audit_log['total_records_collected']}")
    print(f"      - Failures Tracked  : {audit_log['total_failures_recorded']}")
    print(f"      - Duplicates Flagged: {audit_log['duplicates_removed']}")
    print(f"      - Clean Ingested    : {audit_log['final_clean_records']}")

    # 2. Execute Data Processing Pipeline (Phase 1)
    print("\n[Step 3/5] Executing Data Ingestion, Cleaning & Validation Pipeline...")
    pipeline = DataProcessingPipeline(CURRENT_DIR)
    quality_report = pipeline.execute(use_collected_data=True, inject_demo_listings=False)

    print("\n[Step 4/5] Pipeline Data Quality & Validation Summary:")
    print(f"      - Total Records Ingested   : {quality_report['total_records_processed']}")
    print(f"      - Unique Products (Keys)   : {quality_report['unique_product_keys']}")
    print(f"      - Record Tiers Breakdown   : {quality_report['record_type_breakdown']}")
    print(f"      - Validated Product URLs   : {quality_report['validation_metrics']['valid_urls_count']}")
    print(f"      - Validated Prices (NPR)   : {quality_report['validation_metrics']['valid_prices_count']}")
    print(f"      - Registered Platforms     : {quality_report['validation_metrics']['registered_platforms_count']}")
    print(f"      - Multi-Store Products     : {quality_report['duplicate_audit']['multi_store_products_count']}")

    # 3. Display Comparison Results
    print("\n[Step 5/5] Multi-Store Price Comparison & Savings Summary:")
    summary_path = os.path.join(CURRENT_DIR, "data", "processed", "price_comparison_summary.csv")
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        print("\n" + "-" * 75)
        print(summary_df.to_string(index=False))
        print("-" * 75)

    print("\nGenerated Project Artifacts:")
    for artifact in quality_report["artifacts_generated"]:
        rel_path = os.path.relpath(artifact, CURRENT_DIR)
        size_kb = round(os.path.getsize(artifact) / 1024, 2)
        print(f"      ✓ {rel_path} ({size_kb} KB)")

    audit_path = os.path.join(CURRENT_DIR, "reports", "collection_audit.json")
    if os.path.exists(audit_path):
        rel_audit = os.path.relpath(audit_path, CURRENT_DIR)
        print(f"      ✓ {rel_audit} ({round(os.path.getsize(audit_path) / 1024, 2)} KB)")

    print("\n" + "=" * 75)
    print("               PHASE 2 DATA COLLECTION COMPLETED SUCCESSFULLY           ")
    print("=" * 75)


if __name__ == "__main__":
    run_main()
