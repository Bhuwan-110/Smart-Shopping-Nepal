"""
SmartShopping Nepal - Main Application Entry Point
Phases 1, 2 & 3: Collection, Intelligent Matching & Advanced Price Comparison
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
    print("=" * 85)
    print("   SMARTSHOPPING NEPAL - INTELLIGENT PRODUCT MATCHING & PRICE COMPARISON ENGINE  ")
    print("=" * 85)

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
    print(f"      - Records Collected : {audit_log['total_records_collected']}")
    print(f"      - Clean Ingested    : {audit_log['final_clean_records']}")

    # 2. Execute Data Processing & Intelligent Matching (Phase 3)
    print("\n[Step 3/5] Running Intelligent Product Matching & Verification Pipeline...")
    pipeline = DataProcessingPipeline(CURRENT_DIR)
    quality_report = pipeline.execute(use_collected_data=True, inject_demo_listings=False)

    print("\n[Step 4/5] Pipeline & Matching Audit Summary:")
    print(f"      - Total Records Ingested   : {quality_report['total_records_processed']}")
    print(f"      - Unique Canonical Keys    : {quality_report['unique_product_keys']}")
    print(f"      - Multi-Store Deals Found  : {quality_report['matching_metrics']['multi_store_deals_count']}")
    print(f"      - Verified Real Listings   : {quality_report['record_type_breakdown'].get('real', 0)}")
    print(f"      - Sample Catalog Templates : {quality_report['record_type_breakdown'].get('sample', 0)}")

    # 3. Display Advanced Price Comparison Summary
    print("\n[Step 5/5] Multi-Store Price Comparison & Savings Summary (Tier: Real Listings):")
    summary_path = os.path.join(CURRENT_DIR, "data", "processed", "price_comparison_summary.csv")
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        real_summary = summary_df[summary_df["record_type"] == "real"]
        display_cols = [
            "product_name", "cheapest_store", "lowest_total_price",
            "highest_total_price", "average_total_price", "saving_npr", "saving_percent", "number_of_stores"
        ]
        available_cols = [c for c in display_cols if c in real_summary.columns]
        print("\n" + "-" * 85)
        print(real_summary[available_cols].to_string(index=False))
        print("-" * 85)

    print("\nPer-Store Breakdown for Matching Multi-Store Products:")
    detailed_path = os.path.join(CURRENT_DIR, "data", "processed", "detailed_store_comparison.csv")
    if os.path.exists(detailed_path):
        detailed_df = pd.read_csv(detailed_path)
        real_detailed = detailed_df[detailed_df["record_type"] == "real"]
        d_cols = ["product_name", "platform", "price_npr", "shipping_cost_npr", "total_price_npr", "rating", "availability_status"]
        print("-" * 85)
        print(real_detailed[d_cols].to_string(index=False))
        print("-" * 85)

    print("\nGenerated Processed Artifacts in data/processed/:")
    for artifact in quality_report["artifacts_generated"]:
        rel_path = os.path.relpath(artifact, CURRENT_DIR)
        size_kb = round(os.path.getsize(artifact) / 1024, 2)
        print(f"      ✓ {rel_path} ({size_kb} KB)")

    print("\n" + "=" * 85)
    print("      PHASE 3 INTELLIGENT MATCHING & PRICE COMPARISON COMPLETED SUCCESSFULLY      ")
    print("=" * 85)


if __name__ == "__main__":
    run_main()
