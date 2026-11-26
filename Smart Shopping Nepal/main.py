"""
SmartShopping Nepal - Main Application Entry Point
Phases 1, 2, 3 & 5: Collection, Intelligent Matching, Price Comparison & Django Web Integration
"""

import os
import sys
import json
import argparse
import pandas as pd

# Add current directory to Python module search path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.pipeline import DataProcessingPipeline
from src.collectors.manager import CollectionManager


def run_pipeline():
    print("=" * 85)
    print("   SMARTSHOPPING NEPAL - INTELLIGENT PRODUCT MATCHING & WEB PLATFORM ENGINE  ")
    print("=" * 85)

    # 1. Execute Authorized Data Collection (Phase 2)
    print("\n[Step 1/6] Initializing Authorized Platform Connectors...")
    collector = CollectionManager(CURRENT_DIR)
    collector.load_authorized_connectors()
    print(f"      Loaded {len(collector.connectors)} authorized connectors:")
    for conn in collector.connectors:
        print(f"      - {conn.platform_name} (Source: {conn.data_source}, Approved: {conn.is_approved})")

    print("\n[Step 2/6] Running Authorized Product Data Collection...")
    collection_res = collector.run_collection()
    audit_log = collection_res["audit_log"]
    print(f"      - Records Collected : {audit_log['total_records_collected']}")
    print(f"      - Clean Ingested    : {audit_log['final_clean_records']}")

    # 2. Execute Data Processing & Intelligent Matching (Phase 3)
    print("\n[Step 3/6] Running Intelligent Product Matching & Verification Pipeline...")
    pipeline = DataProcessingPipeline(CURRENT_DIR)
    quality_report = pipeline.execute(use_collected_data=True, inject_demo_listings=False)

    print("\n[Step 4/6] Pipeline & Matching Audit Summary:")
    print(f"      - Total Records Ingested   : {quality_report['total_records_processed']}")
    print(f"      - Unique Canonical Keys    : {quality_report['unique_product_keys']}")
    print(f"      - Multi-Store Deals Found  : {quality_report['matching_metrics']['multi_store_deals_count']}")
    print(f"      - Verified Real Listings   : {quality_report['record_type_breakdown'].get('real', 0)}")
    print(f"      - Sample Catalog Templates : {quality_report['record_type_breakdown'].get('sample', 0)}")

    # 3. Display Advanced Price Comparison Summary
    print("\n[Step 5/6] Multi-Store Price Comparison & Savings Summary (Tier: Real Listings):")
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

    # 4. Phase 5: Django Web Application Database Synchronization
    print("\n[Step 6/6] Synchronizing Processed Pipeline Data into Django Database...")
    try:
        if 'DJANGO_SETTINGS_MODULE' not in os.environ:
            os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smart_shopping_web.settings')
        import django
        django.setup()
        from django.core.management import call_command
        call_command('sync_pipeline')
    except Exception as e:
        print(f"      Notice: Django sync skipped or error occurred: {e}")

    print("\nGenerated Processed Artifacts in data/processed/:")
    for artifact in quality_report["artifacts_generated"]:
        rel_path = os.path.relpath(artifact, CURRENT_DIR)
        size_kb = round(os.path.getsize(artifact) / 1024, 2)
        print(f"      ✓ {rel_path} ({size_kb} KB)")

    print("\n" + "=" * 85)
    print("      PHASE 5 DJANGO WEBSITE & PIPELINE INTEGRATION READY!                      ")
    print("      To launch web application: python manage.py runserver                     ")
    print("      Available at: http://127.0.0.1:8000/                                      ")
    print("=" * 85)


def run_server():
    """Runs Django development server."""
    os.system(f"{sys.executable} manage.py runserver")


def run_streamlit():
    """Runs Streamlit Research & Data Analytics Dashboard."""
    streamlit_cmd = os.path.join(CURRENT_DIR, ".venv", "Scripts", "streamlit")
    if not os.path.exists(streamlit_cmd) and not os.path.exists(streamlit_cmd + ".exe"):
        streamlit_cmd = "streamlit"
    os.system(f"{streamlit_cmd} run streamlit_app.py")


def main():
    parser = argparse.ArgumentParser(description="Smart Shopping Nepal Controller")
    parser.add_argument("--serve", "-s", action="store_true", help="Launch the Django web application")
    parser.add_argument("--streamlit", "-d", action="store_true", help="Launch the Streamlit research dashboard")
    args = parser.parse_args()

    if args.serve:
        run_server()
    elif args.streamlit:
        run_streamlit()
    else:
        run_pipeline()


if __name__ == "__main__":
    main()
