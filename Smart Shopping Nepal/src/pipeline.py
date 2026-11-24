"""
Smart Shopping Nepal - End-to-End Data Processing Pipeline Orchestrator.

Integrates Phase 1 & Phase 2:
1. Ingests raw catalog definitions (sample_products.csv) & compliance registry (source_registry.csv).
2. Executes authorized data collection via CollectionManager (Manual verified imports, partner feeds, APIs).
3. Standardizes text, brands, categories, models, and variants.
4. Normalizes prices and shipping costs to NPR (preserving NaNs, never blank as 0).
5. Validates URLs, images, prices, ratings, and source platforms.
6. Generates consistent, backward-compatible product keys.
7. Distinguishes record tiers ('sample', 'demo', 'verified', 'real').
8. Audits duplicates and cross-store availability.
9. Performs multi-store landed cost comparison and savings analysis.
10. Prepares ML features for recommendation models.
11. Exports all processed outputs to data/processed/ and audit reports to reports/.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from src.loader import load_raw_catalog, load_source_registry, save_csv
from src.cleaner import clean_product_dataframe
from src.keygen import add_product_keys
from src.validator import validate_listings_dataframe
from src.deduplicator import identify_duplicates
from src.comparator import generate_comparison_summary, calculate_total_price
from src.ml_preparer import prepare_ml_features
from src.collectors.manager import CollectionManager


class DataProcessingPipeline:
    """
    Orchestrates ingestion, collection, cleaning, validation, deduplication,
    comparison, and ML preparation for Smart Shopping Nepal.
    """

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.base_dir = base_dir

        self.raw_products_path = os.path.join(self.base_dir, "data", "raw", "sample_products.csv")
        self.raw_sources_path = os.path.join(self.base_dir, "data", "raw", "source_registry.csv")
        self.processed_dir = os.path.join(self.base_dir, "data", "processed")
        self.reports_dir = os.path.join(self.base_dir, "reports")

    def execute(
        self,
        use_collected_data: bool = True,
        inject_demo_listings: bool = False
    ) -> Dict[str, Any]:
        """
        Executes the end-to-end data processing pipeline.
        - If use_collected_data=True, runs CollectionManager to ingest verified platform feeds.
        - If inject_demo_listings=True, attaches demo store listings (for offline demo mode).
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

        # 1. Load Raw Source Registry & Catalog
        source_registry = load_source_registry(self.raw_sources_path)
        registered_platforms = set(source_registry["platform"].dropna().unique())
        raw_catalog = load_raw_catalog(self.raw_products_path)

        listings_to_process = []

        # 2. Run Authorized Data Collection (Phase 2)
        collection_summary = {}
        if use_collected_data:
            collector = CollectionManager(self.base_dir)
            collector.load_authorized_connectors()
            collection_res = collector.run_collection()
            collection_summary = collection_res["audit_log"]
            collected_df = collection_res["collected_df"]
            if not collected_df.empty:
                listings_to_process.append(collected_df)

        # 3. Include Canonical Catalog (tagged as 'sample')
        if not raw_catalog.empty:
            cleaned_raw = raw_catalog.copy()
            cleaned_raw["record_type"] = "sample"
            listings_to_process.append(cleaned_raw)

        # 4. Optional Demo Listings
        if inject_demo_listings:
            demo_stores = ["Demo Store A", "Demo Store B", "Demo Store C"]
            demo_prices = [45000.0, 85000.0, 3500.0]
            demo_shipping = [0.0, 200.0, 100.0]
            demo_ratings = [4.5, 4.2, 4.0]
            demo_df = raw_catalog.copy()
            for i in range(min(len(demo_df), len(demo_stores))):
                demo_df.loc[i, "platform"] = demo_stores[i]
                demo_df.loc[i, "price_npr"] = demo_prices[i]
                demo_df.loc[i, "shipping_cost_npr"] = demo_shipping[i]
                demo_df.loc[i, "rating"] = demo_ratings[i]
                demo_df.loc[i, "record_type"] = "demo"
            listings_to_process.append(demo_df)

        # Combine all listings
        if listings_to_process:
            combined_df = pd.concat(listings_to_process, ignore_index=True)
        else:
            combined_df = raw_catalog.copy()

        # 5. Clean & Standardize
        cleaned_df = clean_product_dataframe(combined_df)

        # 6. Generate Consistent Product Identifiers
        cleaned_df = add_product_keys(cleaned_df)

        # 7. Validate URLs, Prices, Ratings, and Source Platforms
        all_known_platforms = registered_platforms.union(
            set(cleaned_df["platform"].dropna().unique())
        )
        validated_df = validate_listings_dataframe(cleaned_df, all_known_platforms)

        # 8. Duplicate Detection & Record Auditing
        flagged_df, duplicate_audit = identify_duplicates(validated_df)

        # 9. Multi-Store Price Comparison & Savings Calculation
        # Only evaluate priced listings for comparison
        summary_df = generate_comparison_summary(flagged_df)

        # 10. Machine Learning Feature Preparation
        ml_features_df = prepare_ml_features(flagged_df)

        # 11. Save Processed Artifacts
        cleaned_output_path = os.path.join(self.processed_dir, "cleaned_products.csv")
        summary_output_path = os.path.join(self.processed_dir, "price_comparison_summary.csv")
        ml_output_path = os.path.join(self.processed_dir, "products_ml_features.csv")
        report_output_path = os.path.join(self.reports_dir, "data_quality_report.json")

        save_csv(flagged_df, cleaned_output_path)
        save_csv(summary_df, summary_output_path)
        save_csv(ml_features_df, ml_output_path)

        # Compile Quality Report
        quality_report = {
            "status": "SUCCESS",
            "total_records_processed": len(flagged_df),
            "unique_product_keys": int(flagged_df["product_key"].nunique()),
            "record_type_breakdown": flagged_df["record_type"].value_counts().to_dict(),
            "missing_values": {
                "price_npr": int(flagged_df["price_npr"].isna().sum()),
                "shipping_cost_npr": int(flagged_df["shipping_cost_npr"].isna().sum()),
                "rating": int(flagged_df["rating"].isna().sum()),
                "review_count": int(flagged_df["review_count"].isna().sum()),
                "product_url": int(flagged_df["product_url"].isna().sum())
            },
            "validation_metrics": {
                "valid_urls_count": int(flagged_df["is_valid_url"].sum()),
                "valid_prices_count": int(flagged_df["is_valid_price"].sum()),
                "registered_platforms_count": int(flagged_df["is_registered_platform"].sum())
            },
            "duplicate_audit": duplicate_audit,
            "collection_audit": collection_summary,
            "artifacts_generated": [
                cleaned_output_path,
                summary_output_path,
                ml_output_path,
                report_output_path
            ]
        }

        with open(report_output_path, "w", encoding="utf-8") as f:
            json.dump(quality_report, f, indent=4)

        return quality_report
