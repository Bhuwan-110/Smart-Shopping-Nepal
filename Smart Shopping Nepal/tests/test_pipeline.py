"""
Smart Shopping Nepal - Comprehensive Pipeline Test Suite.
Validates all requirements specified in Phase 1:
1. CSV loading
2. Data cleaning
3. Missing-value handling (No zero prices, no invented ratings)
4. Duplicate detection
5. Product identifier generation
6. Processed CSV generation and raw data preservation
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.loader import load_raw_catalog, load_source_registry, load_csv
from src.cleaner import (
    clean_text,
    standardize_brand,
    standardize_category,
    parse_currency_to_npr,
    parse_shipping_cost_npr,
    parse_rating,
    parse_review_count,
    determine_record_type,
    clean_product_dataframe,
)
from src.keygen import generate_product_key, add_product_keys
from src.validator import validate_url, validate_rating, validate_price
from src.deduplicator import identify_duplicates
from src.comparator import calculate_total_price, compare_product_by_key, generate_comparison_summary
from src.ml_preparer import prepare_ml_features
from src.pipeline import DataProcessingPipeline


class TestSmartShoppingPipeline(unittest.TestCase):

    def setUp(self):
        self.raw_catalog_path = os.path.join(PROJECT_ROOT, "data", "raw", "sample_products.csv")
        self.raw_sources_path = os.path.join(PROJECT_ROOT, "data", "raw", "source_registry.csv")

    # 1. Test CSV Loading
    def test_csv_loading(self):
        """Checks safe ingestion of raw files and preserves schemas."""
        self.assertTrue(os.path.exists(self.raw_catalog_path), "sample_products.csv must exist")
        self.assertTrue(os.path.exists(self.raw_sources_path), "source_registry.csv must exist")

        catalog_df = load_raw_catalog(self.raw_catalog_path)
        self.assertGreaterEqual(len(catalog_df), 3, "Catalog must load at least 3 baseline products")
        self.assertIn("product_name", catalog_df.columns)
        self.assertIn("category", catalog_df.columns)
        self.assertIn("brand", catalog_df.columns)

        sources_df = load_source_registry(self.raw_sources_path)
        self.assertGreaterEqual(len(sources_df), 2, "Source registry must load at least 2 platforms")
        self.assertIn("platform", sources_df.columns)

    # 2. Test Data Cleaning
    def test_data_cleaning(self):
        """Checks text trimming, case normalization, and currency parsing."""
        self.assertEqual(clean_text("  Samsung Galaxy  "), "samsung galaxy")
        self.assertEqual(standardize_brand("samsung"), "Samsung")
        self.assertEqual(standardize_brand("hp"), "HP")
        self.assertEqual(standardize_category("mobile phone"), "Mobile Phone")

        # Currency string handling
        self.assertEqual(parse_currency_to_npr("Rs. 45,000"), 45000.0)
        self.assertEqual(parse_currency_to_npr("NPR 3,500.50"), 3500.5)
        self.assertEqual(parse_currency_to_npr(85200), 85200.0)

        # Shipping cost parsing
        self.assertEqual(parse_shipping_cost_npr("Free"), 0.0)
        self.assertEqual(parse_shipping_cost_npr("0"), 0.0)
        self.assertEqual(parse_shipping_cost_npr("Rs. 150"), 150.0)

    # 3. Test Missing-Value Handling (CRITICAL RULES)
    def test_missing_value_handling(self):
        """
        Enforces strict rules:
        - Blank price is NEVER 0.
        - Blank rating is NEVER invented or defaulted.
        - Total landed price is NaN if price is NaN.
        """
        # Blank prices must return np.nan
        self.assertTrue(np.isnan(parse_currency_to_npr(None)))
        self.assertTrue(np.isnan(parse_currency_to_npr("")))
        self.assertTrue(np.isnan(parse_currency_to_npr("   ")))
        self.assertTrue(np.isnan(parse_currency_to_npr("NaN")))
        self.assertTrue(np.isnan(parse_currency_to_npr(0)))  # 0 price is treated as unpriced

        # Ratings must preserve NaN
        self.assertTrue(np.isnan(parse_rating(None)))
        self.assertTrue(np.isnan(parse_rating("")))
        self.assertTrue(np.isnan(parse_rating("NaN")))
        # Invalid rating out of bounds must be rejected
        self.assertTrue(np.isnan(parse_rating(6.5)))
        self.assertTrue(np.isnan(parse_rating(-1.0)))

        # Landed total price must be NaN if base price is NaN
        self.assertTrue(np.isnan(calculate_total_price(np.nan, 150.0)))

        # Cleaned dataframe check on raw catalog (all prices are currently blank)
        raw_catalog = load_raw_catalog(self.raw_catalog_path)
        cleaned_df = clean_product_dataframe(raw_catalog)
        self.assertTrue(cleaned_df["price_npr"].isna().all(), "Raw catalog prices must remain NaN")
        self.assertTrue(cleaned_df["rating"].isna().all(), "Raw catalog ratings must remain NaN")
        self.assertTrue(cleaned_df["total_price_npr"].isna().all(), "Total price must remain NaN when price is NaN")

        # Record type classification
        self.assertEqual(determine_record_type(np.nan), "sample")
        self.assertEqual(determine_record_type("Demo Store A"), "demo")
        self.assertEqual(determine_record_type("Daraz Nepal"), "real")

    # 4. Test Duplicate Detection
    def test_duplicate_detection(self):
        """Verifies duplicate identification on same store vs multi-store distribution."""
        sample_data = pd.DataFrame([
            {"product_key": "samsung_a56", "platform": "Store A", "price_npr": 45000.0},
            {"product_key": "samsung_a56", "platform": "Store A", "price_npr": 45000.0},  # Exact Duplicate
            {"product_key": "samsung_a56", "platform": "Store B", "price_npr": 44000.0},  # Multi-store (Valid)
        ])
        flagged, audit = identify_duplicates(sample_data)
        self.assertEqual(audit["duplicate_count"], 1, "Must detect exactly 1 duplicate listing")
        self.assertTrue(flagged.loc[1, "is_duplicate"], "Second identical listing must be marked duplicate")
        self.assertFalse(flagged.loc[2, "is_duplicate"], "Different store listing must not be marked duplicate")

    # 5. Test Product Identifier Generation
    def test_product_key_generation(self):
        """Ensures consistent, slugified product keys matching notebook structure."""
        key = generate_product_key("Samsung", "Galaxy A56", "8GB/256GB")
        self.assertEqual(key, "samsung_galaxy_a56_8gb_256gb")

        key2 = generate_product_key("HP", "Pavilion", "16GB/512GB")
        self.assertEqual(key2, "hp_pavilion_16gb_512gb")

        key3 = generate_product_key("Example Brand", "Example Model", "Standard")
        self.assertEqual(key3, "example_brand_example_model_standard")

        # Fallback for empty variant
        key4 = generate_product_key("Sony", "WH-1000XM5", "")
        self.assertEqual(key4, "sony_wh_1000xm5_standard")

    # 6. Test Processed CSV Generation & File Integrity
    def test_processed_csv_generation(self):
        """Runs the pipeline and verifies generated artifacts and schema validity."""
        pipeline = DataProcessingPipeline(PROJECT_ROOT)
        report = pipeline.execute(inject_demo_listings=True)

        self.assertEqual(report["status"], "SUCCESS")
        self.assertGreater(report["total_records_processed"], 0)

        # Check that artifacts exist
        cleaned_csv = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_products.csv")
        summary_csv = os.path.join(PROJECT_ROOT, "data", "processed", "price_comparison_summary.csv")
        ml_csv = os.path.join(PROJECT_ROOT, "data", "processed", "products_ml_features.csv")
        report_json = os.path.join(PROJECT_ROOT, "reports", "data_quality_report.json")

        self.assertTrue(os.path.exists(cleaned_csv), "cleaned_products.csv must be created")
        self.assertTrue(os.path.exists(summary_csv), "price_comparison_summary.csv must be created")
        self.assertTrue(os.path.exists(ml_csv), "products_ml_features.csv must be created")
        self.assertTrue(os.path.exists(report_json), "data_quality_report.json must be created")

        # Verify summary content
        summary_df = pd.read_csv(summary_csv)
        self.assertIn("product_name", summary_df.columns)
        self.assertIn("cheapest_store", summary_df.columns)
        self.assertIn("total_price_npr", summary_df.columns)
        self.assertIn("saving_npr", summary_df.columns)
        self.assertIn("saving_percent", summary_df.columns)

        # Verify raw files remain intact
        raw_catalog = pd.read_csv(self.raw_catalog_path)
        self.assertEqual(len(raw_catalog), 3, "Original sample_products.csv must remain intact with 3 rows")


if __name__ == "__main__":
    unittest.main()
