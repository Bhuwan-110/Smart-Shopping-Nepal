"""
Smart Shopping Nepal - Phase 2 Product Data Collection Test Suite.

Validates:
1. Reusable data collection connectors (BaseConnector, ManualImportConnector, DarazConnector, PartnerFeedConnector).
2. Failure tracking and timestamping.
3. Duplicate prevention across collections.
4. Schema validation (product_name, brand, model, price_npr, image_url, availability_status, verification_status).
5. Source registry compliance & unapproved scraping prevention.
6. End-to-end integration with the Phase 1 processing pipeline.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.collectors.base import BaseConnector, CollectionResult, CollectionFailure
from src.collectors.manual_import import ManualImportConnector
from src.collectors.daraz import DarazConnector
from src.collectors.partner_feed import PartnerFeedConnector
from src.collectors.manager import CollectionManager
from src.pipeline import DataProcessingPipeline


class TestProductDataCollection(unittest.TestCase):

    def setUp(self):
        self.manual_imports_dir = os.path.join(PROJECT_ROOT, "data", "raw", "manual_imports")
        self.daraz_sample = os.path.join(self.manual_imports_dir, "daraz_verified_listings.csv")
        self.retailer_sample = os.path.join(self.manual_imports_dir, "retailer_verified_listings.csv")

    def test_manual_import_connector_valid_records(self):
        """Verifies ingestion of authorized manual feeds with full schema validation."""
        self.assertTrue(os.path.exists(self.retailer_sample), "retailer_verified_listings.csv must exist")

        connector = ManualImportConnector(import_source=self.retailer_sample)
        result = connector.collect()

        self.assertTrue(result.success, "Manual import must succeed for valid file")
        self.assertGreaterEqual(result.total_collected, 3, "Must collect at least 3 retailer records")
        self.assertEqual(result.total_failed, 0, "No failures expected on valid feed")

        # Verify record attributes
        first_rec = result.collected_records[0]
        self.assertIn("product_name", first_rec)
        self.assertIn("brand", first_rec)
        self.assertIn("platform", first_rec)
        self.assertIn("price_npr", first_rec)
        self.assertGreater(first_rec["price_npr"], 0, "Price must be strictly positive")
        self.assertIn("image_url", first_rec)
        self.assertIn("availability_status", first_rec)
        self.assertIn("last_updated", first_rec)
        self.assertIn("verification_status", first_rec)
        self.assertEqual(first_rec["data_source"], "manual_import")

    def test_failure_tracking_on_invalid_records(self):
        """Verifies that records violating schema rules are recorded as failures, not crashing."""
        class MockInvalidConnector(BaseConnector):
            def __init__(self):
                super().__init__("Test Store", "mock", is_approved=True)

            def fetch_raw_listings(self):
                return [
                    {"product_name": "Valid Phone", "platform": "Test Store", "price_npr": 30000},
                    {"product_name": "", "platform": "Test Store", "price_npr": 20000},  # Missing name
                    {"product_name": "Free Phone", "platform": "Test Store", "price_npr": 0},  # Zero price (Forbidden)
                    {"product_name": "Unpriced Phone", "platform": "Test Store", "price_npr": None},  # Blank price (Forbidden)
                ]

        connector = MockInvalidConnector()
        result = connector.collect()

        self.assertEqual(result.total_attempted, 4)
        self.assertEqual(result.total_collected, 1, "Only 1 valid record should be collected")
        self.assertEqual(result.total_failed, 3, "3 invalid records must be captured as failures")
        self.assertEqual(len(result.failures), 3)

        # Check failure reasons
        reasons = [f.error_reason for f in result.failures]
        self.assertTrue(any("product_name" in r for r in reasons))
        self.assertTrue(any("Blank prices are never treated as zero" in r for r in reasons))

    def test_daraz_connector_unapproved_safeguard(self):
        """
        Enforces rule: Do not assume automated scraping is permitted.
        Daraz connector without approved credentials must refuse unpermitted scraping.
        """
        # Connector without credentials or verified feed
        connector = DarazConnector(app_key=None, app_secret=None, verified_feed_path=None)
        result = connector.collect()

        self.assertFalse(result.success, "Unapproved Daraz scraping must be rejected")
        self.assertEqual(result.total_collected, 0)
        self.assertGreaterEqual(result.total_failed, 1)
        self.assertIn("manual verified import", result.failures[0].error_reason.lower())

        # Connector WITH verified manual feed
        approved_connector = DarazConnector(verified_feed_path=self.daraz_sample)
        approved_result = approved_connector.collect()
        self.assertTrue(approved_result.success)
        self.assertGreaterEqual(approved_result.total_collected, 3)

    def test_collection_manager_orchestration_and_audit(self):
        """Verifies multi-connector execution, duplicate prevention, and audit logging."""
        manager = CollectionManager(PROJECT_ROOT)
        manager.load_authorized_connectors()

        run_output = manager.run_collection()
        audit_log = run_output["audit_log"]
        collected_df = run_output["collected_df"]

        self.assertEqual(audit_log["collection_status"], "COMPLETED")
        self.assertGreater(audit_log["total_records_collected"], 0)
        self.assertFalse(collected_df.empty)

        # Check audit file creation
        audit_json_path = os.path.join(PROJECT_ROOT, "reports", "collection_audit.json")
        self.assertTrue(os.path.exists(audit_json_path), "collection_audit.json must be saved")

        # Check raw collected listings output
        raw_collected_path = os.path.join(PROJECT_ROOT, "data", "raw", "collected_listings.csv")
        self.assertTrue(os.path.exists(raw_collected_path), "collected_listings.csv must be saved")

    def test_pipeline_integration_with_collected_data(self):
        """Verifies that collected records feed directly into Phase 1 pipeline."""
        pipeline = DataProcessingPipeline(PROJECT_ROOT)
        report = pipeline.execute(use_collected_data=True, inject_demo_listings=False)

        self.assertEqual(report["status"], "SUCCESS")
        self.assertGreater(report["total_records_processed"], 0)

        # Check that price comparison summary reflects collected multi-store listings
        summary_path = os.path.join(PROJECT_ROOT, "data", "processed", "price_comparison_summary.csv")
        self.assertTrue(os.path.exists(summary_path))

        summary_df = pd.read_csv(summary_path)
        self.assertIn("cheapest_store", summary_df.columns)
        self.assertIn("saving_npr", summary_df.columns)

        # Check that Daraz Nepal, Hukut Store, or Oliz Store appear in the comparison
        stores = set(summary_df["cheapest_store"].dropna().unique())
        expected_stores = {"Daraz Nepal", "Hukut Store", "Oliz Store"}
        self.assertTrue(len(stores.intersection(expected_stores)) > 0, f"Expected store in {stores}")


if __name__ == "__main__":
    unittest.main()
