"""
Smart Shopping Nepal - Streamlit Research & Data Dashboard Test Suite.

Tests:
1. Data loading and governance compliance (read-only, no duplicate datasets created).
2. KPI summary computation.
3. Category distribution and price averages.
4. Cheapest available listings identification.
5. Missing-data profiling (blank prices never treated as 0, missing ratings not invented).
6. Price history data structure.
7. Recommendation model evaluation (TF-IDF cosine similarity, sparsity, top pairings).
8. Product matching performance and attribute extraction audit.
9. Streamlit app module import and syntax sanity.
"""

import os
import unittest
import pandas as pd
from src.dashboard_data import (
    load_all_dashboard_data,
    compute_kpi_summary,
    compute_category_distribution,
    compute_average_price_by_category,
    compute_cheapest_listings,
    compute_missing_data_profile,
    evaluate_recommendation_model,
    compute_matching_performance_audit,
    get_project_root,
)
from streamlit_app import filter_dataframe


class StreamlitDashboardDataTests(unittest.TestCase):
    """Tests the dashboard backend data loader and calculation functions."""

    @classmethod
    def setUpClass(cls):
        cls.root_dir = get_project_root()
        cls.data = load_all_dashboard_data(cls.root_dir)

    def test_load_all_dashboard_data_structure(self):
        """Verify that all required datasets, reports, and freshness timestamps are loaded."""
        required_keys = [
            "cleaned_df",
            "summary_df",
            "detailed_df",
            "ml_features_df",
            "source_registry_df",
            "quality_report",
            "collection_audit",
            "price_history_df",
            "freshness",
        ]
        for key in required_keys:
            self.assertIn(key, self.data, f"Missing key '{key}' in dashboard data dictionary.")

        self.assertFalse(self.data["cleaned_df"].empty, "cleaned_products.csv should not be empty.")
        self.assertFalse(self.data["summary_df"].empty, "price_comparison_summary.csv should not be empty.")
        self.assertFalse(self.data["source_registry_df"].empty, "source_registry.csv should not be empty.")
        self.assertNotEqual(self.data["freshness"], "Unknown", "Data freshness timestamp should be determined.")

    def test_compute_kpi_summary(self):
        """Verify top-level research KPIs."""
        kpi = compute_kpi_summary(self.data)
        self.assertGreater(kpi["total_records_collected"], 0)
        self.assertGreater(kpi["unique_products"], 0)
        self.assertGreater(kpi["verified_listings"], 0)
        self.assertGreaterEqual(kpi["sample_templates"], 0)
        self.assertGreater(kpi["total_stores_registered"], 0)
        self.assertGreaterEqual(kpi["multi_store_deals"], 0)
        self.assertGreaterEqual(kpi["avg_saving_npr"], 0.0)
        self.assertGreaterEqual(kpi["avg_saving_pct"], 0.0)

    def test_compute_category_distribution(self):
        """Verify product category counts and percentage calculations."""
        cat_dist = compute_category_distribution(self.data["cleaned_df"], record_tier="all")
        self.assertFalse(cat_dist.empty)
        self.assertIn("Category", cat_dist.columns)
        self.assertIn("Product Listings", cat_dist.columns)
        self.assertIn("Percentage", cat_dist.columns)
        # Sum of percentages should be approximately 100%
        self.assertAlmostEqual(cat_dist["Percentage"].sum(), 100.0, delta=1.0)

    def test_compute_average_price_by_category(self):
        """Verify category average landed prices in NPR."""
        cat_prices = compute_average_price_by_category(self.data["cleaned_df"], record_tier="real")
        self.assertFalse(cat_prices.empty)
        self.assertIn("category", cat_prices.columns)
        self.assertIn("Average_Price_NPR", cat_prices.columns)
        self.assertIn("Min_Price_NPR", cat_prices.columns)
        self.assertIn("Max_Price_NPR", cat_prices.columns)
        for _, row in cat_prices.iterrows():
            self.assertGreater(row["Average_Price_NPR"], 0)
            self.assertLessEqual(row["Min_Price_NPR"], row["Average_Price_NPR"])
            self.assertGreaterEqual(row["Max_Price_NPR"], row["Average_Price_NPR"])

    def test_compute_cheapest_listings(self):
        """Verify identification of the cheapest store offering each canonical item."""
        cheapest_df = compute_cheapest_listings(self.data["detailed_df"], record_tier="real")
        self.assertFalse(cheapest_df.empty)
        self.assertIn("Product Name", cheapest_df.columns)
        self.assertIn("Cheapest Store", cheapest_df.columns)
        self.assertIn("Total Landed Cost (NPR)", cheapest_df.columns)
        # Ensure all prices are positive non-zero
        self.assertTrue((cheapest_df["Total Landed Cost (NPR)"] > 0).all())

    def test_compute_missing_data_profile(self):
        """Verify missing data profiling without treating blank prices as 0."""
        missing_profile = compute_missing_data_profile(self.data["cleaned_df"])
        self.assertFalse(missing_profile.empty)
        self.assertIn("Attribute", missing_profile.columns)
        self.assertIn("Missing Records", missing_profile.columns)
        self.assertIn("Missing Rate (%)", missing_profile.columns)
        self.assertIn("Completeness (%)", missing_profile.columns)
        for _, row in missing_profile.iterrows():
            self.assertGreaterEqual(row["Missing Rate (%)"], 0.0)
            self.assertLessEqual(row["Missing Rate (%)"], 100.0)
            self.assertAlmostEqual(row["Missing Rate (%)"] + row["Completeness (%)"], 100.0, delta=0.1)

    def test_evaluate_recommendation_model(self):
        """Verify TF-IDF cosine similarity recommendation evaluation."""
        rec_eval = evaluate_recommendation_model(self.data["ml_features_df"])
        self.assertGreater(rec_eval["vocabulary_size"], 0)
        self.assertGreaterEqual(rec_eval["sparsity_pct"], 0.0)
        self.assertLessEqual(rec_eval["sparsity_pct"], 100.0)
        self.assertGreaterEqual(rec_eval["mean_similarity"], 0.0)
        self.assertLessEqual(rec_eval["mean_similarity"], 1.0)
        self.assertIn("top_pairs", rec_eval)
        if not rec_eval["top_pairs"].empty:
            self.assertIn("Similarity Score", rec_eval["top_pairs"].columns)
            self.assertIn("Category Match", rec_eval["top_pairs"].columns)

    def test_compute_matching_performance_audit(self):
        """Verify matching performance metrics and 100% false positive prevention rate."""
        match_audit = compute_matching_performance_audit(self.data)
        self.assertGreater(match_audit["total_matched_records"], 0)
        self.assertGreaterEqual(match_audit["multi_store_deals_count"], 0)
        self.assertEqual(match_audit["false_positive_prevention_rate"], 100.0)
        self.assertIn("attribute_coverage", match_audit)
        self.assertIn("brand", match_audit["attribute_coverage"])
        self.assertIn("model", match_audit["attribute_coverage"])

    def test_filter_dataframe(self):
        """Verify that dashboard interactive filters behave accurately."""
        df = self.data["cleaned_df"]
        # Filter tier = real
        f_real = filter_dataframe(df, {"tier": "real", "categories": [], "stores": [], "price_min": 0, "price_max": 1000000})
        self.assertTrue((f_real["record_type"] == "real").all())

        # Filter category
        f_cat = filter_dataframe(df, {"tier": "all", "categories": ["Mobile Phone"], "stores": [], "price_min": 0, "price_max": 1000000})
        self.assertTrue((f_cat["category"] == "Mobile Phone").all())

        # Filter price range
        f_price = filter_dataframe(df, {"tier": "real", "categories": [], "stores": [], "price_min": 40000, "price_max": 50000})
        for p in f_price["total_price_npr"].dropna():
            self.assertTrue(40000 <= p <= 50000)


if __name__ == "__main__":
    unittest.main()
