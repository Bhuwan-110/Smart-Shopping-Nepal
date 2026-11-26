"""
Smart Shopping Nepal - Phase 3 Intelligent Matching & Comparison Test Suite.

Validates:
1. Intelligent product matching on labeled positive and negative pairs.
2. Prevention of false positives:
   - Different models (e.g. A56 vs A55).
   - Different variants (e.g. 128GB vs 256GB).
   - Different conditions (New vs Used/Refurbished).
   - Different brands and categories.
3. Advanced price comparison metrics (lowest, highest, average, savings, stores count).
4. Exclusion of out-of-stock listings and invalid prices.
5. Handling and clear indication of missing shipping costs.
6. Tier isolation (Never mixing demo data with real listings).
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.matcher import IntelligentProductMatcher, ProductAttributeExtractor
from src.comparator import calculate_total_price, compare_product_by_key, generate_comparison_summary


class TestIntelligentProductMatching(unittest.TestCase):

    def setUp(self):
        self.matcher = IntelligentProductMatcher(tfidf_threshold=0.45)
        self.extractor = ProductAttributeExtractor()

    # 1. Attribute Extraction Tests
    def test_attribute_extraction(self):
        """Verifies parsing of brand, model, storage, RAM, and condition."""
        title = "Samsung Galaxy A56 5G (8GB RAM / 256GB Storage) Awesome Graphite"
        attrs = self.extractor.extract_all(title)

        self.assertEqual(attrs["extracted_brand"], "Samsung")
        self.assertEqual(attrs["extracted_model"], "Galaxy A56")
        self.assertEqual(attrs["extracted_ram"], "8GB")
        self.assertEqual(attrs["extracted_storage"], "256GB")
        self.assertEqual(attrs["extracted_variant"], "8GB/256GB")
        self.assertEqual(attrs["extracted_condition"], "new")
        self.assertEqual(attrs["canonical_product_key"], "samsung_galaxy_a56_8gb_256gb")

        # Laptop attribute extraction
        laptop_title = "HP Pavilion 15 Laptop (16GB RAM, 512GB SSD) Intel Core i5"
        l_attrs = self.extractor.extract_all(laptop_title)
        self.assertEqual(l_attrs["extracted_brand"], "HP")
        self.assertEqual(l_attrs["extracted_model"], "Pavilion 15")
        self.assertEqual(l_attrs["extracted_variant"], "16GB/512GB")

        # Condition: Used / Refurbished
        used_title = "Apple iPhone 15 (128GB) - Refurbished / Pre-owned"
        u_attrs = self.extractor.extract_all(used_title)
        self.assertEqual(u_attrs["extracted_condition"], "used")
        self.assertTrue(u_attrs["canonical_product_key"].endswith("_used"))

    # 2. Labeled Positive Matching Pairs (Should Match)
    def test_labeled_positive_pairs(self):
        """Pairs representing identical products with minor title variations must match."""
        positive_pairs = [
            (
                {"product_name": "Samsung Galaxy A56 5G (8GB/256GB) Awesome Graphite", "brand": "Samsung"},
                {"product_name": "Samsung Galaxy A56 8GB RAM 256GB ROM Smartphone", "brand": "Samsung"}
            ),
            (
                {"product_name": "HP Pavilion 15 (16GB RAM / 512GB SSD) Laptop", "brand": "HP"},
                {"product_name": "HP Pavilion Laptop 15 16GB/512GB SSD", "brand": "HP"}
            ),
            (
                {"product_name": "Anker Soundcore R50i True Wireless Earbuds", "brand": "Anker"},
                {"product_name": "Anker Soundcore R50i TWS Earbuds - Black", "brand": "Anker"}
            ),
            (
                {"product_name": "Apple iPhone 15 (128GB) Black", "brand": "Apple"},
                {"product_name": "Apple iPhone 15 128GB Storage Mobile", "brand": "Apple"}
            )
        ]

        for p1, p2 in positive_pairs:
            is_match, score, reason = self.matcher.are_products_equivalent(p1, p2)
            self.assertTrue(is_match, f"Failed positive match: '{p1['product_name']}' vs '{p2['product_name']}'. Reason: {reason}")

    # 3. Labeled Negative Pairs (Must NOT Match)
    def test_labeled_negative_different_models(self):
        """Different models must NEVER match, even if titles look very similar."""
        negative_model_pairs = [
            # A56 vs A55
            (
                {"product_name": "Samsung Galaxy A56 5G (8GB/256GB)", "brand": "Samsung"},
                {"product_name": "Samsung Galaxy A55 5G (8GB/256GB)", "brand": "Samsung"}
            ),
            # iPhone 15 vs iPhone 14
            (
                {"product_name": "Apple iPhone 15 (128GB)", "brand": "Apple"},
                {"product_name": "Apple iPhone 14 (128GB)", "brand": "Apple"}
            ),
            # HP Pavilion 15 vs HP Pavilion 14
            (
                {"product_name": "HP Pavilion 15 (16GB/512GB)", "brand": "HP"},
                {"product_name": "HP Pavilion 14 (16GB/512GB)", "brand": "HP"}
            )
        ]

        for p1, p2 in negative_model_pairs:
            is_match, score, reason = self.matcher.are_products_equivalent(p1, p2)
            self.assertFalse(is_match, f"Incorrectly matched different models: '{p1['product_name']}' vs '{p2['product_name']}'")

    def test_labeled_negative_different_variants(self):
        """Different RAM or Storage capacities must NEVER match."""
        negative_variant_pairs = [
            # 256GB vs 128GB
            (
                {"product_name": "Samsung Galaxy A56 (8GB/256GB)", "brand": "Samsung"},
                {"product_name": "Samsung Galaxy A56 (8GB/128GB)", "brand": "Samsung"}
            ),
            # 16GB vs 8GB RAM
            (
                {"product_name": "HP Pavilion 15 (16GB/512GB)", "brand": "HP"},
                {"product_name": "HP Pavilion 15 (8GB/512GB)", "brand": "HP"}
            )
        ]

        for p1, p2 in negative_variant_pairs:
            is_match, score, reason = self.matcher.are_products_equivalent(p1, p2)
            self.assertFalse(is_match, f"Incorrectly matched different variants: '{p1['product_name']}' vs '{p2['product_name']}'")

    def test_labeled_negative_different_conditions(self):
        """New products must NEVER match Used or Refurbished products."""
        p_new = {"product_name": "Samsung Galaxy A56 (8GB/256GB) Brand New", "brand": "Samsung"}
        p_used = {"product_name": "Samsung Galaxy A56 (8GB/256GB) Used / Refurbished", "brand": "Samsung"}

        is_match, score, reason = self.matcher.are_products_equivalent(p_new, p_used)
        self.assertFalse(is_match, "New and Refurbished products must not be grouped together")
        self.assertIn("Condition mismatch", reason)

    # 4. Advanced Price Comparison Calculations Test
    def test_price_comparison_advanced_metrics(self):
        """Verifies calculation of lowest, highest, average, savings, and store counts."""
        sample_listings = pd.DataFrame([
            {
                "product_name": "Samsung Galaxy A56 5G",
                "product_key": "samsung_galaxy_a56_8gb_256gb",
                "brand": "Samsung",
                "category": "Mobile Phone",
                "platform": "Store A",
                "price_npr": 45000.0,
                "shipping_cost_npr": 0.0,
                "total_price_npr": 45000.0,
                "availability_status": "in_stock",
                "record_type": "real"
            },
            {
                "product_name": "Samsung Galaxy A56",
                "product_key": "samsung_galaxy_a56_8gb_256gb",
                "brand": "Samsung",
                "category": "Mobile Phone",
                "platform": "Store B",
                "price_npr": 43500.0,
                "shipping_cost_npr": 200.0,
                "total_price_npr": 43700.0,
                "availability_status": "in_stock",
                "record_type": "real"
            },
            {
                "product_name": "Samsung Galaxy A56",
                "product_key": "samsung_galaxy_a56_8gb_256gb",
                "brand": "Samsung",
                "category": "Mobile Phone",
                "platform": "Store C",
                "price_npr": 44000.0,
                "shipping_cost_npr": 100.0,
                "total_price_npr": 44100.0,
                "availability_status": "in_stock",
                "record_type": "real"
            }
        ])

        res = compare_product_by_key(sample_listings, "samsung_galaxy_a56_8gb_256gb", target_tier="real")

        self.assertEqual(res["cheapest_store"], "Store B")
        self.assertEqual(res["lowest_total_price"], 43700.0)
        self.assertEqual(res["highest_total_price"], 45000.0)
        expected_avg = round((45000.0 + 43700.0 + 44100.0) / 3, 2)
        self.assertEqual(res["average_total_price"], expected_avg)
        self.assertEqual(res["saving_npr"], 1300.0)
        expected_pct = round((1300.0 / 45000.0) * 100, 2)
        self.assertEqual(res["saving_percent"], expected_pct)
        self.assertEqual(res["number_of_stores"], 3)

    def test_out_of_stock_and_invalid_price_exclusion(self):
        """Out of stock and invalid prices must be excluded from calculations."""
        listings = pd.DataFrame([
            {
                "product_name": "Test Phone",
                "product_key": "test_phone_key",
                "platform": "Store Active",
                "price_npr": 30000.0,
                "shipping_cost_npr": 100.0,
                "total_price_npr": 30100.0,
                "availability_status": "in_stock",
                "record_type": "real"
            },
            {
                "product_name": "Test Phone",
                "product_key": "test_phone_key",
                "platform": "Store Cheap But OutOfStock",
                "price_npr": 20000.0,  # Cheaper, but out of stock!
                "shipping_cost_npr": 0.0,
                "total_price_npr": 20000.0,
                "availability_status": "out_of_stock",
                "record_type": "real"
            }
        ])

        res = compare_product_by_key(listings, "test_phone_key", target_tier="real")
        self.assertEqual(res["cheapest_store"], "Store Active", "Out of stock listing must not be selected as cheapest")
        self.assertEqual(res["lowest_total_price"], 30100.0)
        self.assertEqual(res["number_of_stores"], 1)

    def test_tier_isolation_never_mix_demo_and_real(self):
        """Ensures demo data is never mixed with real verified store listings."""
        mixed_listings = pd.DataFrame([
            {
                "product_name": "Samsung Galaxy A56",
                "product_key": "samsung_a56",
                "platform": "Demo Store Fake",
                "price_npr": 10000.0,  # Unrealistic fake demo price
                "total_price_npr": 10000.0,
                "availability_status": "in_stock",
                "record_type": "demo"
            },
            {
                "product_name": "Samsung Galaxy A56",
                "product_key": "samsung_a56",
                "platform": "Daraz Nepal",
                "price_npr": 44999.0,
                "total_price_npr": 44999.0,
                "availability_status": "in_stock",
                "record_type": "real"
            }
        ])

        # Querying real tier must exclude demo store
        real_res = compare_product_by_key(mixed_listings, "samsung_a56", target_tier="real")
        self.assertEqual(real_res["cheapest_store"], "Daraz Nepal")
        self.assertEqual(real_res["lowest_total_price"], 44999.0)
        self.assertEqual(real_res["number_of_stores"], 1)

    def test_four_confidence_match_tiers(self):
        """Verifies 4 confidence tiers: Exact Match, Probable Match, Similar Product, Unmatched Product."""
        # 1. Exact Match (Confidence 1.0)
        exact_p1 = {"product_name": "Samsung Galaxy A56 5G (8GB/256GB)", "brand": "Samsung"}
        exact_p2 = {"product_name": "Samsung Galaxy A56 5G 8GB/256GB Smartphone", "brand": "Samsung"}
        tier_exact = self.matcher.classify_match_tier(exact_p1, exact_p2)
        self.assertEqual(tier_exact["tier"], "Exact Match")
        self.assertEqual(tier_exact["confidence"], 1.0)
        self.assertTrue(tier_exact["is_match"])

        # 2. Probable Match (Confidence 0.85 - 0.95)
        prob_p1 = {"product_name": "Anker Soundcore R50i True Wireless Earbuds", "brand": "Anker"}
        prob_p2 = {"product_name": "Anker Soundcore R50i TWS Earbuds - Black", "brand": "Anker"}
        tier_prob = self.matcher.classify_match_tier(prob_p1, prob_p2)
        self.assertIn(tier_prob["tier"], ["Exact Match", "Probable Match"])
        self.assertGreaterEqual(tier_prob["confidence"], 0.85)
        self.assertTrue(tier_prob["is_match"])

        # 3. Similar Product (Confidence 0.60 - 0.84, is_match False)
        sim_p1 = {"product_name": "Samsung Galaxy A56 5G (8GB/128GB)", "brand": "Samsung", "category": "Mobile Phone"}
        sim_p2 = {"product_name": "Samsung Galaxy A56 5G (8GB/256GB)", "brand": "Samsung", "category": "Mobile Phone"}
        tier_sim = self.matcher.classify_match_tier(sim_p1, sim_p2)
        self.assertEqual(tier_sim["tier"], "Similar Product")
        self.assertGreaterEqual(tier_sim["confidence"], 0.60)
        self.assertLessEqual(tier_sim["confidence"], 0.84)
        self.assertFalse(tier_sim["is_match"])

        # 4. Unmatched Product (Confidence < 0.60, is_match False)
        unmatch_p1 = {"product_name": "Apple iPhone 15 128GB", "brand": "Apple", "category": "Mobile Phone"}
        unmatch_p2 = {"product_name": "Samsung Galaxy A56 256GB", "brand": "Samsung", "category": "Mobile Phone"}
        tier_unmatch = self.matcher.classify_match_tier(unmatch_p1, unmatch_p2)
        self.assertEqual(tier_unmatch["tier"], "Unmatched Product")
        self.assertLess(tier_unmatch["confidence"], 0.60)
        self.assertFalse(tier_unmatch["is_match"])


if __name__ == "__main__":
    unittest.main()
