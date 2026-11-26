"""
Smart Shopping Nepal - Pipeline Synchronization Management Command.

Connects the existing data-processing & collection pipeline to the Django database:
1. Ingests source_registry.csv into Store models.
2. Runs the DataProcessingPipeline (or reads cleaned_products.csv).
3. Upserts canonical Products, ProductVariants, and ProductListings.
4. Records PriceHistory for temporal price tracking.
5. Computes Content-Based Recommendations between products.
6. Records a CollectionLog entry.
"""

import os
from decimal import Decimal
from django.core.management.base import BaseCommand
from django.conf import settings
from django.utils import timezone
from django.utils.text import slugify

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from shopping.models import (
    Product,
    ProductVariant,
    Store,
    ProductListing,
    PriceHistory,
    RecommendationResult,
    CollectionLog,
)
from src.pipeline import DataProcessingPipeline
from src.loader import load_source_registry, load_csv


class Command(BaseCommand):
    help = "Synchronizes processed pipeline data into the Django database."

    def add_arguments(self, parser):
        parser.add_argument(
            '--run-fresh-collection',
            action='store_true',
            help='Triggers a live data collection run before importing.',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE("=" * 65))
        self.stdout.write(self.style.NOTICE("   SMARTSHOPPING NEPAL - DATABASE PIPELINE SYNCHRONIZATION      "))
        self.stdout.write(self.style.NOTICE("=" * 65))

        base_dir = str(settings.BASE_DIR)

        # 1. Optionally trigger fresh collection and processing
        if options['run_fresh_collection']:
            self.stdout.write(self.style.WARNING("Running fresh collection and processing pipeline..."))
            pipeline = DataProcessingPipeline(base_dir)
            pipeline.execute(use_collected_data=True, inject_demo_listings=False)
            self.stdout.write(self.style.SUCCESS("✓ Pipeline executed successfully."))

        # 2. Synchronize Store / Platform Registry
        registry_path = os.path.join(base_dir, "data", "raw", "source_registry.csv")
        self.stdout.write(f"Syncing stores from {registry_path}...")
        if os.path.exists(registry_path):
            sources_df = load_source_registry(registry_path)
            store_count = 0
            for _, row in sources_df.iterrows():
                name = str(row.get("platform", "")).strip()
                if not name or name.lower() in ["nan", "none", ""]:
                    continue

                Store.objects.update_or_create(
                    name=name,
                    defaults={
                        "slug": slugify(name),
                        "data_access_method": str(row.get("data_access_method", "")),
                        "permission_status": str(row.get("permission_status", "")),
                        "api_or_feed_status": str(row.get("api_or_feed_status", "")),
                        "notes": str(row.get("notes", "")),
                        "is_active": True
                    }
                )
                store_count += 1
            self.stdout.write(self.style.SUCCESS(f"✓ Synchronized {store_count} stores/platforms."))

        # 3. Synchronize Products, Variants & Listings from cleaned_products.csv
        cleaned_csv = os.path.join(base_dir, "data", "processed", "cleaned_products.csv")
        if not os.path.exists(cleaned_csv):
            # Run pipeline to generate it if missing
            self.stdout.write(self.style.WARNING("cleaned_products.csv missing. Generating via pipeline..."))
            pipeline = DataProcessingPipeline(base_dir)
            pipeline.execute(use_collected_data=True, inject_demo_listings=False)

        listings_df = load_csv(cleaned_csv)
        self.stdout.write(f"Ingesting {len(listings_df)} product listings into database...")

        product_cache = {}
        ingested_listings = 0

        for _, row in listings_df.iterrows():
            prod_name = str(row.get("product_name", "")).strip()
            prod_key = str(row.get("product_key", "")).strip()
            brand = str(row.get("brand", "Generic")).strip()
            category = str(row.get("category", "General")).strip()
            model = str(row.get("model", "Standard")).strip()
            variant_name = str(row.get("variant", "Standard")).strip()

            if not prod_name or not prod_key or prod_key.lower() in ["nan", "none"]:
                continue

            # Upsert Canonical Product
            if prod_key not in product_cache:
                product, _ = Product.objects.update_or_create(
                    canonical_product_key=prod_key,
                    defaults={
                        "name": prod_name,
                        "brand": brand,
                        "model": model,
                        "category": category,
                        "image_url": str(row.get("image_url", "") or ""),
                        "description": f"{brand} {model} ({variant_name}) in category {category}."
                    }
                )
                product_cache[prod_key] = product
            else:
                product = product_cache[prod_key]

            # Upsert Product Variant
            variant, _ = ProductVariant.objects.get_or_create(
                product=product,
                variant_name=variant_name,
                defaults={"is_active": True}
            )

            # Store lookup or creation
            platform_name = str(row.get("platform", "Unknown")).strip()
            if not platform_name or platform_name.lower() in ["nan", "none"]:
                platform_name = "Catalog Sample"

            store, _ = Store.objects.get_or_create(
                name=platform_name,
                defaults={"slug": slugify(platform_name), "permission_status": "Verified"}
            )

            # Pricing and availability
            price_raw = row.get("price_npr")
            if pd.isna(price_raw) or float(price_raw) <= 0:
                # Canonical catalog item without live price
                continue

            price_dec = Decimal(str(round(float(price_raw), 2)))

            shipping_raw = row.get("shipping_cost_npr")
            shipping_dec = Decimal(str(round(float(shipping_raw), 2))) if pd.notna(shipping_raw) else Decimal("0.00")
            is_ship_known = pd.notna(shipping_raw)

            rating_raw = row.get("rating")
            rating_dec = Decimal(str(round(float(rating_raw), 2))) if pd.notna(rating_raw) else None

            reviews_raw = row.get("review_count")
            reviews_int = int(reviews_raw) if pd.notna(reviews_raw) else None

            rec_type = str(row.get("record_type", "real")).lower()
            if rec_type not in ["real", "verified", "sample", "demo"]:
                rec_type = "real"

            listing, created = ProductListing.objects.update_or_create(
                product=product,
                store=store,
                variant=variant,
                defaults={
                    "title": prod_name,
                    "price_npr": price_dec,
                    "shipping_cost_npr": shipping_dec,
                    "is_shipping_known": is_ship_known,
                    "total_price_npr": price_dec + shipping_dec,
                    "rating": rating_dec,
                    "review_count": reviews_int,
                    "product_url": str(row.get("product_url", "#")),
                    "image_url": str(row.get("image_url", "") or ""),
                    "availability_status": str(row.get("availability_status", "in_stock")),
                    "record_type": rec_type,
                    "is_verified": rec_type in ["real", "verified"],
                    "last_updated": timezone.now()
                }
            )

            # Record temporal Price History
            PriceHistory.objects.create(
                listing=listing,
                price_npr=listing.price_npr,
                shipping_cost_npr=listing.shipping_cost_npr,
                total_price_npr=listing.total_price_npr
            )
            ingested_listings += 1

        self.stdout.write(self.style.SUCCESS(f"✓ Ingested {len(product_cache)} canonical products and {ingested_listings} listings."))

        # 4. Generate Precomputed Recommendations (Content-Based TF-IDF)
        self.stdout.write("Computing product recommendations using TF-IDF...")
        all_products = list(Product.objects.all())
        if len(all_products) >= 2:
            corpus = [f"{p.category} {p.brand} {p.name} {p.description}" for p in all_products]
            vec = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vec.fit_transform(corpus)
            sim_matrix = cosine_similarity(tfidf_matrix)

            rec_count = 0
            for i, p_src in enumerate(all_products):
                # Get top similar products (excluding self)
                sim_scores = [(j, sim_matrix[i][j]) for j in range(len(all_products)) if i != j]
                sim_scores.sort(key=lambda x: x[1], reverse=True)

                for j, score in sim_scores[:4]:
                    if score > 0.05:
                        RecommendationResult.objects.update_or_create(
                            source_product=p_src,
                            recommended_product=all_products[j],
                            defaults={
                                "similarity_score": round(float(score), 4),
                                "algorithm_type": "tfidf_content_based"
                            }
                        )
                        rec_count += 1
            self.stdout.write(self.style.SUCCESS(f"✓ Generated {rec_count} recommendation pairings."))

        # 5. Log collection execution
        CollectionLog.objects.create(
            data_source="pipeline_sync",
            status="SUCCESS",
            records_attempted=len(listings_df),
            records_collected=ingested_listings,
            failures_count=0,
            audit_details={"synced_products": len(product_cache), "synced_listings": ingested_listings}
        )

        self.stdout.write(self.style.SUCCESS("=" * 65))
        self.stdout.write(self.style.SUCCESS("   DATABASE SYNCHRONIZATION COMPLETED SUCCESSFULLY!            "))
        self.stdout.write(self.style.SUCCESS("=" * 65))
