"""
Smart Shopping Nepal - Base Connector & Failure Tracking Infrastructure.

Defines the contract for all platform connectors:
- Standardized product collection schema
- Strict validation rules (Never blank price as 0, never invent ratings)
- Granular failure tracking & collection timestamps
- Data source & verification status provenance
"""

import abc
import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

from src.cleaner import (
    clean_text,
    standardize_brand,
    standardize_category,
    parse_currency_to_npr,
    parse_shipping_cost_npr,
    parse_rating,
    parse_review_count,
)
from src.keygen import generate_product_key
from src.validator import validate_url


@dataclass
class CollectionFailure:
    """Represents a failed record collection or validation attempt."""
    platform: str
    source_identifier: str
    raw_record: Dict[str, Any]
    error_reason: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())


@dataclass
class CollectionResult:
    """Summary and payload resulting from a collection run."""
    platform: str
    data_source: str
    success: bool
    collected_records: List[Dict[str, Any]] = field(default_factory=list)
    failures: List[CollectionFailure] = field(default_factory=list)
    total_attempted: int = 0
    total_collected: int = 0
    total_failed: int = 0
    start_time: str = ""
    end_time: str = ""
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "data_source": self.data_source,
            "success": self.success,
            "total_attempted": self.total_attempted,
            "total_collected": self.total_collected,
            "total_failed": self.total_failed,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "message": self.message,
            "failures": [
                {
                    "source_id": f.source_identifier,
                    "reason": f.error_reason,
                    "timestamp": f.timestamp
                }
                for f in self.failures
            ]
        }


class BaseConnector(abc.ABC):
    """
    Abstract base class for all authorized shopping platform connectors.
    Every connector must define its authorization mechanism and schema validation.
    """

    def __init__(
        self,
        platform_name: str,
        data_source: str,
        is_approved: bool = False,
        default_verification_status: str = "unverified"
    ):
        self.platform_name = platform_name
        self.data_source = data_source  # 'official_api', 'partner_feed', 'manual_import'
        self.is_approved = is_approved
        self.default_verification_status = default_verification_status

    @abc.abstractmethod
    def fetch_raw_listings(self) -> List[Dict[str, Any]]:
        """
        Retrieves raw listing dictionaries from the permitted source.
        Must NOT attempt unauthorized web scraping.
        """
        pass

    def validate_and_normalize_record(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates and standardizes a single collected product record.
        Enforces strict Phase 2 schema rules:
        - Product name, brand, platform, and positive price are required.
        - Blank price is NEVER treated as zero (raises ValueError).
        - Ratings/reviews are preserved as NaN if not explicitly present and valid.
        """
        # 1. Product Name
        prod_name = raw.get("product_name")
        if not prod_name or pd.isna(prod_name) or not str(prod_name).strip():
            raise ValueError("Missing required field: 'product_name'")
        product_name = str(prod_name).strip()

        # 2. Platform / Store Name
        platform = raw.get("platform") or self.platform_name
        if not platform or pd.isna(platform) or not str(platform).strip():
            raise ValueError("Missing required field: 'platform'")
        platform = str(platform).strip()

        # 3. Price in NPR (Never treat blank as zero!)
        raw_price = raw.get("price_npr")
        price_npr = parse_currency_to_npr(raw_price)
        if pd.isna(price_npr) or price_npr <= 0:
            raise ValueError(f"Invalid or missing price '{raw_price}'. Blank prices are never treated as zero.")

        # 4. Brand, Model, Category, Variant
        raw_brand = raw.get("brand")
        brand = standardize_brand(raw_brand) if raw_brand and pd.notna(raw_brand) else "Generic"

        raw_model = raw.get("model")
        model = str(raw_model).strip() if raw_model and pd.notna(raw_model) else "Standard"

        raw_category = raw.get("category")
        category = standardize_category(raw_category) if raw_category and pd.notna(raw_category) else "General"

        raw_variant = raw.get("variant")
        variant = str(raw_variant).strip() if raw_variant and pd.notna(raw_variant) else "Standard"

        # 5. Product Key Generation
        product_key = generate_product_key(brand, model, variant)

        # 6. Shipping Cost in NPR
        raw_shipping = raw.get("shipping_cost_npr")
        shipping_cost_npr = parse_shipping_cost_npr(raw_shipping)

        # 7. Total Landed Price
        s_val = 0.0 if pd.isna(shipping_cost_npr) else float(shipping_cost_npr)
        total_price_npr = price_npr + s_val

        # 8. Rating and Review Count (Never invent values!)
        raw_rating = raw.get("rating")
        rating = parse_rating(raw_rating)

        raw_reviews = raw.get("review_count")
        review_count = parse_review_count(raw_reviews)

        # 9. URLs
        raw_url = raw.get("product_url")
        product_url = str(raw_url).strip() if raw_url and pd.notna(raw_url) else ""

        raw_img = raw.get("image_url")
        image_url = str(raw_img).strip() if raw_img and pd.notna(raw_img) else ""

        # 10. Availability Status
        avail = raw.get("availability_status")
        availability_status = str(avail).strip().lower() if avail and pd.notna(avail) else "in_stock"

        # 11. Timestamps & Verification Metadata
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        last_updated = raw.get("last_updated") or now_iso
        data_source = raw.get("data_source") or self.data_source
        verification_status = raw.get("verification_status") or self.default_verification_status

        return {
            "product_name": product_name,
            "brand": brand,
            "model": model,
            "category": category,
            "variant": variant,
            "platform": platform,
            "price_npr": price_npr,
            "shipping_cost_npr": shipping_cost_npr,
            "total_price_npr": total_price_npr,
            "rating": rating,
            "review_count": review_count,
            "product_url": product_url,
            "image_url": image_url,
            "availability_status": availability_status,
            "last_updated": last_updated,
            "data_source": data_source,
            "verification_status": verification_status,
            "product_key": product_key
        }

    def collect(self) -> CollectionResult:
        """
        Executes the collection pipeline for this connector:
        1. Checks approval status.
        2. Retrieves raw listings.
        3. Validates each record and records any failures.
        4. Returns an encapsulated CollectionResult.
        """
        start_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
        failures: List[CollectionFailure] = []
        collected: List[Dict[str, Any]] = []

        if not self.is_approved:
            end_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
            return CollectionResult(
                platform=self.platform_name,
                data_source=self.data_source,
                success=False,
                collected_records=[],
                failures=[
                    CollectionFailure(
                        platform=self.platform_name,
                        source_identifier=self.platform_name,
                        raw_record={},
                        error_reason="Access is not approved. Automated scraping is prohibited; use manual verified import workflow instead."
                    )
                ],
                total_attempted=0,
                total_collected=0,
                total_failed=1,
                start_time=start_time,
                end_time=end_time,
                message=f"Collection aborted: {self.platform_name} requires explicit authorization or manual verified import."
            )

        try:
            raw_listings = self.fetch_raw_listings()
        except Exception as e:
            end_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
            failure = CollectionFailure(
                platform=self.platform_name,
                source_identifier=self.platform_name,
                raw_record={},
                error_reason=f"Connector fetch error: {str(e)}"
            )
            return CollectionResult(
                platform=self.platform_name,
                data_source=self.data_source,
                success=False,
                collected_records=[],
                failures=[failure],
                total_attempted=0,
                total_collected=0,
                total_failed=1,
                start_time=start_time,
                end_time=end_time,
                message=f"Fetch failed for {self.platform_name}: {str(e)}"
            )

        total_attempted = len(raw_listings)

        for idx, raw in enumerate(raw_listings):
            src_id = raw.get("product_name") or raw.get("product_url") or f"row_{idx}"
            try:
                valid_record = self.validate_and_normalize_record(raw)
                collected.append(valid_record)
            except ValueError as ve:
                failures.append(
                    CollectionFailure(
                        platform=self.platform_name,
                        source_identifier=str(src_id),
                        raw_record=raw,
                        error_reason=str(ve)
                    )
                )
            except Exception as ex:
                failures.append(
                    CollectionFailure(
                        platform=self.platform_name,
                        source_identifier=str(src_id),
                        raw_record=raw,
                        error_reason=f"Unexpected validation error: {str(ex)}"
                    )
                )

        end_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
        success = len(collected) > 0 or total_attempted == 0

        return CollectionResult(
            platform=self.platform_name,
            data_source=self.data_source,
            success=success,
            collected_records=collected,
            failures=failures,
            total_attempted=total_attempted,
            total_collected=len(collected),
            total_failed=len(failures),
            start_time=start_time,
            end_time=end_time,
            message=f"Collected {len(collected)} records with {len(failures)} failures from {self.platform_name}."
        )
