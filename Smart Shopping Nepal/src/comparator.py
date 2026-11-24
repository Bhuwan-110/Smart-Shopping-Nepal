"""
Smart Shopping Nepal - Multi-Store Price Comparison Engine.

Phase 3 Enhancements:
1. Advanced Comparison Metrics:
   - Lowest total price, highest total price, average total price.
   - Potential absolute savings (NPR) and savings percentage.
   - Count of unique stores offering the product.
2. Data Quality & Availability Rules:
   - Excludes out-of-stock listings (availability_status == 'out_of_stock').
   - Excludes invalid or missing prices (never treats blank as 0).
   - Clearly flags missing or unknown shipping costs.
   - Tier Isolation: Never mixes demo data with verified real listings.
   - Preserves backward compatibility with existing summary schema (total_price_npr alias).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


def calculate_total_price(price_npr: Any, shipping_cost_npr: Any = 0.0) -> float:
    """
    Computes landed cost (product price + shipping).
    If price_npr is NaN or None, returns np.nan (never 0).
    """
    if pd.isna(price_npr) or price_npr is None:
        return np.nan

    try:
        p = float(price_npr)
        if p <= 0:
            return np.nan
    except (ValueError, TypeError):
        return np.nan

    s = 0.0
    if pd.notna(shipping_cost_npr) and shipping_cost_npr is not None:
        try:
            s_val = float(shipping_cost_npr)
            s = s_val if s_val >= 0 else 0.0
        except (ValueError, TypeError):
            s = 0.0

    return p + s


def compare_product_by_key(
    data: pd.DataFrame,
    product_key: str,
    target_tier: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compares all store listings for a specific canonical product_key.

    Filters:
    - Only listings matching target_tier ('real' vs 'demo') to prevent mixing.
    - Excludes listings where availability_status == 'out_of_stock'.
    - Excludes listings where price is NaN or <= 0.
    """
    options = data[data["product_key"] == product_key].copy()
    if options.empty:
        return {}

    # Tier isolation: Never mix demo data with verified real listings
    if target_tier:
        options = options[options["record_type"] == target_tier]
        if options.empty:
            return {}

    # Ensure total price calculation
    if "total_price_npr" not in options.columns:
        options["total_price_npr"] = options.apply(
            lambda r: calculate_total_price(r.get("price_npr"), r.get("shipping_cost_npr")),
            axis=1
        )

    # Exclude out_of_stock listings
    available_mask = options["availability_status"].fillna("in_stock").str.lower() != "out_of_stock"
    priced_mask = options["total_price_npr"].notna() & (options["total_price_npr"] > 0)
    valid_options = options[available_mask & priced_mask]

    first_row = options.iloc[0]
    prod_name = first_row.get("product_name", product_key)
    brand = first_row.get("brand", "")
    category = first_row.get("category", "")
    record_tier = target_tier or first_row.get("record_type", "real")

    if valid_options.empty:
        # None of the listings currently have active price data
        return {
            "product_name": prod_name,
            "product_key": product_key,
            "brand": brand,
            "category": category,
            "cheapest_store": "No Price Available",
            "lowest_total_price": np.nan,
            "highest_total_price": np.nan,
            "average_total_price": np.nan,
            "total_price_npr": np.nan,  # Backward-compatible alias
            "number_of_stores": len(options["platform"].dropna().unique()),
            "saving_npr": 0.0,
            "saving_percent": 0.0,
            "record_type": record_tier,
            "has_valid_offers": False,
            "store_listings": []
        }

    # Sort listings by total_price_npr ascending
    sorted_options = valid_options.sort_values("total_price_npr").copy()
    cheapest = sorted_options.iloc[0]
    cheapest_price = float(cheapest["total_price_npr"])
    highest_price = float(sorted_options["total_price_npr"].max())
    average_price = float(sorted_options["total_price_npr"].mean())

    saving_npr = highest_price - cheapest_price
    saving_percent = (saving_npr / highest_price * 100.0) if highest_price > 0 else 0.0

    # Build detailed per-store breakdown
    store_listings = []
    for _, opt in sorted_options.iterrows():
        raw_ship = opt.get("shipping_cost_npr")
        is_ship_known = pd.notna(raw_ship) and raw_ship is not None
        store_listings.append({
            "platform": opt.get("platform", "Unknown"),
            "price_npr": float(opt["price_npr"]) if pd.notna(opt.get("price_npr")) else np.nan,
            "shipping_cost_npr": float(raw_ship) if is_ship_known else None,
            "is_shipping_known": is_ship_known,
            "total_price_npr": float(opt["total_price_npr"]),
            "rating": float(opt["rating"]) if pd.notna(opt.get("rating")) else None,
            "review_count": int(opt["review_count"]) if pd.notna(opt.get("review_count")) else None,
            "product_url": opt.get("product_url", ""),
            "last_updated": opt.get("last_updated", ""),
            "availability_status": opt.get("availability_status", "in_stock")
        })

    return {
        "product_name": cheapest.get("product_name", prod_name),
        "product_key": product_key,
        "brand": brand,
        "category": category,
        "cheapest_store": cheapest.get("platform", "Unknown"),
        "lowest_total_price": cheapest_price,
        "highest_total_price": highest_price,
        "average_total_price": round(average_price, 2),
        "total_price_npr": cheapest_price,  # Backward-compatible alias
        "number_of_stores": len(sorted_options["platform"].dropna().unique()),
        "saving_npr": round(saving_npr, 2),
        "saving_percent": round(saving_percent, 2),
        "record_type": record_tier,
        "has_valid_offers": True,
        "store_listings": store_listings
    }


def generate_comparison_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a full comparison summary across all distinct products.
    Maintains tier isolation (processes real/verified separately from demo).
    Matches and extends data/processed/price_comparison_summary.csv.
    """
    summary: List[Dict[str, Any]] = []

    # Identify existing tiers in data
    tiers = data["record_type"].dropna().unique() if "record_type" in data.columns else ["real"]

    for tier in tiers:
        tier_data = data[data["record_type"] == tier]
        for key in tier_data["product_key"].unique():
            if pd.isna(key) or not str(key).strip():
                continue
            result = compare_product_by_key(data, key, target_tier=tier)
            if result:
                summary.append({
                    "product_name": result["product_name"],
                    "product_key": result["product_key"],
                    "brand": result["brand"],
                    "category": result["category"],
                    "cheapest_store": result["cheapest_store"],
                    "lowest_total_price": result["lowest_total_price"],
                    "highest_total_price": result["highest_total_price"],
                    "average_total_price": result["average_total_price"],
                    "total_price_npr": result["total_price_npr"],  # Backward-compatible alias
                    "number_of_stores": result["number_of_stores"],
                    "saving_npr": result["saving_npr"],
                    "saving_percent": result["saving_percent"],
                    "record_type": result["record_type"]
                })

    return pd.DataFrame(summary)
