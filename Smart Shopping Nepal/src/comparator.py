"""
Smart Shopping Nepal - Multi-Store Price Comparison Engine.

Strict Data Rules:
1. Never treats blank or NaN prices as zero.
2. If prices are missing, marks them as unpriced rather than free.
3. Computes accurate absolute (NPR) and percentage savings.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List


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


def compare_product_by_key(data: pd.DataFrame, product_key: str) -> Dict[str, Any]:
    """
    Compares all store listings for a specific product_key.
    Filters out unpriced listings so blank prices are not treated as 0.
    """
    options = data[data["product_key"] == product_key].copy()
    if options.empty:
        return {}

    # Calculate total price if not already present
    if "total_price_npr" not in options.columns:
        options["total_price_npr"] = options.apply(
            lambda r: calculate_total_price(r.get("price_npr"), r.get("shipping_cost_npr")),
            axis=1
        )

    # Filter for listings that actually have a valid price (> 0)
    valid_options = options[options["total_price_npr"].notna() & (options["total_price_npr"] > 0)]

    first_row = options.iloc[0]
    prod_name = first_row.get("product_name", product_key)

    if valid_options.empty:
        # None of the listings currently have price data
        return {
            "product_name": prod_name,
            "product_key": product_key,
            "cheapest_store": "No Price Available",
            "total_price_npr": np.nan,
            "highest_price_npr": np.nan,
            "number_of_stores": len(options),
            "saving_npr": 0.0,
            "saving_percent": 0.0,
            "listings": []
        }

    sorted_options = valid_options.sort_values("total_price_npr")
    cheapest = sorted_options.iloc[0]
    cheapest_price = float(cheapest["total_price_npr"])
    highest_price = float(sorted_options["total_price_npr"].max())

    saving = highest_price - cheapest_price
    saving_percent = (saving / highest_price * 100.0) if highest_price > 0 else 0.0

    return {
        "product_name": cheapest.get("product_name", prod_name),
        "product_key": product_key,
        "cheapest_store": cheapest.get("platform", "Unknown"),
        "total_price_npr": cheapest_price,
        "highest_price_npr": highest_price,
        "number_of_stores": len(sorted_options),
        "saving_npr": round(saving, 2),
        "saving_percent": round(saving_percent, 2),
        "listings": sorted_options[["product_name", "platform", "total_price_npr"]].to_dict(orient="records")
    }


def generate_comparison_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a full comparison summary across all distinct products.
    Matches the schema of data/processed/price_comparison_summary.csv.
    """
    summary: List[Dict[str, Any]] = []

    for key in data["product_key"].unique():
        if pd.isna(key) or not str(key).strip():
            continue
        result = compare_product_by_key(data, key)
        if result:
            summary.append({
                "product_name": result["product_name"],
                "cheapest_store": result["cheapest_store"],
                "total_price_npr": result["total_price_npr"],
                "number_of_stores": result["number_of_stores"],
                "saving_npr": result["saving_npr"],
                "saving_percent": result["saving_percent"]
            })

    return pd.DataFrame(summary)
