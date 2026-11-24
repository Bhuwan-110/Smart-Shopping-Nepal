"""
Smart Shopping Nepal - Deduplication & Record Audit Module.

Identifies and handles duplicate product records:
1. Exact duplicates: Same product_key, same platform, same price, same variant.
2. Platform price conflicts: Same product_key, same platform, different price.
3. Multi-store distribution: Same product_key across multiple platforms (intended comparison feature).
"""

import pandas as pd
from typing import Dict, Any, Tuple


def identify_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Audits the DataFrame for duplicates without destructive deletion.
    Appends an 'is_duplicate' boolean column and returns an audit metrics dictionary.
    """
    df_flagged = df.copy()

    # If platform is present, duplicate listing means same product_key on same platform
    if "platform" in df_flagged.columns and "product_key" in df_flagged.columns:
        subset_cols = ["product_key", "platform"]
    elif "product_key" in df_flagged.columns:
        subset_cols = ["product_key"]
    else:
        subset_cols = ["product_name"]

    # Flag duplicates (keep='first' marks later occurrences as duplicates)
    df_flagged["is_duplicate"] = df_flagged.duplicated(subset=subset_cols, keep="first")

    # Calculate audit metrics
    total_records = len(df_flagged)
    duplicate_count = int(df_flagged["is_duplicate"].sum())
    unique_products_count = int(df_flagged["product_key"].nunique()) if "product_key" in df_flagged.columns else 0

    # Cross-store presence (number of products available on > 1 store)
    multi_store_products = 0
    if "platform" in df_flagged.columns and "product_key" in df_flagged.columns:
        platform_counts = df_flagged.groupby("product_key")["platform"].nunique()
        multi_store_products = int((platform_counts > 1).sum())

    audit_summary = {
        "total_records": total_records,
        "duplicate_count": duplicate_count,
        "unique_products_count": unique_products_count,
        "multi_store_products_count": multi_store_products,
        "duplicate_rate_percent": round((duplicate_count / total_records * 100), 2) if total_records > 0 else 0.0
    }

    return df_flagged, audit_summary


def remove_duplicate_listings(
    df: pd.DataFrame,
    strategy: str = "keep_first"
) -> pd.DataFrame:
    """
    Safely removes duplicate listings on the same platform based on strategy:
    - 'keep_first': Retains the first encountered row.
    - 'keep_lowest_price': Retains the listing with lowest total price.
    """
    if "platform" not in df.columns or "product_key" not in df.columns:
        return df.drop_duplicates(keep="first")

    if strategy == "keep_lowest_price" and "total_price_npr" in df.columns:
        # Sort by total_price ascending, then drop duplicate product_key + platform
        sorted_df = df.sort_values("total_price_npr", ascending=True)
        return sorted_df.drop_duplicates(subset=["product_key", "platform"], keep="first").reset_index(drop=True)

    return df.drop_duplicates(subset=["product_key", "platform"], keep="first").reset_index(drop=True)
