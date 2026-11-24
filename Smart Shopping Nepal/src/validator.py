"""
Smart Shopping Nepal - Data Validation Module.
Validates URLs, ratings, prices, and cross-references source platforms
against data/raw/source_registry.csv.
"""

import re
import pandas as pd
from typing import Set, Tuple, Any

# Standard URL regex checking for http:// or https:// with valid hostname
URL_REGEX = re.compile(
    r"^(https?:\/\/)?"  # protocol
    r"(([a-zA-Z0-9_-]+\.)+[a-zA-Z]{2,})"  # domain name
    r"(:\d+)?"  # port
    r"(\/[a-zA-Z0-9_\-\.\/~\+\%#\?=&]*)?$",  # path/query
    re.IGNORECASE
)


def validate_url(url: Any) -> bool:
    """
    Checks if a URL string is syntactically valid and uses HTTP/HTTPS.
    Returns False for NaN/empty values.
    """
    if pd.isna(url) or url is None:
        return False
    url_str = str(url).strip()
    if not url_str or url_str.lower() in ["nan", "none", "null"]:
        return False
    return bool(URL_REGEX.match(url_str))


def validate_rating(rating: Any) -> bool:
    """
    Validates if a rating falls within the standard [0.0, 5.0] range.
    Returns False if NaN or out of bounds.
    """
    if pd.isna(rating) or rating is None:
        return False
    try:
        val = float(rating)
        return 0.0 <= val <= 5.0
    except (ValueError, TypeError):
        return False


def validate_review_count(review_count: Any) -> bool:
    """
    Validates if a review count is a non-negative integer.
    """
    if pd.isna(review_count) or review_count is None:
        return False
    try:
        val = int(review_count)
        return val >= 0
    except (ValueError, TypeError):
        return False


def validate_price(price: Any) -> bool:
    """
    Validates if a price is positive and non-zero.
    RULE: Never treat blank or 0 prices as valid product prices.
    """
    if pd.isna(price) or price is None:
        return False
    try:
        val = float(price)
        return val > 0.0
    except (ValueError, TypeError):
        return False


def validate_source_platform(platform: Any, registered_platforms: Set[str]) -> bool:
    """
    Checks if a platform is registered in source_registry.csv.
    """
    if pd.isna(platform) or platform is None:
        return False
    return str(platform).strip().lower() in {p.lower() for p in registered_platforms}


def validate_listings_dataframe(
    df: pd.DataFrame,
    registered_platforms: Set[str]
) -> pd.DataFrame:
    """
    Appends audit/validation flags to listings DataFrame:
    - is_valid_url (bool)
    - is_valid_price (bool)
    - is_valid_rating (bool or None)
    - is_registered_platform (bool)
    """
    validated_df = df.copy()

    validated_df["is_valid_url"] = validated_df["product_url"].apply(validate_url)
    validated_df["is_valid_price"] = validated_df["price_npr"].apply(validate_price)
    validated_df["is_registered_platform"] = validated_df["platform"].apply(
        lambda p: validate_source_platform(p, registered_platforms)
    )

    return validated_df
