"""
Smart Shopping Nepal - Text & Numeric Normalization Module.

Strict Data Rules:
1. Never treat blank prices as zero (returns np.nan).
2. Never invent missing ratings or review counts (preserves np.nan).
3. Clearly distinguishes record types: 'real', 'verified', 'sample', 'demo'.
4. Normalizes all prices and shipping fees into NPR.
"""

import re
import numpy as np
import pandas as pd
from typing import Any, Optional

# Known brand capitalization overrides
BRAND_CASING_MAP = {
    "samsung": "Samsung",
    "hp": "HP",
    "apple": "Apple",
    "dell": "Dell",
    "lenovo": "Lenovo",
    "asus": "Asus",
    "acer": "Acer",
    "xiaomi": "Xiaomi",
    "redmi": "Redmi",
    "realme": "Realme",
    "oneplus": "OnePlus",
    "sony": "Sony",
    "boat": "boAt",
    "noise": "Noise",
    "anker": "Anker",
    "lg": "LG",
    "cg": "CG",
    "philips": "Philips",
    "baltra": "Baltra",
    "yasuda": "Yasuda",
    "midea": "Midea",
    "goldstar": "Goldstar",
    "caliber": "Caliber",
    "adidas": "Adidas",
    "nike": "Nike",
    "puma": "Puma",
    "levi's": "Levi's",
    "levis": "Levi's",
    "zara": "Zara",
    "h&m": "H&M",
    "erke": "Erke",
    "fastrack": "Fastrack",
    "titan": "Titan",
    "ray-ban": "Ray-Ban",
    "wildhorn": "WildHorn",
    "baggit": "Baggit",
    "cetaphil": "Cetaphil",
    "the derma co": "The Derma Co",
    "nivea": "Nivea",
    "mamaearth": "Mamaearth",
    "l'oreal": "L'Oreal",
    "himalaya": "Himalaya",
    "nilkamal": "Nilkamal",
    "solimo": "Solimo",
    "kurl-on": "Kurl-on",
    "cortina": "Cortina",
    "yonex": "Yonex",
    "cosco": "Cosco",
    "nivia": "Nivia",
    "strauss": "Strauss",
    "decathlon": "Decathlon",
    "doms": "Doms",
    "classmate": "Classmate",
    "logitech": "Logitech",
    "sandisk": "SanDisk",
    "baseus": "Baseus",
    "tp-link": "TP-Link",
    "belkin": "Belkin",
    "spigen": "Spigen",
    "remax": "Remax",
    "western digital": "Western Digital",
    "example brand": "Example Brand"
}

CATEGORY_CASING_MAP = {
    # 15 Core Sectors (A through O)
    "mobile phones": "Mobile Phone",
    "mobile phone": "Mobile Phone",
    "mobile": "Mobile Phone",
    "smartphones": "Mobile Phone",
    "laptops and computers": "Laptops and Computers",
    "laptop": "Laptop",
    "laptops": "Laptop",
    "mobile accessories": "Mobile Accessories",
    "computer accessories": "Computer Accessories",
    "audio and wearables": "Audio and Wearables",
    "earbuds": "Earbuds",
    "audio": "Audio and Wearables",
    "smartwatch": "Audio and Wearables",
    "men's clothing": "Men's Clothing",
    "mens clothing": "Men's Clothing",
    "women's clothing": "Women's Clothing",
    "womens clothing": "Women's Clothing",
    "kids' clothing": "Kids' Clothing",
    "kids clothing": "Kids' Clothing",
    "footwear": "Footwear",
    "shoes": "Footwear",
    "fashion accessories": "Fashion Accessories",
    "home appliances": "Home Appliances",
    "appliances": "Home Appliances",
    "beauty and personal care": "Beauty and Personal Care",
    "beauty": "Beauty and Personal Care",
    "furniture and home": "Furniture and Home",
    "furniture": "Furniture and Home",
    "sports and fitness": "Sports and Fitness",
    "sports": "Sports and Fitness",
    "books and stationery": "Books and Stationery",
    "books": "Books and Stationery",
    "stationery": "Books and Stationery",
    "tablet": "Mobile Phones",
    "accessories": "Mobile Accessories"
}


def clean_text(text: Any) -> str:
    """Strips leading/trailing whitespace and converts to lower case."""
    if pd.isna(text) or text is None:
        return ""
    text_str = str(text).strip()
    if text_str.lower() in ["nan", "none", "null"]:
        return ""
    return text_str.lower()


def standardize_brand(brand: Any) -> str:
    """Standardizes brand casing while keeping original value if unknown."""
    if pd.isna(brand) or brand is None:
        return ""
    cleaned = clean_text(brand)
    return BRAND_CASING_MAP.get(cleaned, str(brand).strip().title())


def standardize_category(category: Any) -> str:
    """Standardizes product category casing."""
    if pd.isna(category) or category is None:
        return ""
    cleaned = clean_text(category)
    return CATEGORY_CASING_MAP.get(cleaned, str(category).strip().title())


def parse_currency_to_npr(price: Any) -> float:
    """
    Normalizes price values to numeric NPR float.
    RULE: Never treat blank or missing prices as zero. Returns np.nan.
    Strips currency symbols ('NPR', 'Rs.', 'NRs', 'Rs', commas).
    """
    if pd.isna(price) or price is None:
        return np.nan

    price_str = str(price).strip()
    if not price_str or price_str.lower() in ["nan", "none", "null", ""]:
        return np.nan

    # Strip currency prefixes (e.g. Rs., NPR, NRs, Rs)
    s = re.sub(r"^(npr|rs\.?|nrs\.?|रू\.?)\s*", "", price_str, flags=re.IGNORECASE)
    # Remove commas
    s = s.replace(",", "").strip()

    # Extract valid float number
    match = re.search(r"\d+(\.\d+)?", s)
    if not match:
        return np.nan

    try:
        val = float(match.group(0))
        # In e-commerce, a product price must be > 0. Blank/0 is treated as missing.
        return val if val > 0 else np.nan
    except (ValueError, TypeError):
        return np.nan


def parse_shipping_cost_npr(shipping: Any) -> float:
    """
    Normalizes shipping cost to NPR float.
    - Explicit 'free' or '0' -> 0.0
    - Missing/blank -> np.nan (distinguishes free from unknown shipping)
    """
    if pd.isna(shipping) or shipping is None:
        return np.nan

    shipping_str = str(shipping).strip()
    if not shipping_str or shipping_str.lower() in ["nan", "none", "null", ""]:
        return np.nan

    if shipping_str.lower() in ["free", "0", "0.0"]:
        return 0.0

    s = re.sub(r"^(npr|rs\.?|nrs\.?|रू\.?)\s*", "", shipping_str, flags=re.IGNORECASE)
    s = s.replace(",", "").strip()

    match = re.search(r"\d+(\.\d+)?", s)
    if not match:
        return np.nan

    try:
        val = float(match.group(0))
        return val if val >= 0 else np.nan
    except (ValueError, TypeError):
        return np.nan


def parse_rating(rating: Any) -> float:
    """
    Validates rating on a 0.0 to 5.0 scale.
    RULE: Never invent missing ratings. Returns np.nan.
    """
    if pd.isna(rating) or rating is None:
        return np.nan

    rating_str = str(rating).strip()
    if not rating_str or rating_str.lower() in ["nan", "none", "null", ""]:
        return np.nan

    try:
        val = float(rating_str)
        return val if 0.0 <= val <= 5.0 else np.nan
    except (ValueError, TypeError):
        return np.nan


def parse_review_count(review_count: Any) -> float:
    """
    Normalizes review count to integer/float >= 0.
    RULE: Never invent missing review counts. Returns np.nan.
    """
    if pd.isna(review_count) or review_count is None:
        return np.nan

    cnt_str = str(review_count).strip()
    if not cnt_str or cnt_str.lower() in ["nan", "none", "null", ""]:
        return np.nan

    try:
        val = int(float(cnt_str))
        return float(val) if val >= 0 else np.nan
    except (ValueError, TypeError):
        return np.nan


def determine_record_type(platform: Any, explicit_type: Optional[str] = None) -> str:
    """
    Classifies listings into data tiers:
    - 'sample': Canonical catalog definitions without live store data.
    - 'demo': Simulated demo store data (Demo Store A, B, etc.).
    - 'verified': Manually verified or partner feeds.
    - 'real': Live scraped or API-sourced listings.
    """
    if explicit_type:
        return explicit_type

    if pd.isna(platform) or not str(platform).strip() or str(platform).lower() in ["nan", "none", "null"]:
        return "sample"

    p_lower = str(platform).lower()
    if "demo" in p_lower:
        return "demo"
    if "verified" in p_lower:
        return "verified"

    return "real"


def clean_product_dataframe(
    df: pd.DataFrame,
    default_record_type: Optional[str] = None
) -> pd.DataFrame:
    """
    Full dataframe normalization conforming to all Phase 1 requirements:
    1. Trims and standardizes product names, brands, categories, models.
    2. Generates lowercase clean columns (_clean) for consistent matching.
    3. Normalizes prices and shipping costs into NPR (without treating blank as 0).
    4. Validates ratings and review counts (without inventing values).
    5. Computes total_price_npr (price + shipping, keeping NaN if price is NaN).
    6. Tags record_type ('real', 'verified', 'sample', 'demo').
    """
    cleaned_df = df.copy()

    # 1. Clean and standardize text columns
    for col in ["product_name", "category", "brand", "model", "variant", "platform", "product_url"]:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].apply(lambda x: str(x).strip() if pd.notna(x) else np.nan)

    # Standardize casing for primary display columns
    if "brand" in cleaned_df.columns:
        cleaned_df["brand"] = cleaned_df["brand"].apply(standardize_brand)
    if "category" in cleaned_df.columns:
        cleaned_df["category"] = cleaned_df["category"].apply(standardize_category)

    # 2. Lowercase matching fields (_clean)
    if "product_name" in cleaned_df.columns:
        cleaned_df["product_name_clean"] = cleaned_df["product_name"].apply(clean_text)
    if "category" in cleaned_df.columns:
        cleaned_df["category_clean"] = cleaned_df["category"].apply(clean_text)
    if "brand" in cleaned_df.columns:
        cleaned_df["brand_clean"] = cleaned_df["brand"].apply(clean_text)
    if "model" in cleaned_df.columns:
        cleaned_df["model_clean"] = cleaned_df["model"].apply(clean_text)
    if "variant" in cleaned_df.columns:
        cleaned_df["variant_clean"] = cleaned_df["variant"].apply(clean_text)

    # 3. Numeric Normalization (Strictly preserving NaNs)
    if "price_npr" in cleaned_df.columns:
        cleaned_df["price_npr"] = cleaned_df["price_npr"].apply(parse_currency_to_npr)

    if "shipping_cost_npr" in cleaned_df.columns:
        cleaned_df["shipping_cost_npr"] = cleaned_df["shipping_cost_npr"].apply(parse_shipping_cost_npr)

    if "rating" in cleaned_df.columns:
        cleaned_df["rating"] = cleaned_df["rating"].apply(parse_rating)

    if "review_count" in cleaned_df.columns:
        cleaned_df["review_count"] = cleaned_df["review_count"].apply(parse_review_count)

    # 4. Total Price Calculation: Price + Shipping
    # If price is NaN, total_price is NaN!
    # If shipping is NaN, we treat total as price_npr (with unknown shipping).
    def _compute_total(row):
        p = row.get("price_npr")
        s = row.get("shipping_cost_npr")
        if pd.isna(p):
            return np.nan
        s_val = 0.0 if pd.isna(s) else float(s)
        return float(p) + s_val

    cleaned_df["total_price_npr"] = cleaned_df.apply(_compute_total, axis=1)

    # 5. Record Type Classification
    if "platform" in cleaned_df.columns:
        cleaned_df["record_type"] = cleaned_df["platform"].apply(
            lambda p: determine_record_type(p, default_record_type)
        )
    else:
        cleaned_df["record_type"] = default_record_type or "sample"

    return cleaned_df
