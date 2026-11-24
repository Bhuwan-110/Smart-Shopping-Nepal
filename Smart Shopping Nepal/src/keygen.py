"""
Smart Shopping Nepal - Product Identifier Module.

Generates consistent, deterministic product identifiers:
1. Preserves full backward-compatibility with existing notebook product_key format:
   brand_clean + "_" + model_clean + "_" + variant_clean (with '/' and ' ' replaced by '_').
2. Provides fallback logic when variant or model is empty.
3. Generates a global SKU / canonical product ID.
"""

import pandas as pd
from src.cleaner import clean_text


def generate_product_key(brand: str, model: str, variant: str) -> str:
    """
    Generates a standardized product_key.
    Example: 'Samsung', 'Galaxy A56', '8GB/256GB' -> 'samsung_galaxy_a56_8gb_256gb'
    If variant is empty or None, defaults to 'standard'.
    """
    b = clean_text(brand) or "unknown"
    m = clean_text(model) or "generic"
    v = clean_text(variant) or "standard"

    raw_key = f"{b}_{m}_{v}"
    # Replace spaces, slashes, dashes, and duplicate underscores
    sanitized = raw_key.replace("/", "_").replace(" ", "_").replace("-", "_")
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_")


def add_product_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures every row in the DataFrame has a consistent, compliant product_key.
    Maintains compatibility with existing sample_products.csv and notebook outputs.
    """
    df = df.copy()

    # Ensure clean text fields exist
    for col in ["brand", "model", "variant"]:
        clean_col = f"{col}_clean"
        if clean_col not in df.columns:
            df[clean_col] = df[col].apply(clean_text) if col in df.columns else ""

    df["product_key"] = df.apply(
        lambda r: generate_product_key(r["brand_clean"], r["model_clean"], r["variant_clean"]),
        axis=1
    )

    return df
