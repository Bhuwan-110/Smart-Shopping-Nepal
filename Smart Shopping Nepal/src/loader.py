"""
Smart Shopping Nepal - Reusable Data Loading and Export Functions.
Provides robust ingestion of raw product catalogs, source registries,
and safe saving of processed data artifacts without corrupting schemas.
"""

import os
import pandas as pd
from typing import Optional


def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Safely loads a CSV file into a pandas DataFrame.
    Preserves blank/empty cells as NaN instead of zero or default strings.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Ensure blank strings and whitespace are treated as NaN
    df = pd.read_csv(
        filepath,
        keep_default_na=True,
        na_values=["", " ", "NA", "N/A", "null", "None", "nan", "NaN"],
        **kwargs
    )
    return df


def load_raw_catalog(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Loads the canonical product catalog from data/raw/sample_products.csv.
    """
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "data", "raw", "sample_products.csv")

    df = load_csv(filepath)

    # Ensure baseline column types do not coerce NaN into 0
    nullable_numeric_cols = ["price_npr", "shipping_cost_npr", "rating", "review_count"]
    for col in nullable_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # String columns kept as object / string with None/NaN preserved
    string_cols = [
        "product_name", "category", "brand", "model", "variant",
        "platform", "product_url", "product_key"
    ]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(object)

    return df


def load_source_registry(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Loads the platform compliance registry from data/raw/source_registry.csv.
    """
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "data", "raw", "source_registry.csv")

    return load_csv(filepath)


def save_csv(df: pd.DataFrame, filepath: str, index: bool = False, **kwargs) -> str:
    """
    Saves a DataFrame to CSV, ensuring parent directories exist.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    df.to_csv(filepath, index=index, **kwargs)
    return filepath
