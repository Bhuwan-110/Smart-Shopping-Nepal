"""
Smart Shopping Nepal - Machine Learning Data Preparation Module.

Prepares cleaned datasets with engineered features for Scikit-Learn:
1. NLP / Text feature representation (combined text for TF-IDF / Content-Based Filtering).
2. Category and Brand categorical encoding.
3. Price logarithmic transformation (log_price) for regression / clustering.
4. Missing rating indicators (never inventing ratings, but signaling availability for models).
"""

import numpy as np
import pandas as pd


def prepare_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the cleaned product dataset into ML-ready feature matrices.
    Saved to data/processed/products_ml_features.csv.
    """
    ml_df = df.copy()

    # 1. Combined feature text for NLP / recommendation systems
    def _create_feature_text(row):
        parts = [
            str(row.get("category_clean", "")),
            str(row.get("brand_clean", "")),
            str(row.get("model_clean", "")),
            str(row.get("variant_clean", "")),
            str(row.get("product_name_clean", ""))
        ]
        return " ".join([p for p in parts if p and p != "nan"]).strip()

    ml_df["ml_feature_text"] = ml_df.apply(_create_feature_text, axis=1)

    # 2. Log-transformed price (avoids skewness in clustering/recommendations)
    # Where price is NaN, log_price stays NaN
    if "price_npr" in ml_df.columns:
        ml_df["log_price_npr"] = ml_df["price_npr"].apply(
            lambda x: np.log1p(float(x)) if pd.notna(x) and float(x) > 0 else np.nan
        )

    # 3. Missing Value Indicators for ML algorithms
    # This ensures ML models know whether rating was observed or missing without inventing values
    if "rating" in ml_df.columns:
        ml_df["has_rating"] = ml_df["rating"].apply(lambda r: 1 if pd.notna(r) else 0)
    else:
        ml_df["has_rating"] = 0

    if "price_npr" in ml_df.columns:
        ml_df["has_price"] = ml_df["price_npr"].apply(lambda p: 1 if pd.notna(p) else 0)
    else:
        ml_df["has_price"] = 0

    # 4. Filter or organize key columns for ML consumption
    ml_columns = [
        "product_key",
        "product_name",
        "brand",
        "category",
        "model",
        "variant",
        "ml_feature_text",
        "price_npr",
        "log_price_npr",
        "rating",
        "review_count",
        "has_rating",
        "has_price",
        "record_type"
    ]

    selected_cols = [c for c in ml_columns if c in ml_df.columns]
    return ml_df[selected_cols]
