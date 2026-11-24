
# SmartShopping Nepal

## Project Overview
SmartShopping Nepal is a product price comparison and recommendation system designed to help customers in Nepal compare products across different online shopping platforms.

## Project Objectives
- Compare prices of the same product across online shopping platforms.
- Display product ratings and review counts when reliable data is available.
- Help customers identify suitable products based on price and other relevant factors.
- Provide links to the original product pages.
- Build a data-driven recommendation system.

## Planned Technologies
- Python 3.10+
- Django (Core Web Application & REST APIs)
- Streamlit (Internal Analytics & Model Prototyping Dashboard)
- Jupyter Notebook (EDA & Prototyping)
- Pandas & NumPy (Data processing)
- Scikit-learn (Recommendation Engine & Machine Learning)
- Matplotlib & Seaborn (Data visualization)

## Project Structure
- `data/raw/` — Original collected data (`sample_products.csv`, `source_registry.csv`)
- `data/processed/` — Cleaned and processed data (`price_comparison_summary.csv`)
- `notebooks/` / `Notebook/` — Data analysis and machine learning experiments (`Model Notebook Code .ipynb`)
- `src/` — Reusable Python modules:
  - `loader.py`: Reusable, schema-safe CSV ingestion & export routines
  - `cleaner.py`: Text normalization, NPR currency & shipping normalization, rating validation
  - `keygen.py`: Backward-compatible composite slug product key generator
  - `validator.py`: URL, price, rating, and platform compliance validation
  - `deduplicator.py`: Exact and cross-store duplicate detection and auditing
  - `matcher.py`: Intelligent Product Matching Engine (Attribute extraction, hard gating & Scikit-learn TF-IDF similarity)
  - `comparator.py`: Multi-store comparison engine (Lowest, highest, average landed cost, savings math, store counting)
  - `ml_preparer.py`: Feature engineering for machine learning and recommendation models
  - `pipeline.py`: End-to-end data processing pipeline orchestrator
  - `collectors/`:
    - `base.py`: Abstract BaseConnector with strict schema validation & failure tracking
    - `manual_import.py`: Authorized manual CSV/JSON import connector
    - `daraz.py`: Daraz Open Platform API connector with ToS compliance safeguards
    - `partner_feed.py`: Direct merchant partner feed connector under written agreements
    - `manager.py`: CollectionManager coordinating multi-platform collection & audit logging
- `data/raw/manual_imports/` — Standardized manual import templates & verified store feeds
- `data/processed/` — Processed artifacts (`cleaned_products.csv`, `price_comparison_summary.csv`, `detailed_store_comparison.csv`, `products_ml_features.csv`)
- `models/` — Saved machine learning models & serialization artifacts
- `reports/` — Analysis, evaluation, data quality, and collection audit reports
- `tests/` — Automated test suite (`test_pipeline.py`, `test_collection.py`, `test_matching.py`)
- `main.py` — Application entry point running the full collection, matching, and comparison pipeline

## Data Policy
Product data will be collected strictly from authorized APIs, partner feeds, permitted sources, or manually verified listings. Automated scraping without express written permission is strictly prohibited. Demo and sample data are clearly distinguished from verified merchant listings.

## Project Status
- **Phase 0 — Project Audit & Architectural Baseline**: Completed.
- **Phase 1 — Data Processing Pipeline**: Completed.
- **Phase 2 — Product Data Collection System**: Completed.
- **Phase 3 — Intelligent Product Matching & Price Comparison**: Completed.
  - Multi-attribute extractor (Brand, Model, RAM, Storage, Condition) and Scikit-learn TF-IDF engine implemented in `src/matcher.py`.
  - False positive prevention: strictly prevents mismatching different models (e.g. A56 vs A55), variants (e.g. 128GB vs 256GB), or conditions (New vs Used).
  - Advanced comparison metrics calculated: lowest, highest, average total landed price (`price + shipping`), potential savings (NPR and %), and store availability counts.
  - Tier isolation strictly enforced: demo listings are never mixed with verified real store listings.
  - Comprehensive labeled test suite implemented in `tests/test_matching.py` (19/19 total project tests passing).
