
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
  - `comparator.py`: Landed cost calculation, multi-store comparison, and savings math
  - `ml_preparer.py`: Feature engineering for machine learning and recommendation models
  - `pipeline.py`: End-to-end data processing pipeline orchestrator
  - `collectors/`:
    - `base.py`: Abstract BaseConnector with strict schema validation & failure tracking
    - `manual_import.py`: Authorized manual CSV/JSON import connector
    - `daraz.py`: Daraz Open Platform API connector with ToS compliance safeguards
    - `partner_feed.py`: Direct merchant partner feed connector under written agreements
    - `manager.py`: CollectionManager coordinating multi-platform collection & audit logging
- `data/raw/manual_imports/` — Standardized manual import templates & verified store feeds
- `models/` — Saved machine learning models & serialization artifacts
- `reports/` — Analysis, evaluation, data quality, and collection audit reports
- `tests/` — Automated unit and integration test suite (`test_pipeline.py`, `test_collection.py`)
- `main.py` — Application entry point running the full collection and comparison pipeline

## Data Policy
Product data will be collected strictly from authorized APIs, partner feeds, permitted sources, or manually verified listings. Automated scraping without express written permission is strictly prohibited. Demo and sample data are clearly distinguished from verified merchant listings.

## Project Status
- **Phase 0 — Project Audit & Architectural Baseline**: Completed.
- **Phase 1 — Data Processing Pipeline**: Completed.
- **Phase 2 — Product Data Collection System**: Completed.
  - Reusable platform connectors (`BaseConnector`, `ManualImportConnector`, `DarazConnector`, `PartnerFeedConnector`) and `CollectionManager` implemented.
  - Platform compliance verified in `data/raw/source_registry.csv` with anti-scraping safeguards.
  - Full collection audit trail generated in `reports/collection_audit.json` with failure tracking and timestamps.
  - Standardized manual import templates and verified store feeds created under `data/raw/manual_imports/`.
  - Integrated with Phase 1 pipeline, computing multi-store comparisons across Daraz Nepal, Hukut Store, and Oliz Store.
  - Comprehensive automated test suite passing (11/11 tests passing across `test_pipeline.py` and `test_collection.py`).
