
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
- `models/` — Saved machine learning models & serialization artifacts
- `reports/` — Analysis, evaluation, and data quality reports (`data_quality_report.json`)
- `tests/` — Automated unit and integration test suite (`test_pipeline.py`)
- `main.py` — Application entry point running the full Phase 1 processing pipeline

## Data Policy
Product data will be collected from authorized APIs, partner feeds, permitted sources, or manually verified listings. Demo data will be clearly identified as demo data.

## Project Status
- **Phase 0 — Project Audit & Architectural Baseline**: Completed.
- **Phase 1 — Data Processing Pipeline**: Completed.
  - Reusable loader, cleaner, validator, deduplicator, key generator, and ML feature preparer implemented in `src/`.
  - Strict data rules enforced: blank prices are never treated as zero, missing ratings/reviews are never invented, record types (`sample`, `demo`, `verified`, `real`) are strictly classified.
  - Processed artifacts generated in `data/processed/` (`cleaned_products.csv`, `price_comparison_summary.csv`, `products_ml_features.csv`).
  - Data quality audit generated in `reports/data_quality_report.json`.
  - Comprehensive unit test suite implemented in `tests/test_pipeline.py` (6/6 passing).
  - Raw datasets preserved completely intact.
