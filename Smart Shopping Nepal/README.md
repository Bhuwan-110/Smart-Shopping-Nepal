# SmartShopping Nepal

## Project Overview
SmartShopping Nepal is an end-to-end product price comparison, smart recommendation, and analytics web application built with Python and Django. Designed specifically for customers and online shoppers in Nepal, it normalizes product information, compares true landed costs (Product Price + Shipping Fee in Nepalese Rupees), and provides transparent price insights across verified Nepali e-commerce platforms.

---

## Key Features & Completed Phases

### 1. Phase 1 — Data Processing Pipeline
- Reusable CSV ingestion, schema validation, and data cleaning routines (`src/loader.py`, `src/cleaner.py`).
- NPR currency normalization and delivery fee parsing without ever assuming blank prices are zero.
- Backward-compatible composite slug product keys (`src/keygen.py`).
- Exact and fuzzy cross-store deduplication (`src/deduplicator.py`).
- Clear data tier tagging: `real` (verified listings), `sample` (canonical catalog templates), and `demo`.

### 2. Phase 2 — Authorized Data Collection System
- Strict adherence to data governance policies: automated scraping without express permission is prohibited.
- Modular connector architecture (`src/collectors/`):
  - `manual_import.py`: Authorized manual CSV and JSON imports.
  - `daraz.py`: Official Open Platform API connector with HMAC-SHA256 signing and throttling.
  - `partner_feed.py`: Merchant partner direct feeds.
  - `manager.py`: Centralized orchestrator tracking collection logs and audit trails.

### 3. Phase 3 — Intelligent Product Matching & Price Comparison
- Multi-attribute spec extraction: brand, model, RAM, storage, color, and hardware condition.
- Hard-gating rules: strictly prevents false positives across different models (e.g. Galaxy A56 vs A55) or variants (e.g. 128GB vs 256GB).
- Scikit-learn TF-IDF similarity calculation for flexible cross-store title grouping (`src/matcher.py`).
- Multi-store price comparison engine: lowest price, highest price, average price, potential savings (NPR and %), and store availability counts (`src/comparator.py`).

### 4. Phase 5 — Django Web Application & Relational Database Integration
- Full-stack Django web application running on PostgreSQL (with automatic zero-dependency SQLite fallback for local development).
- **10 Core Website Modules**:
  1. **Homepage** (`/`): Hero search banner, key catalog metrics, featured multi-store deals, browse by category, and recently added catalog products.
  2. **Product Search** (`/search/`): Real-time keyword search with faceted filtering by category, brand, min/max price, and sorting.
  3. **Product Categories** (`/categories/`, `/category/<category_name>/`): Comprehensive taxonomy browsing.
  4. **Product Details** (`/product/<id>/`): Specification breakdown, multi-store comparison table, best price badge, and wishlist toggle.
  5. **Price Comparison** (`/compare/`, `/compare/<product_key>/`): Detailed multi-store comparison matrix highlighting cheapest store, landed costs, and potential savings.
  6. **Product Recommendations** (`/recommendations/`): Smart TF-IDF content-based product pairings.
  7. **Store / Platform Directory** (`/stores/`, `/store/<slug>/`): Compliance registry showing data access methods and authorization status for Nepali e-commerce platforms.
  8. **User Authentication** (`/register/`, `/login/`, `/logout/`): Standard user accounts with session management.
  9. **User Wishlist** (`/wishlist/`, `/wishlist/add/<id>/`, `/wishlist/remove/<id>/`): Saved products with target price drop tracking.
  10. **Operations & Admin Dashboard** (`/dashboard/`): Internal analytics, catalog metrics, category distributions, collection audit logs, and direct link to Django Admin (`/admin/`).

---

## Database Architecture

The relational schema is implemented in `shopping/models.py`:
- `Product`: Canonical product identity (Brand, Model, Category, Canonical Key).
- `ProductVariant`: Hardware variants (RAM, Storage, Color, SKU).
- `Store`: Registered Nepali stores and platforms with compliance audit attributes.
- `ProductListing`: Per-store product offerings, price in NPR, delivery cost, availability, ratings, and live store URL. Automatically computes landed cost (`total_price_npr = price_npr + shipping_cost_npr`).
- `PriceHistory`: Temporal audit log tracking price changes over time.
- `ProductReview`: User ratings and verified reviews.
- `Wishlist`: User saved products and target price drop alerts.
- `RecommendationResult`: Precomputed Scikit-Learn TF-IDF similarity scores.
- `CollectionLog`: Audit trail of ingestion runs and API sync records.

---

## Project Structure

```text
Smart Shopping Nepal/
├── data/
│   ├── raw/
│   │   ├── sample_products.csv             # Baseline canonical catalog
│   │   ├── source_registry.csv             # Authorized platform compliance registry
│   │   └── manual_imports/                 # Authorized partner imports
│   └── processed/
│       ├── cleaned_products.csv            # Cleaned, standardized listings
│       ├── price_comparison_summary.csv    # Multi-store price comparison summary
│       ├── detailed_store_comparison.csv   # Granular per-store comparison
│       └── products_ml_features.csv        # TF-IDF feature matrix
├── notebooks/                              # Jupyter analysis and model experimentation
├── reports/                                # Ingestion and quality audit reports
├── src/                                    # Reusable Python data processing modules
│   ├── cleaner.py                          # Currency, shipping & text cleaner
│   ├── comparator.py                       # Price comparison engine
│   ├── deduplicator.py                     # Duplicate detection
│   ├── keygen.py                           # Canonical key generation
│   ├── loader.py                           # Schema-validated CSV loader
│   ├── matcher.py                          # Scikit-learn TF-IDF matching engine
│   ├── ml_preparer.py                      # Recommendation feature engineering
│   ├── pipeline.py                         # End-to-end data processing pipeline
│   └── collectors/                         # Authorized data collection connectors
├── smart_shopping_web/                     # Django core project package
│   ├── settings.py                         # Dual PostgreSQL / SQLite configuration
│   ├── urls.py                             # Project-level URL routing
│   ├── wsgi.py                             # WSGI server entry
│   └── asgi.py                             # ASGI server entry
├── shopping/                               # Django shopping web application
│   ├── models.py                           # 10 Core relational models
│   ├── views.py                            # View controllers for all 10 modules
│   ├── urls.py                             # App URL routing
│   ├── admin.py                            # Django Admin configuration
│   ├── management/commands/sync_pipeline.py # Ingestion & pipeline sync command
│   └── tests.py                            # Web app test suite
├── static/                                 # Static assets (CSS, JS)
│   └── css/styles.css                      # Custom responsive UI styles
├── templates/                              # HTML templates (Bootstrap 5)
│   ├── base.html                           # Base layout with navbar and footer
│   └── shopping/                           # 10 Module templates
├── tests/                                  # Automated test suites
│   ├── test_pipeline.py                    # Phase 1 pipeline tests
│   ├── test_collection.py                  # Phase 2 collection tests
│   ├── test_matching.py                    # Phase 3 matching tests
│   └── test_django_app.py                  # Phase 5 Django integration tests
├── manage.py                               # Django CLI entrypoint
├── main.py                                 # Pipeline runner & application controller
└── README.md                               # Project documentation
```

---

## Getting Started

### 1. Requirements
- Python 3.10+
- Django 5+
- Pandas, NumPy, Scikit-learn

### 2. Run Pipeline & Ingest Data into Database
Run the complete collection, matching, comparison, and database synchronization pipeline:
```bash
python main.py
```
Or execute the Django synchronization command directly:
```bash
python manage.py sync_pipeline
```

### 3. Launch the Django Web Application
Start the local development server:
```bash
python manage.py runserver
```
or via the main runner:
```bash
python main.py --serve
```
Open your browser at: **`http://127.0.0.1:8000/`**

### 4. Access Django Admin
To access the administrative panel at `http://127.0.0.1:8000/admin/`, create a superuser:
```bash
python manage.py createsuperuser
```

---

## Running Automated Tests

Run the complete test suite (37 passing tests across all phases):
```bash
# 1. Run Django Web & Database Integration Tests (18 tests)
python manage.py test

# 2. Run Data Pipeline, Collection, and Matching Tests (19 tests)
python -m unittest tests.test_pipeline tests.test_collection tests.test_matching
```
All tests pass cleanly with zero warnings or errors.
