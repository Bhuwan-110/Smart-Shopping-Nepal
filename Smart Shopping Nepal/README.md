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

### 2. Phase 2 — Multi-Store Product Collection & Sector-Wise Catalogue
- Comprehensive 15-Sector Product Catalogue across all major consumer categories in Nepal (A through O):
  - Sector A: Mobile Phones (Samsung, Apple, Xiaomi, OnePlus)
  - Sector B: Laptops and Computers (HP, Acer, Dell, Lenovo, Apple)
  - Sector C: Mobile Accessories (Anker, Spigen, Baseus, Remax)
  - Sector D: Computer Accessories (Logitech, SanDisk, Western Digital)
  - Sector E: Audio and Wearables (Anker Soundcore, boAt, Noise, Apple AirPods)
  - Sector F: Men's Clothing (Levi's, Classic Polo, Casual Jackets)
  - Sector G: Women's Clothing (Kurti Sets, Zara Dresses)
  - Sector H: Kids' Clothing (Toddler Sets, Sweatshirts)
  - Sector I: Footwear (Goldstar Trekker, Caliber Casual, Erke)
  - Sector J: Fashion Accessories (Fastrack Aviator, WildHorn Leather Wallets, Titan)
  - Sector K: Home Appliances (Baltra Rice Cookers, CG Microwave Ovens, Mixer Grinders)
  - Sector L: Beauty and Personal Care (Cetaphil Cleansers, The Derma Co Serums, Himalaya)
  - Sector M: Furniture and Home (Nilkamal Ergonomic Chairs, Modern Study Desks)
  - Sector N: Sports and Fitness (Yonex Badminton Rackets, Nivia Match Footballs, Yoga Mats)
  - Sector O: Books and Stationery (Doms Brush Pens, Classmate Spiral Notebooks)
- Complete 24-field schema adherence across all listings.
- Multi-store cross-retailer offerings across competing Nepali stores (Daraz, Hukut, Oliz, Fatafat Sewa, Mobile Mandu, iShop, Bizbazar, SastoDeal, Hamrobazaar, Neo Store, DealAyo, OnePasal, Tukaatu, Hamromall, AfnoBazar, Mero ePasal).
- Strict data governance: automated bot scraping is prohibited; authorized access via official Open Platform APIs, partner merchant feeds, and audited manual imports.
- Modular connector architecture (`src/collectors/`):
  - `manual_import.py`: Authorized manual CSV and JSON imports.
  - `daraz.py`: Official Open Platform API connector with HMAC-SHA256 signing.
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

### 5. Phase 7 — Streamlit Research & Data Analytics Dashboard
- Coexists alongside Django as an internal research, data quality, and model evaluation cockpit (`streamlit_app.py`, `src/dashboard_data.py`).
- **12 Key Dashboard Features**:
  1. Total products collected metric.
  2. Number of verified product listings vs sample templates.
  3. Number of participating stores & active platforms.
  4. Product category distribution (Seaborn countplot & percentage table).
  5. Price comparison analysis (Landed cost spreads, savings breakdown).
  6. Average prices by category (Seaborn barplot with NPR landed values).
  7. Cheapest available listings directory with direct store links.
  8. Price history visualization (temporal price fluctuation line chart per product).
  9. Missing-data analysis (attribute completeness percentages and missingness heatmap).
  10. Data collection status & connector execution audit logs.
  11. Product matching performance (multi-store deal count, 100% false positive prevention rate).
  12. Recommendation model evaluation (TF-IDF cosine similarity distribution, vocabulary size, sparsity, top pairings).

### 6. Phase 8 — Testing, Security Hardening & Production Deployment
- **Security Hardening**:
  - CSRF protection enforced across all forms.
  - Session-based authentication with PBKDF2 password hashing.
  - SQL injection immunity via parameterized Django ORM queries.
  - Environment-based secret key management (`.env.example`).
  - Production security headers: X-Frame-Options (Clickjacking defense), X-Content-Type-Options, SSL redirect options.
  - Custom user-friendly 404 and 500 error pages.
- **Performance Optimization**:
  - `Paginator` implemented for search results and category product listings (12 items/page).
  - Explicit database indexes on `canonical_product_key`, `brand`, `category`, and `total_price_npr`.
  - Optimized QuerySets with `select_related` and `prefetch_related` avoiding N+1 queries.
- **Comprehensive Test Suite**:
  - **56 / 56 passing automated tests** across Django, Security, Performance, Data Pipeline, Matching, and Streamlit Dashboard.

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
├── docs/
│   └── deployment.md                       # Comprehensive production deployment guide
├── notebooks/                              # Jupyter analysis and model experimentation
├── reports/                                # Ingestion and quality audit reports
├── src/                                    # Reusable Python data processing modules
│   ├── cleaner.py                          # Currency, shipping & text cleaner
│   ├── comparator.py                       # Price comparison engine
│   ├── dashboard_data.py                   # Streamlit analytics & KPI engine
│   ├── deduplicator.py                     # Duplicate detection
│   ├── keygen.py                           # Canonical key generation
│   ├── loader.py                           # Schema-validated CSV loader
│   ├── matcher.py                          # Scikit-learn TF-IDF matching engine
│   ├── ml_preparer.py                      # Recommendation feature engineering
│   ├── pipeline.py                         # End-to-end data processing pipeline
│   └── collectors/                         # Authorized data collection connectors
├── smart_shopping_web/                     # Django core project package
│   ├── settings.py                         # Dual PostgreSQL / SQLite & security settings
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
│   ├── 404.html                            # Custom 404 Not Found template
│   ├── 500.html                            # Custom 500 Server Error template
│   ├── base.html                           # Base layout with navbar and footer
│   └── shopping/                           # 10 Module templates
├── tests/                                  # Automated test suites
│   ├── test_pipeline.py                    # Phase 1 pipeline tests
│   ├── test_collection.py                  # Phase 2 collection tests
│   ├── test_matching.py                    # Phase 3 matching tests
│   ├── test_django_app.py                  # Phase 5 Django integration tests
│   ├── test_streamlit_dashboard.py         # Phase 7 Streamlit dashboard tests
│   └── test_security_performance.py        # Phase 8 Security & performance tests
├── streamlit_app.py                        # Streamlit Research Dashboard entry point
├── manage.py                               # Django CLI entrypoint
├── main.py                                 # Pipeline runner & application controller
├── .env.example                            # Production environment template
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

---

## Quickstart & Execution Commands

### 1. Execute End-to-End Pipeline & Ingest Data into Database
```bash
python main.py
```
Or run the Django synchronization management command directly:
```bash
python manage.py sync_pipeline
```

### 2. Launch Customer-Facing Website (Django)
```bash
python manage.py runserver
# Or: python main.py --serve
```
Open in browser: **`http://127.0.0.1:8000/`**

### 3. Launch Internal Research Dashboard (Streamlit)
```bash
streamlit run streamlit_app.py
# Or: python main.py --streamlit
```
Open in browser: **`http://localhost:8501/`**

### 4. Run Complete Automated Test Suite (56 Tests)
```bash
# Django, security, and performance test suite (27 tests)
python manage.py test

# Data pipeline, collection, matching, and Streamlit dashboard tests (29 tests)
python -m unittest tests.test_pipeline tests.test_collection tests.test_matching tests.test_streamlit_dashboard
```
All **56 / 56 tests pass cleanly with 0 failures and 0 errors**.
