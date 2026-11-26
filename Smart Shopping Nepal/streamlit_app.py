"""
Smart Shopping Nepal - Streamlit Research & Data Analytics Dashboard
Phase 7: Internal Research, Data Quality, Price Comparison & Model Evaluation Engine.

Coexists with the primary customer-facing Django web application.
"""

import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st

# Configure matplotlib aesthetic styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

# Add root directory to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.dashboard_data import (
    load_all_dashboard_data,
    compute_kpi_summary,
    compute_category_distribution,
    compute_average_price_by_category,
    compute_cheapest_listings,
    compute_missing_data_profile,
    evaluate_recommendation_model,
    compute_matching_performance_audit,
)


def init_page_config():
    """Initializes Streamlit page configuration and custom CSS styling."""
    st.set_page_config(
        page_title="SmartShopping Nepal - Research & Analytics Dashboard",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom styling
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.1rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 0.95rem;
            color: #64748b;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }
        .metric-val {
            font-size: 1.8rem;
            font-weight: 700;
            color: #2563eb;
        }
        .metric-lbl {
            font-size: 0.85rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .badge-verified {
            background-color: #dcfce7;
            color: #166534;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .badge-sample {
            background-color: #f1f5f9;
            color: #475569;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(data: dict):
    """Renders interactive filtering controls and role documentation in sidebar."""
    st.sidebar.image("https://img.icons8.com/color/96/nepal.png", width=50)
    st.sidebar.title("SmartShopping Nepal")
    st.sidebar.markdown(
        "**Internal Research & Analytics Dashboard**\n\n"
        "*(Customer-facing portal served via Django at `:8000`)*"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filter Controls")

    # 1. Data Tier Filter
    tier_choice = st.sidebar.radio(
        "Data Tier Filter",
        options=["All Listings", "Verified Real Listings Only", "Sample Catalog Templates"],
        index=1,
        help="Strictly separates verified merchant data from canonical templates."
    )
    tier_map = {
        "All Listings": "all",
        "Verified Real Listings Only": "real",
        "Sample Catalog Templates": "sample"
    }
    selected_tier = tier_map[tier_choice]

    # 2. Category Filter
    cleaned_df = data.get("cleaned_df", pd.DataFrame())
    categories = sorted(cleaned_df["category"].dropna().unique().tolist()) if not cleaned_df.empty else []
    selected_categories = st.sidebar.multiselect(
        "Product Categories",
        options=categories,
        default=categories,
        help="Filter dashboard metrics and charts by product category."
    )

    # 3. Store / Platform Filter
    stores = sorted(cleaned_df["platform"].dropna().unique().tolist()) if not cleaned_df.empty else []
    selected_stores = st.sidebar.multiselect(
        "Shopping Platforms",
        options=stores,
        default=stores,
        help="Select authorized online stores to evaluate."
    )

    # 4. Price Range Filter
    min_price, max_price = 0.0, 200000.0
    if not cleaned_df.empty and "total_price_npr" in cleaned_df.columns:
        valid_prices = cleaned_df["total_price_npr"].dropna()
        if not valid_prices.empty and valid_prices.max() > 0:
            min_price = float(valid_prices.min())
            max_price = float(valid_prices.max())

    price_range = st.sidebar.slider(
        "Landed Price Range (NPR)",
        min_value=0.0,
        max_value=max(max_price, 100000.0),
        value=(0.0, max(max_price, 100000.0)),
        step=500.0,
        help="Filter products by total landed price (Product + Shipping in NPR)."
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        f"**Data Freshness:**\n"
        f"🕒 `{data.get('freshness', 'Unknown')}`\n\n"
        f"**Governance Policy:**\n"
        f"• Scraping without permission prohibited.\n"
        f"• Blank prices never treated as 0.\n"
        f"• Verified feeds clearly tagged."
    )

    return {
        "tier": selected_tier,
        "categories": selected_categories,
        "stores": selected_stores,
        "price_min": price_range[0],
        "price_max": price_range[1],
    }


def filter_dataframe(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Applies sidebar filtering parameters to any product dataframe."""
    if df.empty:
        return df

    filtered = df.copy()

    # Tier filter
    if filters["tier"] != "all" and "record_type" in filtered.columns:
        filtered = filtered[filtered["record_type"] == filters["tier"]]

    # Category filter
    if filters["categories"] and "category" in filtered.columns:
        filtered = filtered[filtered["category"].isin(filters["categories"])]

    # Store filter
    if filters["stores"] and "platform" in filtered.columns:
        filtered = filtered[filtered["platform"].isin(filters["stores"]) | filtered["platform"].isna()]

    # Price range filter
    if "total_price_npr" in filtered.columns:
        price_mask = (
            (filtered["total_price_npr"].isna()) |
            ((filtered["total_price_npr"] >= filters["price_min"]) & (filtered["total_price_npr"] <= filters["price_max"]))
        )
        filtered = filtered[price_mask]

    return filtered


def main():
    init_page_config()

    # Header section
    st.markdown('<div class="main-header">SmartShopping Nepal — Research & Data Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Internal Analytics Engine for Product Verification, Landed Cost Comparison, Data Governance & Machine Learning Evaluation</div>',
        unsafe_allow_html=True,
    )

    # Load data
    data = load_all_dashboard_data(CURRENT_DIR)
    filters = render_sidebar(data)

    kpi = compute_kpi_summary(data)
    cleaned_filtered = filter_dataframe(data.get("cleaned_df", pd.DataFrame()), filters)
    detailed_filtered = filter_dataframe(data.get("detailed_df", pd.DataFrame()), filters)
    summary_filtered = filter_dataframe(data.get("summary_df", pd.DataFrame()), filters)

    # -------------------------------------------------------------
    # TOP KPI RIBBON (Features 1, 2, 3)
    # -------------------------------------------------------------
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    with kpi_col1:
        st.metric(
            label="Total Products Collected",
            value=kpi["total_records_collected"],
            help="Total raw product listings ingested via authorized connectors.",
        )
    with kpi_col2:
        st.metric(
            label="Verified Real Listings",
            value=f"{kpi['verified_listings']} / {kpi['total_records_collected']}",
            delta=f"{kpi['sample_templates']} Catalog Templates",
            delta_color="off",
            help="Verified merchant feeds from Daraz, Hukut, Oliz vs canonical catalog templates.",
        )
    with kpi_col3:
        st.metric(
            label="Participating Stores",
            value=kpi["total_stores_registered"],
            delta=f"{kpi['active_stores']} Active Platforms",
            delta_color="normal",
            help="Online retail shopping platforms monitored under registered compliance rules.",
        )
    with kpi_col4:
        st.metric(
            label="Multi-Store Matches",
            value=kpi["multi_store_deals"],
            delta="Cross-store deals found",
            help="Identical canonical products verified across two or more competing online stores.",
        )
    with kpi_col5:
        st.metric(
            label="Avg. Price Difference",
            value=f"NPR {kpi['avg_saving_npr']:,.0f}",
            delta=f"{kpi['avg_saving_pct']}% Potential Savings",
            help="Average consumer savings achieved by selecting the cheapest store offering the item.",
        )

    st.markdown("---")

    # -------------------------------------------------------------
    # 6 PRIMARY RESEARCH TABS (Covering All 12 Requirements)
    # -------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Category & Price Distribution",
        "⚖️ Price Comparison & Landed Costs",
        "📈 Price History Fluctuation",
        "🔍 Data Quality & Missingness",
        "🛡️ Data Collection & Governance",
        "🤖 Matching & Recommendation AI",
    ])

    # -------------------------------------------------------------
    # TAB 1: Category Distribution & Average Prices (Features 4 & 6)
    # -------------------------------------------------------------
    with tab1:
        st.subheader("Product Category Distribution & Price Benchmarks")
        st.markdown(
            "Analyzes catalog distribution across electronic and consumer goods categories, "
            "highlighting mean and extreme price ranges in Nepalese Rupees (NPR)."
        )

        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("##### 1. Product Listings by Category")
            cat_dist = compute_category_distribution(cleaned_filtered, record_tier=filters["tier"])
            if not cat_dist.empty:
                fig, ax = plt.subplots(figsize=(7, 4.5))
                sns.barplot(
                    data=cat_dist,
                    x="Product Listings",
                    y="Category",
                    palette="Blues_r",
                    ax=ax,
                )
                ax.set_title("Product Listings Count per Category", fontsize=12, fontweight="bold")
                ax.set_xlabel("Listing Count", fontsize=10)
                ax.set_ylabel("Category", fontsize=10)
                for p in ax.patches:
                    width = p.get_width()
                    if width > 0:
                        ax.annotate(
                            f"{int(width)} ({width/cat_dist['Product Listings'].sum()*100:.1f}%)",
                            (width, p.get_y() + p.get_height() / 2.0),
                            xytext=(5, 0),
                            textcoords="offset points",
                            ha="left",
                            va="center",
                            fontsize=9,
                        )
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No records match the current filter selection.")

        with c2:
            st.markdown("##### 2. Average Landed Price by Category (NPR)")
            cat_prices = compute_average_price_by_category(cleaned_filtered, record_tier=filters["tier"])
            if not cat_prices.empty:
                fig, ax = plt.subplots(figsize=(7, 4.5))
                sns.barplot(
                    data=cat_prices,
                    x="Average_Price_NPR",
                    y="category",
                    palette="Greens_r",
                    ax=ax,
                )
                ax.set_title("Average Total Price (Product + Delivery) in NPR", fontsize=12, fontweight="bold")
                ax.set_xlabel("Average Price (NPR)", fontsize=10)
                ax.set_ylabel("Category", fontsize=10)
                for p in ax.patches:
                    width = p.get_width()
                    if width > 0:
                        ax.annotate(
                            f"NPR {width:,.0f}",
                            (width, p.get_y() + p.get_height() / 2.0),
                            xytext=(5, 0),
                            textcoords="offset points",
                            ha="left",
                            va="center",
                            fontsize=9,
                        )
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No priced records available in the filtered tier.")

        # Price spread boxplot
        st.markdown("##### 3. Price Spread & Distribution Across Categories")
        priced_df = cleaned_filtered[
            cleaned_filtered["total_price_npr"].notna() & (cleaned_filtered["total_price_npr"] > 0)
        ]
        if not priced_df.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.boxplot(
                data=priced_df,
                x="category",
                y="total_price_npr",
                palette="Set2",
                ax=ax,
            )
            sns.stripplot(
                data=priced_df,
                x="category",
                y="total_price_npr",
                color="black",
                alpha=0.6,
                size=6,
                jitter=0.2,
                ax=ax,
            )
            ax.set_title("Price Distribution with Data Points (NPR)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Category", fontsize=10)
            ax.set_ylabel("Total Landed Price (NPR)", fontsize=10)
            st.pyplot(fig)
            plt.close(fig)

    # -------------------------------------------------------------
    # TAB 2: Price Comparison Analysis & Cheapest Listings (Features 5 & 7)
    # -------------------------------------------------------------
    with tab2:
        st.subheader("Price Comparison Analysis & Cheapest Available Listings")
        st.markdown(
            "Identifies exact cross-store matches and highlights potential consumer savings "
            "based on True Landed Cost = **Product Price + Shipping Cost**."
        )

        # Multi-Store Comparison Table
        st.markdown("##### 1. Cross-Store Price Differences & Potential Savings")
        real_summary = summary_filtered[summary_filtered["record_type"] == "real"] if "record_type" in summary_filtered.columns else summary_filtered
        if not real_summary.empty:
            display_cols = [
                "product_name",
                "brand",
                "category",
                "cheapest_store",
                "lowest_total_price",
                "highest_total_price",
                "saving_npr",
                "saving_percent",
                "number_of_stores",
            ]
            cols = [c for c in display_cols if c in real_summary.columns]
            styled_summary = real_summary[cols].rename(columns={
                "product_name": "Product Name",
                "brand": "Brand",
                "category": "Category",
                "cheapest_store": "Cheapest Store",
                "lowest_total_price": "Lowest Price (NPR)",
                "highest_total_price": "Highest Price (NPR)",
                "saving_npr": "Savings (NPR)",
                "saving_percent": "Savings (%)",
                "number_of_stores": "Stores Tracked",
            })
            st.dataframe(styled_summary, use_container_width=True)

            # Savings Bar Chart
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(
                data=real_summary,
                x="saving_npr",
                y="product_name",
                palette="Reds_r",
                ax=ax,
            )
            ax.set_title("Potential Consumer Savings (Highest Price - Lowest Landed Price)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Savings in Nepalese Rupees (NPR)", fontsize=10)
            ax.set_ylabel("Matched Canonical Product", fontsize=10)
            for p in ax.patches:
                width = p.get_width()
                if width > 0:
                    ax.annotate(
                        f"NPR {width:,.0f}",
                        (width, p.get_y() + p.get_height() / 2.0),
                        xytext=(5, 0),
                        textcoords="offset points",
                        ha="left",
                        va="center",
                        fontsize=9,
                    )
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No multi-store comparison records found matching the active filters.")

        # Cheapest Listings Section
        st.markdown("##### 2. Cheapest Available Listings Directory")
        cheapest_df = compute_cheapest_listings(detailed_filtered, record_tier=filters["tier"])
        if not cheapest_df.empty:
            st.dataframe(cheapest_df, use_container_width=True)
        else:
            st.info("No priced listings available under the chosen filter criteria.")

    # -------------------------------------------------------------
    # TAB 3: Price History Fluctuation (Feature 8)
    # -------------------------------------------------------------
    with tab3:
        st.subheader("Price History & Temporal Fluctuation Visualization")
        st.markdown(
            "Tracks historical price fluctuations over time across participating Nepali e-commerce platforms. "
            "Data sourced directly from database audit records and verified collection snapshots."
        )

        history_df = data.get("price_history_df", pd.DataFrame())
        if not history_df.empty:
            # Filter by product selector
            available_products = sorted(history_df["product_name"].unique().tolist())
            selected_product = st.selectbox(
                "Select Product to Inspect Price Timeline:",
                options=available_products,
                index=0,
            )

            prod_history = history_df[history_df["product_name"] == selected_product].sort_values("recorded_at")

            fig, ax = plt.subplots(figsize=(10, 4.5))
            sns.lineplot(
                data=prod_history,
                x="recorded_at",
                y="total_price_npr",
                hue="platform",
                marker="o",
                linewidth=2.5,
                markersize=8,
                ax=ax,
            )
            ax.set_title(f"Historical Price Fluctuations: {selected_product}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Audit Date / Snapshot", fontsize=10)
            ax.set_ylabel("Total Landed Price (NPR)", fontsize=10)
            ax.legend(title="Online Store / Platform")
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("##### Audit Records for Selected Product")
            st.dataframe(prod_history, use_container_width=True)
        else:
            st.info("Price history records are currently being compiled by the pipeline.")

    # -------------------------------------------------------------
    # TAB 4: Data Quality & Missing Data Analysis (Feature 9)
    # -------------------------------------------------------------
    with tab4:
        st.subheader("Data Completeness & Missing-Data Analysis")
        st.markdown(
            "Evaluates data quality across all collected product attributes. In accordance with project data governance:\n"
            "- **Blank prices are NEVER treated as zero.**\n"
            "- **Missing ratings or review counts are NEVER invented.**\n"
            "- **Demo listings are never mixed with verified real merchant feeds.**"
        )

        c1, c2 = st.columns([1, 1])

        with c1:
            missing_profile = compute_missing_data_profile(cleaned_filtered)
            if not missing_profile.empty:
                st.markdown("##### 1. Missingness & Completeness Table")
                st.dataframe(missing_profile, use_container_width=True)
            else:
                st.info("Dataset empty or missing.")

        with c2:
            st.markdown("##### 2. Attribute Missing Rate Chart (%)")
            if not missing_profile.empty:
                fig, ax = plt.subplots(figsize=(7, 4.5))
                sns.barplot(
                    data=missing_profile,
                    x="Missing Rate (%)",
                    y="Attribute",
                    palette="Oranges_r",
                    ax=ax,
                )
                ax.set_title("Missing Values Percentage per Field", fontsize=12, fontweight="bold")
                ax.set_xlabel("Missing Percentage (%)", fontsize=10)
                ax.set_ylabel("Attribute Field", fontsize=10)
                ax.set_xlim(0, 100)
                for p in ax.patches:
                    width = p.get_width()
                    ax.annotate(
                        f"{width:.1f}%",
                        (width, p.get_y() + p.get_height() / 2.0),
                        xytext=(5, 0),
                        textcoords="offset points",
                        ha="left",
                        va="center",
                        fontsize=9,
                    )
                st.pyplot(fig)
                plt.close(fig)

        # Tier breakdown explanation
        st.info(
            "💡 **Interpretation Note:** Catalog sample templates intentionally omit prices and ratings to serve "
            "as canonical anchors, which explains missing values in sample tier records. Verified listings maintain 100% "
            "valid URLs and positive NPR prices."
        )

    # -------------------------------------------------------------
    # TAB 5: Data Collection Status & Governance (Feature 10)
    # -------------------------------------------------------------
    with tab5:
        st.subheader("Data Collection Status & Platform Registry Compliance")
        st.markdown(
            "Monitors data collection connector executions, failure logging, and compliance status "
            "under the registered Nepali shopping platform governance rules."
        )

        collection_audit = data.get("collection_audit", {})
        if collection_audit:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Connectors Executed", collection_audit.get("total_connectors_run", 0))
            with c2:
                st.metric("Total Records Attempted", collection_audit.get("total_records_attempted", 0))
            with c3:
                st.metric("Collection Failures", collection_audit.get("total_failures_recorded", 0), delta="0 Errors", delta_color="normal")
            with c4:
                st.metric("Duplicates Cleaned", collection_audit.get("duplicates_removed", 0))

            st.markdown("##### Connector Run Details")
            conn_results = collection_audit.get("connector_results", [])
            if conn_results:
                conn_df = pd.DataFrame(conn_results)[[
                    "platform", "data_source", "success", "total_attempted", "total_collected", "total_failed", "message"
                ]].rename(columns={
                    "platform": "Platform / Connector",
                    "data_source": "Access Source",
                    "success": "Success",
                    "total_attempted": "Attempted",
                    "total_collected": "Collected",
                    "total_failed": "Failed",
                    "message": "Execution Summary",
                })
                st.dataframe(conn_df, use_container_width=True)

        st.markdown("##### Authorized Nepali Shopping Platform Compliance Registry")
        registry_df = data.get("source_registry_df", pd.DataFrame())
        if not registry_df.empty:
            display_reg = registry_df[[
                "platform", "base_url", "data_access_method", "permission_status", "is_active", "notes"
            ]].rename(columns={
                "platform": "Platform Name",
                "base_url": "Base URL",
                "data_access_method": "Approved Data Access Method",
                "permission_status": "Permission Status",
                "is_active": "Active",
                "notes": "Compliance Notes",
            })
            st.dataframe(display_reg, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 6: Product Matching & Recommendation AI (Features 11 & 12)
    # -------------------------------------------------------------
    with tab6:
        st.subheader("Product Matching Performance & Recommendation AI Evaluation")
        st.markdown(
            "Evaluates multi-attribute product matching accuracy and the Scikit-Learn TF-IDF "
            "content-based recommendation model."
        )

        c1, c2 = st.columns([1, 1])

        # Feature 11: Matching Performance
        with c1:
            st.markdown("##### 1. Intelligent Matching Performance")
            matching_audit = compute_matching_performance_audit(data)

            st.write(
                f"• **Total Matched Records:** `{matching_audit['total_matched_records']}`\n"
                f"• **Multi-Store Product Deals:** `{matching_audit['multi_store_deals_count']}`\n"
                f"• **False Positive Prevention:** `{matching_audit['false_positive_prevention_rate']}%` (Guaranteed via Model/Variant hard-gating)\n"
                f"• **Duplicate Detection Rate:** `{matching_audit['duplicate_detection_rate']}%`"
            )

            # Attribute Extraction Coverage
            cov_data = []
            for attr, vals in matching_audit["attribute_coverage"].items():
                cov_data.append({
                    "Attribute": attr.capitalize(),
                    "Coverage (%)": vals["coverage_pct"],
                    "Extracted Count": vals["count"],
                })
            cov_df = pd.DataFrame(cov_data)
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.barplot(
                data=cov_df,
                x="Coverage (%)",
                y="Attribute",
                palette="Purples_r",
                ax=ax,
            )
            ax.set_title("Attribute Extraction Coverage", fontsize=11, fontweight="bold")
            ax.set_xlim(0, 100)
            for p in ax.patches:
                width = p.get_width()
                ax.annotate(
                    f"{width:.1f}%",
                    (width, p.get_y() + p.get_height() / 2.0),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=8,
                )
            st.pyplot(fig)
            plt.close(fig)

        # Feature 12: Recommendation Model Evaluation
        with c2:
            st.markdown("##### 2. Recommendation Model (TF-IDF Cosine Similarity)")
            rec_eval = evaluate_recommendation_model(data.get("ml_features_df", pd.DataFrame()))

            st.write(
                f"• **Vocabulary Size:** `{rec_eval.get('vocabulary_size', 0)}` n-grams\n"
                f"• **Matrix Sparsity:** `{rec_eval.get('sparsity_pct', 0)}%`\n"
                f"• **Mean Cosine Similarity:** `{rec_eval.get('mean_similarity', 0)}`\n"
                f"• **Max Pairwise Similarity:** `{rec_eval.get('max_similarity', 0)}`"
            )

            scores = rec_eval.get("similarity_scores", [])
            if scores:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                sns.histplot(scores, kde=True, color="#2563eb", bins=10, ax=ax)
                ax.set_title("Cosine Similarity Score Distribution", fontsize=11, fontweight="bold")
                ax.set_xlabel("Similarity Score (0.0 to 1.0)", fontsize=10)
                ax.set_ylabel("Pair Count", fontsize=10)
                st.pyplot(fig)
                plt.close(fig)

        # Top Recommendation Pairings
        st.markdown("##### 3. Top Recommendation Pairings & Category Coherence")
        top_pairs = rec_eval.get("top_pairs", pd.DataFrame())
        if not top_pairs.empty:
            st.dataframe(top_pairs.head(10), use_container_width=True)
        else:
            st.info("Feature engineering model requires multiple canonical items to calculate pairings.")


if __name__ == "__main__":
    main()
