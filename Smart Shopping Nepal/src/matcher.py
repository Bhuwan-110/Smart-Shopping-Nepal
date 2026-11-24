"""
Smart Shopping Nepal - Intelligent Product Matching Engine.

Multi-stage hybrid product reconciliation:
1. Attribute Extraction:
   - Brand, Category, Model number / series, Specifications (RAM / Storage), Condition (New / Used / Refurbished).
2. Hard Gating:
   - Strict rejection on mismatched brand, category, condition, model number, or RAM/storage variant.
   - Different models (e.g. A56 vs A55) or different variants (e.g. 128GB vs 256GB) are NEVER grouped.
3. TF-IDF & N-gram Text Similarity (Scikit-learn):
   - Computes cosine similarity over normalized product titles.
4. Canonical Product Key Alignment:
   - Assigns consistent slugified keys: {brand_clean}_{model_clean}_{variant_clean}.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.cleaner import clean_text, standardize_brand, standardize_category
from src.keygen import generate_product_key

# Known brands in Nepalese consumer electronics
KNOWN_BRANDS = [
    "samsung", "apple", "hp", "dell", "lenovo", "asus", "acer", "xiaomi",
    "redmi", "realme", "oneplus", "sony", "boat", "noise", "anker", "honor",
    "vivo", "oppo", "poco", "google", "huawei", "msi", "infinix", "example brand"
]

# Condition indicators
CONDITION_USED_KEYWORDS = ["used", "refurbished", "renewed", "second hand", "pre-owned", "open box", "refurb"]

# Common model regex patterns - Ordered from most specific brand series to general patterns
MODEL_PATTERNS = [
    # Laptop series: Pavilion 15, Pavilion 14, Victus, IdeaPad, ZenBook, MacBook
    r"\b(pavilion(?:\s+\d{1,2})?)\b",
    r"\b(victus(?:\s+\d{1,2})?)\b",
    r"\b(ideapad(?:\s+[a-z]?\d{1,3})?)\b",
    r"\b(zenbook(?:\s+\d{1,2})?)\b",
    r"\b(thinkpad(?:\s+[a-z]?\d{1,3})?)\b",
    r"\b(macbook\s+(?:air|pro)(?:\s+\d{1,2})?)\b",
    # Audio series: Soundcore R50i, WH-1000XM5, AirPods Pro, Airdopes 141
    r"\b(soundcore\s+[a-z0-9]+)\b",
    r"\b(wh[-\s]?1000xm\d)\b",
    r"\b(airpods(?:\s+pro)?(?:\s+\d(?:nd|rd)?\s+gen)?)\b",
    r"\b(airdopes\s+\d{2,3})\b",
    # Phone series: Galaxy A56, Galaxy S24, iPhone 15, Redmi Note 13
    r"\b(galaxy\s+[a-z]\d{1,2}[a-z]?)\b",
    r"\b(iphone\s+\d{1,2}(?:\s+(?:pro\s+max|pro|plus|mini))?)\b",
    r"\b(redmi\s+note\s+\d{1,2}(?:\s+pro(?:\s+plus)?)?)\b",
    r"\b(redmi\s+\d{1,2}[a-z]?)\b",
    # Standalone phone model numbers: e.g. A56, A55, S24 (Excluding PC CPU tags like i3, i5, i7, i9, r5, r7)
    r"\b([a-hj-qs-z]\d{1,2}[a-z]?(?:\s+5g)?)\b",
    r"\b(example model)\b"
]


class ProductAttributeExtractor:
    """
    Extracts structured technical specifications and attributes from unstructured product titles.
    """

    @staticmethod
    def extract_condition(title: str) -> str:
        """Identifies product condition (new vs used/refurbished)."""
        t_clean = clean_text(title)
        for kw in CONDITION_USED_KEYWORDS:
            if re.search(r"\b" + re.escape(kw) + r"\b", t_clean):
                return "used"
        return "new"

    @staticmethod
    def extract_brand(title: str, existing_brand: Optional[str] = None) -> str:
        """Extracts brand from title or normalizes existing brand."""
        if existing_brand and pd.notna(existing_brand) and str(existing_brand).strip():
            b = clean_text(existing_brand)
            if b and b != "generic":
                return standardize_brand(b)

        t_clean = clean_text(title)
        for b in KNOWN_BRANDS:
            if re.search(r"\b" + re.escape(b) + r"\b", t_clean):
                return standardize_brand(b)
        return "Generic"

    @staticmethod
    def extract_ram_and_storage(title: str, existing_variant: Optional[str] = None) -> Tuple[Optional[str], Optional[str], str]:
        """
        Extracts RAM (e.g. 8GB) and Storage (e.g. 256GB, 128GB, 512GB).
        Returns: (ram, storage, variant_string)
        """
        t = clean_text(title)
        v_in = clean_text(existing_variant or "")

        # Protect model numbers from being treated as RAM (e.g. iPhone 15, Pavilion 15, Note 13)
        t_cleaned = re.sub(r"\b(iphone|pavilion|victus|note|galaxy)\s+\d{1,2}\b", "", t)
        combined_text = f"{t_cleaned} {v_in}"

        # 1. Match explicit slashed pairs: e.g. 8GB/256GB, 8/256GB, 16GB/512GB
        combo_match = re.search(r"\b(\d{1,2})\s*(?:gb)?\s*[\/]\s*(\d{2,4})\s*(?:gb|tb)?\b", combined_text)
        if combo_match:
            ram_num = int(combo_match.group(1))
            store_str = combo_match.group(2).upper()
            if not store_str.endswith("GB") and not store_str.endswith("TB"):
                store_str = f"{store_str}GB"
            if ram_num <= 64:  # Valid RAM boundary
                return f"{ram_num}GB", store_str, f"{ram_num}GB/{store_str}"

        # 2. Match explicit RAM: e.g. "8GB RAM", "RAM 16GB"
        ram_match = re.search(r"\b(\d{1,2})\s*gb\s*ram\b", combined_text) or re.search(r"\bram\s*[:\-]?\s*(\d{1,2})\s*gb\b", combined_text)
        ram = f"{ram_match.group(1)}GB" if ram_match else None

        # 3. Match Storage: e.g. "256GB Storage", "512GB SSD", or standalone standard storage sizes "(128GB)", "256GB"
        storage_match = re.search(r"\b(64|128|256|512)\s*gb\b", combined_text) or re.search(r"\b(1|2)\s*tb\b", combined_text)
        storage = None
        if storage_match:
            val = storage_match.group(0).upper().replace(" ", "")
            storage = val

        if ram and storage:
            return ram, storage, f"{ram}/{storage}"
        elif storage:
            return None, storage, storage
        elif ram:
            return ram, None, ram

        # 4. Fallback to existing variant if provided
        if existing_variant and pd.notna(existing_variant) and str(existing_variant).strip():
            v_clean = str(existing_variant).strip()
            if v_clean.lower() not in ["standard", "generic", "nan", ""]:
                return None, None, v_clean

        return None, None, "Standard"

    @staticmethod
    def extract_model(title: str, existing_model: Optional[str] = None) -> str:
        """Extracts model name or series number with structural normalization."""
        t_clean = clean_text(title)

        # 1. Laptop series: Pavilion 15, Victus 15, IdeaPad 3, ZenBook 14, MacBook Pro
        m_lap = re.search(r"\b(pavilion|victus|ideapad|zenbook|thinkpad)\b(?:[^\d]*?\b(1[3-7])\b)?", t_clean)
        if m_lap:
            series = m_lap.group(1).title()
            num = m_lap.group(2)
            return f"{series} {num}" if num else series

        m_mac = re.search(r"\b(macbook\s+(?:air|pro)(?:\s+\d{1,2})?)\b", t_clean)
        if m_mac:
            return m_mac.group(1).title()

        # 2. Audio series: Soundcore R50i, WH-1000XM5, AirPods Pro, Airdopes 141
        m_snd = re.search(r"\b(soundcore\s+[a-z0-9]+)\b", t_clean)
        if m_snd:
            return m_snd.group(1).title()

        m_sony = re.search(r"\b(wh[-\s]?1000xm\d)\b", t_clean)
        if m_sony:
            return m_sony.group(1).upper().replace(" ", "-")

        m_airp = re.search(r"\b(airpods(?:\s+pro)?(?:\s+\d(?:nd|rd)?\s+gen)?)\b", t_clean)
        if m_airp:
            return m_airp.group(1).title()

        # 3. Phone series: Galaxy A56, iPhone 15, Redmi Note 13
        m_gal = re.search(r"\b(galaxy\s+[a-z]\d{1,2}[a-z]?)\b", t_clean)
        if m_gal:
            return m_gal.group(1).title()

        m_iph = re.search(r"\b(iphone\s+\d{1,2}(?:\s+(?:pro\s+max|pro|plus|mini))?)\b", t_clean)
        if m_iph:
            return m_iph.group(1).title()

        m_redmi = re.search(r"\b(redmi\s+note\s+\d{1,2}(?:\s+pro(?:\s+plus)?)?)\b", t_clean)
        if m_redmi:
            return m_redmi.group(1).title()

        # 4. Fallback: match other MODEL_PATTERNS
        for pat in MODEL_PATTERNS:
            m = re.search(pat, t_clean)
            if m:
                raw_m = m.group(1).strip()
                raw_m = re.sub(r"\s+5g$", "", raw_m)
                return raw_m.title()

        if existing_model and pd.notna(existing_model) and str(existing_model).strip():
            em = str(existing_model).strip()
            if em.lower() not in ["standard", "generic", "nan", ""]:
                return em.title()

        return "Generic"

    @staticmethod
    def extract_category(title: str, existing_cat: Optional[str] = None) -> str:
        """Infers category from title keywords or standardizes existing."""
        if existing_cat and pd.notna(existing_cat) and str(existing_cat).strip():
            return standardize_category(existing_cat)

        t_clean = clean_text(title)
        if any(w in t_clean for w in ["phone", "smartphone", "galaxy a", "galaxy s", "iphone", "redmi", "realme"]):
            return "Mobile Phone"
        if any(w in t_clean for w in ["laptop", "pavilion", "notebook", "ideapad", "macbook", "zenbook", "thinkpad"]):
            return "Laptop"
        if any(w in t_clean for w in ["earbuds", "earphone", "headphone", "soundcore", "airpods", "airdopes", "tws"]):
            return "Earbuds"
        if any(w in t_clean for w in ["watch", "smartwatch", "band"]):
            return "Smartwatch"

        return "General"

    @classmethod
    def extract_all(cls, title: str, raw_record: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Runs full attribute extraction on a listing title and record metadata."""
        rec = raw_record or {}
        brand = cls.extract_brand(title, rec.get("brand"))
        category = cls.extract_category(title, rec.get("category"))
        model = cls.extract_model(title, rec.get("model"))
        ram, storage, variant = cls.extract_ram_and_storage(title, rec.get("variant"))
        condition = cls.extract_condition(title)

        # Generate canonical product key
        product_key = generate_product_key(brand, model, variant)
        if condition != "new":
            product_key = f"{product_key}_{condition}"

        return {
            "extracted_brand": brand,
            "extracted_category": category,
            "extracted_model": model,
            "extracted_ram": ram,
            "extracted_storage": storage,
            "extracted_variant": variant,
            "extracted_condition": condition,
            "canonical_product_key": product_key
        }


class IntelligentProductMatcher:
    """
    Intelligent product matching system combining attribute gating with
    Scikit-learn TF-IDF text similarity to ensure strict equivalence.
    """

    def __init__(self, tfidf_threshold: float = 0.50):
        self.tfidf_threshold = tfidf_threshold
        self.extractor = ProductAttributeExtractor()

    def are_products_equivalent(
        self,
        listing1: Dict[str, Any],
        listing2: Dict[str, Any]
    ) -> Tuple[bool, float, str]:
        """
        Determines whether two product listings represent the exact same product & variant.
        Returns: (is_match, similarity_score, reason)

        STRICT RULES:
        - Mismatched brand -> REJECT.
        - Mismatched category -> REJECT.
        - Mismatched condition (New vs Used) -> REJECT.
        - Mismatched model (e.g. A56 vs A55) -> REJECT.
        - Mismatched RAM or Storage -> REJECT.
        """
        t1 = str(listing1.get("product_name", ""))
        t2 = str(listing2.get("product_name", ""))

        attr1 = self.extractor.extract_all(t1, listing1)
        attr2 = self.extractor.extract_all(t2, listing2)

        # 1. Hard Gate: Brand
        if attr1["extracted_brand"].lower() != attr2["extracted_brand"].lower():
            return False, 0.0, f"Brand mismatch: {attr1['extracted_brand']} != {attr2['extracted_brand']}"

        # 2. Hard Gate: Category
        if attr1["extracted_category"].lower() != attr2["extracted_category"].lower():
            return False, 0.0, f"Category mismatch: {attr1['extracted_category']} != {attr2['extracted_category']}"

        # 3. Hard Gate: Condition (Never match New with Used/Refurbished!)
        if attr1["extracted_condition"] != attr2["extracted_condition"]:
            return False, 0.0, f"Condition mismatch: {attr1['extracted_condition']} != {attr2['extracted_condition']}"

        # 4. Hard Gate: Model
        m1 = clean_text(attr1["extracted_model"])
        m2 = clean_text(attr2["extracted_model"])
        if m1 != m2 and m1 != "generic" and m2 != "generic":
            return False, 0.0, f"Model mismatch: {m1} != {m2}"

        # 5. Hard Gate: Variant / Specifications (Storage & RAM)
        v1 = clean_text(attr1["extracted_variant"])
        v2 = clean_text(attr2["extracted_variant"])
        if v1 != v2 and v1 != "standard" and v2 != "standard":
            return False, 0.0, f"Variant/Spec mismatch: {v1} != {v2}"

        # 6. Scikit-learn TF-IDF Word Similarity
        # Clean noisy marketing tokens before computing similarity
        def _strip_marketing(text: str) -> str:
            cleaned = clean_text(text)
            for noise in ["smartphone", "mobile", "official", "warranty", "laptop", "earbuds", "tws", "original", "black", "awesome", "graphite"]:
                cleaned = re.sub(r"\b" + noise + r"\b", "", cleaned)
            return cleaned.strip()

        core_t1 = _strip_marketing(t1) or clean_text(t1)
        core_t2 = _strip_marketing(t2) or clean_text(t2)

        corpus = [core_t1, core_t2]
        try:
            vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            tfidf_matrix = vectorizer.fit_transform(corpus)
            score = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        except Exception:
            score = 0.0

        # High-Fidelity Match: When Brand, Category, Condition, Model, and Variant all align
        if m1 == m2 and m1 != "generic" and v1 == v2:
            final_score = max(score, 0.95)
            return True, final_score, f"Equivalent match on attributes and canonical key: {attr1['canonical_product_key']}"

        # Standard models with matching brand & category
        if m1 == m2 and v1 == v2 and score >= self.tfidf_threshold:
            return True, score, f"Equivalent match (canonical key: {attr1['canonical_product_key']})"

        if score >= self.tfidf_threshold and (m1 == "generic" or m2 == "generic"):
            return True, score, f"Fuzzy equivalent match (canonical key: {attr1['canonical_product_key']})"

        return False, score, f"Text similarity {score:.2f} below threshold {self.tfidf_threshold}"

    def reconcile_listings_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a DataFrame of product listings, extracting specifications,
        validating condition, and aligning to canonical product keys.
        """
        reconciled_df = df.copy()

        extracted_rows = []
        for _, row in reconciled_df.iterrows():
            title = str(row.get("product_name", ""))
            extracted = self.extractor.extract_all(title, row.to_dict())
            extracted_rows.append(extracted)

        extracted_df = pd.DataFrame(extracted_rows)

        # Update columns with canonical values while keeping compatibility
        reconciled_df["brand"] = extracted_df["extracted_brand"]
        reconciled_df["category"] = extracted_df["extracted_category"]
        reconciled_df["model"] = extracted_df["extracted_model"]
        reconciled_df["variant"] = extracted_df["extracted_variant"]
        reconciled_df["condition"] = extracted_df["extracted_condition"]
        reconciled_df["canonical_product_key"] = extracted_df["canonical_product_key"]

        # Ensure product_key matches canonical key
        reconciled_df["product_key"] = reconciled_df["canonical_product_key"]

        return reconciled_df
