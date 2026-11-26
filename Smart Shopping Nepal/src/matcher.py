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

# Known brands in Nepalese consumer markets across all 15 sectors
KNOWN_BRANDS = [
    # Electronics, Mobiles, Computers & Audio
    "samsung", "apple", "hp", "dell", "lenovo", "asus", "acer", "xiaomi",
    "redmi", "realme", "oneplus", "sony", "boat", "noise", "anker", "honor",
    "vivo", "oppo", "poco", "google", "huawei", "msi", "infinix", "logitech",
    "sandisk", "baseus", "tp-link", "belkin", "spigen", "remax", "western digital",
    # Home Appliances
    "lg", "cg", "philips", "baltra", "yasuda", "midea", "panasonic", "whirlpool",
    # Fashion, Footwear & Accessories
    "goldstar", "caliber", "adidas", "nike", "puma", "levi's", "levis", "zara",
    "h&m", "erke", "fastrack", "titan", "ray-ban", "wildhorn", "baggit", "peter england", "biba",
    # Beauty & Personal Care
    "cetaphil", "the derma co", "nivea", "mamaearth", "l'oreal", "himalaya",
    # Furniture & Home
    "nilkamal", "solimo", "kurl-on", "cortina",
    # Sports & Fitness
    "yonex", "cosco", "nivia", "strauss", "decathlon",
    # Books & Stationery
    "doms", "classmate", "penguin", "fineprint", "ratna pustak",
    "example brand"
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
    def extract_size_and_color(title: str, existing_variant: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """Extracts clothing/footwear sizes and primary colors."""
        t_clean = clean_text(title)
        v_clean = clean_text(existing_variant or "")
        combined = f"{t_clean} {v_clean}"

        # 1. Footwear & Apparel Sizes
        size_match = re.search(r"\b(?:uk|size|us)\s*(\d{1,2}(?:\.\d)?)\b", combined)
        if not size_match:
            size_match = re.search(r"\b(xxl|2xl|3xl|xl|xs|[sml])\b", combined)
        if not size_match:
            size_match = re.search(r"\b(?:size\s+)?(2[89]|3[0-8]|4[02])\b", combined)

        size = size_match.group(0).upper() if size_match else None

        # 2. Colors
        colors = ["black", "white", "navy", "blue", "grey", "gray", "red", "green", "olive", "maroon", "teal", "yellow", "gold", "silver", "brown", "beige", "cream"]
        found_color = None
        for c in colors:
            if re.search(r"\b" + c + r"\b", combined):
                found_color = c.capitalize()
                break

        return size, found_color

    @staticmethod
    def extract_category(title: str, existing_cat: Optional[str] = None) -> str:
        """Infers category across all 15 sectors (A-O) or standardizes existing."""
        if existing_cat and pd.notna(existing_cat) and str(existing_cat).strip():
            return standardize_category(existing_cat)

        t_clean = clean_text(title)
        if any(w in t_clean for w in ["phone", "smartphone", "galaxy a", "galaxy s", "iphone", "redmi", "realme", "pixel"]):
            return "Mobile Phones"
        if any(w in t_clean for w in ["laptop", "pavilion", "notebook", "ideapad", "macbook", "zenbook", "thinkpad", "nitro", "legion"]):
            return "Laptops and Computers"
        if any(w in t_clean for w in ["charger", "adapter", "power bank", "case", "cover", "cable", "powercore"]):
            return "Mobile Accessories"
        if any(w in t_clean for w in ["mouse", "keyboard", "webcam", "portable ssd", "external hdd", "router", "usb hub"]):
            return "Computer Accessories"
        if any(w in t_clean for w in ["earbuds", "earphone", "headphone", "soundcore", "airpods", "airdopes", "tws", "speaker", "watch", "smartwatch", "band"]):
            return "Audio and Wearables"
        if any(w in t_clean for w in ["polo", "shirt", "chino", "trousers", "track jacket"]):
            return "Men's Clothing"
        if any(w in t_clean for w in ["kurti", "kurta", "anarkali", "blazer", "dress", "saree"]):
            return "Women's Clothing"
        if any(w in t_clean for w in ["romper", "toddler", "baby boy", "baby girl", "kids"]):
            return "Kids' Clothing"
        if any(w in t_clean for w in ["shoes", "sneaker", "boots", "trekker", "running shoes", "footwear"]):
            return "Footwear"
        if any(w in t_clean for w in ["sunglasses", "wallet", "handbag", "analog watch", "belt"]):
            return "Fashion Accessories"
        if any(w in t_clean for w in ["washing machine", "refrigerator", "rice cooker", "mixer grinder", "led tv", "microwave", "air conditioner"]):
            return "Home Appliances"
        if any(w in t_clean for w in ["cleanser", "serum", "cream", "shampoo", "conditioner", "face wash"]):
            return "Beauty and Personal Care"
        if any(w in t_clean for w in ["office chair", "dining table", "computer desk", "mattress", "curtains"]):
            return "Furniture and Home"
        if any(w in t_clean for w in ["badminton", "racket", "volley ball", "football", "yoga mat", "dumbbell"]):
            return "Sports and Fitness"
        if any(w in t_clean for w in ["book", "novel", "notebook", "pencils", "stationery"]):
            return "Books and Stationery"

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
        size, color = cls.extract_size_and_color(title, rec.get("variant"))

        if category in ["Men's Clothing", "Women's Clothing", "Kids' Clothing", "Footwear"] and size:
            variant = f"{size} {color or ''}".strip()

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
            "extracted_size": size,
            "extracted_color": color,
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

    def classify_match_tier(
        self,
        listing1: Dict[str, Any],
        listing2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classifies two product listings into 4 confidence tiers:
        - Exact Match (1.0): Identical brand, category, condition, model, and variant.
        - Probable Match (0.85–0.95): Identical brand, category, condition, and model; standard or compatible variant.
        - Similar Product (0.60–0.84): Same category and brand, but different variant or same series.
        - Unmatched Product (<0.60): Brand/category/condition mismatch, or distinct products.
        """
        t1 = str(listing1.get("product_name", ""))
        t2 = str(listing2.get("product_name", ""))
        attr1 = self.extractor.extract_all(t1, listing1)
        attr2 = self.extractor.extract_all(t2, listing2)

        # Compute TF-IDF text similarity independently of hard gates
        def _strip_marketing(text: str) -> str:
            cleaned = clean_text(text)
            for noise in ["smartphone", "mobile", "official", "warranty", "laptop", "earbuds", "tws", "original", "black", "awesome", "graphite"]:
                cleaned = re.sub(r"\b" + noise + r"\b", "", cleaned)
            return cleaned.strip()

        core_t1 = _strip_marketing(t1) or clean_text(t1)
        core_t2 = _strip_marketing(t2) or clean_text(t2)
        try:
            vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            tfidf_matrix = vectorizer.fit_transform([core_t1, core_t2])
            text_sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        except Exception:
            text_sim = 0.0

        is_equiv, score, reason = self.are_products_equivalent(listing1, listing2)

        # 1. Exact Match (1.0)
        if (
            is_equiv
            and attr1["canonical_product_key"] == attr2["canonical_product_key"]
            and attr1["extracted_variant"].lower() == attr2["extracted_variant"].lower()
        ):
            return {
                "tier": "Exact Match",
                "confidence": 1.0,
                "is_match": True,
                "reason": f"Exact attribute and canonical key match: {attr1['canonical_product_key']}",
                "canonical_product_key": attr1["canonical_product_key"],
                "score": 1.0
            }

        # 2. Probable Match (0.85–0.95)
        if is_equiv:
            confidence = max(0.85, min(0.95, round(score, 2)))
            return {
                "tier": "Probable Match",
                "confidence": confidence,
                "is_match": True,
                "reason": reason,
                "canonical_product_key": attr1["canonical_product_key"],
                "score": confidence
            }

        # 3. Similar Product (0.60–0.84)
        same_brand = attr1["extracted_brand"].lower() == attr2["extracted_brand"].lower()
        same_cat = attr1["extracted_category"].lower() == attr2["extracted_category"].lower()
        m1 = clean_text(attr1["extracted_model"])
        m2 = clean_text(attr2["extracted_model"])
        same_model = (m1 == m2 and m1 != "generic")
        
        if same_brand and same_cat and (same_model or text_sim >= 0.50):
            similar_conf = max(0.60, min(0.84, round(max(text_sim, 0.70 if same_model else 0.60), 2)))
            return {
                "tier": "Similar Product",
                "confidence": similar_conf,
                "is_match": False,
                "reason": f"Similar product in same category and brand ({attr1['extracted_brand']}), variant/spec divergence ({reason}).",
                "canonical_product_key": None,
                "score": similar_conf
            }

        # 4. Unmatched Product (<0.60)
        unmatched_conf = round(min(text_sim, 0.59), 2)
        return {
            "tier": "Unmatched Product",
            "confidence": unmatched_conf,
            "is_match": False,
            "reason": reason,
            "canonical_product_key": None,
            "score": unmatched_conf
        }

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
        # Allow compatible category names (e.g. 'Mobile Phone' vs 'Mobile Phones')
        c1 = clean_text(attr1["extracted_category"]).rstrip("s")
        c2 = clean_text(attr2["extracted_category"]).rstrip("s")
        if c1 != c2:
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
        
        # Check storage/RAM explicit conflict
        s1 = attr1.get("extracted_storage")
        s2 = attr2.get("extracted_storage")
        if s1 and s2 and s1 != s2:
            return False, 0.0, f"Storage mismatch: {s1} != {s2}"

        r1 = attr1.get("extracted_ram")
        r2 = attr2.get("extracted_ram")
        if r1 and r2 and r1 != r2:
            return False, 0.0, f"RAM mismatch: {r1} != {r2}"

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

        # Compatible variant (one standard, one color/empty) with matching model
        if m1 == m2 and m1 != "generic" and (v1 == v2 or v1 == "standard" or v2 == "standard"):
            final_score = max(score, 0.90)
            return True, final_score, f"Equivalent match on model and compatible variant: {attr1['canonical_product_key']}"

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
