"""
Smart Shopping Nepal - Core Relational Models.

Implements database schema for:
1. Product & ProductVariant (decoupled canonical entities)
2. Store (Platform registry & compliance)
3. ProductListing & PriceHistory (multi-store pricing, ratings, URLs)
4. ProductReview (community ratings & verified reviews)
5. Wishlist (user saved products & target price alerts)
6. RecommendationResult (ML similarity mappings)
7. CollectionLog (data governance & pipeline audit logs)
"""

from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.utils.text import slugify


class Product(models.Model):
    """
    Canonical product representation.
    Ensures identical products are not duplicated in the catalog.
    """
    name = models.CharField(max_length=255, db_index=True)
    brand = models.CharField(max_length=100, db_index=True)
    model = models.CharField(max_length=100, db_index=True)
    category = models.CharField(max_length=100, db_index=True)
    canonical_product_key = models.CharField(max_length=255, unique=True, db_index=True)
    description = models.TextField(blank=True, default='')
    image_url = models.URLField(max_length=1000, blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['brand', 'name']
        verbose_name = 'Canonical Product'
        verbose_name_plural = 'Canonical Products'
        indexes = [
            models.Index(fields=['brand', 'category']),
            models.Index(fields=['canonical_product_key']),
        ]

    def __str__(self):
        if self.brand and self.name.lower().startswith(self.brand.lower()):
            return self.name
        return f"{self.brand} {self.name}"

    def get_lowest_price_listing(self):
        """Returns the active listing offering the lowest total landed price."""
        return self.listings.filter(
            availability_status='in_stock',
            total_price_npr__gt=0
        ).order_by('total_price_npr').first()

    def get_price_summary(self):
        """Calculates lowest, highest, average price and potential savings."""
        active_listings = self.listings.filter(
            availability_status='in_stock',
            total_price_npr__gt=0
        ).order_by('total_price_npr')

        if not active_listings.exists():
            return {
                "cheapest_store": "No Price Available",
                "lowest_price": None,
                "highest_price": None,
                "average_price": None,
                "saving_npr": 0.0,
                "saving_percent": 0.0,
                "store_count": 0
            }

        lowest = active_listings.first()
        highest = active_listings.last()
        prices = [float(l.total_price_npr) for l in active_listings]
        avg = round(sum(prices) / len(prices), 2)
        saving = float(highest.total_price_npr - lowest.total_price_npr)
        saving_pct = round((saving / float(highest.total_price_npr) * 100), 2) if highest.total_price_npr > 0 else 0.0

        return {
            "cheapest_store": lowest.store.name,
            "lowest_price": float(lowest.total_price_npr),
            "highest_price": float(highest.total_price_npr),
            "average_price": avg,
            "saving_npr": round(saving, 2),
            "saving_percent": saving_pct,
            "store_count": active_listings.values('store').distinct().count()
        }


class ProductVariant(models.Model):
    """
    Specific hardware/storage/RAM/color variants of a canonical product.
    """
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='variants')
    variant_name = models.CharField(max_length=100)  # e.g., '8GB/256GB'
    ram = models.CharField(max_length=50, blank=True, default='')
    storage = models.CharField(max_length=50, blank=True, default='')
    color = models.CharField(max_length=50, blank=True, default='')
    sku = models.CharField(max_length=100, blank=True, default='')
    is_active = models.BooleanField(default=True)

    class Meta:
        unique_together = ('product', 'variant_name')
        verbose_name = 'Product Variant'
        verbose_name_plural = 'Product Variants'

    def __str__(self):
        return f"{self.product.name} ({self.variant_name})"


class Store(models.Model):
    """
    Registered online retail platforms & e-commerce stores in Nepal.
    Mirrors source_registry.csv governance rules.
    """
    name = models.CharField(max_length=150, unique=True)
    slug = models.SlugField(max_length=150, unique=True, blank=True)
    base_url = models.URLField(max_length=500, blank=True, default='')
    data_access_method = models.CharField(max_length=255, blank=True, default='')
    permission_status = models.CharField(max_length=150, blank=True, default='Unverified')
    api_or_feed_status = models.CharField(max_length=255, blank=True, default='')
    notes = models.TextField(blank=True, default='')
    logo_url = models.URLField(max_length=1000, blank=True, default='')
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['name']
        verbose_name = 'Store / Platform'
        verbose_name_plural = 'Stores & Platforms'

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name


class ProductListing(models.Model):
    """
    Specific store listing for a product.
    Tracks price, shipping, rating, availability, and live store URL.
    """
    RECORD_TIERS = (
        ('real', 'Verified Real Listing'),
        ('verified', 'Partner Verified Feed'),
        ('sample', 'Canonical Catalog Template'),
        ('demo', 'Demonstration Record'),
    )

    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='listings')
    variant = models.ForeignKey(ProductVariant, on_delete=models.SET_NULL, null=True, blank=True, related_name='listings')
    store = models.ForeignKey(Store, on_delete=models.CASCADE, related_name='listings')
    title = models.CharField(max_length=300)
    price_npr = models.DecimalField(max_digits=12, decimal_places=2)  # Never 0 for valid listing
    shipping_cost_npr = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    is_shipping_known = models.BooleanField(default=True)
    total_price_npr = models.DecimalField(max_digits=12, decimal_places=2, db_index=True)
    rating = models.DecimalField(max_digits=3, decimal_places=2, null=True, blank=True)
    review_count = models.PositiveIntegerField(null=True, blank=True)
    product_url = models.URLField(max_length=1000)
    image_url = models.URLField(max_length=1000, blank=True, default='')
    availability_status = models.CharField(max_length=50, default='in_stock', db_index=True)
    record_type = models.CharField(max_length=30, choices=RECORD_TIERS, default='real', db_index=True)
    is_verified = models.BooleanField(default=True)
    last_updated = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['total_price_npr']
        verbose_name = 'Product Listing'
        verbose_name_plural = 'Product Listings'
        indexes = [
            models.Index(fields=['product', 'store']),
            models.Index(fields=['total_price_npr']),
            models.Index(fields=['record_type', 'availability_status']),
        ]

    def save(self, *args, **kwargs):
        # Automatically enforce landed price formula
        s = self.shipping_cost_npr if self.shipping_cost_npr is not None else 0
        self.total_price_npr = self.price_npr + s
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.product.name} @ {self.store.name} - NPR {self.total_price_npr}"


class PriceHistory(models.Model):
    """
    Temporal audit trail tracking historical price fluctuations over time.
    """
    listing = models.ForeignKey(ProductListing, on_delete=models.CASCADE, related_name='price_history')
    price_npr = models.DecimalField(max_digits=12, decimal_places=2)
    shipping_cost_npr = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    total_price_npr = models.DecimalField(max_digits=12, decimal_places=2)
    recorded_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ['-recorded_at']
        verbose_name = 'Price History Record'
        verbose_name_plural = 'Price History Records'

    def __str__(self):
        return f"{self.listing.product.name} ({self.listing.store.name}) on {self.recorded_at.strftime('%Y-%m-%d')}: NPR {self.total_price_npr}"


class ProductReview(models.Model):
    """
    User ratings and reviews for products.
    """
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='user_reviews')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='product_reviews')
    rating = models.PositiveSmallIntegerField(default=5)  # 1 to 5
    review_title = models.CharField(max_length=200, blank=True, default='')
    review_text = models.TextField(blank=True, default='')
    is_verified_buyer = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        unique_together = ('product', 'user')
        verbose_name = 'Product Review'
        verbose_name_plural = 'Product Reviews'

    def __str__(self):
        return f"{self.user.username} - {self.product.name} ({self.rating}/5)"


class Wishlist(models.Model):
    """
    User saved products for price drop monitoring.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='wishlist_items')
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='in_wishlists')
    target_price_npr = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    notify_price_drop = models.BooleanField(default=True)
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-added_at']
        unique_together = ('user', 'product')
        verbose_name = 'Wishlist Item'
        verbose_name_plural = 'Wishlist Items'

    def __str__(self):
        return f"{self.user.username} -> {self.product.name}"


class RecommendationResult(models.Model):
    """
    Precomputed recommendation relationships (Scikit-Learn TF-IDF / Content-Based).
    """
    source_product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='recommendations_from')
    recommended_product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='recommended_to')
    similarity_score = models.FloatField(default=0.0)
    algorithm_type = models.CharField(max_length=100, default='content_based_tfidf')
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-similarity_score']
        unique_together = ('source_product', 'recommended_product')
        verbose_name = 'Recommendation Mapping'
        verbose_name_plural = 'Recommendation Mappings'

    def __str__(self):
        return f"{self.source_product.name} -> {self.recommended_product.name} ({self.similarity_score:.2f})"


class CollectionLog(models.Model):
    """
    Data collection and pipeline audit logging.
    """
    store = models.ForeignKey(Store, on_delete=models.SET_NULL, null=True, blank=True, related_name='collection_logs')
    data_source = models.CharField(max_length=100)
    status = models.CharField(max_length=50, default='SUCCESS')
    records_attempted = models.PositiveIntegerField(default=0)
    records_collected = models.PositiveIntegerField(default=0)
    failures_count = models.PositiveIntegerField(default=0)
    audit_details = models.JSONField(default=dict, blank=True)
    executed_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ['-executed_at']
        verbose_name = 'Collection Audit Log'
        verbose_name_plural = 'Collection Audit Logs'

    def __str__(self):
        return f"[{self.executed_at.strftime('%Y-%m-%d %H:%M')}] {self.data_source} - {self.status}"

