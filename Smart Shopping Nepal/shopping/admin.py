"""
Smart Shopping Nepal - Django Admin Registration.

Configures administrative panels for all core models with:
- Search capabilities
- Filtering & faceted navigation
- Inline listing & variant management
- Read-only audit timestamps
"""

from django.contrib import admin
from .models import (
    Product,
    ProductVariant,
    Store,
    ProductListing,
    PriceHistory,
    ProductReview,
    Wishlist,
    RecommendationResult,
    CollectionLog,
)


class ProductVariantInline(admin.TabularInline):
    model = ProductVariant
    extra = 1
    fields = ('variant_name', 'ram', 'storage', 'color', 'sku', 'is_active')


class ProductListingInline(admin.TabularInline):
    model = ProductListing
    extra = 0
    fields = ('store', 'title', 'price_npr', 'shipping_cost_npr', 'total_price_npr', 'availability_status', 'record_type', 'is_verified')
    readonly_fields = ('total_price_npr',)


@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ('name', 'brand', 'model', 'category', 'canonical_product_key', 'created_at')
    list_filter = ('category', 'brand', 'created_at')
    search_fields = ('name', 'brand', 'model', 'canonical_product_key')
    inlines = [ProductVariantInline, ProductListingInline]
    readonly_fields = ('created_at', 'updated_at')


@admin.register(ProductVariant)
class ProductVariantAdmin(admin.ModelAdmin):
    list_display = ('product', 'variant_name', 'ram', 'storage', 'color', 'sku', 'is_active')
    list_filter = ('is_active', 'storage', 'ram')
    search_fields = ('product__name', 'variant_name', 'sku')


@admin.register(Store)
class StoreAdmin(admin.ModelAdmin):
    list_display = ('name', 'data_access_method', 'permission_status', 'is_active', 'created_at')
    list_filter = ('permission_status', 'is_active')
    search_fields = ('name', 'base_url', 'notes')
    prepopulated_fields = {'slug': ('name',)}


@admin.register(ProductListing)
class ProductListingAdmin(admin.ModelAdmin):
    list_display = (
        'product', 'store', 'price_npr', 'shipping_cost_npr',
        'total_price_npr', 'record_type', 'availability_status',
        'is_verified', 'last_updated'
    )
    list_filter = ('record_type', 'availability_status', 'is_verified', 'store')
    search_fields = ('title', 'product__name', 'store__name')
    readonly_fields = ('total_price_npr',)


@admin.register(PriceHistory)
class PriceHistoryAdmin(admin.ModelAdmin):
    list_display = ('listing', 'price_npr', 'shipping_cost_npr', 'total_price_npr', 'recorded_at')
    list_filter = ('recorded_at',)
    search_fields = ('listing__title', 'listing__product__name', 'listing__store__name')
    readonly_fields = ('recorded_at',)


@admin.register(ProductReview)
class ProductReviewAdmin(admin.ModelAdmin):
    list_display = ('product', 'user', 'rating', 'review_title', 'is_verified_buyer', 'created_at')
    list_filter = ('rating', 'is_verified_buyer', 'created_at')
    search_fields = ('product__name', 'user__username', 'review_title', 'review_text')
    readonly_fields = ('created_at',)


@admin.register(Wishlist)
class WishlistAdmin(admin.ModelAdmin):
    list_display = ('user', 'product', 'target_price_npr', 'notify_price_drop', 'added_at')
    list_filter = ('notify_price_drop', 'added_at')
    search_fields = ('user__username', 'product__name')
    readonly_fields = ('added_at',)


@admin.register(RecommendationResult)
class RecommendationResultAdmin(admin.ModelAdmin):
    list_display = ('source_product', 'recommended_product', 'similarity_score', 'algorithm_type', 'updated_at')
    list_filter = ('algorithm_type',)
    search_fields = ('source_product__name', 'recommended_product__name')
    readonly_fields = ('updated_at',)


@admin.register(CollectionLog)
class CollectionLogAdmin(admin.ModelAdmin):
    list_display = ('executed_at', 'data_source', 'store', 'status', 'records_attempted', 'records_collected', 'failures_count')
    list_filter = ('status', 'data_source', 'executed_at')
    search_fields = ('data_source', 'status')
    readonly_fields = ('executed_at',)
