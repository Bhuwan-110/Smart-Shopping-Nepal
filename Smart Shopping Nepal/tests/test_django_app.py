"""
Smart Shopping Nepal - Django Web Application & Database Integration Test Suite.

Tests:
1. Core relational models (Product, ProductVariant, Store, ProductListing, PriceHistory,
   ProductReview, Wishlist, RecommendationResult, CollectionLog).
2. Business logic & price calculation formulas (Landed cost, price summary, savings math).
3. All 10 website view modules and HTTP status codes.
4. User authentication and Wishlist interaction flows.
5. Management command (sync_pipeline) execution.
"""

import os
import django

if 'DJANGO_SETTINGS_MODULE' not in os.environ:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smart_shopping_web.settings')
django.setup()

from decimal import Decimal
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from django.core.management import call_command
from shopping.models import (
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


class ModelBusinessLogicTests(TestCase):
    """Tests for database schemas, constraints, and business logic methods."""

    def setUp(self):
        self.store_a = Store.objects.create(
            name="Daraz Nepal",
            base_url="https://daraz.com.np",
            data_access_method="Official Open Platform API",
            permission_status="Partner API Approved"
        )
        self.store_b = Store.objects.create(
            name="Hamrobazaar",
            base_url="https://hamrobazaar.com",
            data_access_method="Authorized Merchant Feed",
            permission_status="Written Permission"
        )
        self.product = Product.objects.create(
            name="Samsung Galaxy A55 5G",
            brand="Samsung",
            model="Galaxy A55 5G",
            category="Smartphones",
            canonical_product_key="samsung-galaxy-a55-5g"
        )

    def test_store_slug_auto_generation(self):
        """Verify Store slug is automatically generated from name."""
        self.assertEqual(self.store_a.slug, "daraz-nepal")
        self.assertEqual(str(self.store_a), "Daraz Nepal")

    def test_product_str(self):
        """Verify Product string representation."""
        self.assertEqual(str(self.product), "Samsung Galaxy A55 5G")

    def test_listing_landed_cost_calculation(self):
        """
        Verify that total_price_npr is automatically computed as:
        Total Price = Product Price + Shipping Cost
        """
        listing = ProductListing.objects.create(
            product=self.product,
            store=self.store_a,
            title="Samsung Galaxy A55 5G 8GB/256GB - Awesome Navy",
            price_npr=Decimal("53999.00"),
            shipping_cost_npr=Decimal("150.00"),
            product_url="https://daraz.com.np/products/samsung-a55",
            availability_status="in_stock",
            record_type="real"
        )
        self.assertEqual(listing.total_price_npr, Decimal("54149.00"))

    def test_listing_landed_cost_with_zero_or_null_shipping(self):
        """Verify free/null shipping behaves correctly."""
        listing = ProductListing.objects.create(
            product=self.product,
            store=self.store_b,
            title="Samsung Galaxy A55 5G 8GB/256GB",
            price_npr=Decimal("52499.00"),
            shipping_cost_npr=None,
            is_shipping_known=False,
            product_url="https://hamrobazaar.com/products/samsung-a55",
            availability_status="in_stock",
            record_type="real"
        )
        self.assertEqual(listing.total_price_npr, Decimal("52499.00"))

    def test_product_price_summary_and_savings_calculation(self):
        """
        Verify multi-store lowest price, highest price, average, and savings metrics.
        """
        # Store A: 53999 + 150 = 54149
        ProductListing.objects.create(
            product=self.product,
            store=self.store_a,
            title="Samsung Galaxy A55 Store A",
            price_npr=Decimal("53999.00"),
            shipping_cost_npr=Decimal("150.00"),
            product_url="https://storea.com/p1",
            availability_status="in_stock",
            record_type="real"
        )
        # Store B: 51999 + 200 = 52199
        ProductListing.objects.create(
            product=self.product,
            store=self.store_b,
            title="Samsung Galaxy A55 Store B",
            price_npr=Decimal("51999.00"),
            shipping_cost_npr=Decimal("200.00"),
            product_url="https://storeb.com/p1",
            availability_status="in_stock",
            record_type="real"
        )

        summary = self.product.get_price_summary()
        self.assertEqual(summary["cheapest_store"], "Hamrobazaar")
        self.assertEqual(summary["lowest_price"], 52199.00)
        self.assertEqual(summary["highest_price"], 54149.00)
        self.assertEqual(summary["store_count"], 2)
        self.assertEqual(summary["saving_npr"], 1950.00)
        expected_pct = round((1950.00 / 54149.00) * 100, 2)
        self.assertEqual(summary["saving_percent"], expected_pct)

    def test_product_variant_creation(self):
        """Verify ProductVariant relationship."""
        variant = ProductVariant.objects.create(
            product=self.product,
            variant_name="8GB/256GB Awesome Iceblue",
            ram="8GB",
            storage="256GB",
            color="Awesome Iceblue"
        )
        self.assertEqual(str(variant), "Samsung Galaxy A55 5G (8GB/256GB Awesome Iceblue)")

    def test_price_history_and_review(self):
        """Verify PriceHistory and ProductReview persistence."""
        listing = ProductListing.objects.create(
            product=self.product,
            store=self.store_a,
            title="Samsung Galaxy A55",
            price_npr=Decimal("50000.00"),
            shipping_cost_npr=Decimal("100.00"),
            product_url="https://daraz.com.np/item",
            availability_status="in_stock",
            record_type="real"
        )
        history = PriceHistory.objects.create(
            listing=listing,
            price_npr=listing.price_npr,
            shipping_cost_npr=listing.shipping_cost_npr,
            total_price_npr=listing.total_price_npr
        )
        self.assertEqual(history.total_price_npr, Decimal("50100.00"))

        user = User.objects.create_user(username="testnepal", password="Password123!")
        review = ProductReview.objects.create(
            product=self.product,
            user=user,
            rating=5,
            review_title="Superb battery life",
            review_text="Best value smartphone in Nepal right now."
        )
        self.assertEqual(review.rating, 5)
        self.assertEqual(str(review), "testnepal - Samsung Galaxy A55 5G (5/5)")


class WebsiteModuleViewsTests(TestCase):
    """Tests all 10 frontend view modules and HTTP response codes."""

    def setUp(self):
        self.client = Client()
        self.store = Store.objects.create(
            name="SastoDeal",
            base_url="https://sastodeal.com",
            data_access_method="Partner Data Feed",
            permission_status="Partner Agreement Approved"
        )
        self.product = Product.objects.create(
            name="Redmi Note 13 Pro",
            brand="Xiaomi",
            model="Note 13 Pro",
            category="Smartphones",
            canonical_product_key="xiaomi-redmi-note-13-pro"
        )
        self.listing = ProductListing.objects.create(
            product=self.product,
            store=self.store,
            title="Xiaomi Redmi Note 13 Pro 8GB/256GB",
            price_npr=Decimal("34999.00"),
            shipping_cost_npr=Decimal("100.00"),
            product_url="https://sastodeal.com/item/redmi-13-pro",
            availability_status="in_stock",
            record_type="real"
        )
        self.user = User.objects.create_user(username="buyer_pokhara", password="SafePassword123")

    def test_homepage_view(self):
        """Module 1: Homepage returns 200 and loads products."""
        response = self.client.get(reverse('shopping:home'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Smart Shopping Nepal")
        self.assertContains(response, "Redmi Note 13 Pro")

    def test_product_search_view(self):
        """Module 2: Search returns 200 and filters results."""
        response = self.client.get(reverse('shopping:search') + '?q=Redmi')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Redmi Note 13 Pro")

    def test_categories_and_detail_views(self):
        """Module 3: Categories listing and Category detail."""
        res_cat = self.client.get(reverse('shopping:categories'))
        self.assertEqual(res_cat.status_code, 200)
        self.assertContains(res_cat, "Smartphones")

        res_detail = self.client.get(reverse('shopping:category_detail', kwargs={'category_name': 'Smartphones'}))
        self.assertEqual(res_detail.status_code, 200)
        self.assertContains(res_detail, "Redmi Note 13 Pro")

    def test_product_detail_view(self):
        """Module 4: Product detail page."""
        response = self.client.get(reverse('shopping:product_detail', kwargs={'pk': self.product.pk}))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Xiaomi")
        self.assertContains(response, "SastoDeal")
        self.assertContains(response, "35,099")

    def test_price_comparison_views(self):
        """Module 5: Price comparison list and single product comparison."""
        res_list = self.client.get(reverse('shopping:compare_list'))
        self.assertEqual(res_list.status_code, 200)

        res_detail = self.client.get(reverse('shopping:compare_detail', kwargs={'product_key': self.product.canonical_product_key}))
        self.assertEqual(res_detail.status_code, 200)
        self.assertContains(res_detail, "Price Comparison")

    def test_recommendations_view(self):
        """Module 6: Recommendations page."""
        # Create second product and recommendation pairing
        product_b = Product.objects.create(
            name="Realme 12 Pro",
            brand="Realme",
            model="12 Pro",
            category="Smartphones",
            canonical_product_key="realme-12-pro"
        )
        RecommendationResult.objects.create(
            source_product=self.product,
            recommended_product=product_b,
            similarity_score=0.85
        )
        response = self.client.get(reverse('shopping:recommendations'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Realme 12 Pro")

    def test_stores_directory_and_detail(self):
        """Module 7: Store platform directory and single store view."""
        res_dir = self.client.get(reverse('shopping:stores'))
        self.assertEqual(res_dir.status_code, 200)
        self.assertContains(res_dir, "SastoDeal")

        res_store = self.client.get(reverse('shopping:store_detail', kwargs={'slug': self.store.slug}))
        self.assertEqual(res_store.status_code, 200)
        self.assertContains(res_store, "Partner Agreement Approved")

    def test_user_registration_and_login(self):
        """Module 8: User registration and login flow."""
        # Registration GET
        res_reg_get = self.client.get(reverse('shopping:register'))
        self.assertEqual(res_reg_get.status_code, 200)

        # Registration POST with UserCreationForm fields
        res_reg_post = self.client.post(reverse('shopping:register'), {
            'username': 'newuser_ktm',
            'password1': 'StrongPassword123!',
            'password2': 'StrongPassword123!'
        }, follow=True)
        self.assertEqual(res_reg_post.status_code, 200)
        self.assertTrue(User.objects.filter(username='newuser_ktm').exists())

        # Login POST
        res_login = self.client.post(reverse('shopping:login'), {
            'username': 'newuser_ktm',
            'password': 'StrongPassword123!'
        }, follow=True)
        self.assertEqual(res_login.status_code, 200)
        self.assertTrue(res_login.context['user'].is_authenticated)

    def test_wishlist_functionality(self):
        """Module 9: User Wishlist add, view, and remove."""
        self.client.login(username="buyer_pokhara", password="SafePassword123")

        # Add to wishlist
        add_res = self.client.post(reverse('shopping:add_to_wishlist', kwargs={'product_id': self.product.id}), {
            'target_price': '33000'
        }, follow=True)
        self.assertEqual(add_res.status_code, 200)
        self.assertTrue(Wishlist.objects.filter(user=self.user, product=self.product).exists())

        # View wishlist
        view_res = self.client.get(reverse('shopping:wishlist'))
        self.assertEqual(view_res.status_code, 200)
        self.assertContains(view_res, "Redmi Note 13 Pro")

        # Remove from wishlist
        remove_res = self.client.get(reverse('shopping:remove_from_wishlist', kwargs={'product_id': self.product.id}), follow=True)
        self.assertEqual(remove_res.status_code, 200)
        self.assertFalse(Wishlist.objects.filter(user=self.user, product=self.product).exists())

    def test_admin_operations_dashboard(self):
        """Module 10: Admin / Operations dashboard."""
        response = self.client.get(reverse('shopping:dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Operations & Ingestion Dashboard")
        self.assertContains(response, "Smartphones")


class ManagementCommandSyncPipelineTests(TestCase):
    """Tests the sync_pipeline management command."""

    def test_sync_pipeline_execution(self):
        """Verify python manage.py sync_pipeline runs cleanly and populates database."""
        call_command('sync_pipeline')
        self.assertGreater(Store.objects.count(), 0)
        self.assertGreater(Product.objects.count(), 0)
        self.assertGreater(ProductListing.objects.count(), 0)
        self.assertGreater(CollectionLog.objects.count(), 0)
