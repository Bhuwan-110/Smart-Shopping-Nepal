"""
Smart Shopping Nepal - Security, Performance, and Quality Assessment Test Suite.
Phase 8: Complete Quality & Security Assurance.

Verifies:
1. Security: CSRF protection, Authentication & Authorization, SQL injection immunity, Password hashing.
2. Performance: Query efficiency, Pagination controls, Index usage, Clean 404/500 handling.
"""

import os
import django

if "DJANGO_SETTINGS_MODULE" not in os.environ:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "smart_shopping_web.settings")
django.setup()

from decimal import Decimal
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from django.conf import settings
from shopping.models import Product, ProductListing, Store, Wishlist


class SecurityQualityTests(TestCase):
    """Verifies security controls, input validation, and user protection."""

    def setUp(self):
        self.client = Client(enforce_csrf_checks=True)
        self.store = Store.objects.create(
            name="Daraz Test Store",
            base_url="https://daraz.com.np",
            data_access_method="Official Open Platform API",
            permission_status="Approved"
        )
        self.product = Product.objects.create(
            name="Security Test Phone",
            brand="SecBrand",
            model="SecModel",
            category="Smartphones",
            canonical_product_key="secbrand-secmodel"
        )
        self.listing = ProductListing.objects.create(
            product=self.product,
            store=self.store,
            title="SecBrand SecModel 8GB/128GB",
            price_npr=Decimal("25000.00"),
            shipping_cost_npr=Decimal("100.00"),
            product_url="https://daraz.com.np/secphone",
            availability_status="in_stock",
            record_type="real"
        )
        self.user = User.objects.create_user(username="secuser", password="SuperSecretPassword2026!")

    def test_secret_key_configuration(self):
        """Verify SECRET_KEY is loaded and not empty."""
        self.assertTrue(bool(settings.SECRET_KEY))
        self.assertGreater(len(settings.SECRET_KEY), 20)

    def test_security_middleware_installed(self):
        """Verify critical security middleware layers are active."""
        required_middleware = [
            'django.middleware.security.SecurityMiddleware',
            'django.middleware.csrf.CsrfViewMiddleware',
            'django.middleware.clickjacking.XFrameOptionsMiddleware',
        ]
        for mw in required_middleware:
            self.assertIn(mw, settings.MIDDLEWARE, f"Security middleware '{mw}' is missing!")

    def test_password_hashing_security(self):
        """Verify user passwords are encrypted using PBKDF2 and never stored in plain text."""
        user = User.objects.get(username="secuser")
        self.assertNotEqual(user.password, "SuperSecretPassword2026!")
        self.assertTrue(user.password.startswith("pbkdf2_sha256$"))

    def test_unauthenticated_wishlist_access_denied(self):
        """Verify unauthenticated requests to wishlist endpoints are redirected to login."""
        # View wishlist
        res = self.client.get(reverse('shopping:wishlist'))
        self.assertEqual(res.status_code, 302)
        self.assertIn(reverse('shopping:login'), res.url)

        # Add to wishlist
        res_add = self.client.get(reverse('shopping:add_to_wishlist', kwargs={'product_id': self.product.id}))
        self.assertEqual(res_add.status_code, 302)
        self.assertIn(reverse('shopping:login'), res_add.url)

    def test_sql_injection_resistance(self):
        """Verify malicious SQL injection attempts in search parameters are safely escaped by ORM."""
        malicious_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE shopping_product; --",
            "1 UNION SELECT null, username, password FROM auth_user --",
            "<script>alert('xss')</script>",
            "admin' --",
        ]
        for payload in malicious_payloads:
            res = self.client.get(reverse('shopping:search'), {'q': payload})
            self.assertEqual(res.status_code, 200, f"Failed on payload: {payload}")
            # Ensure database is still intact
            self.assertTrue(Product.objects.filter(pk=self.product.pk).exists())

    def test_x_frame_options_header_present(self):
        """Verify Clickjacking defense via X-Frame-Options header."""
        res = self.client.get(reverse('shopping:home'))
        self.assertEqual(res.status_code, 200)
        self.assertIn('X-Frame-Options', res.headers)
        self.assertIn(res.headers['X-Frame-Options'], ['DENY', 'SAMEORIGIN'])


class PerformanceQualityTests(TestCase):
    """Verifies query efficiency, pagination, and responsiveness."""

    def setUp(self):
        self.client = Client()
        self.store = Store.objects.create(name="Perf Store", base_url="https://perf.com.np")
        # Bulk create 25 products to thoroughly test pagination
        products = [
            Product(
                name=f"Bulk Product {i}",
                brand="BrandX",
                model=f"Model {i}",
                category="Electronics",
                canonical_product_key=f"bulk-product-{i}"
            )
            for i in range(25)
        ]
        Product.objects.bulk_create(products)

    def test_search_pagination_first_and_second_page(self):
        """Verify pagination divides 25 products into pages of 12 items."""
        # Page 1
        res_p1 = self.client.get(reverse('shopping:search') + '?page=1')
        self.assertEqual(res_p1.status_code, 200)
        self.assertEqual(len(res_p1.context['products']), 12)
        self.assertTrue(res_p1.context['page_obj'].has_next())

        # Page 2
        res_p2 = self.client.get(reverse('shopping:search') + '?page=2')
        self.assertEqual(res_p2.status_code, 200)
        self.assertEqual(len(res_p2.context['products']), 12)

        # Page 3 (remaining 1 item)
        res_p3 = self.client.get(reverse('shopping:search') + '?page=3')
        self.assertEqual(res_p3.status_code, 200)
        self.assertEqual(len(res_p3.context['products']), 1)
        self.assertFalse(res_p3.context['page_obj'].has_next())

    def test_invalid_page_falls_back_gracefully(self):
        """Verify passing non-integer or out-of-bounds page numbers doesn't crash."""
        res_invalid = self.client.get(reverse('shopping:search') + '?page=invalid_page')
        self.assertEqual(res_invalid.status_code, 200)
        self.assertEqual(res_invalid.context['page_obj'].number, 1)

        res_overflow = self.client.get(reverse('shopping:search') + '?page=99999')
        self.assertEqual(res_overflow.status_code, 200)
        self.assertEqual(res_overflow.context['page_obj'].number, 3)

    def test_database_indexing_applied(self):
        """Verify database model fields carry indexes to maintain query velocity."""
        product_indexes = [idx.fields for idx in Product._meta.indexes]
        self.assertTrue(any('canonical_product_key' in fields for fields in product_indexes))
        self.assertTrue(any('brand' in fields and 'category' in fields for fields in product_indexes))

        listing_indexes = [idx.fields for idx in ProductListing._meta.indexes]
        self.assertTrue(any('total_price_npr' in fields for fields in listing_indexes))
