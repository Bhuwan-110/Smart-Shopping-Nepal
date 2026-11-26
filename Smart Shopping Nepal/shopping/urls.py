"""
Shopping Application URL Routing.
Covers all 10 website modules required in Phase 5.
"""

from django.urls import path
from . import views

app_name = 'shopping'

urlpatterns = [
    # 1. Homepage
    path('', views.home_view, name='home'),

    # 2. Product Search & Catalog Filter
    path('search/', views.product_search_view, name='search'),

    # 3. Product Categories
    path('categories/', views.category_list_view, name='categories'),
    path('category/<str:category_name>/', views.category_detail_view, name='category_detail'),

    # 4. Product Details
    path('product/<int:pk>/', views.product_detail_view, name='product_detail'),
    path('product/key/<str:product_key>/', views.product_detail_by_key_view, name='product_detail_by_key'),

    # 5. Price Comparison
    path('compare/', views.price_comparison_list_view, name='compare_list'),
    path('compare/<str:product_key>/', views.price_comparison_detail_view, name='compare_detail'),

    # 6. Product Recommendations
    path('recommendations/', views.recommendations_overview_view, name='recommendations'),
    path('recommendations/<int:pk>/', views.product_recommendations_view, name='product_recommendations'),

    # 7. Store / Platform Directory & Governance
    path('stores/', views.store_list_view, name='stores'),
    path('store/<slug:slug>/', views.store_detail_view, name='store_detail'),

    # 8. User Registration & Authentication
    path('register/', views.user_register_view, name='register'),
    path('login/', views.user_login_view, name='login'),
    path('logout/', views.user_logout_view, name='logout'),

    # 9. User Wishlist
    path('wishlist/', views.wishlist_view, name='wishlist'),
    path('wishlist/add/<int:product_id>/', views.add_to_wishlist_view, name='add_to_wishlist'),
    path('wishlist/remove/<int:product_id>/', views.remove_from_wishlist_view, name='remove_from_wishlist'),

    # 10. Admin / Operations Dashboard
    path('dashboard/', views.admin_dashboard_view, name='dashboard'),
]
