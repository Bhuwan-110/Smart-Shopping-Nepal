"""
Shopping Application Views.
Implements presentation, comparison logic, recommendations, search, and user accounts.
"""

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Q, Min, Max, Avg, Count
from django.http import HttpRequest, HttpResponse, JsonResponse

from .models import (
    Product,
    ProductVariant,
    Store,
    ProductListing,
    PriceHistory,
    ProductReview,
    Wishlist,
    RecommendationResult,
    CollectionLog
)


def home_view(request: HttpRequest) -> HttpResponse:
    """Module 1: Homepage with Hero, Search, Featured Deals, Categories, and Stores."""
    products = Product.objects.all().prefetch_related('listings', 'listings__store')
    categories = Product.objects.values('category').annotate(count=Count('id')).order_by('-count')
    stores = Store.objects.filter(is_active=True)

    # Highlight multi-store deals (products with >= 2 listings)
    deal_products = []
    for p in products:
        summary = p.get_price_summary()
        if summary.get('store_count', 0) >= 2 and summary.get('saving_npr', 0) > 0:
            deal_products.append({
                'product': p,
                'summary': summary,
                'lowest_listing': p.get_lowest_price_listing()
            })

    # Sort deals by highest saving percentage
    deal_products.sort(key=lambda x: x['summary']['saving_percent'], reverse=True)

    recent_products = products.order_by('-created_at')[:8]

    # Precomputed AI recommendations for Homepage
    recommendations = RecommendationResult.objects.select_related(
        'source_product', 'recommended_product'
    ).order_by('-similarity_score')[:4]

    # Recent store updates
    recent_updates = ProductListing.objects.filter(
        availability_status='in_stock',
        price_npr__gt=0
    ).select_related('product', 'store').order_by('-last_updated')[:6]

    context = {
        'deal_products': deal_products[:6],
        'recent_products': recent_products,
        'recommendations': recommendations,
        'recent_updates': recent_updates,
        'categories': categories,
        'stores': stores,
        'total_products': products.count(),
        'total_listings': ProductListing.objects.count(),
        'total_stores': stores.count(),
    }
    return render(request, 'shopping/home.html', context)


def product_search_view(request: HttpRequest) -> HttpResponse:
    """Module 2: Product Search with full query, category, brand, and price filters."""
    query = request.GET.get('q', '').strip()
    selected_category = request.GET.get('category', '').strip()
    selected_brand = request.GET.get('brand', '').strip()
    min_price = request.GET.get('min_price', '').strip()
    max_price = request.GET.get('max_price', '').strip()
    sort_by = request.GET.get('sort', 'relevance')

    products = Product.objects.all().prefetch_related('listings', 'listings__store')

    if query:
        products = products.filter(
            Q(name__icontains=query) |
            Q(brand__icontains=query) |
            Q(model__icontains=query) |
            Q(category__icontains=query) |
            Q(canonical_product_key__icontains=query)
        )

    if selected_category:
        products = products.filter(category__iexact=selected_category)

    if selected_brand:
        products = products.filter(brand__iexact=selected_brand)

    # Annotate with lowest total price for price filtering and sorting
    products = products.annotate(min_price=Min('listings__total_price_npr'))

    if min_price:
        try:
            products = products.filter(min_price__gte=float(min_price))
        except ValueError:
            pass

    if max_price:
        try:
            products = products.filter(min_price__lte=float(max_price))
        except ValueError:
            pass

    if sort_by == 'price_low':
        products = products.order_by('min_price')
    elif sort_by == 'price_high':
        products = products.order_by('-min_price')
    elif sort_by == 'name':
        products = products.order_by('name')
    else:
        products = products.order_by('-created_at', 'id')

    all_categories = Product.objects.values_list('category', flat=True).distinct().order_by('category')
    all_brands = Product.objects.values_list('brand', flat=True).distinct().order_by('brand')

    total_results = products.count()
    paginator = Paginator(products, 12)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    context = {
        'products': page_obj,
        'page_obj': page_obj,
        'query': query,
        'selected_category': selected_category,
        'selected_brand': selected_brand,
        'min_price': min_price,
        'max_price': max_price,
        'sort_by': sort_by,
        'all_categories': all_categories,
        'all_brands': all_brands,
        'total_results': total_results,
    }
    return render(request, 'shopping/search.html', context)


def category_list_view(request: HttpRequest) -> HttpResponse:
    """Module 3a: Category directory."""
    categories = Product.objects.values('category').annotate(
        product_count=Count('id'),
        min_price=Min('listings__total_price_npr'),
        max_price=Max('listings__total_price_npr')
    ).order_by('category')

    context = {'categories': categories}
    return render(request, 'shopping/categories.html', context)


def category_detail_view(request: HttpRequest, category_name: str) -> HttpResponse:
    """Module 3b: Category products view."""
    products = Product.objects.filter(category__iexact=category_name).prefetch_related('listings', 'listings__store').order_by('-created_at', 'id')
    brands = products.values_list('brand', flat=True).distinct()
    total_count = products.count()

    paginator = Paginator(products, 12)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    context = {
        'category_name': category_name,
        'products': page_obj,
        'page_obj': page_obj,
        'brands': brands,
        'total_count': total_count
    }
    return render(request, 'shopping/category_detail.html', context)


def product_detail_view(request: HttpRequest, pk: int) -> HttpResponse:
    """Module 4: Product detail view with specifications, variants, and reviews."""
    product = get_object_or_404(Product.objects.prefetch_related('listings', 'listings__store', 'variants', 'user_reviews'), pk=pk)
    summary = product.get_price_summary()
    listings = product.listings.all().order_by('total_price_npr')
    variants = product.variants.filter(is_active=True)
    reviews = product.user_reviews.all()
    user_in_wishlist = False
    if request.user.is_authenticated:
        user_in_wishlist = Wishlist.objects.filter(user=request.user, product=product).exists()

    # Get similar recommendations
    recommendations = RecommendationResult.objects.filter(source_product=product).select_related('recommended_product')[:4]

    context = {
        'product': product,
        'summary': summary,
        'listings': listings,
        'variants': variants,
        'reviews': reviews,
        'recommendations': recommendations,
        'user_in_wishlist': user_in_wishlist,
    }
    return render(request, 'shopping/product_detail.html', context)


def product_detail_by_key_view(request: HttpRequest, product_key: str) -> HttpResponse:
    """Module 4b: Slug/Key based product lookup."""
    product = get_object_or_404(Product, canonical_product_key=product_key)
    return redirect('shopping:product_detail', pk=product.pk)


def price_comparison_list_view(request: HttpRequest) -> HttpResponse:
    """Module 5a: Global price comparison index."""
    products = Product.objects.all().prefetch_related('listings', 'listings__store')

    comparison_data = []
    for p in products:
        summary = p.get_price_summary()
        listings = p.listings.all().order_by('total_price_npr')
        comparison_data.append({
            'product': p,
            'summary': summary,
            'listings': listings,
            'has_multi_store': summary.get('store_count', 0) > 1
        })

    # Put multi-store deals first
    comparison_data.sort(key=lambda x: (x['has_multi_store'], x['summary'].get('saving_percent', 0)), reverse=True)

    context = {
        'comparison_data': comparison_data,
        'total_compared': len(comparison_data)
    }
    return render(request, 'shopping/compare_list.html', context)


def price_comparison_detail_view(request: HttpRequest, product_key: str) -> HttpResponse:
    """Module 5b: Dedicated side-by-side comparison page for a single product."""
    product = get_object_or_404(Product.objects.prefetch_related('listings', 'listings__store'), canonical_product_key=product_key)
    summary = product.get_price_summary()
    listings = product.listings.all().order_by('total_price_npr')

    context = {
        'product': product,
        'summary': summary,
        'listings': listings,
    }
    return render(request, 'shopping/compare_detail.html', context)


def recommendations_overview_view(request: HttpRequest) -> HttpResponse:
    """Module 6a: Overview of recommended products and smart deals."""
    deals = Product.objects.annotate(
        store_count=Count('listings__store', distinct=True),
        lowest_price=Min('listings__total_price_npr'),
        highest_price=Max('listings__total_price_npr')
    ).filter(store_count__gte=2).order_by('-highest_price')

    rec_mappings = RecommendationResult.objects.select_related('source_product', 'recommended_product')[:12]

    context = {
        'deals': deals,
        'rec_mappings': rec_mappings
    }
    return render(request, 'shopping/recommendations.html', context)


def product_recommendations_view(request: HttpRequest, pk: int) -> HttpResponse:
    """Module 6b: Recommendations tailored for a specific product."""
    product = get_object_or_404(Product, pk=pk)
    recs = RecommendationResult.objects.filter(source_product=product).select_related('recommended_product')

    context = {
        'product': product,
        'recommendations': recs,
    }
    return render(request, 'shopping/product_recommendations.html', context)


def store_list_view(request: HttpRequest) -> HttpResponse:
    """Module 7: Store directory & compliance transparency."""
    stores = Store.objects.all().annotate(active_listings_count=Count('listings'))

    context = {
        'stores': stores,
    }
    return render(request, 'shopping/stores.html', context)


def store_detail_view(request: HttpRequest, slug: str) -> HttpResponse:
    """Module 7b: Individual store listings view."""
    store = get_object_or_404(Store, slug=slug)
    listings = store.listings.select_related('product').order_by('total_price_npr')

    context = {
        'store': store,
        'listings': listings,
        'total_listings': listings.count()
    }
    return render(request, 'shopping/store_detail.html', context)


def user_register_view(request: HttpRequest) -> HttpResponse:
    """Module 8a: User registration."""
    if request.user.is_authenticated:
        return redirect('shopping:home')

    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f"Welcome to Smart Shopping Nepal, {user.username}!")
            return redirect('shopping:home')
        else:
            messages.error(request, "Registration error. Please check the requirements below.")
    else:
        form = UserCreationForm()

    return render(request, 'shopping/register.html', {'form': form})


def user_login_view(request: HttpRequest) -> HttpResponse:
    """Module 8b: User login."""
    if request.user.is_authenticated:
        return redirect('shopping:home')

    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f"Welcome back, {user.username}!")
            next_url = request.GET.get('next') or 'shopping:home'
            return redirect(next_url)
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()

    return render(request, 'shopping/login.html', {'form': form})


def user_logout_view(request: HttpRequest) -> HttpResponse:
    """Module 8c: User logout."""
    logout(request)
    messages.info(request, "You have been logged out.")
    return redirect('shopping:home')


@login_required
def wishlist_view(request: HttpRequest) -> HttpResponse:
    """Module 9a: User Wishlist page."""
    items = Wishlist.objects.filter(user=request.user).select_related('product')
    wishlist_data = []

    for item in items:
        summary = item.product.get_price_summary()
        wishlist_data.append({
            'item': item,
            'product': item.product,
            'summary': summary,
            'lowest_listing': item.product.get_lowest_price_listing()
        })

    context = {
        'wishlist_data': wishlist_data,
        'total_items': len(wishlist_data)
    }
    return render(request, 'shopping/wishlist.html', context)


@login_required
def add_to_wishlist_view(request: HttpRequest, product_id: int) -> HttpResponse:
    """Module 9b: Add product to wishlist."""
    product = get_object_or_404(Product, pk=product_id)
    obj, created = Wishlist.objects.get_or_create(user=request.user, product=product)

    if created:
        messages.success(request, f"{product.name} added to your Wishlist!")
    else:
        messages.info(request, f"{product.name} is already in your Wishlist.")

    next_url = request.META.get('HTTP_REFERER') or 'shopping:wishlist'
    return redirect(next_url)


@login_required
def remove_from_wishlist_view(request: HttpRequest, product_id: int) -> HttpResponse:
    """Module 9c: Remove product from wishlist."""
    product = get_object_or_404(Product, pk=product_id)
    Wishlist.objects.filter(user=request.user, product=product).delete()
    messages.success(request, f"{product.name} removed from your Wishlist.")

    next_url = request.META.get('HTTP_REFERER') or 'shopping:wishlist'
    return redirect(next_url)


def admin_dashboard_view(request: HttpRequest) -> HttpResponse:
    """Module 10: Admin / Operations Dashboard with inventory statistics and scraping logs."""
    total_products = Product.objects.count()
    total_listings = ProductListing.objects.count()
    total_stores = Store.objects.count()
    real_listings_count = ProductListing.objects.filter(record_type='real').count()
    demo_listings_count = ProductListing.objects.filter(record_type='demo').count()

    categories_count = Product.objects.values('category').annotate(count=Count('id'))
    recent_logs = CollectionLog.objects.select_related('store').order_by('-executed_at')[:10]
    multi_store_deals_count = 0

    for p in Product.objects.all():
        if p.listings.values('store').distinct().count() > 1:
            multi_store_deals_count += 1

    context = {
        'total_products': total_products,
        'total_listings': total_listings,
        'total_stores': total_stores,
        'real_listings_count': real_listings_count,
        'demo_listings_count': demo_listings_count,
        'categories_count': categories_count,
        'recent_logs': recent_logs,
        'multi_store_deals_count': multi_store_deals_count,
    }
    return render(request, 'shopping/dashboard.html', context)
