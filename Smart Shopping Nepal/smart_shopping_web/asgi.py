"""
ASGI config for smart_shopping_web project.
"""
import os
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smart_shopping_web.settings')
application = get_asgi_application()
