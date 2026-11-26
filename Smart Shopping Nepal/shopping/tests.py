"""
Smart Shopping Nepal - Application Test Suite Export.
Delegates to tests/test_django_app.py and tests/test_security_performance.py.
"""

from tests.test_django_app import (
    ModelBusinessLogicTests,
    WebsiteModuleViewsTests,
    ManagementCommandSyncPipelineTests,
)
from tests.test_security_performance import (
    SecurityQualityTests,
    PerformanceQualityTests,
)
