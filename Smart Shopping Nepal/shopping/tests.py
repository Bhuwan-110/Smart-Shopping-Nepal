"""
Smart Shopping Nepal - Application Test Suite Export.
Delegates to tests/test_django_app.py for unified test execution.
"""

from tests.test_django_app import (
    ModelBusinessLogicTests,
    WebsiteModuleViewsTests,
    ManagementCommandSyncPipelineTests,
)
