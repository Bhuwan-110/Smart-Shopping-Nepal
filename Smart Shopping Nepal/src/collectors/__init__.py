"""
Smart Shopping Nepal - Product Data Collection Package.
Manages authorized data connectors, failure tracking, and verified import workflows.
"""

from src.collectors.base import BaseConnector, CollectionResult, CollectionFailure
from src.collectors.manual_import import ManualImportConnector
from src.collectors.daraz import DarazConnector
from src.collectors.partner_feed import PartnerFeedConnector
from src.collectors.manager import CollectionManager

__all__ = [
    "BaseConnector",
    "CollectionResult",
    "CollectionFailure",
    "ManualImportConnector",
    "DarazConnector",
    "PartnerFeedConnector",
    "CollectionManager",
]
