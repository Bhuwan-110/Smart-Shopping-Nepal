"""
Smart Shopping Nepal - Daraz Nepal Connector.

Adheres strictly to the Daraz Open Platform Terms of Service:
1. Automated web scraping of Daraz without express written consent is prohibited.
2. Official API integration requires approved Open Platform credentials (DARAZ_APP_KEY, DARAZ_APP_SECRET).
3. If credentials are not configured or access is not approved, this connector cleanly
   reports unapproved status without inventing fake credentials or endpoints,
   and delegates to the Manual Verified Import workflow.
"""

import os
from typing import List, Dict, Any, Optional

from src.collectors.base import BaseConnector


class DarazConnector(BaseConnector):
    """
    Official Open Platform connector for Daraz Nepal.
    Enforces compliance and prevents unauthorized bot scraping.
    """

    def __init__(
        self,
        app_key: Optional[str] = None,
        app_secret: Optional[str] = None,
        verified_feed_path: Optional[str] = None
    ):
        self.app_key = app_key or os.getenv("DARAZ_APP_KEY")
        self.app_secret = app_secret or os.getenv("DARAZ_APP_SECRET")
        self.verified_feed_path = verified_feed_path

        # Determine authorization status
        has_credentials = bool(self.app_key and self.app_secret)
        has_verified_feed = bool(self.verified_feed_path and os.path.exists(self.verified_feed_path))

        is_approved = has_credentials or has_verified_feed
        data_source = "official_api" if has_credentials else "manual_verified_feed"

        super().__init__(
            platform_name="Daraz Nepal",
            data_source=data_source,
            is_approved=is_approved,
            default_verification_status="verified_api" if has_credentials else "verified_manual"
        )

    def fetch_raw_listings(self) -> List[Dict[str, Any]]:
        """
        Retrieves listings if credentials exist or from verified partner feed.
        RULE: Does not attempt unauthorized scraping or invent fake API endpoints.
        """
        if self.app_key and self.app_secret:
            # When legitimate merchant credentials exist, query official API
            # For this Phase, we structure the API request schema
            return self._fetch_from_official_api()
        elif self.verified_feed_path and os.path.exists(self.verified_feed_path):
            from src.loader import load_csv
            df = load_csv(self.verified_feed_path)
            return df.to_dict(orient="records")
        else:
            raise PermissionError(
                "Daraz Nepal catalog access requires approved Open Platform credentials (DARAZ_APP_KEY, DARAZ_APP_SECRET). "
                "Automated public scraping is prohibited by Daraz Terms of Service. "
                "Use the manual verified import workflow instead."
            )

    def _fetch_from_official_api(self) -> List[Dict[str, Any]]:
        """
        Stub for official authenticated API client using authorized credentials.
        """
        # In a production environment with approved merchant keys, this issues signed HMAC-SHA256
        # requests to the official Daraz Open Platform gateway (api.daraz.com.np/rest).
        return []
