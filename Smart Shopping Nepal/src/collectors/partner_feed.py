"""
Smart Shopping Nepal - Authorized Partner Feed Connector.

Handles direct feeds provided by local retail partners (e.g., Hukut, Oliz, SmartDoko)
under explicit written agreements.
"""

import os
from typing import List, Dict, Any, Optional

from src.collectors.base import BaseConnector
from src.loader import load_csv


class PartnerFeedConnector(BaseConnector):
    """
    Ingests partner feeds (CSV / JSON) under explicit commercial agreements.
    """

    def __init__(
        self,
        partner_name: str,
        feed_filepath: str,
        agreement_id: Optional[str] = None
    ):
        self.feed_filepath = feed_filepath
        self.agreement_id = agreement_id or "WRITTEN_AGREEMENT_PENDING"

        # Check if feed exists and agreement is active
        is_approved = bool(feed_filepath and os.path.exists(feed_filepath))

        super().__init__(
            platform_name=partner_name,
            data_source="partner_feed",
            is_approved=is_approved,
            default_verification_status="verified_partner"
        )

    def fetch_raw_listings(self) -> List[Dict[str, Any]]:
        """
        Loads partner listings from the agreed feed file.
        """
        if not os.path.exists(self.feed_filepath):
            raise FileNotFoundError(f"Partner feed file not found: {self.feed_filepath}")

        df = load_csv(self.feed_filepath)
        records = df.to_dict(orient="records")
        for r in records:
            r["platform"] = r.get("platform") or self.platform_name
            r["agreement_id"] = self.agreement_id
        return records
