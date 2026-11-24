"""
Smart Shopping Nepal - Manual Verified Import Connector.

The primary approved, human-verified ingestion workflow:
- Ingests structured CSV/JSON product feeds prepared by merchants or audited manually.
- Fully authorized and compliant with data governance policies.
- Enforces strict schema validation, timestamping, and failure logging.
"""

import os
import glob
import pandas as pd
from typing import List, Dict, Any, Optional

from src.collectors.base import BaseConnector
from src.loader import load_csv


class ManualImportConnector(BaseConnector):
    """
    Ingests manually verified product catalog files.
    Ideal when external scraping is forbidden by platform Terms of Service.
    """

    def __init__(
        self,
        import_source: Optional[str] = None,
        platform_name: str = "Manual Verified Import",
        verification_status: str = "verified_manual"
    ):
        super().__init__(
            platform_name=platform_name,
            data_source="manual_import",
            is_approved=True,  # Manual verified imports are always authorized by definition
            default_verification_status=verification_status
        )
        self.import_source = import_source

    def fetch_raw_listings(self) -> List[Dict[str, Any]]:
        """
        Reads listings from a single CSV/JSON file or all CSVs in an import directory.
        """
        listings: List[Dict[str, Any]] = []

        if not self.import_source:
            # Default directory for manual imports
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            import_dir = os.path.join(base_dir, "data", "raw", "manual_imports")
            if not os.path.exists(import_dir):
                return []
            csv_files = glob.glob(os.path.join(import_dir, "*.csv"))
        elif os.path.isdir(self.import_source):
            csv_files = glob.glob(os.path.join(self.import_source, "*.csv"))
        elif os.path.isfile(self.import_source):
            csv_files = [self.import_source]
        else:
            raise FileNotFoundError(f"Specified manual import source does not exist: {self.import_source}")

        for filepath in csv_files:
            # Ignore template files
            if "template" in os.path.basename(filepath).lower():
                continue

            try:
                df = load_csv(filepath)
                # Convert records to dictionary
                records = df.to_dict(orient="records")
                for r in records:
                    # Inject import source provenance
                    r["import_filename"] = os.path.basename(filepath)
                    listings.append(r)
            except Exception as e:
                # Let collector track file read errors as failure
                raise RuntimeError(f"Error reading manual import file {filepath}: {str(e)}")

        return listings
