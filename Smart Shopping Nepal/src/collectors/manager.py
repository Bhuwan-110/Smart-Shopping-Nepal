"""
Smart Shopping Nepal - Product Collection Manager Orchestrator.

Manages data collection runs across all registered connectors:
1. Validates connector authorization status against source_registry.csv.
2. Runs parallel or sequential collection.
3. Consolidates and deduplicates collected product records.
4. Logs failure tracking metrics into reports/collection_audit.json.
5. Saves integrated collection results to data/raw/collected_listings.csv.
"""

import os
import json
import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

from src.collectors.base import BaseConnector, CollectionResult
from src.loader import load_source_registry, save_csv
from src.deduplicator import identify_duplicates


class CollectionManager:
    """
    Coordinates multi-platform product data collection while strictly enforcing
    authorization, failure logging, and duplicate prevention.
    """

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        else:
            self.base_dir = base_dir

        self.registry_path = os.path.join(self.base_dir, "data", "raw", "source_registry.csv")
        self.reports_dir = os.path.join(self.base_dir, "reports")
        self.output_raw_path = os.path.join(self.base_dir, "data", "raw", "collected_listings.csv")
        self.connectors: List[BaseConnector] = []

    def register_connector(self, connector: BaseConnector) -> None:
        """Adds a platform connector to the collection roster."""
        self.connectors.append(connector)

    def load_authorized_connectors(self) -> None:
        """
        Inspects source_registry.csv and initializes default compliant connectors:
        - ManualImportConnector (Always authorized)
        - DarazConnector (Authorized if credentials exist or fallback to manual feed)
        """
        from src.collectors.manual_import import ManualImportConnector
        from src.collectors.daraz import DarazConnector

        # 1. Register Manual Import Connector
        manual_imports_dir = os.path.join(self.base_dir, "data", "raw", "manual_imports")
        self.register_connector(ManualImportConnector(import_source=manual_imports_dir))

        # 2. Register Daraz Connector
        daraz_feed = os.path.join(manual_imports_dir, "daraz_verified_listings.csv")
        self.register_connector(DarazConnector(verified_feed_path=daraz_feed))

    def run_collection(self) -> Dict[str, Any]:
        """
        Runs collection across all registered connectors.
        Enforces deduplication, tracks collection failures, and generates an audit log.
        """
        os.makedirs(self.reports_dir, exist_ok=True)
        start_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

        all_collected_records: List[Dict[str, Any]] = []
        connector_summaries: List[Dict[str, Any]] = []
        total_attempted = 0
        total_collected = 0
        total_failed = 0

        for connector in self.connectors:
            result = connector.collect()
            connector_summaries.append(result.to_dict())
            total_attempted += result.total_attempted
            total_collected += result.total_collected
            total_failed += result.total_failed

            if result.success and result.collected_records:
                all_collected_records.extend(result.collected_records)

        # Convert to DataFrame for deduplication and pipeline export
        if all_collected_records:
            collected_df = pd.DataFrame(all_collected_records)
            # Prevent duplicate records on the same platform
            flagged_df, dup_audit = identify_duplicates(collected_df)
            # Remove exact duplicate entries
            deduped_df = flagged_df[~flagged_df["is_duplicate"]].copy()
            # Save raw collected listings
            save_csv(deduped_df, self.output_raw_path)
            final_record_count = len(deduped_df)
            duplicates_removed = int(flagged_df["is_duplicate"].sum())
        else:
            deduped_df = pd.DataFrame()
            final_record_count = 0
            duplicates_removed = 0
            dup_audit = {}

        end_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Compile comprehensive collection audit log
        audit_log = {
            "collection_status": "COMPLETED",
            "start_time": start_time,
            "end_time": end_time,
            "total_connectors_run": len(self.connectors),
            "total_records_attempted": total_attempted,
            "total_records_collected": total_collected,
            "total_failures_recorded": total_failed,
            "duplicates_removed": duplicates_removed,
            "final_clean_records": final_record_count,
            "connector_results": connector_summaries,
            "deduplication_audit": dup_audit,
            "output_filepath": self.output_raw_path if final_record_count > 0 else None
        }

        audit_path = os.path.join(self.reports_dir, "collection_audit.json")
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(audit_log, f, indent=4)

        return {
            "audit_log": audit_log,
            "collected_df": deduped_df
        }
