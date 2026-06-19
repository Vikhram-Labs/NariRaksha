import os
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger

class DataIngestor:
    """
    Ingests and normalizes raw safety-related knowledge sources:
    - BNS (Bharatiya Nyaya Sanhita)
    - POSH (Prevention of Sexual Harassment) Guidelines
    - Cybercrime resources
    - Government advisories
    """

    def __init__(self, raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def ingest_bns_laws(self, file_path: str) -> List[Dict[str, Any]]:
        """Parses BNS laws."""
        logger.info(f"Ingesting BNS laws from {file_path}")
        # Placeholder for actual PDF/Text extraction logic
        # For now, assumes a pre-structured JSON format or text
        processed_data = []
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    processed_data = json.load(f)
                else:
                    # Dummy processing for text
                    text = f.read()
                    processed_data = [{"source": "BNS", "content": text}]
        
        self._save_processed("bns_processed.json", processed_data)
        return processed_data

    def ingest_posh_guidelines(self, file_path: str) -> List[Dict[str, Any]]:
        """Parses POSH guidelines."""
        logger.info(f"Ingesting POSH guidelines from {file_path}")
        processed_data = []
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                 if file_path.endswith('.json'):
                    processed_data = json.load(f)
                 else:
                    text = f.read()
                    processed_data = [{"source": "POSH", "content": text}]
        
        self._save_processed("posh_processed.json", processed_data)
        return processed_data

    def _save_processed(self, filename: str, data: List[Dict[str, Any]]) -> None:
        out_path = self.processed_data_dir / filename
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved processed data to {out_path}")

if __name__ == "__main__":
    ingestor = DataIngestor()
    # ingestor.ingest_bns_laws("data/raw/bns.json")
