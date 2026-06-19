import json
import argparse
from loguru import logger
from collections import Counter

class DataQualityReporter:
    """
    Generates data quality reports for NariRaksha-100K datasets.
    Checks for class balance, severity distribution, and language representation.
    """
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        
    def generate_report(self):
        logger.info(f"Analyzing data quality for {self.dataset_path}")
        
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
        except FileNotFoundError:
            logger.error(f"Dataset {self.dataset_path} not found.")
            return

        if not data:
            logger.warning("Dataset is empty.")
            return

        total_samples = len(data)
        
        risks = [d.get('risk_type') for d in data]
        severities = [d.get('severity') for d in data]
        langs = [d.get('language') for d in data]
        
        report = {
            "total_samples": total_samples,
            "risk_distribution": dict(Counter(risks)),
            "severity_distribution": dict(Counter(severities)),
            "language_distribution": dict(Counter(langs)),
            "missing_values": self._check_missing(data)
        }
        
        out_path = self.dataset_path.replace(".jsonl", "_quality_report.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Quality report generated and saved to {out_path}")
        
    def _check_missing(self, data: list) -> dict:
        missing = {}
        for item in data:
            for k, v in item.items():
                if v is None or v == "":
                    missing[k] = missing.get(k, 0) + 1
        return missing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()
    
    reporter = DataQualityReporter(args.dataset_path)
    reporter.generate_report()
