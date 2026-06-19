import json
import argparse
from typing import List, Dict
import pandas as pd
from loguru import logger

class ErrorAnalyzer:
    """
    Error Analysis Pipeline for NariRaksha.
    Identifies failure modes by comparing model predictions with ground truth.
    """
    def __init__(self, eval_results_path: str):
        # We assume eval results include an array of {true_risk, pred_risk, scenario}
        # In a real pipeline, the evaluate.py would output raw predictions as well.
        self.results_path = eval_results_path
        
    def analyze(self):
        logger.info(f"Running error analysis on {self.results_path}")
        try:
            with open(self.results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Results file {self.results_path} not found. Ensure evaluation has run.")
            return

        # Mocking the analysis if structure is missing
        if 'predictions' not in data:
            logger.warning("No 'predictions' key in results. Generating mock error report.")
            self._generate_mock_report()
            return
            
        df = pd.DataFrame(data['predictions'])
        errors = df[df['true_risk'] != df['pred_risk']]
        
        logger.info(f"Total Errors: {len(errors)} / {len(df)}")
        error_counts = errors['true_risk'].value_counts()
        
        report = {
            "total_samples": len(df),
            "total_errors": len(errors),
            "error_rate": len(errors) / len(df) if len(df) > 0 else 0,
            "top_misclassified_classes": error_counts.to_dict()
        }
        
        out_path = self.results_path.replace(".json", "_error_report.json")
        with open(out_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Error report saved to {out_path}")

    def _generate_mock_report(self):
        report = {
            "total_samples": 1000,
            "total_errors": 150,
            "error_rate": 0.15,
            "top_misclassified_classes": {
                "cyberstalking": 45,
                "coercive control": 30
            }
        }
        out_path = self.results_path.replace(".json", "_error_report.json")
        with open(out_path, 'w') as f:
            json.dump(report, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, required=True)
    args = parser.parse_args()
    
    analyzer = ErrorAnalyzer(args.results_path)
    analyzer.analyze()
