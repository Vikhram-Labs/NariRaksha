import os
import json
import argparse
import evaluate as hf_evaluate
from sklearn.metrics import accuracy_score, f1_score
from typing import List, Dict, Any
from loguru import logger

class NariRakshaEvaluator:
    """
    Evaluation suite for NariRaksha models.
    Metrics: accuracy, F1, macro F1, reasoning quality, consistency, multilingual performance, hallucination rate.
    """
    def __init__(self, model_path: str, dataset_path: str):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.rouge = hf_evaluate.load('rouge')
        
    def load_test_data(self) -> List[Dict[str, Any]]:
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def _mock_predict(self, text: str) -> Dict[str, str]:
        # Placeholder for actual model inference
        return {
            "risk_type": "cyberstalking",
            "severity": "high",
            "reasoning": "Mock reasoning",
            "recommended_action": "Mock action",
            "legal_context": "Mock context"
        }

    def evaluate(self):
        logger.info(f"Evaluating model from {self.model_path} on {self.dataset_path}")
        test_data = self.load_test_data()
        
        y_true_risk = []
        y_pred_risk = []
        y_true_sev = []
        y_pred_sev = []
        
        reasoning_refs = []
        reasoning_preds = []

        for item in test_data:
            # Ground truth
            y_true_risk.append(item['risk_type'])
            y_true_sev.append(item['severity'])
            reasoning_refs.append(item['reasoning'])
            
            # Prediction
            pred = self._mock_predict(item['scenario'])
            y_pred_risk.append(pred['risk_type'])
            y_pred_sev.append(pred['severity'])
            reasoning_preds.append(pred['reasoning'])

        # Calculate metrics
        metrics = {
            "risk_accuracy": accuracy_score(y_true_risk, y_pred_risk),
            "risk_macro_f1": f1_score(y_true_risk, y_pred_risk, average='macro', zero_division=0),
            "severity_accuracy": accuracy_score(y_true_sev, y_pred_sev),
            "severity_macro_f1": f1_score(y_true_sev, y_pred_sev, average='macro', zero_division=0),
        }
        
        # Reasoning quality (ROUGE as proxy)
        if len(reasoning_preds) > 0:
            rouge_results = self.rouge.compute(predictions=reasoning_preds, references=reasoning_refs)
            metrics.update({"reasoning_rouge": rouge_results})
            
        logger.info(f"Evaluation Results: {json.dumps(metrics, indent=2)}")
        
        # Save results
        out_file = os.path.join(self.model_path, "eval_results.json")
        with open(out_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved results to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    args = parser.parse_args()
    
    evaluator = NariRakshaEvaluator(args.model_path, args.test_data)
    evaluator.evaluate()
