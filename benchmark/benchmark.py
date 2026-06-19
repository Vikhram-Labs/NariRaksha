import os
import json
import argparse
from loguru import logger

class BenchmarkCreator:
    """
    Creates specialized benchmark datasets for NariRaksha.
    Splits datasets into varied demographics, edge cases, and adversarial examples.
    """
    def __init__(self, dataset_path: str, output_dir: str):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
            
    def create_adversarial_split(self, data: list):
        # Placeholder logic: Find examples with low confidence or high complexity
        adversarial = [d for d in data if d.get('confidence', 1.0) < 0.90]
        out_path = os.path.join(self.output_dir, "adversarial_benchmark.jsonl")
        with open(out_path, 'w', encoding='utf-8') as f:
            for item in adversarial:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Created adversarial benchmark with {len(adversarial)} examples")

    def create_multilingual_split(self, data: list):
        # Group by language
        langs = {}
        for d in data:
            langs.setdefault(d['language'], []).append(d)
            
        out_path = os.path.join(self.output_dir, "multilingual_benchmark.jsonl")
        with open(out_path, 'w', encoding='utf-8') as f:
            for item in data:
                if item['language'] != 'English':
                    f.write(json.dumps(item) + '\n')
        logger.info(f"Created multilingual benchmark with {sum(len(v) for k,v in langs.items() if k != 'English')} examples")

    def run(self):
        logger.info("Running benchmark creator...")
        data = self.load_data()
        self.create_adversarial_split(data)
        self.create_multilingual_split(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="benchmark/splits")
    args = parser.parse_args()
    
    creator = BenchmarkCreator(args.dataset, args.out_dir)
    creator.run()
