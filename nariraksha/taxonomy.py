from typing import Dict, List
import json
from loguru import logger

class SafetyTaxonomyBuilder:
    """
    Builds and manages a hierarchical safety taxonomy for NariRaksha.
    Maps high-level categories to specific risk types and keywords.
    """
    def __init__(self):
        self.taxonomy = {
            "Digital Safety": {
                "cyberstalking": ["tracking", "unwanted messages", "doxing"],
                "online harassment": ["trolling", "abusive comments", "threats"],
                "deepfake abuse": ["morphed images", "synthetic media"],
                "revenge content": ["NCII", "non-consensual sharing"]
            },
            "Physical Safety": {
                "domestic violence": ["physical abuse", "verbal abuse", "home"],
                "public safety threats": ["street harassment", "stalking", "transit"]
            },
            "Psychological & Economic": {
                "coercive control": ["isolation", "monitoring", "gaslighting"],
                "financial exploitation": ["withholding money", "forced debt"],
                "blackmail": ["extortion", "threats to expose"]
            }
        }
        
    def export_taxonomy(self, output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.taxonomy, f, indent=2)
        logger.info(f"Taxonomy exported to {output_path}")

if __name__ == "__main__":
    builder = SafetyTaxonomyBuilder()
    # builder.export_taxonomy("data/taxonomy.json")
