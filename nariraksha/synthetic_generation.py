import json
import random
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from loguru import logger

class NariRakshaExample(BaseModel):
    scenario: str = Field(..., description="The safety scenario or situation")
    language: str = Field(..., description="Language of the scenario (e.g., English, Hindi)")
    risk_type: str = Field(..., description="Type of risk (e.g., cyberstalking, domestic violence)")
    severity: str = Field(..., description="Severity level: low, medium, high, critical")
    reasoning: str = Field(..., description="Step-by-step reasoning explaining the risk and context")
    recommended_action: str = Field(..., description="Actionable advice for the victim")
    legal_context: str = Field(..., description="Applicable Indian laws (e.g., BNS sections)")
    confidence: float = Field(..., description="Confidence score of the assessment")

class SyntheticDataGenerator:
    """
    Generates synthetic safety scenarios for NariRaksha-100K dataset.
    """
    RISK_CATEGORIES = [
        "cyberstalking", "online harassment", "workplace harassment",
        "domestic violence", "coercive control", "blackmail",
        "extortion", "trafficking indicators", "public safety threats",
        "deepfake abuse", "revenge content", "impersonation",
        "financial exploitation", "emotional manipulation"
    ]
    
    SEVERITY_LEVELS = ["low", "medium", "high", "critical"]
    LANGUAGES = ["English", "Tamil", "Hindi", "Telugu", "Kannada", "Malayalam", "Bengali", "Marathi"]

    def __init__(self, output_dir: str = "datasets/synthetic"):
        self.output_dir = output_dir
        import os
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_prompt(self, risk: str, severity: str, language: str) -> str:
        """Creates a prompt for the LLM to generate a scenario."""
        prompt = f"""
        Generate a realistic women's safety scenario in India.
        Risk Type: {risk}
        Severity: {severity}
        Language: {language}
        
        The scenario should be culturally relevant to India, varying demographics (urban/rural).
        Output the response strictly as a JSON object matching this schema:
        {{
            "scenario": "...",
            "language": "{language}",
            "risk_type": "{risk}",
            "severity": "{severity}",
            "reasoning": "...",
            "recommended_action": "...",
            "legal_context": "...",
            "confidence": 0.95
        }}
        """
        return prompt

    def generate_batch(self, batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        Mock generation of a batch. In production, this would call an LLM API 
        (e.g., OpenAI, Anthropic, or local open-source model).
        """
        logger.info(f"Generating batch of {batch_size} synthetic examples...")
        examples = []
        for _ in range(batch_size):
            risk = random.choice(self.RISK_CATEGORIES)
            severity = random.choice(self.SEVERITY_LEVELS)
            language = random.choice(self.LANGUAGES)
            
            # This is a mock example. You would use LangChain or OpenAI here.
            mock_example = {
                "scenario": f"A woman in a rural setting is facing {risk} with {severity} severity.",
                "language": language,
                "risk_type": risk,
                "severity": severity,
                "reasoning": f"This constitutes {risk} because...",
                "recommended_action": "Contact local authorities and national helpline 1091.",
                "legal_context": "BNS Section XX, IT Act Section 67",
                "confidence": round(random.uniform(0.85, 0.99), 2)
            }
            # Validate with Pydantic
            validated = NariRakshaExample(**mock_example)
            examples.append(validated.model_dump())
            
        return examples

    def save_dataset(self, data: List[Dict[str, Any]], filename: str = "nariraksha_synthetic.jsonl"):
        out_path = f"{self.output_dir}/{filename}"
        with open(out_path, 'a', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"Appended {len(data)} examples to {out_path}")

if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    batch = generator.generate_batch(5)
    generator.save_dataset(batch)
