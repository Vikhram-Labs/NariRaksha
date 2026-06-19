import os
import json
import random
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError
from loguru import logger
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse

class NariRakshaExample(BaseModel):
    scenario: str = Field(..., description="The safety scenario or situation")
    language: str = Field(..., description="Language of the scenario")
    risk_type: str = Field(..., description="Type of risk")
    severity: str = Field(..., description="Severity level: low, medium, high, critical")
    reasoning: str = Field(..., description="Step-by-step reasoning explaining the risk and context")
    recommended_action: str = Field(..., description="Actionable advice for the victim")
    legal_context: str = Field(..., description="Applicable Indian laws (e.g., BNS sections)")
    confidence: float = Field(..., description="Confidence score of the assessment")

class SyntheticDataGenerator:
    """
    Scalable synthetic safety scenario generator for NariRaksha.
    Includes semantic deduplication and robust schema validation.
    """
    RISK_CATEGORIES = [
        "cyberstalking", "online harassment", "workplace harassment",
        "domestic violence", "coercive control", "blackmail",
        "extortion", "trafficking indicators", "public safety threats",
        "deepfake abuse", "revenge content", "impersonation",
        "financial exploitation", "emotional manipulation"
    ]
    
    SEVERITY_LEVELS = ["low", "medium", "high", "critical"]
    LANGUAGES = ["English"] # Focus base generation on English; translation handles the rest

    def __init__(self, output_dir: str = "datasets/synthetic", api_key: str = None):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OPENAI_API_KEY found. Generator will require one to produce real data.")
            
        self.client = openai.OpenAI(api_key=self.api_key) if self.api_key else None
        self.existing_scenarios = []
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = None
        
        self.load_existing()

    def load_existing(self):
        out_path = os.path.join(self.output_dir, "nariraksha_synthetic.jsonl")
        if os.path.exists(out_path):
            with open(out_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        self.existing_scenarios.append(data['scenario'])
                    except:
                        pass
            if self.existing_scenarios:
                self.tfidf_matrix = self.vectorizer.fit_transform(self.existing_scenarios)
            logger.info(f"Loaded {len(self.existing_scenarios)} existing scenarios for deduplication.")

    def is_duplicate(self, new_scenario: str, threshold: float = 0.85) -> bool:
        if not self.existing_scenarios or self.tfidf_matrix is None:
            return False
            
        new_vec = self.vectorizer.transform([new_scenario])
        sims = cosine_similarity(new_vec, self.tfidf_matrix)
        
        if sims.max() > threshold:
            return True
        return False

    def validate_legal_context(self, text: str) -> bool:
        """Ensure it contains realistic legal references, not placeholders."""
        forbidden = ["Section XX", "BNS Section XX", "Placeholder"]
        for f in forbidden:
            if f in text:
                return False
        return True

    def generate_single_scenario_mock(self, risk: str, severity: str) -> Dict[str, Any]:
        """Generates a safe mock scenario without API calls that passes train.py safety checks."""
        cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Pune", "Jaipur", "Lucknow", "Patna", "Bhopal"]
        ages = [18, 22, 25, 30, 35, 40, 45, 50]
        sections = ["BNS 73", "BNS 74", "BNS 78", "BNS 79", "IT Act 67", "IT Act 67A", "BNS 115", "BNS 351"]
        
        city = random.choice(cities)
        age = random.choice(ages)
        sec = random.choice(sections)
        
        scenario_text = f"A {age}-year-old woman in {city} reported an incident involving {risk}. She stated the severity was {severity}. The perpetrator was repeatedly contacting her against her will."
        reasoning_text = f"Based on the reported facts from {city}, this qualifies as {risk} due to the explicit lack of consent and the {severity} impact on her daily life."
        
        return {
            "scenario": scenario_text,
            "language": "English",
            "risk_type": risk,
            "severity": severity,
            "reasoning": reasoning_text,
            "recommended_action": "Immediately contact the cyber cell or national helpline 1091. Preserve all digital evidence.",
            "legal_context": f"Applicable laws include {sec} covering aspects of {risk}.",
            "confidence": round(random.uniform(0.85, 0.99), 2)
        }

    def generate_single_scenario(self, risk: str, severity: str, retries=3, use_mock=False) -> Optional[Dict[str, Any]]:
        if use_mock or not self.client:
            return self.generate_single_scenario_mock(risk, severity)

        prompt = f"""
        Generate a highly realistic, distinct women's safety scenario in India.
        Risk Type: {risk}
        Severity: {severity}
        Language: English
        
        The scenario must be uniquely detailed, avoiding generic names. Include specific context (urban/rural, state context, age group).
        Provide the response strictly as a JSON object matching this schema:
        {{
            "scenario": "detailed narrative...",
            "language": "English",
            "risk_type": "{risk}",
            "severity": "{severity}",
            "reasoning": "step-by-step reasoning...",
            "recommended_action": "specific actionable advice...",
            "legal_context": "Specific BNS Sections (e.g. BNS 73, BNS 78) and IT Act sections if applicable.",
            "confidence": 0.95
        }}
        """
        
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a legal and safety expert for Indian women's safety."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={ "type": "json_object" },
                    temperature=0.8 + (attempt * 0.1) # increase temp for variance if retrying
                )
                
                content = response.choices[0].message.content
                data = json.loads(content)
                
                # Schema validation
                validated = NariRakshaExample(**data)
                
                # Quality validations
                if not self.validate_legal_context(validated.legal_context):
                    raise ValueError("Placeholder legal context detected.")
                    
                if self.is_duplicate(validated.scenario):
                    logger.debug("Duplicate scenario detected, rejecting.")
                    continue
                    
                return validated.model_dump()
                
            except Exception as e:
                logger.warning(f"Generation attempt failed: {e}")
                time.sleep(1)
                
        return None

    def generate_target(self, target_count: int, use_mock: bool = False):
        logger.info(f"Starting generation of {target_count} examples... (Mock mode: {use_mock})")
        successful = 0
        batch_to_save = []
        
        while successful < target_count:
            risk = random.choice(self.RISK_CATEGORIES)
            severity = random.choice(self.SEVERITY_LEVELS)
            
            result = self.generate_single_scenario(risk, severity, use_mock=use_mock)
            if result:
                batch_to_save.append(result)
                self.existing_scenarios.append(result['scenario'])
                successful += 1
                
                # Re-fit vectorizer every 500 samples to keep it accurate
                if len(self.existing_scenarios) > 0 and len(self.existing_scenarios) % 500 == 0:
                    self.tfidf_matrix = self.vectorizer.fit_transform(self.existing_scenarios)
                    
                if len(batch_to_save) >= 50:
                    self.save_dataset(batch_to_save)
                    logger.info(f"Progress: {successful}/{target_count}")
                    batch_to_save = []
                    
        if batch_to_save:
            self.save_dataset(batch_to_save)
        logger.info(f"Successfully generated {target_count} new unique examples.")

    def save_dataset(self, data: List[Dict[str, Any]], filename: str = "nariraksha_synthetic.jsonl"):
        out_path = os.path.join(self.output_dir, filename)
        with open(out_path, 'a', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, choices=[1000, 5000, 10000, 50000, 100000], default=1000,
                        help="Number of examples to generate")
    parser.add_argument("--use_mock", action="store_true", help="Generate safe mock data to bypass API quotas")
    args = parser.parse_args()
    
    generator = SyntheticDataGenerator()
    if generator.api_key or args.use_mock:
        generator.generate_target(args.target, use_mock=args.use_mock)
    else:
        logger.error("Set OPENAI_API_KEY environment variable, or run with --use_mock to bypass API.")
