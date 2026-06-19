import os
import argparse
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from loguru import logger

class NariRakshaPredictor:
    """
    Inference pipeline for NariRaksha models.
    Supports 4-bit quantized loading for T4 GPUs.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        logger.info(f"Loading model from {model_path} in 4-bit...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.model.eval()

    def predict(self, scenario: str) -> dict:
        prompt = f"### Instruction:\nAnalyze the following scenario and provide safety reasoning, risk assessment, and legal context.\n\nScenario:\n{scenario}\n\n### Response:\n{{"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        # Reconstruct JSON (since we forced generation to start with '{')
        try:
            result = json.loads("{" + generated_text.strip())
            return result
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON output. Returning raw text.")
            return {"raw_output": "{" + generated_text}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--scenario", type=str, required=True)
    args = parser.parse_args()
    
    predictor = NariRakshaPredictor(args.model_path)
    result = predictor.predict(args.scenario)
    print(json.dumps(result, indent=2))
