import os
import sys

# Fix libnvJitLink CUDA version mismatch on Colab T4 (CUDA 12.1)
_cuda_paths = ["/usr/local/cuda/lib64", "/usr/local/cuda-12.1/lib64", "/usr/lib/x86_64-linux-gnu"]
_existing = ":".join(p for p in _cuda_paths if os.path.exists(p))
if _existing:
    os.environ["LD_LIBRARY_PATH"] = _existing + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import argparse
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from loguru import logger

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def format_prompt(example):
    """Formats the instruction for the model."""
    prompt = f"### Instruction:\nAnalyze the following scenario and provide safety reasoning, risk assessment, and legal context.\n\nScenario:\n{example['scenario']}\n\n"
    
    response = f"### Response:\n" \
               f"{{\n" \
               f'  "risk_type": "{example["risk_type"]}",\n' \
               f'  "severity": "{example["severity"]}",\n' \
               f'  "reasoning": "{example["reasoning"]}",\n' \
               f'  "recommended_action": "{example["recommended_action"]}",\n' \
               f'  "legal_context": "{example["legal_context"]}"\n' \
               f"}}\n"
    
    return {"text": prompt + response}

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the training config yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    model_id = config["model_id"]
    output_dir = config["output_dir"]
    dataset_path = config["dataset_path"]
    
    logger.info(f"Starting training for {model_id}")
    logger.info("Loading dataset...")
    
    logger.info("Loading tokenizer for safety checks...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)
    
    # --- PRE-FLIGHT SAFETY CHECKS ---
    if len(dataset) < 1000:
        logger.error(f"Dataset too small! Contains {len(dataset)} examples. Minimum 1000 required to prevent overfitting.")
        return
        
    placeholder_phrases = ["BNS Section XX", "This constitutes", "Placeholder"]
    for i in range(min(100, len(dataset))):
        text = dataset[i]['text']
        for phrase in placeholder_phrases:
            if phrase in text:
                logger.error(f"Placeholder data '{phrase}' detected in sample {i}. Training aborted to prevent data contamination.")
                return

    logger.info("Calculating sequence lengths...")
    token_lengths = [len(tokenizer(ex['text'])['input_ids']) for ex in dataset]
    avg_tokens = sum(token_lengths) / len(token_lengths)
    if avg_tokens < 100:
        logger.error(f"Average token count too low ({avg_tokens:.1f}). Expected > 100 for sufficient reasoning quality.")
        return
        
    token_lengths.sort()
    p95_length = token_lengths[int(len(token_lengths) * 0.95)]
    dynamic_max_seq = min(config.get("max_seq_length", 2048), p95_length + 128)
    logger.info(f"Safety checks passed. Scaling max_seq_length to {dynamic_max_seq} (95th percentile + padding).")
    
    # QLoRA Quantization Config (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    
    logger.info("Loading model...")
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare for QLoRA
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    )
    model = get_peft_model(model, peft_config)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.get("batch_size", 4),
        gradient_accumulation_steps=config.get("grad_accum", 4),
        optim="paged_adamw_8bit",
        save_steps=config.get("save_steps", 100),
        logging_steps=config.get("logging_steps", 10),
        learning_rate=float(config.get("learning_rate", 2e-4)),
        weight_decay=0.001,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_grad_norm=0.3,
        max_steps=config.get("max_steps", 1000),
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none" # Or wandb
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=dynamic_max_seq,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )
    
    logger.info("Starting training loop...")
    trainer.train()
    
    logger.info(f"Saving model to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train()
