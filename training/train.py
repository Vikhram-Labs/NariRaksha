import os
import sys
import argparse
import yaml
import torch
from datasets import load_dataset
from loguru import logger

# Try Unsloth first (preferred for T4), fallback to standard transformers
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
    logger.info("Unsloth detected — using FastLanguageModel for optimized T4 training.")
except ImportError:
    UNSLOTH_AVAILABLE = False
    logger.warning("Unsloth not found — falling back to standard transformers + bitsandbytes.")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


def load_config(config_path: str) -> dict:
    """Load YAML training configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def format_prompt(example: dict) -> dict:
    """Formats a NariRaksha example as an instruction-response pair."""
    prompt = (
        "### Instruction:\n"
        "Analyze the following scenario and provide safety reasoning, "
        "risk assessment, and legal context.\n\n"
        f"Scenario:\n{example['scenario']}\n\n"
    )
    response = (
        "### Response:\n"
        "{\n"
        f'  "risk_type": "{example["risk_type"]}",\n'
        f'  "severity": "{example["severity"]}",\n'
        f'  "reasoning": "{example["reasoning"]}",\n'
        f'  "recommended_action": "{example["recommended_action"]}",\n'
        f'  "legal_context": "{example["legal_context"]}"\n'
        "}\n"
    )
    return {"text": prompt + response}


def run_preflight_checks(dataset, tokenizer, config: dict) -> int:
    """
    Validates dataset quality before training begins.
    Returns the dynamic max_seq_length to use.
    Raises SystemExit on failure.
    """
    # Check dataset size
    if len(dataset) < 1000:
        logger.error(
            f"Dataset too small: {len(dataset)} examples. "
            "Minimum 1000 required. Run: python nariraksha/synthetic_generation.py --target 1000 --use_mock"
        )
        sys.exit(1)

    # Check for placeholder contamination
    PLACEHOLDER_PHRASES = ["BNS Section XX", "This constitutes", "Placeholder"]
    for i in range(min(100, len(dataset))):
        text = dataset[i]["text"]
        for phrase in PLACEHOLDER_PHRASES:
            if phrase in text:
                logger.error(
                    f"Placeholder contamination detected: '{phrase}' in sample {i}. "
                    "Regenerate dataset with: python nariraksha/synthetic_generation.py --target 1000 --use_mock"
                )
                sys.exit(1)

    # Calculate token lengths and check average
    logger.info("Calculating sequence lengths for dynamic max_seq_length...")
    token_lengths = [len(tokenizer(ex["text"])["input_ids"]) for ex in dataset]
    avg_tokens = sum(token_lengths) / len(token_lengths)

    if avg_tokens < 50:
        logger.error(
            f"Average token count too low ({avg_tokens:.1f}). "
            "Dataset quality is insufficient for training."
        )
        sys.exit(1)

    token_lengths.sort()
    p95 = token_lengths[int(len(token_lengths) * 0.95)]
    dynamic_max_seq = min(config.get("max_seq_length", 2048), p95 + 128)

    logger.info(
        f"Pre-flight passed ✅ | "
        f"Examples: {len(dataset)} | "
        f"Avg tokens: {avg_tokens:.0f} | "
        f"max_seq_length: {dynamic_max_seq} (p95={p95})"
    )
    return dynamic_max_seq


def train_with_unsloth(config: dict, dataset, dynamic_max_seq: int):
    """QLoRA training via Unsloth FastLanguageModel (optimized for T4)."""
    model_id = config["model_id"]
    output_dir = config["output_dir"]

    logger.info(f"Loading {model_id} via Unsloth 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=dynamic_max_seq,
        dtype=None,  # Auto: bfloat16 if supported, else float16
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        ),
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        random_state=42,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.get("batch_size", 2),
        gradient_accumulation_steps=config.get("grad_accum", 8),
        optim="adamw_8bit",
        learning_rate=float(config.get("learning_rate", 2e-4)),
        weight_decay=0.01,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_grad_norm=0.3,
        max_steps=config.get("max_steps", 1000),
        warmup_steps=config.get("warmup_steps", 50),
        lr_scheduler_type="cosine",
        save_steps=config.get("save_steps", 200),
        logging_steps=config.get("logging_steps", 10),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        max_seq_length=dynamic_max_seq,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    logger.info("Starting Unsloth QLoRA training...")
    trainer.train()

    logger.info(f"Saving model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Training complete ✅")


def train_with_transformers(config: dict, dataset, dynamic_max_seq: int):
    """Fallback QLoRA training via standard transformers + bitsandbytes."""
    model_id = config["model_id"]
    output_dir = config["output_dir"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    logger.info(f"Loading {model_id} via transformers 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    )
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.get("batch_size", 2),
        gradient_accumulation_steps=config.get("grad_accum", 8),
        optim="paged_adamw_8bit",
        learning_rate=float(config.get("learning_rate", 2e-4)),
        weight_decay=0.01,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_grad_norm=0.3,
        max_steps=config.get("max_steps", 1000),
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        save_steps=config.get("save_steps", 200),
        logging_steps=config.get("logging_steps", 10),
        report_to="none",
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

    logger.info("Starting standard QLoRA training...")
    trainer.train()

    logger.info(f"Saving model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Training complete ✅")


def train():
    parser = argparse.ArgumentParser(description="NariRaksha QLoRA Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    model_id = config["model_id"]
    dataset_path = config["dataset_path"]

    logger.info(f"Starting training for {model_id}")
    logger.info(f"Config: {args.config} | Dataset: {dataset_path}")

    # Load tokenizer first (for safety checks)
    if UNSLOTH_AVAILABLE:
        from unsloth import FastLanguageModel
        _, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id, max_seq_length=512, dtype=None, load_in_4bit=False
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Load and format dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

    # Run pre-flight safety checks
    dynamic_max_seq = run_preflight_checks(dataset, tokenizer, config)

    # Train using best available backend
    if UNSLOTH_AVAILABLE:
        train_with_unsloth(config, dataset, dynamic_max_seq)
    else:
        train_with_transformers(config, dataset, dynamic_max_seq)


if __name__ == "__main__":
    train()
