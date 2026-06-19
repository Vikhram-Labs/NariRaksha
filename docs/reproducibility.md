# NariRaksha Reproducibility Guide

This document outlines the exact steps to reproduce the NariRaksha dataset generation, multilingual expansion, model training, and evaluation pipelines from scratch. Our goal is 100% reproducibility on Google Colab T4 GPUs.

## 1. Environment Setup

To ensure identical dependencies, use the provided `requirements-colab.txt`.

```bash
# Clone repository
git clone https://github.com/your-org/Nariraksha.git
cd Nariraksha

# Install dependencies (pinned versions for reproducibility)
pip install -r requirements-colab.txt
pip install -e .
```

## 2. Dataset Generation (NariRaksha-100K)

The NariRaksha-100K dataset is synthetically generated and translated. To reproduce:

1. **Taxonomy & Ingestion:**
   ```bash
   python nariraksha/taxonomy.py
   python nariraksha/data_ingestion.py
   ```
   *Note: This ingests the BNS and POSH guidelines from `data/raw` into `data/processed`.*

2. **Synthetic Scenario Generation:**
   ```bash
   python nariraksha/synthetic_generation.py
   ```
   *This uses the risk profiles defined in the taxonomy to generate balanced scenarios across severities and demographics.*

3. **Multilingual Translation:**
   ```bash
   python nariraksha/translation.py
   ```
   *Translates the English scenarios into 7 Indian languages using IndicTrans2.*

## 3. Model Training (QLoRA)

We provide configurations for Qwen 2.5 3B, Gemma 2 2B, and SmolLM 1.7B. All are designed to fit within the 16GB VRAM of a Colab T4 instance using 4-bit quantization and gradient checkpointing.

To reproduce the Qwen 3B model:

```bash
python training/train.py --config configs/qwen.yaml
```

**Key Hyperparameters (Qwen 3B):**
- LoRA Rank (r): 32
- LoRA Alpha: 64
- Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Batch Size: 2 (with Gradient Accumulation: 8)
- Learning Rate: 2e-4
- Optimizer: paged_adamw_8bit
- Precision: bf16 (if supported) or fp16

## 4. Evaluation & Benchmarking

To run the evaluation suite (accuracy, F1, reasoning quality ROUGE):

```bash
# Create specific benchmark splits (adversarial, multilingual)
python benchmark/benchmark.py --dataset datasets/synthetic/nariraksha_synthetic.jsonl

# Run evaluation on the test set
python evaluation/evaluate.py --model_path models/nariraksha-qwen3-3b --test_data benchmark/splits/adversarial_benchmark.jsonl
```

## 5. Artifact Publishing

To push reproducible artifacts to HuggingFace:

```bash
python scripts/publish_dataset.py --dataset_dir datasets/synthetic --repo_id YOUR_ORG/NariRaksha-100K --token HF_TOKEN
python scripts/publish_model.py --model_dir models/nariraksha-qwen3-3b --repo_id YOUR_ORG/NariRaksha-Qwen-3B --token HF_TOKEN
```
