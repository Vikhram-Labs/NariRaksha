# NariRaksha 🛡️

**NariRaksha** is the world's first open multilingual women's safety reasoning dataset and small language model (SLM) focused on India. Designed to run efficiently on Google Colab T4 GPUs, this project empowers developers, researchers, and NGOs to build localized safety applications, policy analysis tools, and support systems.

## Project Objectives
1. **Multilingual Knowledge Processing**: Ingestion of BNS, POSH, advisories, and cybercrime laws across 8 Indian languages.
2. **NariRaksha-100K Dataset**: A high-quality reasoning dataset detailing scenarios, risk types, severities, and legal contexts.
3. **Optimized SLMs**: QLoRA-based fine-tuning for Qwen, Gemma, SmolLM, and Phi models tailored for resource-constrained hardware (Google Colab T4).

## Supported Languages
- English, Tamil, Hindi, Telugu, Kannada, Malayalam, Bengali, Marathi

## Quick Start on Colab
1. Clone the repository
2. Install requirements:
   ```bash
   pip install -r requirements-colab.txt
   pip install -e .
   ```
3. Generate synthetic data:
   ```bash
   python nariraksha/synthetic_generation.py
   ```
4. Train the model (e.g., Qwen-3B):
   ```bash
   python training/train.py --config configs/qwen.yaml
   ```
5. Evaluate:
   ```bash
   python evaluation/evaluate.py --model_path models/nariraksha-qwen3-3b --test_data datasets/synthetic/nariraksha_synthetic.jsonl
   ```

## Repository Structure
- `nariraksha/`: Core package for data ingestion, synthetic generation, and translation.
- `training/`: QLoRA and PEFT training scripts.
- `evaluation/`: Benchmark suite and testing framework.
- `configs/`: Training hyperparameter configurations.
- `scripts/`: HuggingFace publish scripts.
- `data/` and `datasets/`: Local storage for raw and processed datasets.

## HuggingFace Publishing
To publish the dataset and model to the Hugging Face Hub:
```bash
python scripts/publish_dataset.py --dataset_dir datasets/synthetic --repo_id your_username/NariRaksha-100K --token YOUR_TOKEN
python scripts/publish_model.py --model_dir models/nariraksha-qwen3-3b --repo_id your_username/NariRaksha-Qwen-3B --token YOUR_TOKEN
```

## Quality Standard
Adheres to production-grade standard equivalent to top AI research repositories with strong typing, robust logging (Loguru), and high modularity.
