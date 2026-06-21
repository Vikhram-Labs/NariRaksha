# 🛡️ NariRaksha

**NariRaksha** is an open-source Indian women's safety Small Language Model (SLM) fine-tuned to provide structured safety assessments — risk classification, severity scoring, legal context under BNS 2023 / IT Act, and actionable recommendations.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vikhram-Labs/NariRaksha/blob/main/notebooks/NariRaksha_Colab.ipynb)

---

## 🚀 Quickstart — Google Colab (recommended)

**One click → open notebook → run cells top to bottom.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vikhram-Labs/NariRaksha/blob/main/notebooks/NariRaksha_Colab.ipynb)

1. Click the badge above
2. Set Runtime → T4 GPU
3. Fill in Cell 3 (model, steps, HF token)
4. Run all cells

No local setup. No CUDA conflicts. Everything runs inside the notebook.

---

## 🏗️ Architecture (v2 — clean rewrite)

```
NariRaksha-SLM/
├── notebooks/
│   └── NariRaksha_Colab.ipynb   ← ✅ Start here (Colab one-click)
├── nariraksha/
│   ├── __init__.py
│   └── data.py                  ← Dataset generation + prompt formatting
├── configs/
│   └── default.yaml             ← Reference config (notebook overrides these)
└── README.md
```

**Key design decisions:**
- **Unsloth only** — no direct `bitsandbytes` imports; eliminates all `libnvJitLink` CUDA errors
- **Single notebook** — entire pipeline (install → generate → train → publish) in 10 cells
- **Mock data by default** — no API key needed to start training
- **HF-native publishing** — push model + dataset to Hugging Face from the notebook

---

## 📦 What the notebook does

| Cell | Action |
|------|--------|
| 1 | Install `unsloth`, `datasets`, `huggingface_hub` |
| 2 | Clone / pull this repo |
| 3 | **Configure** (model, steps, HF token) ← edit this |
| 4 | Generate synthetic safety dataset |
| 5 | Load model (Qwen2.5-3B / Phi-3 / Gemma-2) |
| 6 | QLoRA fine-tune with Unsloth |
| 7 | Save model (LoRA + merged fp16) |
| 8 | Quick inference test |
| 9 | Publish model + dataset → Hugging Face |
| 10 | Evaluate (risk accuracy, severity accuracy) |

---

## 🗂️ Dataset schema

```json
{
  "scenario":           "A 24-year-old woman in Pune received threatening messages...",
  "language":           "English",
  "risk_type":          "blackmail / extortion",
  "severity":           "critical",
  "reasoning":          "The act constitutes blackmail under BNS 308(4)...",
  "recommended_action": "File FIR immediately, preserve all screenshots...",
  "legal_context":      "BNS Section 308(4), IT Act Section 66C",
  "confidence":         0.96
}
```

---

## 🧠 Supported models (all tested on T4)

| Model | Size | HF ID |
|-------|------|-------|
| Qwen2.5-3B-Instruct | 3B | `Qwen/Qwen2.5-3B-Instruct` |
| Phi-3-mini | 3.8B | `unsloth/Phi-3-mini-4k-instruct` |
| Gemma-2-2B | 2B | `unsloth/gemma-2-2b-it` |

---

## ⚖️ Legal coverage

| Category | Applicable Law |
|---|---|
| Cyberstalking | BNS §78, IT Act §66E |
| Online harassment | BNS §351, IT Act §67 |
| Workplace harassment | POSH Act 2013 |
| Domestic violence | PWDVA 2005 |
| Deepfake / NCII | IT Act §67A, BNS §73, DPDPA 2023 |
| Trafficking | ITPA, BNS §§143-144 |

---

## 📄 License

Code: Apache 2.0 · Dataset: CC-BY 4.0
