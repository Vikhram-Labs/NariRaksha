# Repository Audit Report

## 1. Executive Summary
The NariRaksha repository, while structurally sound and modular, currently relies on stub implementations and mock data generators. This explains why only 15 examples were generated and why training commenced on placeholder texts. The pipeline is not currently connected to an actual LLM API or robust data extraction engine.

## 2. Component Breakdown

### A. Data Ingestion (`nariraksha/data_ingestion.py`)
- **Status:** STUB
- **Issues:** 
  - `ingest_bns_laws()` and `ingest_posh_guidelines()` read files but do not perform semantic chunking, PDF OCR, or structural parsing.
  - They default to creating a single dictionary `[{"source": "...", "content": text}]`.
  - **Dead Code Path:** No actual integration with the retrieval or generation modules in an automated pipeline.

### B. Synthetic Generation (`nariraksha/synthetic_generation.py`)
- **Status:** MOCK / PLACEHOLDER
- **Issues:**
  - `generate_batch()` does not call any LLM. It hardcodes strings like: `"reasoning": "This constitutes {risk} because..."` and `"legal_context": "BNS Section XX"`.
  - The `__main__` execution block is hardcoded to generate exactly `5` examples (`generator.generate_batch(5)`).
  - No deduplication or semantic similarity filtering exists.

### C. Translation Pipeline (`nariraksha/translation.py`)
- **Status:** MOCK
- **Issues:**
  - `use_mock=True` is the default. It bypasses `IndicTrans2` initialization and prepends `[Lang Translation of]:` to English text.

### D. Training Safety (`training/train.py`)
- **Status:** INCOMPLETE SAFETY CHECKS
- **Issues:**
  - The script blindly loads the `.jsonl` file regardless of size or quality.
  - No verification of token limits, dataset scale (e.g., checking for >1000 samples), or detection of placeholder substrings like `BNS Section XX`.

### E. Configuration (`configs/*.yaml`)
- **Status:** VALID but could be safer.
- **Issues:**
  - `max_seq_length` is statically set to 2048/1024, which may cause OOM on T4 if gradient accumulation and batch size aren't perfectly aligned with sequence distributions.

## 3. Remediation Plan
1. **Rewrite `synthetic_generation.py`**: Implement OpenAI/LiteLLM integration with parallel async generation, Pydantic strict parsing, and a TF-IDF-based semantic deduplication filter.
2. **Add Training Guards**: Inject rigorous pre-flight checks into `train.py`.
3. **Connect Pipelines**: Ensure the generator actually produces scalable datasets (1K, 10K, 100K).
