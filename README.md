# Simplification Benchmark Pipeline

End-to-end evaluation harness for LLM text simplification, optimised for a single H100 80 GB GPU running Ollama alongside API-hosted models.

---

## File Overview

| File | Purpose |
|---|---|
| `step4_run_and_score.py` | Main pipeline script (run → score → judge → results) |
| `config.py` | All constants, model lists, prompts |
| `api_client.py` | Unified `call_model()` for every provider |
| `requirements.txt` | Python dependencies |
| `.env.example` | API key template |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API keys

```bash
cp .env.example .env
# Edit .env and fill in your keys, then:
export $(cat .env | xargs)
```

### 3. Pull Ollama models (if using local inference)

```bash
ollama pull llama3.1:70b
ollama pull gemma3:27b
```

### 4. Prepare input data

Your `--input-dir` must contain:

```
input_dir/
  simp_master_table.jsonl     # dataset items
  simp_probecore_v1.jsonl     # evaluation probes
  simp_probes_all.jsonl       # all generated probes (for pipeline overview)
```

**`simp_master_table.jsonl` — one record per line:**
```json
{
  "item_id": "ose_001",
  "dataset": "OneStopEnglish",
  "original": "The text to simplify...",
  "target_level": "a2",
  "ose_level": "advanced",
  "tier": 1,
  "text_length_words": 120
}
```

**`simp_probecore_v1.jsonl` — one probe per line:**
```json
{
  "probe_id": "probe_inv_001",
  "base_item_id": "ose_001",
  "probe_family": "invariance",
  "probe_subtype": "paraphrase",
  "probe_text": "A paraphrased version of the original...",
  "base_text": "The original text...",
  "target_level": "a2",
  "tier": 1,
  "dataset": "OneStopEnglish",
  "generator_model": "",
  "critical_detail": ""
}
```

`probe_family` options: `invariance` | `directional` | `shortcut`

---

## Running the Pipeline

### Full pipeline (all 4 phases)

```bash
python step4_run_and_score.py \
  --input-dir  ./data \
  --output-dir ./results \
  --phase all
```

### Individual phases

```bash
# Phase 1 — Generate simplifications
python step4_run_and_score.py --input-dir ./data --output-dir ./results --phase run

# Phase 2 — Rule-based scoring
python step4_run_and_score.py --input-dir ./data --output-dir ./results --phase score

# Phase 3 — LLM judge evaluation
python step4_run_and_score.py --input-dir ./data --output-dir ./results --phase judge

# Phase 4 — Aggregate results into 13 CSV/JSON files
python step4_run_and_score.py --input-dir ./data --output-dir ./results --phase results
```

### Run specific models only

```bash
python step4_run_and_score.py \
  --input-dir ./data --output-dir ./results \
  --models llama-3.1-70b,gemma-3-27b
```

### All CLI options

| Flag | Default | Description |
|---|---|---|
| `--input-dir` | *(required)* | Directory with input JSONL files |
| `--output-dir` | *(required)* | Directory for all outputs |
| `--phase` | `all` | `run` \| `score` \| `judge` \| `results` \| `all` |
| `--models` | all models | Comma-separated model names to run |
| `--delay` | `0.5` | Seconds between sequential API calls |
| `--api-threads` | `4` | Concurrent threads for API models |
| `--ollama-url` | `http://localhost:11434` | Ollama server URL |

---

## Output Files (13 files in `results/simp_scores/`)

| File | Description |
|---|---|
| `instance_level_results.csv` | Per-probe score for every model |
| `reliability_summary.csv` | Per-model summary (invariance / directional / shortcut / global) |
| `reliability_by_dataset.csv` | Reliability broken down by dataset |
| `reliability_by_subtype.csv` | Reliability broken down by probe subtype |
| `reliability_by_tier.csv` | Reliability broken down by tier |
| `transform_quality_comparison.csv` | Transform quality across subtypes |
| `pipeline_overview.csv` | High-level pipeline counts |
| `paper_table_ready.csv` | Camera-ready table for publication |
| `judge_decisions_detail.csv` | Every judge decision with confidence & failures |
| `judge_disagreement_analysis.csv` | Probes where judges disagreed |
| `peer_review_agreement.csv` | Inter-judge agreement rate by subtype |
| `disagreement_criteria_frequency.csv` | Most common failure reasons |
| `dataset_characteristics.csv` | Dataset statistics |

---

## Adding a New Model

Edit `config.py` → `EXPERIMENTAL_MODELS`:

```python
{
    "name":       "my-model",          # unique label used in output files
    "provider":   "openai",            # openai | anthropic | groq | mistral | ollama
    "model_id":   "gpt-4o",            # provider-specific model string
    "max_tokens": 2048,
},
```

For Ollama models, also add an entry to `OLLAMA_MODEL_CONFIGS` in `step4_run_and_score.py` to tune `num_ctx`, `num_predict`, and worker concurrency for your GPU.

---

## VRAM Layout (H100 80 GB)

| Model | Quantisation | VRAM | Workers (parallel) |
|---|---|---|---|
| llama3.1:70b | q4_K_M | ~40 GB | 6 |
| gemma3:27b | q4_K_M | ~17 GB | 6 |
| **Total** | | **~70 GB** | — |

Both models run simultaneously. Expected wall time: ~90 min for 10,814 probes.
