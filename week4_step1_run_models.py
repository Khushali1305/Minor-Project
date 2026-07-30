"""
week4_step1_run_models.py
=========================
Generates rewrites for all models on every base item and probe item
in SRS-ProbeCore-v5-frozen.json.

MODELS UNDER EVALUATION
------------------------
  rule_baseline : deterministic EARS restructurer (keyword-pattern, no ML)
  flan_t5_large : google/flan-t5-large
  flan_t5_xl    : google/flan-t5-xl
  llama_8b      : meta-llama/Llama-3.1-8B-Instruct
  llama_70b     : meta-llama/Llama-3.1-70B-Instruct
  qwen_32b      : Qwen/Qwen2.5-32B-Instruct
  gemma_27b     : google/gemma-3-27b-it
  gpt4          : gpt-4.1  (OpenAI API)

BUGS FIXED IN THIS VERSION
---------------------------
  1. FLAN-T5 / Qwen fail — torch_dtype deprecated -> use dtype instead
  2. Llama / Gemma 401 Unauthorized — HF_TOKEN not passed to HF env variable
     Fixed: os.environ["HUGGING_FACE_HUB_TOKEN"] set before any HF call
  3. GPT-4.1 restarts from scratch on resume — save was only after full model
     Fixed: checkpoint save every CHECKPOINT_EVERY records during GPT-4.1 run
     Resume now skips individual completed jobs, not just whole models

CHECKPOINT BEHAVIOUR
--------------------
  For HuggingFace models  : saves after each full model (fast, batch inference)
  For GPT-4.1             : saves every 50 jobs during the run
  On resume               : skips any job already in file with run_status=ok

USAGE
-----
  # First run
  python3 week4_step1_run_models.py

  # Resume after GPU session timeout — safe to run repeatedly
  python3 week4_step1_run_models.py --resume

  # Run only specific models
  python3 week4_step1_run_models.py --models gpt4

  # Smoke test — 20 jobs per model
  python3 week4_step1_run_models.py --smoke --models rule_baseline,gpt4

ENVIRONMENT VARIABLES
---------------------
  export OPENAI_API_KEY=your_key      # for gpt4
  export HF_TOKEN=your_hf_token       # for Llama and Gemma (gated models)

DEPENDENCIES
------------
  pip install --upgrade openai transformers torch accelerate bitsandbytes
"""

import json
import os
import re
import time
import argparse
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "input_benchmark"   : "/teamspace/studios/this_studio/parse/SRS-ProbeCore-v5-frozen.json",
    "output_file"       : "/teamspace/studios/this_studio/parse/week4_model_outputs.json",

    "hf_models": {
        "flan_t5_large" : "google/flan-t5-large",
        "flan_t5_xl"    : "google/flan-t5-xl",
        "llama_8b"      : "meta-llama/Llama-3.1-8B-Instruct",
        "llama_70b"     : "meta-llama/Llama-3.1-70B-Instruct",
        "qwen_32b"      : "Qwen/Qwen2.5-32B-Instruct",
        "gemma_27b"     : "google/gemma-3-27b-it",
    },

    "openai_model"      : "gpt-4.1",
    "openai_api_key"    : os.environ.get("OPENAI_API_KEY", ""),

    # HF_TOKEN — also set into os.environ so HuggingFace Hub library picks it up
    "hf_token"          : os.environ.get("HF_TOKEN", ""),

    "max_new_tokens"    : 256,
    "temperature"       : 0.0,
    "batch_size"        : 8,
    "max_items"         : None,
    "device"            : "auto",

    # Save to disk every N completed records during GPT-4.1 run.
    # Lower = safer but slightly slower. 50 means at most 50 API calls are
    # lost if the session dies mid-run.
    "checkpoint_every"  : 50,
}


# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

EARS_SYSTEM_PROMPT = """You are a requirements engineering assistant.
Your task is to rewrite a free-text software requirement into a structured
EARS (Easy Approach to Requirements Syntax) template form.

EARS templates use:
  Ubiquitous  : The <system> shall <action>.
  Event-driven: WHEN <trigger>, the <system> shall <action>.
  Unwanted    : IF <condition>, the <system> shall <action>.
  State-driven: WHILE <state>, the <system> shall <action>.
  Optional    : WHERE <feature>, the <system> shall <action>.

Rules you must follow:
1. Preserve the original obligation strength (shall/should/may/must).
   Do NOT upgrade "may" or "should" to "shall" unless the original says so.
2. Preserve all conditions, exceptions, and actors from the original.
3. Do NOT invent new behavior, add new clauses, or remove existing ones.
4. Output ONLY the rewritten requirement. No explanation, no commentary.
"""

EARS_USER_PROMPT = """Rewrite the following requirement into EARS format:

Requirement: {requirement}

EARS rewrite:"""


# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED BASELINE
# ─────────────────────────────────────────────────────────────────────────────

class RuleBasedEARSBaseline:
    MODAL_MAP = {
        "shall": "SHALL", "must": "SHALL",
        "should": "SHOULD", "will": "SHALL",
        "may": "MAY", "can": "MAY",
    }
    TRIGGER_PATTERNS = [
        (r"\bif\b",          "IF",   "UnwantedBehavior"),
        (r"\bwhen(ever)?\b", "WHEN", "EventDriven"),
        (r"\bwhile\b",       "WHILE","StateDriven"),
        (r"\bonce\b",        "WHEN", "EventDriven"),
        (r"\bafter\b",       "WHEN", "EventDriven"),
    ]

    def rewrite(self, requirement):
        req = requirement.strip().rstrip(".")

        modal_out = "SHALL"
        for modal_in, token in self.MODAL_MAP.items():
            if re.search(r"\b" + modal_in + r"\b", req, re.IGNORECASE):
                modal_out = token
                break

        trigger_keyword = None
        for pattern, keyword, _ in self.TRIGGER_PATTERNS:
            if re.search(pattern, req, re.IGNORECASE):
                trigger_keyword = keyword
                break

        subject_match = re.match(
            r"^(.*?)\s+(?:shall|must|should|will|may|can)\b",
            req, re.IGNORECASE
        )
        subject = subject_match.group(1).strip() if subject_match else ""
        if not subject or len(subject.split()) > 8:
            subject = "the system"

        action_match = re.search(
            r"\b(?:shall|must|should|will|may|can)\b\s+(.*)",
            req, re.IGNORECASE
        )
        action = action_match.group(1).strip() if action_match else req

        if trigger_keyword:
            cond_match = re.search(
                r"\b(?:if|when|whenever|while|once|after)\b(.+?)"
                r"(?:,\s*|\s+(?:the\s+)?(?:\w+\s+)?(?:shall|must|should|will|may|can)\b)",
                req, re.IGNORECASE
            )
            condition = cond_match.group(1).strip() if cond_match else ""
            if condition:
                return (
                    f"{trigger_keyword} {condition}, {subject} "
                    f"{modal_out} {action}."
                )

        return f"{subject.capitalize()} {modal_out} {action}."


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace model runner
# Fix 1: torch_dtype -> dtype (transformers deprecation)
# Fix 2: os.environ set for HF token so HuggingFace Hub library sees it
# ─────────────────────────────────────────────────────────────────────────────

def _set_hf_token_env(hf_token):
    """
    Set HF token in all environment variables that HuggingFace libraries check.
    This is required because some versions of the HF Hub library read the token
    from the environment directly rather than from the token= argument.
    """
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        os.environ["HF_TOKEN"]               = hf_token
        logging.info("  HF_TOKEN set in environment.")
    else:
        logging.warning(
            "  HF_TOKEN is empty. Llama and Gemma (gated models) will fail with 401. "
            "Set it with: export HF_TOKEN=your_token"
        )


def load_hf_model(model_id, model_key, config):
    """
    Load a HuggingFace model and tokenizer.
    Returns (model, tokenizer, is_seq2seq).

    Fixes applied:
    - torch_dtype kwarg deprecated in new transformers -> use dtype instead
    - HF token set in environment before any Hub call
    - FLAN-T5 loaded without float16 (seq2seq models are small, no quantization needed)
    - Large causal models use 4-bit quantization (fits 70B on single A100)
    """
    try:
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForSeq2SeqLM,
            AutoModelForCausalLM, BitsAndBytesConfig,
        )
    except ImportError:
        raise ImportError("pip install --upgrade transformers torch accelerate bitsandbytes")

    hf_token   = config["hf_token"] or None
    is_seq2seq = "t5" in model_id.lower()

    # Critical: set env var so HF Hub library uses it for gated model auth
    _set_hf_token_env(hf_token)

    logging.info(f"Loading {model_key} ({model_id}) ...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Decoder-only (causal) models require left-padding for correct batched
    # generation. Without this, the model sees padding on the right which
    # corrupts attention alignment and produces wrong/empty outputs.
    if not is_seq2seq:
        tokenizer.padding_side = "left"

    load_kwargs = dict(
        pretrained_model_name_or_path=model_id,
        token=hf_token,
        trust_remote_code=True,
    )

    large_models = {"llama_70b", "qwen_32b", "gemma_27b"}

    if is_seq2seq:
        # FLAN-T5: small seq2seq model, no quantization, load in float32
        # dtype kwarg replaces deprecated torch_dtype in new transformers
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
            load_kwargs["dtype"]      = torch.float32
        else:
            load_kwargs["device_map"] = "cpu"

    elif model_key in large_models and torch.cuda.is_available():
        # Large causal LMs: 4-bit quantization to fit on single GPU
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        load_kwargs["quantization_config"] = bnb
        load_kwargs["device_map"]          = "auto"

    elif torch.cuda.is_available():
        # Small causal LMs (llama_8b): float16 on GPU
        # dtype kwarg replaces deprecated torch_dtype
        load_kwargs["device_map"] = "auto"
        load_kwargs["dtype"]      = torch.float16

    else:
        load_kwargs["device_map"] = "cpu"

    ModelClass = AutoModelForSeq2SeqLM if is_seq2seq else AutoModelForCausalLM
    model      = ModelClass.from_pretrained(**load_kwargs)
    model.eval()
    logging.info(f"  {model_key} loaded.")
    return model, tokenizer, is_seq2seq


# Large models (32B / 70B / 27B): batching them together is slow and produces
# OOM errors. Use batch_size=1 so each item goes through independently.
_LARGE_MODEL_KEYS = {"llama_70b", "qwen_32b", "gemma_27b"}


def run_hf_model(model, tokenizer, is_seq2seq, requirements, config,
                 model_key=""):
    """
    Run batched inference on a HuggingFace model.

    Changes vs. original:
    - batch_size forced to 1 for large models (32B/70B/27B) — batching them
      causes silent multi-hour hangs with no output.
    - Per-batch timing + progress log so you can see it's working.
    - Device detection fixed for 4-bit quantized models (they report 'cpu'
      from next(parameters()) even when running on GPU via bitsandbytes).
    """
    import torch

    # Large models: batch_size=1 is actually faster end-to-end because
    # batched runs with right-length-variance cause huge padding waste.
    if model_key in _LARGE_MODEL_KEYS:
        batch_size = 1
        logging.info(
            f"  [{model_key}] Using batch_size=1 (large model — batching causes hangs)"
        )
    else:
        batch_size = config["batch_size"]

    # Device detection: 4-bit quantized models wrap params in bnb and
    # report 'cpu' from next(parameters()). Use cuda:0 explicitly when
    # CUDA is available and the first param is on cpu (quantized case).
    first_param_device = next(model.parameters()).device
    if str(first_param_device) == "cpu" and torch.cuda.is_available():
        target_device = torch.device("cuda:0")
        logging.info(
            f"  [{model_key}] 4-bit quantized model detected — "
            f"inputs will be sent to cuda:0"
        )
    else:
        target_device = first_param_device

    results   = []
    n_batches = (len(requirements) + batch_size - 1) // batch_size
    run_start = time.time()

    for batch_idx, i in enumerate(range(0, len(requirements), batch_size)):
        batch      = requirements[i : i + batch_size]
        batch_start = time.time()

        if is_seq2seq:
            prompts = [
                EARS_USER_PROMPT.format(requirement=r).strip() for r in batch
            ]
        else:
            prompts = []
            for r in batch:
                messages = [
                    {"role": "system", "content": EARS_SYSTEM_PROMPT.strip()},
                    {"role": "user",   "content": EARS_USER_PROMPT.format(requirement=r).strip()},
                ]
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    prompt = (
                        f"<|system|>{EARS_SYSTEM_PROMPT.strip()}<|end|>\n"
                        f"<|user|>{EARS_USER_PROMPT.format(requirement=r).strip()}<|end|>\n"
                        f"<|assistant|>"
                    )
                prompts.append(prompt)

        inputs = tokenizer(
            prompts, return_tensors="pt",
            padding=True, truncation=True, max_length=512,
        )
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config["max_new_tokens"],
                do_sample=False,
                # temperature must NOT be set when do_sample=False — passing
                # temperature=None can trigger a warning/error in some
                # transformers versions; omitting it is the correct approach.
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, output in enumerate(outputs):
            if is_seq2seq:
                decoded = tokenizer.decode(output, skip_special_tokens=True)
            else:
                input_len = inputs["input_ids"][j].shape[0]
                decoded   = tokenizer.decode(
                    output[input_len:], skip_special_tokens=True
                )
            results.append(decoded.strip())

        # ── Progress log — critical so you know it hasn't hung ────────────────
        elapsed      = time.time() - batch_start
        total_elapsed = time.time() - run_start
        done         = batch_idx + 1
        remaining    = n_batches - done
        eta_sec      = (total_elapsed / done) * remaining if done > 0 else 0
        logging.info(
            f"  [{model_key}] batch {done}/{n_batches} "
            f"({i + len(batch)}/{len(requirements)} reqs) | "
            f"{elapsed:.1f}s/batch | ETA {eta_sec/60:.1f} min"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI runner — GPT-4.1
# Fix 3: checkpoint save every N jobs so GPU session timeout does not lose
#         all progress. Resume skips individual completed jobs.
# ─────────────────────────────────────────────────────────────────────────────

def run_openai_model_with_checkpoint(
    pending_jobs, existing, all_records, config, timestamp
):
    """
    Run GPT-4.1 on pending_jobs with mid-run checkpoint saves.

    Key difference from the old run_openai_model():
    - Processes one job at a time (not batched) so we can save after each one
    - Saves to disk every config["checkpoint_every"] completed jobs
    - On next resume, already-saved jobs are skipped via the existing index
    - This means the A100 1-hour timeout loses at most checkpoint_every calls

    Returns (saved_ok, saved_empty) counts.
    """
    try:
        from openai import OpenAI, RateLimitError
    except ImportError:
        raise ImportError("pip install openai")

    client          = OpenAI(api_key=config["openai_api_key"])
    checkpoint_every = config["checkpoint_every"]
    saved_ok        = 0
    saved_empty     = 0
    completed_since_save = 0

    for i, job in enumerate(pending_jobs):
        req         = job["input_text"]
        rewrite     = ""
        attempt     = 0
        max_retries = 5
        backoff     = 2.0

        while attempt <= max_retries:
            try:
                response = client.chat.completions.create(
                    model=config["openai_model"],
                    messages=[
                        {"role": "system", "content": EARS_SYSTEM_PROMPT.strip()},
                        {"role": "user",   "content": EARS_USER_PROMPT.format(requirement=req).strip()},
                    ],
                    temperature=0.0,
                    max_tokens=config["max_new_tokens"],
                )
                rewrite = response.choices[0].message.content.strip()
                break

            except RateLimitError as e:
                attempt += 1
                if attempt > max_retries:
                    logging.warning(
                        f"GPT-4.1 rate limit — max retries exceeded on job {i}: {e}"
                    )
                else:
                    wait = backoff ** attempt
                    logging.info(
                        f"  Rate limited — retrying job {i} in {wait:.0f}s "
                        f"(attempt {attempt}/{max_retries}) ..."
                    )
                    time.sleep(wait)

            except Exception as e:
                logging.warning(f"GPT-4.1 error on job {i}: {e}")
                break

        is_empty   = not rewrite.strip()
        run_status = "empty" if is_empty else "ok"

        if is_empty:
            saved_empty += 1
            logging.warning(
                f"  Empty rewrite — {job['item_id']} | "
                f"probe_id={job.get('probe_id')}"
            )
        else:
            saved_ok += 1

        record = {
            "item_id"      : job["item_id"],
            "input_type"   : job["input_type"],
            "probe_id"     : job["probe_id"],
            "probe_family" : job["probe_family"],
            "input_text"   : job["input_text"],
            "target_norm"  : job["target_norm"],
            "model_id"     : "gpt4",
            "rewrite"      : rewrite,
            "run_status"   : run_status,
            "run_timestamp": timestamp,
        }
        all_records.append(record)
        if run_status == "ok":
            existing[(job["item_id"], job["probe_id"], "gpt4")] = record

        completed_since_save += 1

        # Mid-run checkpoint save — critical for A100 1-hour session limit
        if completed_since_save >= checkpoint_every:
            with open(config["output_file"], "w") as f:
                json.dump(all_records, f, indent=2)
            logging.info(
                f"  Checkpoint saved — job {i+1}/{len(pending_jobs)} | "
                f"ok so far: {saved_ok} | empty: {saved_empty}"
            )
            completed_since_save = 0

    # Final save after all jobs complete
    with open(config["output_file"], "w") as f:
        json.dump(all_records, f, indent=2)

    return saved_ok, saved_empty


# ─────────────────────────────────────────────────────────────────────────────
# Build flat job list
# ─────────────────────────────────────────────────────────────────────────────

def build_jobs(data, max_items=None):
    jobs = []
    for item in data:
        iid  = item["item_id"]
        norm = item.get("target_norm", "EARS")
        base = item["requirement_text"]

        jobs.append({
            "item_id"     : iid,
            "input_type"  : "base",
            "probe_id"    : None,
            "probe_family": None,
            "input_text"  : base,
            "target_norm" : norm,
        })

        for probe in item.get("probe_neighborhoods", []):
            jobs.append({
                "item_id"     : iid,
                "input_type"  : "probe",
                "probe_id"    : probe.get("probe_id"),
                "probe_family": probe.get("probe_family"),
                "input_text"  : probe.get("probe_text", ""),
                "target_norm" : norm,
            })

    if max_items:
        jobs = jobs[:max_items]
    return jobs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default=(
            "rule_baseline,flan_t5_large,flan_t5_xl,"
            "llama_8b,llama_70b,qwen_32b,gemma_27b,gpt4"
        ),
        help="Comma-separated model keys to run",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Resume a partial run. Skips any job already saved with "
            "run_status=ok. Safe to run repeatedly."
        ),
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test — first 20 jobs per model only",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args          = parse_args()
    models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.smoke:
        CONFIG["max_items"] = 20
        logging.info("Smoke test — max 20 jobs per model")

    with open(CONFIG["input_benchmark"]) as f:
        data = json.load(f)
    logging.info(f"Benchmark: {len(data)} items")

    jobs = build_jobs(data, CONFIG["max_items"])
    logging.info(f"Jobs per model: {len(jobs)}")

    # Resume: load only run_status=ok records as done
    # Empty/error records are retried automatically
    existing    = {}   # (item_id, probe_id, model_id) -> record
    all_records = []

    if Path(CONFIG["output_file"]).exists():
        with open(CONFIG["output_file"]) as f:
            try:
                saved = json.load(f)
            except json.JSONDecodeError:
                logging.error(
                    "week4_model_outputs.json is corrupted (JSONDecodeError). "
                    "Cannot resume. Delete the file and start fresh."
                )
                return

        all_records = saved
        for rec in saved:
            if rec.get("run_status") == "ok" and rec.get("rewrite", "").strip():
                key = (rec["item_id"], rec.get("probe_id"), rec["model_id"])
                existing[key] = rec

        if args.resume:
            logging.info(
                f"Resume: {len(existing)} ok records loaded, "
                f"{len(saved) - len(existing)} empty/error will be retried"
            )
        else:
            logging.info(
                f"Output file found with {len(existing)} ok records. "
                f"Use --resume to continue, or delete the file to start fresh."
            )
            # If not --resume, still respect existing ok records
            # to avoid re-running models that already completed

    rule_baseline = RuleBasedEARSBaseline()

    for model_key in models_to_run:
        logging.info(f"\n{'='*55}")
        logging.info(f"Model: {model_key}")
        logging.info(f"{'='*55}")

        pending = [
            j for j in jobs
            if (j["item_id"], j["probe_id"], model_key) not in existing
        ]

        if not pending:
            logging.info(f"  All {len(jobs)} jobs done for {model_key}. Skipping.")
            continue

        logging.info(f"  Pending: {len(pending)} / {len(jobs)}")
        timestamp = datetime.now(timezone.utc).isoformat()

        # ── Rule baseline ─────────────────────────────────────────────────────
        # Deterministic + fast, but we still checkpoint for consistency so
        # --resume works correctly after any interruption.
        if model_key == "rule_baseline":
            checkpoint_every     = CONFIG["checkpoint_every"]
            saved_ok             = 0
            saved_empty          = 0
            completed_since_save = 0

            for item_idx, job in enumerate(pending):
                rewrite    = rule_baseline.rewrite(job["input_text"])
                is_empty   = not rewrite.strip()
                run_status = "empty" if is_empty else "ok"
                saved_empty += is_empty
                saved_ok    += not is_empty
                record = {
                    "item_id"      : job["item_id"],
                    "input_type"   : job["input_type"],
                    "probe_id"     : job["probe_id"],
                    "probe_family" : job["probe_family"],
                    "input_text"   : job["input_text"],
                    "target_norm"  : job["target_norm"],
                    "model_id"     : model_key,
                    "rewrite"      : rewrite,
                    "run_status"   : run_status,
                    "run_timestamp": timestamp,
                }
                all_records.append(record)
                if run_status == "ok":
                    existing[(job["item_id"], job["probe_id"], model_key)] = record

                completed_since_save += 1
                if completed_since_save >= checkpoint_every:
                    with open(CONFIG["output_file"], "w") as f:
                        json.dump(all_records, f, indent=2)
                    logging.info(
                        f"  [rule_baseline] Checkpoint saved — "
                        f"item {item_idx + 1}/{len(pending)} | "
                        f"ok: {saved_ok} | empty: {saved_empty}"
                    )
                    completed_since_save = 0

            with open(CONFIG["output_file"], "w") as f:
                json.dump(all_records, f, indent=2)
            logging.info(
                f"  [rule_baseline] Done — ok: {saved_ok} | empty: {saved_empty} | "
                f"total in file: {len(all_records)}"
            )

        # ── GPT-4.1 — checkpoint save every N jobs ────────────────────────────
        elif model_key == "gpt4":
            if not CONFIG["openai_api_key"]:
                logging.error(
                    "OPENAI_API_KEY not set. "
                    "Run: export OPENAI_API_KEY=your_key"
                )
                continue
            logging.info(
                f"  GPT-4.1: checkpoint save every {CONFIG['checkpoint_every']} jobs. "
                f"Resume is safe if session times out."
            )
            saved_ok, saved_empty = run_openai_model_with_checkpoint(
                pending, existing, all_records, CONFIG, timestamp
            )
            logging.info(
                f"  GPT-4.1 complete — ok: {saved_ok} | empty: {saved_empty} | "
                f"total in file: {len(all_records)}"
            )

        # ── HuggingFace models ────────────────────────────────────────────────
        # Per-item checkpoint saves (same pattern as GPT-4.1) so a Lightning AI
        # session timeout does not lose all progress. Use --resume to continue.
        elif model_key in CONFIG["hf_models"]:
            hf_id = CONFIG["hf_models"][model_key]
            try:
                model, tokenizer, is_seq2seq = load_hf_model(
                    hf_id, model_key, CONFIG
                )
            except Exception as e:
                logging.error(f"Failed to load {model_key}: {e}")
                continue

            import torch

            checkpoint_every     = CONFIG["checkpoint_every"]
            saved_ok             = 0
            saved_empty          = 0
            completed_since_save = 0

            logging.info(
                f"  [{model_key}] checkpoint save every {checkpoint_every} items. "
                f"Resume is safe if session times out."
            )

            # Determine target device once (handles 4-bit quantized models)
            first_param_device = next(model.parameters()).device
            if str(first_param_device) == "cpu" and torch.cuda.is_available():
                target_device = torch.device("cuda:0")
            else:
                target_device = first_param_device

            for item_idx, job in enumerate(pending):
                # ── Build prompt ──────────────────────────────────────────────
                req = job["input_text"]
                if is_seq2seq:
                    prompt_text = EARS_USER_PROMPT.format(requirement=req).strip()
                else:
                    messages = [
                        {"role": "system", "content": EARS_SYSTEM_PROMPT.strip()},
                        {"role": "user",   "content": EARS_USER_PROMPT.format(requirement=req).strip()},
                    ]
                    try:
                        prompt_text = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    except Exception:
                        prompt_text = (
                            f"<|system|>{EARS_SYSTEM_PROMPT.strip()}<|end|>\n"
                            f"<|user|>{EARS_USER_PROMPT.format(requirement=req).strip()}<|end|>\n"
                            f"<|assistant|>"
                        )

                # ── Tokenize & generate ───────────────────────────────────────
                rewrite = ""
                try:
                    inputs = tokenizer(
                        [prompt_text], return_tensors="pt",
                        padding=True, truncation=True, max_length=512,
                    )
                    inputs = {k: v.to(target_device) for k, v in inputs.items()}

                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=CONFIG["max_new_tokens"],
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                        )

                    if is_seq2seq:
                        rewrite = tokenizer.decode(
                            output_ids[0], skip_special_tokens=True
                        ).strip()
                    else:
                        input_len = inputs["input_ids"][0].shape[0]
                        rewrite   = tokenizer.decode(
                            output_ids[0][input_len:], skip_special_tokens=True
                        ).strip()

                except Exception as e:
                    logging.warning(
                        f"  [{model_key}] Generation error on item {item_idx} "
                        f"({job['item_id']}): {e}"
                    )

                # ── Record result ─────────────────────────────────────────────
                is_empty   = not rewrite.strip()
                run_status = "empty" if is_empty else "ok"
                saved_empty += is_empty
                saved_ok    += not is_empty

                if is_empty:
                    logging.warning(
                        f"  [{model_key}] Empty — {job['item_id']} | "
                        f"probe_id={job.get('probe_id')}"
                    )

                record = {
                    "item_id"      : job["item_id"],
                    "input_type"   : job["input_type"],
                    "probe_id"     : job["probe_id"],
                    "probe_family" : job["probe_family"],
                    "input_text"   : job["input_text"],
                    "target_norm"  : job["target_norm"],
                    "model_id"     : model_key,
                    "rewrite"      : rewrite,
                    "run_status"   : run_status,
                    "run_timestamp": timestamp,
                }
                all_records.append(record)
                if run_status == "ok":
                    existing[(job["item_id"], job["probe_id"], model_key)] = record

                completed_since_save += 1

                # ── Mid-run checkpoint ────────────────────────────────────────
                if completed_since_save >= checkpoint_every:
                    with open(CONFIG["output_file"], "w") as f:
                        json.dump(all_records, f, indent=2)
                    logging.info(
                        f"  [{model_key}] Checkpoint saved — "
                        f"item {item_idx + 1}/{len(pending)} | "
                        f"ok: {saved_ok} | empty: {saved_empty}"
                    )
                    completed_since_save = 0

            # Final save after all items for this model
            del model
            torch.cuda.empty_cache()

            with open(CONFIG["output_file"], "w") as f:
                json.dump(all_records, f, indent=2)
            logging.info(
                f"  [{model_key}] Done — ok: {saved_ok} | empty: {saved_empty} | "
                f"total in file: {len(all_records)}"
            )

        else:
            valid = ["rule_baseline", "gpt4"] + list(CONFIG["hf_models"].keys())
            logging.warning(f"Unknown model key '{model_key}'. Valid: {valid}")
            continue

    # ── Final summary ─────────────────────────────────────────────────────────
    logging.info("\nAll models complete.")
    logging.info(f"Output: {CONFIG['output_file']} | Total: {len(all_records)}")

    empties = [r for r in all_records if r.get("run_status") == "empty"]
    if empties:
        by_model = dict(Counter([r["model_id"] for r in empties]))
        logging.warning(f"Empty records by model: {by_model}")
        logging.warning("Re-run with --resume to retry these.")
    else:
        logging.info("All rewrites successful — no empty records.")


if __name__ == "__main__":
    main()
