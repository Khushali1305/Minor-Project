"""
diagnose.py
===========
Run this on your Lightning.ai machine BEFORE running step1.
It tells you exactly what is installed, what version, and what needs fixing.

Usage:
    python3 diagnose.py
"""

import sys
import os
import subprocess

print("=" * 60)
print("LIGHTNING.AI ENVIRONMENT DIAGNOSIS")
print("=" * 60)
print(f"Python: {sys.version}")
print()

# ── 1. Check transformers version and model classes ───────────────────────────
print("── transformers ─────────────────────────────────────────────")
try:
    import transformers
    v = transformers.__version__
    print(f"Version: {v}")

    # Parse version to check if it meets minimums
    parts = v.split(".")
    major, minor = int(parts[0]), int(parts[1])

    if major == 4 and minor < 37:
        print(f"PROBLEM: Version {v} is too old.")
        print(f"  T5ForConditionalGeneration needs >= 4.0 (OK)")
        print(f"  Qwen2ForCausalLM needs >= 4.37")
        print(f"  GemmaForCausalLM needs >= 4.38")
        print(f"  FIX: pip install 'transformers>=4.40.0' --upgrade")
    elif major == 4 and minor < 40:
        print(f"WARNING: Version {v} may be missing some model classes.")
        print(f"  FIX: pip install 'transformers>=4.40.0' --upgrade")
    else:
        print(f"Version OK (>= 4.40)")

    # Check specific classes
    classes = [
        "T5ForConditionalGeneration",
        "Qwen2ForCausalLM",
        "LlamaForCausalLM",
        "GemmaForCausalLM",
        "Gemma3ForCausalLM",
    ]
    print()
    for cls in classes:
        try:
            getattr(transformers, cls)
            print(f"  {cls}: OK")
        except AttributeError:
            print(f"  {cls}: MISSING — transformers too old")

except ImportError:
    print("NOT INSTALLED — run: pip install 'transformers>=4.40.0'")

print()

# ── 2. Check torch ────────────────────────────────────────────────────────────
print("── torch ────────────────────────────────────────────────────")
try:
    import torch
    print(f"Version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram:.1f} GB")
    else:
        print("WARNING: No GPU — models will run very slowly on CPU")
except ImportError:
    print("NOT INSTALLED — run: pip install torch")

print()

# ── 3. Check accelerate and bitsandbytes ─────────────────────────────────────
print("── accelerate / bitsandbytes ────────────────────────────────")
for pkg in ["accelerate", "bitsandbytes"]:
    try:
        mod = __import__(pkg)
        print(f"{pkg}: {mod.__version__}")
    except ImportError:
        print(f"{pkg}: NOT INSTALLED — run: pip install {pkg}")

print()

# ── 4. Check sentencepiece (needed for some tokenizers) ──────────────────────
print("── sentencepiece / tokenizers ───────────────────────────────")
for pkg in ["sentencepiece", "tokenizers"]:
    try:
        mod = __import__(pkg)
        v = getattr(mod, "__version__", "installed")
        print(f"{pkg}: {v}")
    except ImportError:
        print(f"{pkg}: NOT INSTALLED — run: pip install {pkg}")

print()

# ── 5. Check HuggingFace token ───────────────────────────────────────────────
print("── HuggingFace token ────────────────────────────────────────")
hf_token = os.environ.get("HF_TOKEN", "")
hub_token = os.environ.get("HUGGING_FACE_HUB_TOKEN", "")

if hf_token:
    print(f"HF_TOKEN in env: YES (length={len(hf_token)})")
else:
    print("HF_TOKEN in env: NOT SET")
    print("  FIX: export HF_TOKEN=your_token")

if hub_token:
    print(f"HUGGING_FACE_HUB_TOKEN in env: YES")
else:
    print("HUGGING_FACE_HUB_TOKEN in env: NOT SET")

# Check if token is cached on disk
from pathlib import Path
token_paths = [
    Path.home() / ".cache" / "huggingface" / "token",
    Path.home() / ".huggingface" / "token",
]
for tp in token_paths:
    if tp.exists():
        print(f"Token cached on disk: YES ({tp})")
        break
else:
    print("Token cached on disk: NO")

# Verify token works with a live API call
if hf_token:
    try:
        from huggingface_hub import HfApi
        api  = HfApi(token=hf_token)
        user = api.whoami()
        print(f"Token valid: YES — logged in as '{user['name']}'")

        # Check access to gated models
        print()
        print("  Checking gated model access:")
        gated = {
            "meta-llama/Llama-3.1-8B-Instruct" : "llama_8b",
            "meta-llama/Llama-3.1-70B-Instruct": "llama_70b",
            "google/gemma-3-27b-it"             : "gemma_27b",
        }
        for repo_id, key in gated.items():
            try:
                info = api.model_info(repo_id, token=hf_token)
                print(f"  {key} ({repo_id}): ACCESS OK")
            except Exception as e:
                err = str(e)
                if "401" in err or "403" in err or "gated" in err.lower():
                    print(f"  {key} ({repo_id}): ACCESS DENIED")
                    print(f"    Go to https://huggingface.co/{repo_id}")
                    print(f"    Click 'Agree and access repository' to accept licence")
                else:
                    print(f"  {key} ({repo_id}): ERROR — {err[:80]}")
    except ImportError:
        print("huggingface_hub not installed — run: pip install huggingface_hub")
    except Exception as e:
        print(f"Token INVALID or expired: {e}")
        print("  FIX: Get a new token from https://huggingface.co/settings/tokens")
        print("       Then: export HF_TOKEN=your_new_token")

print()

# ── 6. Check OpenAI ──────────────────────────────────────────────────────────
print("── OpenAI ───────────────────────────────────────────────────")
openai_key = os.environ.get("OPENAI_API_KEY", "")
if openai_key:
    print(f"OPENAI_API_KEY in env: YES (length={len(openai_key)})")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        client.models.list()
        print("OPENAI_API_KEY valid: YES")
    except Exception as e:
        print(f"OPENAI_API_KEY valid: NO — {e}")
else:
    print("OPENAI_API_KEY: NOT SET — run: export OPENAI_API_KEY=your_key")

print()

# ── 7. Check pip upgrade path ────────────────────────────────────────────────
print("── What needs to be installed / upgraded ────────────────────")
needs = []

try:
    import transformers
    v = transformers.__version__
    parts = v.split(".")
    if int(parts[0]) == 4 and int(parts[1]) < 40:
        needs.append("pip install 'transformers>=4.40.0' --upgrade")
except ImportError:
    needs.append("pip install 'transformers>=4.40.0'")

for pkg in ["accelerate", "bitsandbytes", "sentencepiece"]:
    try:
        __import__(pkg)
    except ImportError:
        needs.append(f"pip install {pkg}")

try:
    import openai
except ImportError:
    needs.append("pip install openai")

try:
    import anthropic
except ImportError:
    needs.append("pip install anthropic")

try:
    import bert_score
except ImportError:
    needs.append("pip install bert-score")

if needs:
    print("Run these commands:")
    for cmd in needs:
        print(f"  {cmd}")
else:
    print("All packages OK. No installs needed.")

print()
print("=" * 60)
print("Diagnosis complete.")
print("=" * 60)
