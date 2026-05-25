#!/usr/bin/env python3
"""
Priority 2 Fix — Semantic Probe Balance
========================================
Adds missing semantic probes so all 7 models have equal coverage.

Current state (from probecore analysis):
  flan-t5-large   2035  (baseline — match this)
  flan-t5-xl      2034  (gap: +1)
  gpt-4.1         1159  (gap: +876)
  llama-3.1-70b    864  (gap: +1171)
  qwen2.5-32b      841  (gap: +1194)
  gemma-3-27b      549  (gap: +1486)
  llama-3.1-8b     397  (gap: +1638)
  ────────────────────────────────────
  Total new probes needed: 6,366

How it works:
  1. Loads existing probecore → finds which base items each model already covered
  2. Loads simp_master_table.jsonl → gets remaining base items per model
  3. Calls each model (Ollama or API) to paraphrase each base text
  4. Assigns tier (1=self-contained, 2=context-dependent) via heuristic
  5. Creates probe records with patch_ prefixed IDs
  6. Backs up probecore, then appends new probes

Machine: H100 (Ollama models run in parallel threads)
Time   : ~45 min
Cost   : ~$2.50 (H100 GPU + GPT-4.1 API)

Usage:
  python patch_semantic_probes.py \\
    --probecore output/simp_probecore_v1.jsonl \\
    --master    output/simp_master_table.jsonl \\
    --ollama-url http://localhost:11434
"""
import json
import os
import re
import time
import argparse
import threading
import shutil
import uuid
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


# ── Config ────────────────────────────────────────────────────────────────────

TARGET_COUNT = 2035   # match flan-t5-large (highest)

# Model routing — must match your config.py providers
MODEL_ROUTES = {
    'flan-t5-xl':    {'provider': 'ollama', 'model_id': 'flan-t5-xl',                'workers': 16},
    'llama-3.1-8b':  {'provider': 'ollama', 'model_id': 'llama3.1:8b-instruct-q8_0', 'workers': 16},
    'llama-3.1-70b': {'provider': 'ollama', 'model_id': 'llama3.1:70b',              'workers': 6},
    'qwen2.5-32b':   {'provider': 'ollama', 'model_id': 'qwen2.5:32b',               'workers': 8},
    'gemma-3-27b':   {'provider': 'ollama', 'model_id': 'gemma3:27b',                'workers': 6},
    'gpt-4.1':       {'provider': 'openai', 'model_id': 'gpt-4.1',                   'workers': 2},
}

# Paraphrase prompt (mirrors config.py PARAPHRASE_*)
PARA_SYSTEM = (
    "You are a paraphrasing engine. Rewrite the input using different "
    "words and sentence structure while preserving EXACTLY the same meaning "
    "and reading difficulty. Output ONLY the rewritten text, nothing else."
)
PARA_USER = (
    "Rewrite using different words and structure. "
    "Keep exact same meaning.\n\nTEXT: {text}\n\nREWRITTEN:"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(p):
    if not os.path.exists(p):
        return []
    with open(p, 'r', encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(d, p):
    with open(p, 'a', encoding='utf-8') as f:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')
        f.flush()


_file_locks = defaultdict(threading.Lock)


def safe_append(d, p):
    """Thread-safe JSONL append."""
    with _file_locks[p]:
        append_jsonl(d, p)


def assign_tier(text):
    """
    Tier 1 — self-contained: explicit subject, no anaphoric references.
    Tier 2 — context-dependent: starts with pronoun or has anaphoric signals.
    """
    t = text.lower().strip()

    # Starts with pronoun → depends on prior context
    if re.match(r'^(he|she|it|they|his|her|its|their|this|these|those|such)\b', t):
        return 2

    # Contains anaphoric phrases mid-sentence
    anaphoric = [
        'this was', 'this has', 'this is', 'these are', 'those are',
        'it has been', 'it was found', 'they have', 'they were',
        'the former', 'the latter', 'such as this',
    ]
    if any(a in t for a in anaphoric):
        return 2

    return 1


def make_probe_id(model_name, base_item_id):
    short = str(uuid.uuid4())[:8]
    safe  = re.sub(r'[^a-zA-Z0-9]', '_', model_name)
    return f"patch_sem_{safe}_{base_item_id}_{short}"


# ── Paraphrase API calls ──────────────────────────────────────────────────────

def ollama_paraphrase(model_id, text, base_url, timeout=120):
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": PARA_SYSTEM},
            {"role": "user",   "content": PARA_USER.format(text=text)},
        ],
        "stream":     False,
        "keep_alive": "2h",
        "options": {
            "num_ctx":     1024,
            "num_predict": 300,
            "temperature": 0.3,   # slight variation for diversity
            "top_p":       0.9,
        },
    }
    r = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "").strip()


def openai_paraphrase(model_id, text):
    import openai
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": PARA_SYSTEM},
            {"role": "user",   "content": PARA_USER.format(text=text)},
        ],
        max_tokens=300,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# ── Ollama lifecycle helpers ──────────────────────────────────────────────────

def prewarm_ollama(model_id, base_url):
    try:
        requests.post(
            f"{base_url}/api/chat",
            json={
                "model":      model_id,
                "messages":   [{"role": "user", "content": "hi"}],
                "stream":     False,
                "keep_alive": "2h",
                "options":    {"num_ctx": 512, "num_predict": 1},
            },
            timeout=120,
        ).raise_for_status()
        print(f"  [Ollama] {model_id} warmed ✓", flush=True)
    except Exception as e:
        print(f"  [Ollama] pre-warm warning {model_id}: {e}", flush=True)


def unload_ollama(model_id, base_url):
    try:
        requests.post(
            f"{base_url}/api/chat",
            json={"model": model_id, "messages": [], "keep_alive": 0, "stream": False},
            timeout=30,
        )
        print(f"  [Ollama] unloaded {model_id}", flush=True)
    except Exception:
        pass


# ── Generate probes for one model ─────────────────────────────────────────────

def generate_for_model(model_name, items_to_cover, route, base_url,
                       probecore_path, save_dir):
    """
    Generates semantic probes for one model.
    Runs in its own thread — thread-safe writes via safe_append.
    """
    provider = route['provider']
    model_id = route['model_id']
    workers  = route['workers']
    out_path = os.path.join(save_dir, f"patch_sem_{model_name}.jsonl")

    # Resume — skip already completed items
    done_bases = set()
    if os.path.exists(out_path):
        for r in load_jsonl(out_path):
            done_bases.add(r.get('base_item_id', ''))
    pending = [it for it in items_to_cover if it['item_id'] not in done_bases]

    if not pending:
        print(f"  [{model_name}] all done, skipping", flush=True)
        return 0

    print(
        f"\n  [{model_name}] {len(pending)} probes to generate "
        f"(provider={provider}, workers={workers})",
        flush=True,
    )

    if provider == 'ollama':
        prewarm_ollama(model_id, base_url)

    saved  = 0
    errors = 0
    t0     = time.time()

    def call_one(item):
        text = item['original']
        if provider == 'ollama':
            para = ollama_paraphrase(model_id, text, base_url)
        elif provider == 'openai':
            para = openai_paraphrase(model_id, text)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        return item, para

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(call_one, it): it for it in pending}

        for i, fut in enumerate(as_completed(futures)):
            item = futures[fut]
            try:
                item_, para = fut.result()
                if not para or len(para.split()) < 3:
                    errors += 1
                    continue

                probe = {
                    "probe_id":        make_probe_id(model_name, item['item_id']),
                    "probe_family":    "invariance",
                    "probe_subtype":   "semantic",
                    "generator_model": model_name,
                    "base_item_id":    item['item_id'],
                    "base_text":       item['original'],
                    "probe_text":      para,
                    "target_level":    item.get('target_level', 'simpler'),
                    "dataset":         item['dataset'],
                    "tier":            assign_tier(item['original']),
                    "source":          "patch_v1",
                }
                # Write to per-model temp file, then append to shared probecore
                safe_append(probe, out_path)
                safe_append(probe, probecore_path)
                saved += 1

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"    [{model_name}] ERR {item['item_id']}: {e}", flush=True)

            if (i + 1) % 200 == 0 or (i + 1) == len(pending):
                elapsed = time.time() - t0
                rate    = (i + 1) / elapsed if elapsed > 0 else 0
                eta     = (len(pending) - (i + 1)) / rate if rate > 0 else 0
                print(
                    f"    [{model_name}] {i+1}/{len(pending)} | "
                    f"{rate:.1f}/sec | ETA {eta/60:.0f}min | "
                    f"saved={saved} errors={errors}",
                    flush=True,
                )

    if provider == 'ollama':
        unload_ollama(model_id, base_url)

    elapsed = time.time() - t0
    print(
        f"  [{model_name}] DONE {saved} probes in {elapsed/60:.1f}min | "
        f"errors={errors}",
        flush=True,
    )
    return saved


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Patch semantic probe balance — fill gaps so all models reach TARGET_COUNT."
    )
    p.add_argument('--probecore',  default='output/simp_probecore_v1.jsonl',
                   help='Path to the probecore JSONL file (will be appended to)')
    p.add_argument('--master',     default='output/simp_master_table.jsonl',
                   help='Path to the master table JSONL file')
    p.add_argument('--ollama-url', default='http://localhost:11434',
                   help='Base URL for the local Ollama server')
    p.add_argument('--target',     type=int, default=TARGET_COUNT,
                   help=f'Target probe count per model (default: {TARGET_COUNT})')
    p.add_argument('--models',     default=None,
                   help='Comma-separated model names to patch. Default: all with gaps.')
    p.add_argument('--save-dir',   default='output/patch_temp',
                   help='Temp directory for per-model patch files')
    a = p.parse_args()

    os.makedirs(a.save_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────
    print("Loading probecore...", flush=True)
    probes = load_jsonl(a.probecore)
    print(f"  {len(probes):,} existing probes", flush=True)

    print("Loading master table...", flush=True)
    items = load_jsonl(a.master)
    print(f"  {len(items):,} base items", flush=True)

    # Index master table: item_id → item
    item_map = {it['item_id']: it for it in items}

    # ── Compute coverage gaps ──────────────────────────────────────
    sem_probes = [p for p in probes if p.get('probe_subtype') == 'semantic']
    coverage   = defaultdict(set)   # model → set of base_item_ids already covered
    for p in sem_probes:
        coverage[p['generator_model']].add(p['base_item_id'])

    # All base items eligible for semantic probes (ASSET + TSAR, not OSE)
    eligible_ids = [
        it['item_id'] for it in items
        if it.get('dataset') in ('ASSET', 'TSAR2025')
    ]

    print(f"\nEligible base items for semantic probes: {len(eligible_ids):,}", flush=True)
    print(f"Target per model: {a.target}\n", flush=True)

    print(f"  {'Model':<22} {'Have':>6}  {'Need':>6}  {'Gap':>6}")
    print('  ' + '-' * 43)

    models_to_patch  = {}
    filter_names     = set(a.models.split(',')) if a.models else None

    for model_name, route in MODEL_ROUTES.items():
        if filter_names and model_name not in filter_names:
            continue
        have = len(coverage[model_name])
        gap  = max(0, a.target - have)
        print(f"  {model_name:<22} {have:>6}  {a.target:>6}  {gap:>+6}")
        if gap > 0:
            # Pick uncovered items for this model
            uncovered = [iid for iid in eligible_ids if iid not in coverage[model_name]]
            selected  = uncovered[:gap]
            models_to_patch[model_name] = {
                'route': route,
                'items': [item_map[iid] for iid in selected if iid in item_map],
            }

    if not models_to_patch:
        print("\n✅ All models already at target. Nothing to do.", flush=True)
        return

    total_new = sum(len(v['items']) for v in models_to_patch.values())
    print(f"\nTotal new probes to generate: {total_new:,}", flush=True)

    # ── Backup probecore before any writes ────────────────────────
    backup_path = a.probecore + '.bak'
    shutil.copy2(a.probecore, backup_path)
    print(f"\n✅ Probecore backed up → {backup_path}", flush=True)

    # ── Split Ollama vs API models ────────────────────────────────
    ollama_models = {
        k: v for k, v in models_to_patch.items()
        if v['route']['provider'] == 'ollama'
    }
    api_models = {
        k: v for k, v in models_to_patch.items()
        if v['route']['provider'] != 'ollama'
    }

    t_start = time.time()

    # ── API models run in a background daemon thread ──────────────
    api_thread = None
    if api_models:
        def run_api():
            for mn, cfg in api_models.items():
                generate_for_model(
                    mn, cfg['items'], cfg['route'],
                    a.ollama_url, a.probecore, a.save_dir,
                )

        api_thread = threading.Thread(target=run_api, daemon=True, name='api')
        api_thread.start()
        print(f"\n[API] Background thread started for: {list(api_models.keys())}", flush=True)

    # ── Ollama models run in parallel threads ─────────────────────
    # ── WITH THIS (sequential — safe) ────────────────────────
    if ollama_models:
        n = len(ollama_models)
        # Sort: smallest model first so big ones don't block small ones
        ordered = sorted(
            ollama_models.items(),
            key=lambda x: x[1]['route'].get('workers', 8)  # proxy for size
        )
        print(f"\n[Ollama] Running {n} models SEQUENTIALLY (VRAM safety)...", flush=True)
        for mn, cfg in ordered:
            print(f"\n{'='*50}", flush=True)
            print(f"  Starting: {mn}", flush=True)
            print(f"{'='*50}", flush=True)
            generate_for_model(
                mn, cfg['items'], cfg['route'],
                a.ollama_url, a.probecore, a.save_dir
            )
            # Force unload before next model loads
            unload_ollama(cfg['route']['model_id'], a.ollama_url)
            print(f"  Waiting 15s for VRAM to clear...", flush=True)
            time.sleep(15)
        print("\n[Ollama] All Ollama models complete.", flush=True)

    if api_thread:
        print("[API] Waiting for API thread...", flush=True)
        api_thread.join()

    # ── Verify final distribution ─────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"  VERIFICATION — NEW PROBECORE DISTRIBUTION", flush=True)
    print(f"{'='*60}", flush=True)

    new_probes = load_jsonl(a.probecore)
    sem_new    = [p for p in new_probes if p.get('probe_subtype') == 'semantic']
    gen_new    = Counter(p.get('generator_model', '?') for p in sem_new)

    print(f"\n  Total probes  : {len(new_probes):,}  (was {len(probes):,})")
    print(f"  New added     : {len(new_probes) - len(probes):,}\n")
    print(f"  {'Model':<22} {'Count':>7}  Status")
    print(f"  {'-'*45}")

    for mn in sorted(gen_new, key=lambda x: -gen_new[x]):
        count  = gen_new[mn]
        status = '✅' if count >= a.target else f'⚠️  (gap={a.target - count})'
        print(f"  {mn:<22} {count:>7}  {status}")

    # Family distribution
    fam_new = Counter(p.get('probe_family') for p in new_probes)
    print(f"\n  Family distribution:")
    for k, v in fam_new.most_common():
        pct = v / len(new_probes) * 100
        print(f"    {k:<20} {v:>6}  ({pct:.1f}%)")

    elapsed = time.time() - t_start
    print(f"\n  ✅ Patch complete in {elapsed / 60:.1f} min", flush=True)
    print(f"  Backup at: {backup_path}", flush=True)
    print(f"\n  Next step: re-run step4 --phase run  (generate outputs for new probes)", flush=True)
    print(f"  Then     : step4 --phase score        (semantic probes now judge-only)", flush=True)


if __name__ == '__main__':
    main()