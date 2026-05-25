#!/usr/bin/env python3
"""
Runs a single model on a specific chunk of probes.
Copy this + the chunk file to each studio.

Usage:
  python run_chunk.py \
    --chunk-file  output/splits/probes_chunk_1_of_2.jsonl \
    --master      output/simp_master_table.jsonl \
    --model-name  llama-3.1-70b \
    --out-file    output/llama-3.1-70b_chunk1.jsonl
"""
import json, os, time, argparse, requests
from config import EXPERIMENTAL_MODELS, SIMPLIFY_SYSTEM, SIMPLIFY_USER_CEFR, SIMPLIFY_USER_GENERIC

def load_jsonl(p):
    if not os.path.exists(p): return []
    with open(p,'r',encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]

def append_jsonl(d, p):
    with open(p,'a',encoding='utf-8') as f:
        f.write(json.dumps(d,ensure_ascii=False)+'\n'); f.flush()

def get_prompt(text, target):
    if target in ('a2','b1'):
        return SIMPLIFY_USER_CEFR.format(
            target={'a2':'A2 (elementary)','b1':'B1 (intermediate)'}[target],
            text=text)
    return SIMPLIFY_USER_GENERIC.format(text=text)

p = argparse.ArgumentParser()
p.add_argument('--chunk-file',  required=True)
p.add_argument('--model-name',  required=True)
p.add_argument('--out-file',    required=True)
p.add_argument('--ollama-url',  default='http://localhost:11434')
a = p.parse_args()

# Find model config
mcfg = next((m for m in EXPERIMENTAL_MODELS if m['name'] == a.model_name), None)
if not mcfg:
    raise ValueError(f"Model {a.model_name} not found in config")

# Load chunk probes
probes = load_jsonl(a.chunk_file)

# Resume — skip already done
done = set()
if os.path.exists(a.out_file):
    for r in load_jsonl(a.out_file):
        done.add(r['item_id'])
pending = [p for p in probes if p['probe_id'] not in done]
print(f"Chunk: {len(probes)} total, {len(pending)} pending, {len(done)} done")

from api_client import call_model
errors = 0
for idx, probe in enumerate(pending):
    try:
        output = call_model(
            get_prompt(probe['probe_text'], probe.get('target_level','simpler')),
            SIMPLIFY_SYSTEM,
            mcfg['provider'],
            mcfg['model_id'],
            mcfg.get('max_tokens', 2048)
        ).strip()
        rec = {
            'item_id':       probe['probe_id'],
            'model':         a.model_name,
            'output':        output,
            'output_words':  len(output.split()),
            'input_words':   len(probe['probe_text'].split()),
            'type':          'probe',
            'target':        probe.get('target_level','simpler'),
            'dataset':       probe.get('dataset',''),
            'probe_family':  probe.get('probe_family',''),
            'base_item_id':  probe.get('base_item_id',''),
        }
        append_jsonl(rec, a.out_file)
    except Exception as e:
        errors += 1
        if errors <= 5: print(f"  ERR {probe['probe_id']}: {e}", flush=True)
    if (idx+1) % 100 == 0:
        print(f"  [{idx+1}/{len(pending)}] errors={errors}", flush=True)

print(f"Done. saved={len(pending)-errors} errors={errors}")
