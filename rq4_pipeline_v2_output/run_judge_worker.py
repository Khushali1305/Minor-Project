#!/usr/bin/env python3
"""
Judge Worker — single API key, single shard, resumable.
Supports multiple workers on the same shard via --key-id and --num-keys.
Each worker takes every Nth probe (round-robin), writes to its own temp file.
Merge all worker files after completion.

Single worker (default):
  MISTRAL_API_KEY=key1 python run_judge_worker.py \
      --probecore share1/simp_probecore_v1.jsonl \
      --out-dir   out1/simp_judge_outputs \
      --judge-name mistral-large-3 \
      --provider   mistral \
      --model-id   mistral-large-latest \
      --delay      2.5

3 workers on same shard (3 terminals, 3 different API keys):
  MISTRAL_API_KEY=key1 python run_judge_worker.py \
      --probecore share1/simp_probecore_v1.jsonl \
      --out-dir   out1/simp_judge_outputs \
      --judge-name mistral-large-3 --provider mistral \
      --model-id mistral-large-latest \
      --delay 2.5 --key-id 0 --num-keys 3

  MISTRAL_API_KEY=key2 python run_judge_worker.py \
      --probecore share1/simp_probecore_v1.jsonl \
      --out-dir   out1/simp_judge_outputs \
      --judge-name mistral-large-3 --provider mistral \
      --model-id mistral-large-latest \
      --delay 2.5 --key-id 1 --num-keys 3

  MISTRAL_API_KEY=key3 python run_judge_worker.py \
      --probecore share1/simp_probecore_v1.jsonl \
      --out-dir   out1/simp_judge_outputs \
      --judge-name mistral-large-3 --provider mistral \
      --model-id mistral-large-latest \
      --delay 2.5 --key-id 2 --num-keys 3

After all workers done, merge:
  python run_judge_worker.py --merge \
      --out-dir out1/simp_judge_outputs \
      --judge-name mistral-large-3 \
      --num-keys 3
"""
import json, re, os, time, argparse, threading


def load_jsonl(p):
    if not os.path.exists(p):
        return []
    with open(p, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(d, p, lock):
    with lock:
        with open(p, 'a', encoding='utf-8') as f:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
            f.flush()


def call_with_retry(prompt, system, provider, model_id, max_tokens, delay):
    from api_client import call_model
    time.sleep(delay)
    for attempt in range(8):
        try:
            return call_model(prompt, system, provider, model_id, max_tokens)
        except Exception as e:
            err = str(e)
            is_rate_limit = (
                '429' in err or
                'rate_limit' in err.lower() or
                'rate limit' in err.lower()
            )
            if is_rate_limit and attempt < 7:
                wait = 3 * (2 ** attempt)  # 3,6,12,24,48,96,192s
                print(f"  [429] attempt {attempt+1}/8, sleeping {wait}s...", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Max retries exceeded for {model_id}")


def do_merge(args):
    """Merge all worker temp files into one final decisions file."""
    all_recs = []
    seen_ids = set()
    for kid in range(args.num_keys):
        fpath = f"{args.out_dir}/{args.judge_name}_decisions_worker{kid}.jsonl"
        if not os.path.exists(fpath):
            print(f"  WARNING: missing worker file {fpath}", flush=True)
            continue
        recs = load_jsonl(fpath)
        # Deduplicate in case of any overlap
        for r in recs:
            if r['probe_id'] not in seen_ids:
                all_recs.append(r)
                seen_ids.add(r['probe_id'])
        print(f"  worker{kid}: {len(recs)} records loaded", flush=True)

    out_path = f"{args.out_dir}/{args.judge_name}_decisions.jsonl"
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in all_recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"\nMerged {len(all_recs)} total records → {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser(
        description="Single-judge worker: one shard, one API key, fully resumable."
    )
    p.add_argument('--probecore',  default=None,
                   help='Path to shard probecore JSONL')
    p.add_argument('--out-dir',    required=True,
                   help='Output directory for judge decisions JSONL')
    p.add_argument('--judge-name', required=True,
                   help='Judge name used as output filename prefix')
    p.add_argument('--provider',   default=None,
                   help='API provider: mistral | anthropic | openai | groq')
    p.add_argument('--model-id',   default=None,
                   help='Model ID string passed to the provider API')
    p.add_argument('--max-tokens', type=int, default=150,
                   help='Max tokens for judge response (default: 150)')
    p.add_argument('--delay',      type=float, default=2.5,
                   help=(
                       'Seconds to sleep before each call. '
                       'Recommended: 2.5 for Mistral free tier, '
                       '1.5 for Anthropic paid tier. Default: 2.5'
                   ))
    p.add_argument('--key-id',     type=int, default=0,
                   help=(
                       'Worker index when running multiple keys on one shard. '
                       'Each worker takes probes where index %% num_keys == key_id. '
                       'Default: 0 (single worker, takes all probes)'
                   ))
    p.add_argument('--num-keys',   type=int, default=1,
                   help=(
                       'Total number of parallel workers on this shard. '
                       'Default: 1 (single worker). '
                       'Set to 3 for 3 keys splitting the shard 3 ways.'
                   ))
    p.add_argument('--merge',      action='store_true',
                   help=(
                       'Merge all worker temp files into one final decisions file. '
                       'Run after all workers complete. '
                       'Requires --out-dir, --judge-name, --num-keys.'
                   ))
    a = p.parse_args()

    # ── Merge mode ────────────────────────────────────────────────────────────
    if a.merge:
        print(f"Merging {a.num_keys} worker files for {a.judge_name}...", flush=True)
        do_merge(a)
        return

    # ── Validate required args for run mode ───────────────────────────────────
    if not a.probecore:
        print("ERROR: --probecore required in run mode", flush=True)
        return
    if not a.provider or not a.model_id:
        print("ERROR: --provider and --model-id required in run mode", flush=True)
        return

    from config import JUDGE_SYSTEM, JUDGE_USER

    # Load probes — only invariance family
    all_probes = load_jsonl(a.probecore)
    inv_probes = [pr for pr in all_probes if pr.get('probe_family') == 'invariance']

    # Split probes across workers via round-robin on original order
    # Worker 0 takes probes 0,3,6,9...
    # Worker 1 takes probes 1,4,7,10...
    # Worker 2 takes probes 2,5,8,11...
    my_probes = [pr for i, pr in enumerate(inv_probes) if i % a.num_keys == a.key_id]

    print(f"\nShard:             {a.probecore}", flush=True)
    print(f"Judge:             {a.judge_name} ({a.provider} / {a.model_id})", flush=True)
    print(f"Worker:            {a.key_id} of {a.num_keys} "
          f"(my probes: {len(my_probes)}/{len(inv_probes)})", flush=True)
    print(f"Delay per call:    {a.delay}s (~{60/a.delay:.0f} req/min max)", flush=True)

    # Each worker writes to its own temp file — no race condition
    os.makedirs(a.out_dir, exist_ok=True)
    out_path = (
        f"{a.out_dir}/{a.judge_name}_decisions_worker{a.key_id}.jsonl"
        if a.num_keys > 1
        else f"{a.out_dir}/{a.judge_name}_decisions.jsonl"
    )

    # Resume — skip already completed probe_ids in this worker's file
    done    = set(r['probe_id'] for r in load_jsonl(out_path))
    pending = [pr for pr in my_probes if pr['probe_id'] not in done]
    print(f"Done: {len(done)} | Pending: {len(pending)}", flush=True)

    if not pending:
        print("Nothing to do — worker complete!", flush=True)
        if a.num_keys > 1:
            print(
                f"Run merge when all workers done:\n"
                f"  python run_judge_worker.py --merge "
                f"--out-dir {a.out_dir} "
                f"--judge-name {a.judge_name} "
                f"--num-keys {a.num_keys}",
                flush=True
            )
        return

    est_min = len(pending) * a.delay / 60
    print(f"ETA (no 429s):     {est_min:.0f} min", flush=True)
    print(f"Output:            {out_path}\n", flush=True)

    write_lock = threading.Lock()
    errors     = 0
    completed  = 0
    start_time = time.time()

    for idx, pm in enumerate(pending):
        pid    = pm['probe_id']
        prompt = JUDGE_USER.format(
            source=pm.get('base_text', '')[:500],
            probe=pm.get('probe_text', '')[:500]
        )
        try:
            resp = call_with_retry(
                prompt, JUDGE_SYSTEM,
                a.provider, a.model_id, a.max_tokens, a.delay
            )
            resp = re.sub(r'^```json\s*', '', resp.strip())
            resp = re.sub(r'\s*```$', '', resp)
            v    = json.loads(resp)
            rec  = {
                'probe_id':           pid,
                'judge':              a.judge_name,
                'equivalent':         v.get('equivalent', False),
                'confidence':         v.get('confidence', 'low'),
                'failures':           v.get('failures', []),
                'reason':             v.get('reason', ''),
                'probe_subtype':      pm.get('probe_subtype', ''),
                'generator_model':    pm.get('generator_model', ''),
                'tier':               pm.get('tier', 1),
                'dataset':            pm.get('dataset', ''),
                'base_text_preview':  pm.get('base_text', '')[:200],
                'probe_text_preview': pm.get('probe_text', '')[:200],
            }
        except json.JSONDecodeError as e:
            errors += 1
            rec = {
                'probe_id': pid, 'judge': a.judge_name,
                'equivalent': None, 'confidence': 'parse_error',
                'failures': ['JSON_PARSE'], 'reason': str(e)[:200]
            }
        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  ERR {pid}: {e}", flush=True)
            rec = {
                'probe_id': pid, 'judge': a.judge_name,
                'equivalent': None, 'confidence': 'error',
                'failures': ['API_ERROR'], 'reason': str(e)[:200]
            }

        append_jsonl(rec, out_path, write_lock)
        completed += 1

        if completed % 50 == 0:
            elapsed = time.time() - start_time
            rate    = completed / elapsed
            eta_min = (len(pending) - completed) / rate / 60
            pct     = completed / len(pending) * 100
            print(
                f"  [{completed}/{len(pending)}] {pct:.1f}% | "
                f"ETA: {eta_min:.0f} min | "
                f"rate: {rate*60:.1f} req/min | "
                f"errors={errors}",
                flush=True
            )

    elapsed_total = (time.time() - start_time) / 60
    print(
        f"\nDone! completed={completed} errors={errors} "
        f"time={elapsed_total:.1f}min",
        flush=True
    )
    print(f"Output: {out_path}", flush=True)

    if a.num_keys > 1:
        print(
            f"\nRun merge when ALL {a.num_keys} workers are done:\n"
            f"  python run_judge_worker.py --merge "
            f"--out-dir {a.out_dir} "
            f"--judge-name {a.judge_name} "
            f"--num-keys {a.num_keys}",
            flush=True
        )


if __name__ == '__main__':
    main()