
#!/usr/bin/env python3
"""
Combine shard1 + shard2 + shard3 outputs and run phase_results.

What this does:
  1. Merges simp_scores/*_rule_scores.jsonl from all 3 shards
  2. Merges simp_judge_outputs/*_decisions.jsonl from all 3 shards
  3. Builds judge_agreement.jsonl from merged decisions
  4. Runs phase_results on combined data

Usage:
  python combine_and_run_results.py \
      --shard-out-dirs results/out/shard1 results/out/shard2 results/out/shard3 \
      --combined-dir   results/out/combined \
      --probecore      results/shard1/simp_probecore_v1.jsonl \
      --judge-names    mistral-large-3 claude-judge

  Then phase_results runs automatically, or manually:
  python step4_run_and_score1.py --phase results \
      --input-dir  results/out/combined \
      --output-dir results/out/combined
"""
import json, os, argparse
from collections import defaultdict

def load_jsonl(p):
    if not os.path.exists(p): return []
    with open(p, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]

def save_jsonl(data, p):
    with open(p, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

def save_csv(rows, path, fieldnames=None):
    import csv
    if not rows: return
    if not fieldnames: fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

# ── Step 1: Merge rule scores ─────────────────────────────────────────────────
def merge_rule_scores(shard_dirs, combined_scores_dir):
    os.makedirs(combined_scores_dir, exist_ok=True)
    by_model = defaultdict(list)

    for shard_dir in shard_dirs:
        scores_dir = f"{shard_dir}/simp_scores"
        if not os.path.exists(scores_dir):
            print(f"  WARNING: {scores_dir} not found")
            continue
        for fname in os.listdir(scores_dir):
            if not fname.endswith('_rule_scores.jsonl'):
                continue
            model = fname.replace('_rule_scores.jsonl', '')
            recs  = load_jsonl(f"{scores_dir}/{fname}")
            by_model[model].extend(recs)
            print(f"  {shard_dir.split('/')[-1]} / {model}: {len(recs)} rule scores")

    for model, recs in by_model.items():
        # Deduplicate by probe_id
        seen = set()
        deduped = []
        for r in recs:
            pid = r.get('probe_id', '')
            if pid not in seen:
                deduped.append(r)
                seen.add(pid)
        out = f"{combined_scores_dir}/{model}_rule_scores.jsonl"
        save_jsonl(deduped, out)
        print(f"  → {model}: {len(deduped)} combined rule scores")

    return by_model.keys()

# ── Step 2: Merge judge decisions ─────────────────────────────────────────────
def merge_judge_decisions(shard_dirs, combined_judge_dir, judge_names):
    os.makedirs(combined_judge_dir, exist_ok=True)

    for jname in judge_names:
        all_recs = []
        seen     = set()

        for shard_dir in shard_dirs:
            judge_dir = f"{shard_dir}/simp_judge_outputs"
            # Try merged file first, then worker files
            candidates = [
                f"{judge_dir}/{jname}_decisions.jsonl",
            ]
            # Also pick up any worker temp files not yet merged
            if os.path.exists(judge_dir):
                for fname in os.listdir(judge_dir):
                    if fname.startswith(f"{jname}_decisions_worker"):
                        candidates.append(f"{judge_dir}/{fname}")

            for fpath in candidates:
                if not os.path.exists(fpath):
                    continue
                recs = load_jsonl(fpath)
                for r in recs:
                    pid = r.get('probe_id', '')
                    if pid and pid not in seen:
                        all_recs.append(r)
                        seen.add(pid)
                print(f"  {shard_dir.split('/')[-1]} / {jname}: {len(recs)} from {os.path.basename(fpath)}")

        out = f"{combined_judge_dir}/{jname}_decisions.jsonl"
        save_jsonl(all_recs, out)
        equiv_t = sum(1 for r in all_recs if r.get('equivalent') is True)
        equiv_f = sum(1 for r in all_recs if r.get('equivalent') is False)
        errors  = sum(1 for r in all_recs if r.get('equivalent') is None)
        print(f"  → {jname}: {len(all_recs)} total | true={equiv_t} false={equiv_f} errors={errors}")

# ── Step 3: Build judge_agreement.jsonl ───────────────────────────────────────
def build_agreement(combined_judge_dir, judge_names):
    all_decisions = defaultdict(dict)
    for jname in judge_names:
        path = f"{combined_judge_dir}/{jname}_decisions.jsonl"
        for d in load_jsonl(path):
            all_decisions[d['probe_id']][jname] = d

    agree_recs    = []
    disagree_recs = []

    for pid, judges in all_decisions.items():
        jlist = list(judges.values())
        if len(jlist) < 2:
            # Single judge — include with agreed=True
            j = jlist[0]
            if j.get('equivalent') is None:
                continue
            agree_recs.append({
                'probe_id': pid,
                'j1_name':  j['judge'], 'j1_equiv': j.get('equivalent'),
                'j2_name':  '',          'j2_equiv': None,
                'agreed':   True,
            })
            continue

        j1, j2 = jlist[0], jlist[1]
        if j1.get('equivalent') is None or j2.get('equivalent') is None:
            continue

        agreed = (j1.get('equivalent') == j2.get('equivalent'))
        rec = {
            'probe_id': pid,
            'j1_name':  j1['judge'], 'j1_equiv': j1.get('equivalent'),
            'j2_name':  j2['judge'], 'j2_equiv': j2.get('equivalent'),
            'agreed':   agreed,
        }
        agree_recs.append(rec)
        if not agreed:
            disagree_recs.append(rec)

    save_jsonl(agree_recs,    f"{combined_judge_dir}/judge_agreement.jsonl")
    save_jsonl(disagree_recs, f"{combined_judge_dir}/judge_disagreements.jsonl")

    total   = len(agree_recs)
    n_agree = sum(1 for r in agree_recs if r['agreed'])
    print(f"  → Agreement: {n_agree}/{total} = {n_agree/total:.1%}" if total else "  → No agreement data")
    print(f"  → Disagreements: {len(disagree_recs)}")

# ── Step 4: Symlink/copy probecore so phase_results can find it ───────────────
def setup_combined_input(combined_dir, probecore_path):
    """phase_results reads probecore from input_dir/simp_probecore_v1.jsonl"""
    target = f"{combined_dir}/simp_probecore_v1.jsonl"
    if not os.path.exists(target):
        import shutil
        shutil.copy(probecore_path, target)
        print(f"  Copied probecore → {target}")
    else:
        print(f"  Probecore already exists at {target}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shard-out-dirs', nargs='+', required=True,
                   help='Output dirs e.g. results/out/shard1 results/out/shard2 results/out/shard3')
    p.add_argument('--combined-dir',   required=True,
                   help='Dir for combined output e.g. results/out/combined')
    p.add_argument('--probecore',      required=True,
                   help='Path to FULL simp_probecore_v1.jsonl (not sharded)')
    p.add_argument('--judge-names',    nargs='+', required=True,
                   help='Judge names e.g. mistral-large-3 claude-judge')
    p.add_argument('--run-results',    action='store_true',
                   help='Auto-run phase_results after combining')
    a = p.parse_args()

    combined_scores_dir = f"{a.combined_dir}/simp_scores"
    combined_judge_dir  = f"{a.combined_dir}/simp_judge_outputs"
    os.makedirs(a.combined_dir, exist_ok=True)

    print("\n=== Step 1: Merging rule scores ===")
    merge_rule_scores(a.shard_out_dirs, combined_scores_dir)

    print("\n=== Step 2: Merging judge decisions ===")
    merge_judge_decisions(a.shard_out_dirs, combined_judge_dir, a.judge_names)

    print("\n=== Step 3: Building judge_agreement.jsonl ===")
    build_agreement(combined_judge_dir, a.judge_names)

    print("\n=== Step 4: Setting up probecore ===")
    setup_combined_input(a.combined_dir, a.probecore)

    print(f"""
=== DONE — Combined data ready in {a.combined_dir} ===

Now run phase_results:
  python step4_run_and_score1.py --phase results \\
      --input-dir  {a.combined_dir} \\
      --output-dir {a.combined_dir}
""")

    if a.run_results:
        print("=== Auto-running phase_results ===")
        import sys
        sys.argv = [
            'step4_run_and_score1.py',
            '--phase',      'results',
            '--input-dir',  a.combined_dir,
            '--output-dir', a.combined_dir,
        ]
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "step4", "step4_run_and_score1.py")
        mod = importlib.util.load_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()

if __name__ == '__main__':
    main()
