#!/usr/bin/env python3
"""
Merge all judge decision files from shard1/2/3 into one final file per judge.
Handles both worker temp files (_worker0/1/2.jsonl) and direct files (_decisions.jsonl).
Run this after all workers complete.

Usage:
  python merge_judge_outputs.py \
      --base-dir /path/to/results/out \
      --final-dir /path/to/final_judge_outputs
"""
import json, os, argparse
from collections import defaultdict


def load_jsonl(p):
    if not os.path.exists(p):
        return []
    with open(p, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def find_judge_files(shard_dir):
    """Find all judge decision files in a shard's judge output dir."""
    judge_out = os.path.join(shard_dir, 'simp_judge_outputs')
    if not os.path.exists(judge_out):
        return {}
    files_by_judge = defaultdict(list)
    for fname in sorted(os.listdir(judge_out)):
        if not fname.endswith('.jsonl'):
            continue
        fpath = os.path.join(judge_out, fname)
        # Match: judge-name_decisions.jsonl or judge-name_decisions_workerN.jsonl
        if '_decisions' in fname:
            # Extract judge name (everything before _decisions)
            judge_name = fname.split('_decisions')[0]
            files_by_judge[judge_name].append(fpath)
    return files_by_judge


def main():
    p = argparse.ArgumentParser(
        description="Merge all shard judge outputs into one file per judge."
    )
    p.add_argument('--base-dir',  required=True,
                   help='Base dir containing shard1/, shard2/, shard3/ subdirs')
    p.add_argument('--final-dir', required=True,
                   help='Output dir for merged final decision files')
    p.add_argument('--shards', default='shard1,shard2,shard3',
                   help='Comma-separated shard names (default: shard1,shard2,shard3)')
    a = p.parse_args()

    os.makedirs(a.final_dir, exist_ok=True)
    shards = a.shards.split(',')

    # Collect all files grouped by judge name
    all_files_by_judge = defaultdict(list)
    print(f"\nScanning shards: {shards}")
    for shard in shards:
        shard_dir = os.path.join(a.base_dir, shard.strip())
        if not os.path.exists(shard_dir):
            print(f"  WARNING: {shard_dir} not found — skipping")
            continue
        by_judge = find_judge_files(shard_dir)
        if not by_judge:
            print(f"  WARNING: no judge files found in {shard_dir}/simp_judge_outputs")
            continue
        for judge_name, files in by_judge.items():
            all_files_by_judge[judge_name].extend(files)
            for f in files:
                lines = sum(1 for _ in open(f))
                print(f"  {shard}/{judge_name}: {os.path.basename(f)} ({lines} records)")

    if not all_files_by_judge:
        print("\nNo judge files found. Check --base-dir path.")
        return

    # Merge per judge
    print(f"\n{'='*60}")
    print("Merging...")
    for judge_name, files in sorted(all_files_by_judge.items()):
        all_recs = []
        seen_ids = set()
        dup_count = 0

        for fpath in files:
            recs = load_jsonl(fpath)
            for r in recs:
                pid = r.get('probe_id', '')
                if pid and pid not in seen_ids:
                    all_recs.append(r)
                    seen_ids.add(pid)
                elif pid:
                    dup_count += 1

        out_path = os.path.join(a.final_dir, f"{judge_name}_decisions.jsonl")
        with open(out_path, 'w', encoding='utf-8') as f:
            for r in all_recs:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

        # Stats
        equiv_true  = sum(1 for r in all_recs if r.get('equivalent') is True)
        equiv_false = sum(1 for r in all_recs if r.get('equivalent') is False)
        errors      = sum(1 for r in all_recs if r.get('equivalent') is None)

        print(f"\n  {judge_name}:")
        print(f"    Total records:  {len(all_recs)}")
        print(f"    equivalent=true:  {equiv_true} ({equiv_true/len(all_recs)*100:.1f}%)")
        print(f"    equivalent=false: {equiv_false} ({equiv_false/len(all_recs)*100:.1f}%)")
        print(f"    errors/null:      {errors}")
        if dup_count:
            print(f"    duplicates removed: {dup_count}")
        print(f"    → {out_path}")

    # Final check — how many unique probe_ids across all judges
    print(f"\n{'='*60}")
    print(f"Final files in {a.final_dir}:")
    for fname in sorted(os.listdir(a.final_dir)):
        fpath = os.path.join(a.final_dir, fname)
        lines = sum(1 for _ in open(fpath))
        print(f"  {fname}: {lines} records")
    print("\nDone! Now run --phase results pointing to this final-dir as your judge output dir.")


if __name__ == '__main__':
    main()