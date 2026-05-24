#!/usr/bin/env python3
"""Step 3: Generate audit sample → manual review → freeze Simp-ProbeCore v1
Usage:
  python step3_audit.py --input-dir ./output --output-dir ./output --mode sample   # generates CSV
  python step3_audit.py --input-dir ./output --output-dir ./output --mode freeze   # after manual audit
"""
import json, csv, os, random, argparse
from collections import Counter, defaultdict
from config import AUDIT_SAMPLE_SIZE

random.seed(42)

def load_jsonl(p):
    with open(p) as f: return [json.loads(l) for l in f if l.strip()]

def stratified_sample(probes, n):
    groups = defaultdict(list)
    for p in probes: groups[(p['probe_family'],p['probe_subtype'],p.get('tier',1))].append(p)
    per = max(2, n // len(groups)) if groups else 0
    sample = []
    for k, items in sorted(groups.items()):
        sample.extend(random.sample(items, min(per, len(items))))
    if len(sample) > n: sample = random.sample(sample, n)
    elif len(sample) < n:
        used = {p['probe_id'] for p in sample}
        rest = [p for p in probes if p['probe_id'] not in used]
        random.shuffle(rest)
        sample.extend(rest[:n - len(sample)])
    return sample

def mode_sample(a):
    probes = load_jsonl(f"{a.input_dir}/simp_probes_all.jsonl")
    # Only sample from rule-passing probes
    eligible = [p for p in probes if p.get('rule_pass', True)]
    print(f"Total probes: {len(probes)}, rule-pass: {len(eligible)}")

    sample = stratified_sample(eligible, AUDIT_SAMPLE_SIZE)

    csv_path = f"{a.output_dir}/simp_audit_form.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['probe_id','probe_family','probe_subtype','tier','dataset','generator_model',
                     'base_text_preview','probe_text_preview','expected_relation','operation',
                     'VERDICT','REJECT_REASON','NOTES'])
        for p in sample:
            w.writerow([p['probe_id'], p['probe_family'], p['probe_subtype'], p.get('tier',''),
                        p.get('dataset',''), p.get('generator_model',''),
                        p.get('base_text','')[:200].replace('\n',' '),
                        p.get('probe_text','')[:200].replace('\n',' '),
                        p.get('expected_relation',''), p.get('operation',''),
                        '','',''])

    with open(f"{a.output_dir}/simp_audit_sample.jsonl",'w',encoding='utf-8') as f:
        for p in sample: f.write(json.dumps(p,ensure_ascii=False)+'\n')

    sf = Counter(p['probe_family'] for p in sample)
    ss = Counter(p['probe_subtype'] for p in sample)
    print(f"\nAudit sample: {len(sample)} probes")
    print(f"  Family: {dict(sf)}")
    print(f"  Subtype: {dict(ss)}")
    print(f"\nSaved: {csv_path}")
    print(f"\n{'='*60}")
    print(f"MANUAL AUDIT REQUIRED")
    print(f"{'='*60}")
    print(f"1. Open {csv_path}")
    print(f"2. Mark VERDICT: KEEP / FIX / REJECT")
    print(f"3. Save as: {a.output_dir}/simp_audit_completed.csv")
    print(f"4. Run: python step3_audit.py --mode freeze")
    print(f"{'='*60}")

def mode_freeze(a):
    audit_path = f"{a.input_dir}/simp_audit_completed.csv"
    if not os.path.exists(audit_path):
        print(f"ERROR: {audit_path} not found. Complete audit first.")
        return

    # Load audit results
    verdicts = {}
    with open(audit_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            verdicts[row['probe_id'].strip()] = row.get('VERDICT','').strip().upper()

    vc = Counter(verdicts.values())
    keep_rate = vc.get('KEEP',0) / len(verdicts) if verdicts else 0
    print(f"Audit: {len(verdicts)} verdicts — {dict(vc)}")
    print(f"KEEP rate: {keep_rate:.1%}")

    # Find subtypes with >50% reject rate → drop entire subtype
    probes = load_jsonl(f"{a.input_dir}/simp_probes_all.jsonl")
    probe_map = {p['probe_id']:p for p in probes}
    by_sub = defaultdict(lambda:{'k':0,'r':0,'t':0})
    for pid, v in verdicts.items():
        p = probe_map.get(pid)
        if not p: continue
        sub = p.get('probe_subtype','?')
        by_sub[sub]['t'] += 1
        if v == 'KEEP': by_sub[sub]['k'] += 1
        elif v == 'REJECT': by_sub[sub]['r'] += 1

    drop_subs = set()
    for sub, c in by_sub.items():
        rr = c['r']/c['t'] if c['t'] else 0
        status = "DROP" if rr > 0.5 else "OK"
        if rr > 0.5: drop_subs.add(sub)
        print(f"  {sub}: keep={c['k']} reject={c['r']} rate={rr:.0%} → {status}")

    # Filter
    rejected_ids = {pid for pid, v in verdicts.items() if v == 'REJECT'}
    filtered = [p for p in probes
                if p['probe_id'] not in rejected_ids
                and p.get('probe_subtype') not in drop_subs
                and p.get('rule_pass', True)]

    print(f"\nFiltered: {len(probes)} → {len(filtered)}")
    print(f"  Individually rejected: {len(rejected_ids)}")
    print(f"  Subtypes dropped: {drop_subs or 'none'}")
    print(f"  Rule-filtered: {sum(1 for p in probes if not p.get('rule_pass',True))}")

    out = f"{a.output_dir}/simp_probecore_v1.jsonl"
    with open(out,'w',encoding='utf-8') as f:
        for p in filtered: f.write(json.dumps(p,ensure_ascii=False)+'\n')

    stats = {'total':len(filtered),'by_family':dict(Counter(p['probe_family'] for p in filtered)),
             'by_subtype':dict(Counter(p['probe_subtype'] for p in filtered)),
             'by_tier':dict(Counter(p.get('tier',1) for p in filtered)),
             'keep_rate':keep_rate,'dropped_subtypes':list(drop_subs)}
    with open(f"{a.output_dir}/simp_probecore_stats.json",'w') as f: json.dump(stats,f,indent=2)

    print(f"\nFrozen: {out}")
    print(f"Stats: {stats['by_family']}")
    print(f"✓ Step 3 freeze complete. Simp-ProbeCore v1 is frozen.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--mode', choices=['sample','freeze'], required=True)
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    if a.mode == 'sample': mode_sample(a)
    else: mode_freeze(a)

if __name__=='__main__': main()
