#!/usr/bin/env python3
"""Step 5: Cross-domain analysis — SRS vs Simplification reliability comparison
Usage:
  python step5_cross_domain.py --simp-dir ./output/simp_scores --output-dir ./output/rq4_results
  python step5_cross_domain.py --simp-dir ./output/simp_scores --srs-dir ./srs_scores --output-dir ./output/rq4_results
"""
import json, csv, os, argparse
from collections import defaultdict

def load_json(p):
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)

def load_jsonl(p):
    if not os.path.exists(p): return []
    with open(p) as f: return [json.loads(l) for l in f if l.strip()]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--simp-dir', required=True)
    p.add_argument('--srs-dir', default=None)
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)

    simp = load_json(f"{a.simp_dir}/reliability_summary.json")
    if not simp: print("ERROR: No simplification scores"); return

    srs = load_json(f"{a.srs_dir}/reliability_summary.json") if a.srs_dir else None

    # ── Table 1: Cross-domain (if SRS available) ──
    if srs:
        print("=== Cross-Domain Comparison ===")
        srs_m = {s['model']:s for s in srs}
        simp_m = {s['model']:s for s in simp}
        common = sorted(set(srs_m) & set(simp_m))
        if common:
            rows = []
            for m in common:
                s, si = srs_m[m], simp_m[m]
                rows.append({'model':m,
                    'srs_inv':s.get('inv_rel',0),'simp_inv':si.get('inv_rel',0),
                    'srs_dir':s.get('dir_rel',0),'simp_dir':si.get('dir_rel',0),
                    'srs_sc':s.get('sc_rel',0),'simp_sc':si.get('sc_rel',0),
                    'srs_global':s.get('global_rel',0),'simp_global':si.get('global_rel',0)})
            with open(f"{a.output_dir}/rq4_cross_domain.csv",'w',newline='') as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
            print(f"  Saved rq4_cross_domain.csv ({len(rows)} models)")
            for r in rows:
                print(f"  {r['model']:<20} SRS={r['srs_global']:.1%} SIMP={r['simp_global']:.1%}")
        else:
            print("  No common models")

    # ── Table 2: Tier comparison ──
    print("\n=== Tier Comparison ===")
    tier_rows = []
    for s in simp:
        for fam in ['invariance','directional','shortcut']:
            t1 = s.get(f'{fam}_t1', 0); t2 = s.get(f'{fam}_t2', 0)
            n1 = s.get(f'{fam}_t1_n', 0); n2 = s.get(f'{fam}_t2_n', 0)
            tier_rows.append({'model':s['model'],'family':fam,
                'tier1':t1,'n1':n1,'tier2':t2,'n2':n2,'gap':round(t1-t2,4)})
    with open(f"{a.output_dir}/rq4_tier_comparison.csv",'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=tier_rows[0].keys()); w.writeheader(); w.writerows(tier_rows)
    print(f"  Saved rq4_tier_comparison.csv")

    # ── Table 3: Most diagnostic family ──
    print("\n=== Most Diagnostic Family ===")
    for s in simp:
        fams = {'invariance':s.get('inv_rel',1),'directional':s.get('dir_rel',1),'shortcut':s.get('sc_rel',1)}
        weakest = min(fams, key=fams.get)
        print(f"  {s['model']:<20} weakest={weakest} ({fams[weakest]:.1%})")

    # ── Table 4: Paper-ready table ──
    print("\n=== Paper Table ===")
    paper_path = f"{a.output_dir}/rq4_paper_table.csv"
    with open(paper_path,'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['Model','Invariance','Directional','Shortcut','Global','N'])
        for s in simp:
            w.writerow([s['model'],f"{s['inv_rel']:.3f}",f"{s['dir_rel']:.3f}",
                        f"{s['sc_rel']:.3f}",f"{s['global_rel']:.3f}",s['total']])
    print(f"  Saved: {paper_path}")

    # ── Save full analysis ──
    analysis = {
        'simplification_summary': simp,
        'srs_summary': srs,
        'tier_analysis': tier_rows,
        'rq4_findings': {
            'framework_transfers': 'FILL after reviewing tables',
            'diagnostic_ranking_same_across_domains': 'FILL',
            'tier_gap_meaningful': 'FILL',
            'notes': []
        }
    }
    with open(f"{a.output_dir}/rq4_analysis.json",'w') as f: json.dump(analysis,f,indent=2)
    print(f"\n✓ Step 5 complete. Review CSVs and update rq4_analysis.json.")

if __name__=='__main__': main()
