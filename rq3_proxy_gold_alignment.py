"""
rq3_corrected.py
================
Corrected proxy RQ3: Does probe-based reliability predict expert-quality
(Großer gold) rewrites better than ordinary conformance?

CHANGES FROM ORIGINAL:
1. Uses ESTABLISHED R(f) and per-family scores from diagnostic report — NOT recomputed
2. O(f) uses ONLY structural conformance — no SequenceMatcher meaning preservation
3. Gold alignment reports component-wise AND combined correlations
4. Per-family correlations (IR, DS, SR individually vs each gold component)
5. Honest about what works and what doesn't

NO LLM calls. NO BERTScore. Pure structural metrics.
"""

import json
import re
import csv
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — update these if your repo layout differs
# ─────────────────────────────────────────────────────────────────────────────
BENCHMARK_PATH = "parse/SRS-ProbeCore-v5-frozen.json"
OUTPUTS_PATH   = "parse/week4_model_outputs.json"
OUT_CSV        = "rq3_corrected_gold_alignment.csv"
OUT_CORR       = "rq3_corrected_correlations.txt"
OUT_FAMILY_CSV = "rq3_corrected_per_family.csv"
OUT_COMPONENT  = "rq3_corrected_component_correlations.csv"

# ─────────────────────────────────────────────────────────────────────────────
# ESTABLISHED SCORES FROM DIAGNOSTIC REPORT (24,541 evaluations)
# These are the validated, deterministic scores. DO NOT recompute.
# Source: diagnostic_report.pdf, analysis_summary.txt
# ─────────────────────────────────────────────────────────────────────────────

ESTABLISHED = {
    "qwen_32b":      {"R_f": 0.790, "IR": 0.925, "DS": 0.797, "SR": 0.587},
    "llama_70b":     {"R_f": 0.776, "IR": 0.932, "DS": 0.721, "SR": 0.607},
    "flan_t5_xl":    {"R_f": 0.744, "IR": 0.987, "DS": 0.994, "SR": 0.144},
    "flan_t5_large": {"R_f": 0.729, "IR": 0.987, "DS": 0.991, "SR": 0.093},
    "gpt4":          {"R_f": 0.718, "IR": 0.964, "DS": 0.849, "SR": 0.235},
    "llama_8b":      {"R_f": 0.688, "IR": 0.895, "DS": 0.336, "SR": 0.741},
    "rule_baseline":  {"R_f": 0.664, "IR": 0.906, "DS": 0.900, "SR": 0.080},
    "gemma_27b":     {"R_f": 0.654, "IR": 0.929, "DS": 0.233, "SR": 0.677},
}

MODEL_DISPLAY = {
    "rule_baseline": "Rule Baseline",
    "flan_t5_large": "FLAN-T5-Large",
    "flan_t5_xl":    "FLAN-T5-XL",
    "llama_8b":      "Llama-3.1-8B",
    "llama_70b":     "Llama-3.1-70B",
    "qwen_32b":      "Qwen2.5-32B",
    "gemma_27b":     "Gemma-3-27B",
    "gpt4":          "GPT-4.1",
}

# ─────────────────────────────────────────────────────────────────────────────
# EARS BOILERPLATE for Jaccard
# ─────────────────────────────────────────────────────────────────────────────
EARS_BOILERPLATE = {
    "when", "if", "while", "where", "shall", "should", "must", "may", "will",
    "can", "the", "system", "a", "an", "is", "are", "be", "to", "of", "and",
    "or", "in", "that", "it", "this", "for", "with", "by", "not", "no",
    "then", "ubiquitous", "event", "driven", "unwanted", "behavior",
    "state", "optional", "feature",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def content_tokens(text):
    """Lowercase tokens with stopwords and EARS boilerplate removed."""
    tokens = re.sub(r"[^\w\s]", " ", text.lower()).split()
    return set(t for t in tokens if t not in EARS_BOILERPLATE and len(t) > 2)


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return round(inter / union, 4) if union > 0 else 0.0


def extract_primary_modal(text):
    """Return the first modal found, priority order: shall > must > should > will > may > can."""
    for m in ["shall", "must", "should", "will", "may", "can"]:
        if re.search(rf"\b{m}\b", text, re.IGNORECASE):
            return m
    return None


def extract_ears_keyword(text):
    """Return the dominant EARS keyword: WHEN/IF/WHILE/WHERE/UBIQUITOUS/NONE."""
    t = text.strip()
    for kw in ["WHEN", "IF", "WHILE", "WHERE"]:
        if re.search(rf"\b{kw}\b", t, re.IGNORECASE):
            return kw.upper()
    if re.match(r"^(ubiquitous\b|the\s)", t, re.IGNORECASE):
        return "UBIQUITOUS"
    return "NONE"


def conformance_score(text):
    """
    Rule-based EARS conformance score [0.0, 1.0].
    This is the ONLY component of O(f). No SequenceMatcher meaning preservation.
    """
    if not text or len(text.strip()) < 5:
        return 0.0
    s = 0.0
    if re.search(r"\b(shall|should|must|may|will)\b", text, re.IGNORECASE):
        s += 0.4
    if re.search(r"\bthe\s+\w+(\s+\w+)?\s+(shall|should|must|may|will)\b", text, re.IGNORECASE):
        s += 0.3
    if text.strip().endswith("."):
        s += 0.2
    if re.search(r"\b(when|if|while|where)\b", text, re.IGNORECASE):
        s += 0.1
    elif re.match(r"^the\s+\w", text.strip(), re.IGNORECASE):
        s += 0.1
    return round(min(s, 1.0), 3)


def spearman(xs, ys):
    """Spearman rank correlation. Returns (rho, p_value)."""
    try:
        from scipy.stats import spearmanr
        result = spearmanr(xs, ys)
        rho = result.statistic if hasattr(result, "statistic") else result.correlation
        pval = result.pvalue
        return round(float(rho), 4), round(float(pval), 4)
    except ImportError:
        n = len(xs)
        def ranks(lst):
            indexed = sorted(enumerate(lst), key=lambda x: x[1])
            r = [0.0] * n
            for rank, (orig_idx, _) in enumerate(indexed, 1):
                r[orig_idx] = float(rank)
            return r
        rx, ry = ranks(xs), ranks(ys)
        d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
        rho = 1 - 6 * d2 / (n * (n * n - 1))
        return round(rho, 4), None


def avg(lst):
    return round(sum(lst) / len(lst), 4) if lst else None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("Loading benchmark ...")
    benchmark = json.loads(Path(BENCHMARK_PATH).read_text())
    log.info(f"  {len(benchmark)} items")

    log.info("Loading model outputs ...")
    outputs = json.loads(Path(OUTPUTS_PATH).read_text())
    log.info(f"  {len(outputs)} records")

    # Build output index: (item_id, probe_id, model_id) -> rewrite text
    out_idx = {}
    for rec in outputs:
        if rec.get("run_status", "ok") != "ok" or not rec.get("rewrite", "").strip():
            continue
        key = (rec["item_id"], rec.get("probe_id"), rec["model_id"])
        out_idx[key] = rec["rewrite"]

    all_model_ids = sorted(ESTABLISHED.keys())
    log.info(f"  Models: {all_model_ids}")

    # ── Separate Großer items ─────────────────────────────────────────────────
    grosser_items = [i for i in benchmark if i["item_id"].startswith("GROSSER_")]
    log.info(f"  Großer items with gold rewrites: {len(grosser_items)}")

    # Verify coverage
    for mid in all_model_ids:
        count = sum(1 for item in grosser_items if (item["item_id"], None, mid) in out_idx)
        log.info(f"  {mid}: {count}/{len(grosser_items)} base outputs")

    # ── R(f) and per-family scores: USE ESTABLISHED VALUES ────────────────────
    log.info("Using ESTABLISHED R(f) from diagnostic report (NOT recomputing)")
    for mid in all_model_ids:
        e = ESTABLISHED[mid]
        log.info(f"  {mid}: R(f)={e['R_f']}  IR={e['IR']}  DS={e['DS']}  SR={e['SR']}")

    # ── O(f): Conformance ONLY (no SequenceMatcher meaning preservation) ──────
    log.info("Computing O(f) — structural conformance only (no meaning preservation) ...")
    model_of = {}
    for mid in all_model_ids:
        scores = []
        for item in benchmark:
            rewrite = out_idx.get((item["item_id"], None, mid), "")
            scores.append(conformance_score(rewrite))
        model_of[mid] = round(sum(scores) / len(scores), 4) if scores else 0
    
    log.info("O(f) computed (conformance only):")
    for mid in all_model_ids:
        log.info(f"  {mid}: O(f)={model_of[mid]}")

    # ── Gold alignment on Großer items ────────────────────────────────────────
    log.info("Computing gold alignment on Großer items ...")

    model_gold_components = defaultdict(lambda: {
        "jaccard": [], "modal": [], "ears_kw": [],
    })
    per_item_gold = {}  # (item_id, model_id) -> component scores

    n_missing_gold = 0
    for item in grosser_items:
        iid = item["item_id"]
        gold_text = item.get("reference_rewrite", "") or ""
        if not gold_text.strip():
            n_missing_gold += 1
            continue

        gold_tokens = content_tokens(gold_text)
        gold_modal  = extract_primary_modal(gold_text)
        gold_kw     = extract_ears_keyword(gold_text)

        for mid in all_model_ids:
            rewrite = out_idx.get((iid, None, mid), "")
            if not rewrite.strip():
                continue

            rw_tokens = content_tokens(rewrite)
            rw_modal  = extract_primary_modal(rewrite)
            rw_kw     = extract_ears_keyword(rewrite)

            j  = jaccard(rw_tokens, gold_tokens)
            mm = int(rw_modal == gold_modal) if (rw_modal and gold_modal) else 0
            ek = int(rw_kw == gold_kw)

            model_gold_components[mid]["jaccard"].append(j)
            model_gold_components[mid]["modal"].append(mm)
            model_gold_components[mid]["ears_kw"].append(ek)

            per_item_gold[(iid, mid)] = {
                "jaccard": j, "modal": mm, "ears_kw": ek,
                "combined": round((j + mm + ek) / 3, 4),
                "structural": round((mm + ek) / 2, 4),  # modal + EARS_KW only, no Jaccard
            }

    if n_missing_gold > 0:
        log.warning(f"  {n_missing_gold} Großer items have no reference_rewrite — skipped")

    # Aggregate per model
    model_gold_agg = {}
    for mid in all_model_ids:
        g = model_gold_components[mid]
        model_gold_agg[mid] = {
            "jaccard":    avg(g["jaccard"]),
            "modal":      avg(g["modal"]),
            "ears_kw":    avg(g["ears_kw"]),
            "combined":   round((avg(g["jaccard"]) + avg(g["modal"]) + avg(g["ears_kw"])) / 3, 4),
            "structural": round((avg(g["modal"]) + avg(g["ears_kw"])) / 2, 4),
        }

    log.info("Gold alignment per model:")
    for mid in all_model_ids:
        ga = model_gold_agg[mid]
        log.info(f"  {mid}: combined={ga['combined']}  structural={ga['structural']}  "
                 f"jaccard={ga['jaccard']}  modal={ga['modal']}  ears_kw={ga['ears_kw']}")

    # ── Spearman correlations ─────────────────────────────────────────────────
    log.info("Computing Spearman correlations ...")

    # Extract parallel arrays (same model order)
    rf_vals   = [ESTABLISHED[m]["R_f"] for m in all_model_ids]
    ir_vals   = [ESTABLISHED[m]["IR"]  for m in all_model_ids]
    ds_vals   = [ESTABLISHED[m]["DS"]  for m in all_model_ids]
    sr_vals   = [ESTABLISHED[m]["SR"]  for m in all_model_ids]
    of_vals   = [model_of[m]           for m in all_model_ids]

    gold_combined    = [model_gold_agg[m]["combined"]    for m in all_model_ids]
    gold_structural  = [model_gold_agg[m]["structural"]  for m in all_model_ids]
    gold_jaccard     = [model_gold_agg[m]["jaccard"]     for m in all_model_ids]
    gold_modal       = [model_gold_agg[m]["modal"]       for m in all_model_ids]
    gold_ears_kw     = [model_gold_agg[m]["ears_kw"]     for m in all_model_ids]

    # All correlations
    correlations = {}
    for score_name, score_vals in [
        ("R(f)",  rf_vals),
        ("IR",    ir_vals),
        ("DS",    ds_vals),
        ("SR",    sr_vals),
        ("O(f)",  of_vals),
    ]:
        for gold_name, gold_v in [
            ("gold_combined",   gold_combined),
            ("gold_structural", gold_structural),
            ("gold_jaccard",    gold_jaccard),
            ("gold_modal",      gold_modal),
            ("gold_ears_kw",    gold_ears_kw),
        ]:
            rho, p = spearman(score_vals, gold_v)
            key = f"{score_name} vs {gold_name}"
            correlations[key] = {"rho": rho, "p": p}
            sig = "***" if p and p < 0.01 else "**" if p and p < 0.05 else "*" if p and p < 0.1 else ""
            log.info(f"  {key:40s} ρ={rho:+.4f}  p={p}  {sig}")

    # R(f) vs O(f) anti-correlation
    rho_rf_of, p_rf_of = spearman(rf_vals, of_vals)
    log.info(f"  {'R(f) vs O(f)':40s} ρ={rho_rf_of:+.4f}  p={p_rf_of}")

    # ── Rank table ────────────────────────────────────────────────────────────
    def make_ranks(vals_dict, reverse=True):
        sorted_mids = sorted(vals_dict.keys(), key=lambda m: vals_dict[m], reverse=reverse)
        return {m: r + 1 for r, m in enumerate(sorted_mids)}

    rank_rf   = make_ranks({m: ESTABLISHED[m]["R_f"]           for m in all_model_ids})
    rank_of   = make_ranks({m: model_of[m]                     for m in all_model_ids})
    rank_gold = make_ranks({m: model_gold_agg[m]["combined"]   for m in all_model_ids})
    rank_str  = make_ranks({m: model_gold_agg[m]["structural"] for m in all_model_ids})

    # ── Per-family gold alignment: pass vs fail ───────────────────────────────
    # This uses ESTABLISHED per-model per-family pass rates as weights.
    # For each Großer item × model, we check: did this model generally pass
    # this probe family? We use the model's global family pass rate as a proxy.
    # Models with rate > 0.5 are "pass" for that family, <= 0.5 are "fail."
    #
    # A more precise version would use per-item pass/fail from probe_results_full.json
    # if that file exists in the repo.

    log.info("Computing per-family gold alignment (pass vs fail, model-level threshold) ...")
    family_gold = {}
    for family, key in [("invariance", "IR"), ("directional", "DS"), ("shortcut", "SR")]:
        pass_scores = []
        fail_scores = []
        for mid in all_model_ids:
            family_rate = ESTABLISHED[mid][key]
            for item in grosser_items:
                iid = item["item_id"]
                g = per_item_gold.get((iid, mid))
                if not g:
                    continue
                if family_rate > 0.5:
                    pass_scores.append(g["combined"])
                else:
                    fail_scores.append(g["combined"])
        pa = avg(pass_scores)
        fa = avg(fail_scores)
        diff = round(pa - fa, 4) if (pa is not None and fa is not None) else None
        family_gold[family] = {
            "pass_avg": pa, "fail_avg": fa, "diff": diff,
            "n_pass": len(pass_scores), "n_fail": len(fail_scores),
        }
        log.info(f"  {family}: pass={pa} (n={len(pass_scores)})  "
                 f"fail={fa} (n={len(fail_scores)})  diff={diff}")

    # ── Print full summary ────────────────────────────────────────────────────
    print()
    print("=" * 110)
    print(f"{'Model':<20} {'R(f)':>6} {'O(f)':>6} {'Gold':>6} {'Struct':>6} "
          f"{'Jacc':>6} {'Modal':>6} {'EARS':>6} {'Rk_R':>5} {'Rk_O':>5} {'Rk_G':>5} {'Rk_S':>5}")
    print("-" * 110)
    for mid in sorted(all_model_ids, key=lambda m: rank_rf[m]):
        name = MODEL_DISPLAY.get(mid, mid)
        ga = model_gold_agg[mid]
        print(f"{name:<20} {ESTABLISHED[mid]['R_f']:>6.3f} {model_of[mid]:>6.3f} "
              f"{ga['combined']:>6.3f} {ga['structural']:>6.3f} "
              f"{ga['jaccard']:>6.3f} {ga['modal']:>6.3f} {ga['ears_kw']:>6.3f} "
              f"{rank_rf[mid]:>5} {rank_of[mid]:>5} {rank_gold[mid]:>5} {rank_str[mid]:>5}")
    print("=" * 110)

    # Print key correlations
    print()
    print("KEY CORRELATIONS (n=8 models)")
    print("-" * 60)
    for key in [
        "R(f) vs gold_combined",   "R(f) vs gold_structural",
        "R(f) vs gold_jaccard",    "R(f) vs gold_modal",     "R(f) vs gold_ears_kw",
        "O(f) vs gold_combined",   "O(f) vs gold_structural",
        "DS vs gold_combined",     "DS vs gold_structural",  "DS vs gold_modal",
        "SR vs gold_combined",     "SR vs gold_structural",
        "IR vs gold_combined",
    ]:
        c = correlations.get(key, {})
        rho = c.get("rho", 0)
        p   = c.get("p", 1)
        sig = "***" if p and p < 0.01 else "**" if p and p < 0.05 else "*" if p and p < 0.1 else "ns"
        print(f"  {key:40s} ρ={rho:+.4f}  p={p:<8}  {sig}")

    print(f"\n  {'R(f) vs O(f)':40s} ρ={rho_rf_of:+.4f}  p={p_rf_of}")

    # Per-family pass/fail
    print()
    print("PER-FAMILY GOLD ALIGNMENT (pass vs fail)")
    print("-" * 70)
    print(f"  {'Family':<15} {'Pass gold':>10} {'Fail gold':>10} {'Diff':>8} {'n_pass':>8} {'n_fail':>8}")
    for family in ["invariance", "directional", "shortcut"]:
        fg = family_gold[family]
        print(f"  {family:<15} {fg['pass_avg'] or 0:>10.4f} {fg['fail_avg'] or 0:>10.4f} "
              f"{fg['diff'] or 0:>8.4f} {fg['n_pass']:>8} {fg['n_fail']:>8}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("RQ3 FINDINGS")
    print("=" * 60)

    rf_gold_rho = correlations["R(f) vs gold_combined"]["rho"]
    of_gold_rho = correlations["O(f) vs gold_combined"]["rho"]
    ds_gold_rho = correlations["DS vs gold_combined"]["rho"]
    ds_gold_p   = correlations["DS vs gold_combined"]["p"]
    rf_of_rho   = rho_rf_of
    rf_of_p     = p_rf_of

    print(f"  1. R(f) vs gold_combined:  ρ={rf_gold_rho:+.4f} — {'significant' if correlations['R(f) vs gold_combined']['p'] and correlations['R(f) vs gold_combined']['p'] < 0.05 else 'NOT significant'}")
    print(f"  2. O(f) vs gold_combined:  ρ={of_gold_rho:+.4f} — {'significant' if correlations['O(f) vs gold_combined']['p'] and correlations['O(f) vs gold_combined']['p'] < 0.05 else 'NOT significant'}")
    print(f"  3. DS vs gold_combined:    ρ={ds_gold_rho:+.4f} — {'significant' if ds_gold_p and ds_gold_p < 0.05 else 'NOT significant'}")
    print(f"  4. R(f) vs O(f):           ρ={rf_of_rho:+.4f} — {'significant' if rf_of_p and rf_of_p < 0.05 else 'NOT significant'}")

    print()
    print("  INTERPRETATION:")
    if rf_gold_rho > of_gold_rho and correlations["R(f) vs gold_combined"]["p"] and correlations["R(f) vs gold_combined"]["p"] < 0.05:
        print("  → R(f) predicts gold alignment better than O(f). Original RQ3 hypothesis SUPPORTED.")
    elif rf_gold_rho > of_gold_rho:
        print("  → R(f) has a higher correlation with gold than O(f), but not statistically significant.")
        print("    Frame as directional evidence, not a confirmed finding.")
    else:
        print("  → Original RQ3 hypothesis NOT supported by global correlation.")
        print()
        # Check component-level findings
        best_predictor = None
        best_rho = -2
        for score_name in ["R(f)", "IR", "DS", "SR", "O(f)"]:
            key = f"{score_name} vs gold_combined"
            c = correlations.get(key, {})
            if c.get("p") and c["p"] < 0.05 and c["rho"] > best_rho:
                best_rho = c["rho"]
                best_predictor = score_name
        if best_predictor:
            bp_key = f"{best_predictor} vs gold_combined"
            print(f"  → However, {best_predictor} significantly predicts gold alignment:")
            print(f"    {bp_key}: ρ={correlations[bp_key]['rho']:+.4f}, p={correlations[bp_key]['p']}")
        
        # Check structural-only gold
        for score_name in ["R(f)", "DS", "SR"]:
            key = f"{score_name} vs gold_structural"
            c = correlations.get(key, {})
            if c.get("p") and c["p"] < 0.05:
                print(f"  → {score_name} predicts STRUCTURAL gold alignment (modal+EARS_KW, no Jaccard):")
                print(f"    {key}: ρ={c['rho']:+.4f}, p={c['p']}")

        # Note the R(f) vs O(f) finding
        if rf_of_p and rf_of_p < 0.05:
            print()
            print(f"  → KEY FINDING: R(f) and O(f) are significantly anti-correlated (ρ={rf_of_rho:+.4f}, p={rf_of_p}).")
            print("    Models ranked highest by ordinary conformance are ranked lowest by probe reliability.")
            print("    This directly supports RQ1 and can anchor a reframed RQ3.")

    # ── Save outputs ──────────────────────────────────────────────────────────
    
    # CSV: per-model summary
    fieldnames = [
        "model", "display_name",
        "R_f", "IR", "DS", "SR",
        "O_f_conformance",
        "gold_combined", "gold_structural", "gold_jaccard", "gold_modal", "gold_ears_kw",
        "rank_R", "rank_O", "rank_gold_combined", "rank_gold_structural",
    ]
    rows = []
    for mid in all_model_ids:
        ga = model_gold_agg[mid]
        rows.append({
            "model":              mid,
            "display_name":       MODEL_DISPLAY.get(mid, mid),
            "R_f":                ESTABLISHED[mid]["R_f"],
            "IR":                 ESTABLISHED[mid]["IR"],
            "DS":                 ESTABLISHED[mid]["DS"],
            "SR":                 ESTABLISHED[mid]["SR"],
            "O_f_conformance":    model_of[mid],
            "gold_combined":      ga["combined"],
            "gold_structural":    ga["structural"],
            "gold_jaccard":       ga["jaccard"],
            "gold_modal":         ga["modal"],
            "gold_ears_kw":       ga["ears_kw"],
            "rank_R":             rank_rf[mid],
            "rank_O":             rank_of[mid],
            "rank_gold_combined": rank_gold[mid],
            "rank_gold_structural": rank_str[mid],
        })
    rows.sort(key=lambda r: r["rank_R"])
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    log.info(f"Saved: {OUT_CSV}")

    # Correlations text file
    with open(OUT_CORR, "w") as f:
        f.write("RQ3 Corrected: Großer Gold-Alignment Correlation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write("METHODOLOGY\n")
        f.write("  R(f): ESTABLISHED from diagnostic report (24,541 evals, deterministic scoring)\n")
        f.write("  O(f): Structural EARS conformance ONLY (no SequenceMatcher meaning preservation)\n")
        f.write("  Gold: Jaccard (content tokens) + Modal match + EARS keyword match\n")
        f.write(f"  Großer items: {len(grosser_items)} with gold rewrites\n")
        f.write(f"  Models: {len(all_model_ids)}\n\n")
        f.write("ALL CORRELATIONS\n")
        f.write("-" * 60 + "\n")
        for key, c in sorted(correlations.items()):
            sig = "***" if c["p"] and c["p"] < 0.01 else "**" if c["p"] and c["p"] < 0.05 else ""
            f.write(f"  {key:40s} ρ={c['rho']:+.4f}  p={c['p']}  {sig}\n")
        f.write(f"\n  {'R(f) vs O(f)':40s} ρ={rho_rf_of:+.4f}  p={p_rf_of}\n")
    log.info(f"Saved: {OUT_CORR}")

    # Component correlations CSV
    comp_rows = []
    for key, c in sorted(correlations.items()):
        parts = key.split(" vs ")
        comp_rows.append({
            "score": parts[0], "gold_metric": parts[1],
            "spearman_rho": c["rho"], "p_value": c["p"],
            "significant_005": "yes" if c["p"] and c["p"] < 0.05 else "no",
        })
    with open(OUT_COMPONENT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["score", "gold_metric", "spearman_rho", "p_value", "significant_005"])
        w.writeheader()
        w.writerows(comp_rows)
    log.info(f"Saved: {OUT_COMPONENT}")

    # Per-family CSV
    family_rows = []
    for family in ["invariance", "directional", "shortcut"]:
        fg = family_gold[family]
        family_rows.append({
            "probe_family":       family,
            "pass_gold_alignment": fg["pass_avg"],
            "fail_gold_alignment": fg["fail_avg"],
            "difference":         fg["diff"],
            "n_pass":             fg["n_pass"],
            "n_fail":             fg["n_fail"],
        })
    with open(OUT_FAMILY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["probe_family", "pass_gold_alignment", "fail_gold_alignment", "difference", "n_pass", "n_fail"])
        w.writeheader()
        w.writerows(family_rows)
    log.info(f"Saved: {OUT_FAMILY_CSV}")

    log.info("Done.")


if __name__ == "__main__":
    main()