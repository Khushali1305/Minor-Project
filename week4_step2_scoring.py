"""
week4_step2_scoring.py
======================
Computes the full scoring protocol from methodology.md §7
against model outputs from week4_step1_run_models.py.

SCORING PROTOCOL
----------------
Probe-level (§7.1) — binary pass/fail per probe pair:
  Invariance  : BERTScore F1(f(x), f(x')) >= TAU_INVARIANCE  -> PASS (stable)
  Directional : modal strength shifts in correct direction    -> PASS
  Shortcut    : claude-sonnet-4-6 judge: distractor absent   -> PASS

Family-level (§7.2):
  invariance_reliability   = PASS_inv / N_inv
  directional_sensitivity  = PASS_dir / N_dir
  shortcut_resistance      = PASS_shc / N_shc

Global reliability (§7.3):
  R(f) = (invariance_reliability + directional_sensitivity + shortcut_resistance) / 3

Ordinary evaluation baseline (§7.4):
  EARS conformance  : rule-based structural check on f(x)
  Meaning pres.     : BERTScore F1(f(x), x)  [rewrite vs source]

BUGS FIXED FROM PREVIOUS VERSION
----------------------------------
  1. Judge called OpenAI -> now calls Anthropic (claude-sonnet-4-6)
  2. Fallback check used openai_api_key -> now uses anthropic_api_key
  3. list[str] type hints (Python 3.9+) -> removed, compatible with 3.8+
  4. def f(v) in main() shadowed loop variable f -> renamed to fmt_val
  5. compute_bertscore_f1 used results.insert() with gap indices -> pre-allocated list
  6. Records with run_status=empty now excluded from scoring index

OUTPUTS
-------
  week4_probe_scores.json    : per-probe pass/fail with evidence
  week4_model_scores.json    : per-model family + global + ordinary scores
  week4_results_table.csv    : camera-ready results table

USAGE
-----
  python3 week4_step2_scoring.py

  # Rule-based shortcut fallback (no Anthropic API needed)
  python3 week4_step2_scoring.py --no_llm_judge

  # Different BERTScore threshold
  python3 week4_step2_scoring.py --tau 0.80

ENVIRONMENT VARIABLES
---------------------
  export ANTHROPIC_API_KEY=your_key    # required for shortcut judge

DEPENDENCIES
------------
  pip install anthropic bert-score
"""

import json
import os
import re
import csv
import argparse
import logging
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "benchmark_file"    : "SRS-ProbeCore-v5-frozen.json",
    "model_outputs_file": "week4_model_outputs.json",
    "probe_scores_file" : "week4_probe_scores.json",
    "model_scores_file" : "week4_model_scores.json",
    "results_table_csv" : "week4_results_table.csv",

    # BERTScore F1 threshold for invariance pass/fail
    # 0.85 is conservative — tune on a small validation set if needed
    "tau_invariance"    : 0.85,

    # LLM judge — Anthropic claude-sonnet-4-6
    # Chosen over Opus: binary PASS/FAIL task does not need frontier reasoning
    # Chosen over Haiku: needs reliable detection of paraphrased distractors
    "judge_model"       : "claude-sonnet-4-6",
    "anthropic_api_key" : os.environ.get("ANTHROPIC_API_KEY", ""),

    # Modal strength scale (from evaluation_settings.md §6)
    # 3=strongest obligation, 1=weakest
    "modal_strength": {
        "shall": 3, "must": 3,
        "should": 2, "will": 2,
        "may": 1, "can": 1,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# EARS CONFORMANCE CHECKER — ordinary evaluation baseline
# ─────────────────────────────────────────────────────────────────────────────

EARS_KEYWORDS  = re.compile(r"\b(WHEN|IF|WHILE|WHERE)\b", re.IGNORECASE)
MODAL_PATTERN  = re.compile(r"\b(SHALL|SHOULD|MUST|MAY|WILL)\b", re.IGNORECASE)
SYSTEM_SUBJECT = re.compile(
    r"\bthe\s+\w+(\s+\w+)?\s+(SHALL|SHOULD|MUST|MAY|WILL)\b", re.IGNORECASE
)


def ears_conformance_score(rewrite):
    """
    Rule-based EARS conformance score in [0.0, 1.0].
    +0.4 : modal verb present (SHALL/SHOULD/MUST/MAY/WILL)
    +0.3 : system subject before modal ("the <X> SHALL")
    +0.2 : ends with period
    +0.1 : EARS trigger keyword OR valid Ubiquitous form
    """
    if not rewrite or len(rewrite.strip()) < 5:
        return 0.0
    text  = rewrite.strip()
    score = 0.0
    if MODAL_PATTERN.search(text):
        score += 0.4
    if SYSTEM_SUBJECT.search(text):
        score += 0.3
    if text.endswith("."):
        score += 0.2
    if EARS_KEYWORDS.search(text):
        score += 0.1
    elif re.match(r"^the\s+\w+", text, re.IGNORECASE):
        score += 0.1
    return round(min(score, 1.0), 3)


# ─────────────────────────────────────────────────────────────────────────────
# MODAL STRENGTH EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def extract_modal_strength(text, modal_map):
    """Return the highest modal strength found in text. 0 if none found."""
    text_lower = text.lower()
    best = 0
    for modal, strength in modal_map.items():
        if re.search(r"\b" + modal + r"\b", text_lower):
            best = max(best, strength)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# BERTSCORE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def compute_bertscore_f1(candidates, references):
    """
    BERTScore F1 between parallel candidate and reference lists.
    Returns list of float scores in [0, 1].
    Empty strings are given score 0.0 without being passed to the model.

    Bug fix from previous version: uses pre-allocated list instead of
    results.insert() to avoid index-ordering errors when items are filtered.
    """
    try:
        from bert_score import score as bs_score
    except ImportError:
        raise ImportError("pip install bert-score")

    n       = len(candidates)
    results = [0.0] * n   # pre-allocate — fix for insert() bug

    valid_c, valid_r, valid_idx = [], [], []
    for i, (c, r) in enumerate(zip(candidates, references)):
        if c.strip() and r.strip():
            valid_c.append(c)
            valid_r.append(r)
            valid_idx.append(i)

    if not valid_c:
        return results

    _, _, F1 = bs_score(
        valid_c, valid_r,
        lang="en",
        model_type="bert-base-uncased",
        verbose=False,
        device=None,
    )

    for idx, f1_val in zip(valid_idx, F1.tolist()):
        results[idx] = round(float(f1_val), 4)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# LLM JUDGE — shortcut resistance via Anthropic claude-sonnet-4-6
# ─────────────────────────────────────────────────────────────────────────────

SHORTCUT_JUDGE_SYSTEM = """You are an expert requirements engineering evaluator.
You will be given:
  1. A base requirement (clean, no distractor)
  2. A distractor phrase that was inserted into the probe input
  3. A model's rewrite of the probe input

Your task: determine whether the model's rewrite has incorporated the distractor
content — either verbatim or in semantically equivalent form.

A reliable model should produce a rewrite reflecting ONLY the base requirement,
ignoring the distractor. The distractor is an NFR or functional cue that should
NOT appear in the structural rewrite.

Respond with EXACTLY one of:
  PASS  — the rewrite does NOT incorporate the distractor content
  FAIL  — the rewrite DOES incorporate the distractor content

Then on a new line give a one-sentence explanation.
"""

SHORTCUT_JUDGE_USER = """Base requirement : {base_text}
Distractor phrase: {distractor}
Model rewrite    : {rewrite}

Does the rewrite incorporate the distractor? Answer PASS or FAIL:"""


def judge_shortcut_batch(base_texts, distractors, rewrites, config):
    """
    Call claude-sonnet-4-6 as shortcut judge via Anthropic API.
    Returns list of {"pass": bool, "explanation": str}.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    client  = anthropic.Anthropic(api_key=config["anthropic_api_key"])
    results = []

    for base, distractor, rewrite in zip(base_texts, distractors, rewrites):
        try:
            response = client.messages.create(
                model=config["judge_model"],
                max_tokens=150,
                system=SHORTCUT_JUDGE_SYSTEM.strip(),
                messages=[{
                    "role"   : "user",
                    "content": SHORTCUT_JUDGE_USER.format(
                        base_text=base,
                        distractor=distractor,
                        rewrite=rewrite,
                    ).strip(),
                }],
            )
            content     = response.content[0].text.strip()
            lines       = content.split("\n", 1)
            verdict     = lines[0].strip().upper()
            explanation = lines[1].strip() if len(lines) > 1 else ""
            passed      = verdict.startswith("PASS")
            results.append({"pass": passed, "explanation": explanation})

        except Exception as e:
            logging.warning(f"Anthropic judge error: {e}")
            results.append({"pass": False, "explanation": f"error: {e}"})

    return results


def rule_based_shortcut_check(distractor, rewrite):
    """
    Fallback shortcut check — no LLM required.
    Checks key-token overlap between distractor and rewrite.
    Less reliable than the LLM judge for paraphrased distractors.
    """
    stop = {"a","an","the","and","or","in","to","of","for","with","by","at","on"}
    tokens = set(re.sub(r"[^\w\s]", "", distractor.lower()).split()) - stop

    if not tokens:
        return {"pass": True, "explanation": "rule_based: no content tokens in distractor"}

    rewrite_lower = rewrite.lower()
    overlap = sum(1 for t in tokens if t in rewrite_lower)
    ratio   = overlap / len(tokens)
    passed  = ratio < 0.5

    return {
        "pass": passed,
        "explanation": (
            f"rule_based: {overlap}/{len(tokens)} distractor tokens "
            f"found in rewrite (ratio={ratio:.2f})"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def score_all(benchmark, outputs, args, config):
    """
    Main scoring function.
    Returns (probe_scores, model_scores, ordinary_scores).
    """

    # Index outputs: (item_id, probe_id, model_id) -> rewrite
    # Exclude empty/error records so they don't corrupt scores
    output_index = {}
    skipped_empty = 0
    for rec in outputs:
        if rec.get("run_status", "ok") != "ok" or not rec.get("rewrite", "").strip():
            skipped_empty += 1
            continue
        key = (rec["item_id"], rec.get("probe_id"), rec["model_id"])
        output_index[key] = rec["rewrite"]

    if skipped_empty:
        logging.warning(
            f"Excluded {skipped_empty} empty/error records from scoring index"
        )

    all_model_ids = sorted({rec["model_id"] for rec in outputs})
    logging.info(f"Models: {all_model_ids}")

    probe_scores    = []
    model_tallies   = defaultdict(lambda: defaultdict(lambda: {"pass": 0, "total": 0}))
    ordinary_scores = defaultdict(list)

    # ── Ordinary evaluation (base items only) ────────────────────────────────
    logging.info("Ordinary evaluation ...")

    for model_id in all_model_ids:
        base_jobs = [
            (item["item_id"], item["requirement_text"],
             output_index.get((item["item_id"], None, model_id), ""))
            for item in benchmark
        ]

        for iid, base_text, rewrite in base_jobs:
            ordinary_scores[model_id].append({
                "item_id"     : iid,
                "conformance" : ears_conformance_score(rewrite),
                "rewrite"     : rewrite,
                "base_text"   : base_text,
            })

        candidates = [j[2] for j in base_jobs]
        references = [j[1] for j in base_jobs]
        bs_scores  = compute_bertscore_f1(candidates, references)

        for i in range(len(base_jobs)):
            ordinary_scores[model_id][i]["meaning_preservation"] = bs_scores[i]

    # ── Invariance probes — BERTScore batch ──────────────────────────────────
    logging.info("Invariance scoring ...")

    inv_jobs = []
    for item in benchmark:
        iid = item["item_id"]
        for probe in item.get("probe_neighborhoods", []):
            if probe.get("probe_family") != "invariance":
                continue
            pid = probe.get("probe_id")
            for model_id in all_model_ids:
                base_rw  = output_index.get((iid, None, model_id), "")
                probe_rw = output_index.get((iid, pid, model_id), "")
                inv_jobs.append((iid, pid, model_id, base_rw, probe_rw, probe))

    logging.info(f"  {len(inv_jobs)} invariance pairs ...")

    if inv_jobs:
        bs = compute_bertscore_f1(
            [j[3] for j in inv_jobs],
            [j[4] for j in inv_jobs],
        )
        for (iid, pid, model_id, base_rw, probe_rw, probe), score in zip(inv_jobs, bs):
            passed = score >= config["tau_invariance"]
            probe_scores.append({
                "item_id"      : iid,
                "probe_id"     : pid,
                "probe_family" : "invariance",
                "model_id"     : model_id,
                "pass"         : int(passed),
                "bertscore_f1" : score,
                "base_rewrite" : base_rw,
                "probe_rewrite": probe_rw,
                "evidence"     : f"BERTScore F1={score:.4f} tau={config['tau_invariance']}",
            })
            model_tallies[model_id]["invariance"]["total"] += 1
            if passed:
                model_tallies[model_id]["invariance"]["pass"] += 1

    # ── Directional probes — rule-based modal strength ────────────────────────
    logging.info("Directional scoring ...")

    for item in benchmark:
        iid = item["item_id"]
        for probe in item.get("probe_neighborhoods", []):
            if probe.get("probe_family") != "directional":
                continue
            pid        = probe.get("probe_id")
            operation  = probe.get("operation", "")
            base_text  = item["requirement_text"]
            probe_text = probe.get("probe_text", "")

            if "weaken" in operation:
                expected = "weaken"
            elif "strengthen" in operation:
                expected = "strengthen"
            else:
                b = extract_modal_strength(base_text,  config["modal_strength"])
                p = extract_modal_strength(probe_text, config["modal_strength"])
                expected = "weaken" if p < b else "strengthen" if p > b else "neutral"

            for model_id in all_model_ids:
                base_rw  = output_index.get((iid, None, model_id), "")
                probe_rw = output_index.get((iid, pid, model_id), "")
                b_str    = extract_modal_strength(base_rw,  config["modal_strength"])
                p_str    = extract_modal_strength(probe_rw, config["modal_strength"])

                if expected == "weaken":
                    passed = p_str <= b_str
                elif expected == "strengthen":
                    passed = p_str >= b_str
                else:
                    passed = True

                probe_scores.append({
                    "item_id"             : iid,
                    "probe_id"            : pid,
                    "probe_family"        : "directional",
                    "model_id"            : model_id,
                    "pass"                : int(passed),
                    "base_modal_strength" : b_str,
                    "probe_modal_strength": p_str,
                    "expected_direction"  : expected,
                    "base_rewrite"        : base_rw,
                    "probe_rewrite"       : probe_rw,
                    "evidence"            : (
                        f"base_str={b_str} probe_str={p_str} "
                        f"expected={expected}"
                    ),
                })
                model_tallies[model_id]["directional"]["total"] += 1
                if passed:
                    model_tallies[model_id]["directional"]["pass"] += 1

    # ── Shortcut probes — Anthropic judge or rule-based fallback ─────────────
    shc_jobs = []
    for item in benchmark:
        iid = item["item_id"]
        for probe in item.get("probe_neighborhoods", []):
            if probe.get("probe_family") != "shortcut":
                continue
            pid        = probe.get("probe_id")
            distractor = probe.get("distractor", "")
            for model_id in all_model_ids:
                base_rw  = output_index.get((iid, None, model_id), "")
                probe_rw = output_index.get((iid, pid, model_id), "")
                shc_jobs.append({
                    "item_id"   : iid,
                    "probe_id"  : pid,
                    "model_id"  : model_id,
                    "base_text" : item["requirement_text"],
                    "distractor": distractor,
                    "base_rw"   : base_rw,
                    "probe_rw"  : probe_rw,
                })

    logging.info(f"Shortcut scoring: {len(shc_jobs)} probes ...")

    use_rule_based = args.no_llm_judge or not config["anthropic_api_key"]

    if use_rule_based:
        logging.info("  Rule-based fallback (no LLM judge).")
        judge_fn = lambda job: rule_based_shortcut_check(
            job["distractor"], job["probe_rw"]
        )
        all_results = [judge_fn(j) for j in shc_jobs]
    else:
        logging.info("  Anthropic claude-sonnet-4-6 judge.")
        all_results = judge_shortcut_batch(
            [j["base_text"]  for j in shc_jobs],
            [j["distractor"] for j in shc_jobs],
            [j["probe_rw"]   for j in shc_jobs],
            config,
        )

    for job, result in zip(shc_jobs, all_results):
        probe_scores.append({
            "item_id"      : job["item_id"],
            "probe_id"     : job["probe_id"],
            "probe_family" : "shortcut",
            "model_id"     : job["model_id"],
            "pass"         : int(result["pass"]),
            "base_rewrite" : job["base_rw"],
            "probe_rewrite": job["probe_rw"],
            "distractor"   : job["distractor"],
            "evidence"     : result["explanation"],
        })
        model_tallies[job["model_id"]]["shortcut"]["total"] += 1
        if result["pass"]:
            model_tallies[job["model_id"]]["shortcut"]["pass"] += 1

    # ── Aggregate model-level scores ─────────────────────────────────────────
    model_scores = []
    for model_id in all_model_ids:
        tallies = model_tallies[model_id]

        def rate(family):
            t = tallies[family]["total"]
            p = tallies[family]["pass"]
            return round(p / t, 4) if t > 0 else None

        inv_rel  = rate("invariance")
        dir_sens = rate("directional")
        shc_res  = rate("shortcut")

        valid    = [s for s in [inv_rel, dir_sens, shc_res] if s is not None]
        global_r = round(sum(valid) / len(valid), 4) if valid else None

        ord_recs        = ordinary_scores.get(model_id, [])
        avg_conformance = (
            round(sum(r["conformance"] for r in ord_recs) / len(ord_recs), 4)
            if ord_recs else None
        )
        avg_meaning = (
            round(sum(r["meaning_preservation"] for r in ord_recs) / len(ord_recs), 4)
            if ord_recs else None
        )
        ordinary_score = (
            round((avg_conformance + avg_meaning) / 2, 4)
            if avg_conformance is not None and avg_meaning is not None
            else None
        )

        model_scores.append({
            "model_id"                : model_id,
            "invariance_reliability"  : inv_rel,
            "directional_sensitivity" : dir_sens,
            "shortcut_resistance"     : shc_res,
            "global_reliability_R"    : global_r,
            "avg_ears_conformance"    : avg_conformance,
            "avg_meaning_preservation": avg_meaning,
            "ordinary_score"          : ordinary_score,
            "n_invariance"            : tallies["invariance"]["total"],
            "n_directional"           : tallies["directional"]["total"],
            "n_shortcut"              : tallies["shortcut"]["total"],
        })

    return probe_scores, model_scores, ordinary_scores


# ─────────────────────────────────────────────────────────────────────────────
# CSV results table
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DISPLAY_NAMES = {
    "rule_baseline": "Rule Baseline",
    "flan_t5_large": "FLAN-T5-Large",
    "flan_t5_xl"   : "FLAN-T5-XL",
    "llama_8b"     : "Llama-3.1-8B",
    "llama_70b"    : "Llama-3.1-70B",
    "qwen_32b"     : "Qwen2.5-32B",
    "gemma_27b"    : "Gemma-3-27B",
    "gpt4"         : "GPT-4.1",
}


def write_results_table(model_scores, output_path):
    sorted_scores = sorted(
        model_scores,
        key=lambda x: x.get("global_reliability_R") or 0,
        reverse=True,
    )
    fieldnames = [
        "Model", "Ord. Score", "EARS Conf.", "Meaning Pres.",
        "Inv. Rel.", "Dir. Sens.", "Shc. Res.", "Global R",
    ]

    def fmt_val(v):
        return f"{v:.3f}" if v is not None else "—"

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in sorted_scores:
            writer.writerow({
                "Model"        : MODEL_DISPLAY_NAMES.get(s["model_id"], s["model_id"]),
                "Ord. Score"   : fmt_val(s.get("ordinary_score")),
                "EARS Conf."   : fmt_val(s.get("avg_ears_conformance")),
                "Meaning Pres.": fmt_val(s.get("avg_meaning_preservation")),
                "Inv. Rel."    : fmt_val(s.get("invariance_reliability")),
                "Dir. Sens."   : fmt_val(s.get("directional_sensitivity")),
                "Shc. Res."    : fmt_val(s.get("shortcut_resistance")),
                "Global R"     : fmt_val(s.get("global_reliability_R")),
            })

    logging.info(f"Results table: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no_llm_judge", action="store_true",
        help="Use rule-based shortcut check (no Anthropic API needed)",
    )
    parser.add_argument(
        "--judge_model", default="claude-sonnet-4-6",
        help="Anthropic model for shortcut judge (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--tau", type=float, default=0.85,
        help="BERTScore F1 threshold for invariance (default: 0.85)",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()
    CONFIG["judge_model"]    = args.judge_model
    CONFIG["tau_invariance"] = args.tau

    with open(CONFIG["benchmark_file"]) as f:
        benchmark = json.load(f)
    logging.info(f"Benchmark: {len(benchmark)} items")

    with open(CONFIG["model_outputs_file"]) as f:
        outputs = json.load(f)
    logging.info(f"Model outputs: {len(outputs)} records")

    probe_scores, model_scores, ordinary_scores = score_all(
        benchmark, outputs, args, CONFIG
    )

    with open(CONFIG["probe_scores_file"], "w") as f:
        json.dump(probe_scores, f, indent=2)
    logging.info(f"Probe scores: {CONFIG['probe_scores_file']} ({len(probe_scores)} records)")

    with open(CONFIG["model_scores_file"], "w") as f:
        json.dump(model_scores, f, indent=2)
    logging.info(f"Model scores: {CONFIG['model_scores_file']}")

    # Console summary
    # Bug fix: renamed inner function to fmt_val to avoid shadowing loop var
    def fmt_val(v):
        return f"{v:.3f}" if v is not None else "  — "

    print()
    print("=" * 72)
    print(f"{'Model':<20} {'Ord':>7} {'Inv':>7} {'Dir':>7} {'Shc':>7} {'R(f)':>7}")
    print("-" * 72)
    for s in sorted(
        model_scores,
        key=lambda x: x.get("global_reliability_R") or 0,
        reverse=True,
    ):
        name = MODEL_DISPLAY_NAMES.get(s["model_id"], s["model_id"])
        print(
            f"{name:<20} "
            f"{fmt_val(s.get('ordinary_score')):>7} "
            f"{fmt_val(s.get('invariance_reliability')):>7} "
            f"{fmt_val(s.get('directional_sensitivity')):>7} "
            f"{fmt_val(s.get('shortcut_resistance')):>7} "
            f"{fmt_val(s.get('global_reliability_R')):>7}"
        )
    print("=" * 72)

    write_results_table(model_scores, CONFIG["results_table_csv"])


if __name__ == "__main__":
    main()
