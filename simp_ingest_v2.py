#!/usr/bin/env python3
"""
Week 5 Deadline A (v2): Simplification Data Backbone
=====================================================
Change from v1: OneStopEnglish uses FULL articles as
paragraph-level probe items instead of 2-3 sentence excerpts.

Outputs:
  - simp_master_table_v2.json: normalized base items
  - simp_data_card_v2.json: dataset statistics
  - simp_transfer_examples_v2.json: 10 curated examples
"""

import json
import re
import os
from collections import Counter, defaultdict

UPLOAD = "/teamspace/studios/this_studio/transfer_data"
OUTPUT = "/teamspace/studios/this_studio/parse"

# ─── Load Raw Data ────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

tsar = load_jsonl(f"{UPLOAD}/tsar2025.json")
ose = load_jsonl(f"{UPLOAD}/onestop_english.json")
asset_test = load_jsonl(f"{UPLOAD}/asset_simplification_test.json")
asset_val = load_jsonl(f"{UPLOAD}/asset_simplification_val.json")
asset_ratings = load_jsonl(f"{UPLOAD}/asset_ratings.json")

print(f"Loaded: TSAR={len(tsar)}, OSE={len(ose)}, "
      f"ASSET_test={len(asset_test)}, ASSET_val={len(asset_val)}, "
      f"ASSET_ratings={len(asset_ratings)}")

master_items = []

# ─── 1. TSAR 2025 ────────────────────────────────────────────────────────
# 20 unique originals × 2 CEFR targets (a2, b1).
# Paragraph-level texts (~77 words). Explicit CEFR targets.
# Role: Main simplification backbone.

print("\n=== Processing TSAR 2025 ===")

for row in tsar:
    text = row["original"].strip()
    ref = row["reference"].strip()
    wc = len(text.split())

    if wc < 15:
        continue

    master_items.append({
        "item_id": f"TSAR_{row['text_id']}",
        "dataset": "TSAR2025",
        "original": text,
        "source_level": "b2+",
        "target_level": row["target_cefr"].lower(),
        "references": [ref],
        "n_references": 1,
        "text_length_words": wc,
        "text_length_chars": len(text),
        "suitable_for_probing": True,
        "probe_roles": ["invariance", "directional", "shortcut"],
        "provenance": f"TSAR2025_trial_{row['dataset_id']}",
    })

tsar_count = len([i for i in master_items if i["dataset"] == "TSAR2025"])
print(f"  Ingested: {tsar_count} items")

# ─── 2. OneStopEnglish (FULL ARTICLES) ────────────────────────────────────
# 64 articles × 3 levels (Advanced, Intermediate, Elementary).
# Full articles (~685 words). Same topic rewritten at different
# difficulty levels by different writers.
#
# Design: Use FULL Advanced articles as base items.
# Attach Intermediate and Elementary versions as references.
# This preserves all content and avoids alignment assumptions.
#
# Trade-off: Articles are longer than SRS-ProbeCore items (~30 words)
# but this is acceptable because simplification naturally operates
# at paragraph/document level, unlike requirement correction which
# is sentence-level. The paper can note this length difference
# as a domain property, not a limitation.

print("\n=== Processing OneStopEnglish (full articles) ===")

def clean_article(text):
    """Remove article number prefix and clean whitespace."""
    text = re.sub(r'^\d+\s+[^\n]+\n', '', text.strip())
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Group articles by level
ose_by_level = defaultdict(list)
for row in ose:
    level = row["label_text"].strip()
    cleaned = clean_article(row["text"])
    ose_by_level[level].append(cleaned)

# Verify equal counts across levels
for level, articles in ose_by_level.items():
    print(f"  {level}: {len(articles)} articles, "
          f"avg length: {sum(len(a.split()) for a in articles)//len(articles)} words")

n_articles = min(len(ose_by_level.get("Advance", [])),
                 len(ose_by_level.get("Intermediate", [])),
                 len(ose_by_level.get("Elementary", [])))

print(f"  Aligned triplets available: {n_articles}")

for i in range(n_articles):
    adv_text = ose_by_level["Advance"][i]
    int_text = ose_by_level["Intermediate"][i]
    ele_text = ose_by_level["Elementary"][i]

    adv_wc = len(adv_text.split())

    # Skip articles that are too short (corrupted or stub articles)
    if adv_wc < 50:
        continue

    # Advanced → Intermediate (approx B1)
    master_items.append({
        "item_id": f"OSE_{i:03d}_adv2int",
        "dataset": "OneStopEnglish",
        "original": adv_text,
        "source_level": "advanced",
        "target_level": "b1",
        "references": [int_text],
        "n_references": 1,
        "text_length_words": adv_wc,
        "text_length_chars": len(adv_text),
        "suitable_for_probing": True,
        "probe_roles": ["invariance", "directional", "shortcut"],
        "provenance": f"OSE_article_{i}_adv2int",
    })

    # Advanced → Elementary (approx A2)
    master_items.append({
        "item_id": f"OSE_{i:03d}_adv2ele",
        "dataset": "OneStopEnglish",
        "original": adv_text,
        "source_level": "advanced",
        "target_level": "a2",
        "references": [ele_text],
        "n_references": 1,
        "text_length_words": adv_wc,
        "text_length_chars": len(adv_text),
        "suitable_for_probing": True,
        "probe_roles": ["invariance", "directional", "shortcut"],
        "provenance": f"OSE_article_{i}_adv2ele",
    })

ose_count = len([i for i in master_items if i["dataset"] == "OneStopEnglish"])
print(f"  Ingested: {ose_count} items ({ose_count//2} articles × 2 target levels)")

# ─── 3. ASSET ─────────────────────────────────────────────────────────────
# 2,359 sentence-level items with 10 reference simplifications each.
# Role: Invariance + shortcut probes at sentence level.

print("\n=== Processing ASSET ===")

for split_name, split_data in [("test", asset_test), ("val", asset_val)]:
    for i, row in enumerate(split_data):
        text = row["original"].strip()
        simps = [s.strip() for s in row["simplifications"] if s.strip()]
        wc = len(text.split())

        # Filter: skip sentences shorter than 12 words
        if wc < 12:
            continue
        # Filter: skip if all simplifications are identical to original
        if all(s == text for s in simps):
            continue

        master_items.append({
            "item_id": f"ASSET_{split_name}_{i:04d}",
            "dataset": "ASSET",
            "original": text,
            "source_level": "complex",
            "target_level": "simpler",
            "references": simps[:10],
            "n_references": len(simps[:10]),
            "text_length_words": wc,
            "text_length_chars": len(text),
            "suitable_for_probing": wc >= 15,
            "probe_roles": ["invariance", "shortcut"],
            "provenance": f"ASSET_{split_name}_{i}",
        })

asset_count = len([i for i in master_items if i["dataset"] == "ASSET"])
print(f"  Ingested: {asset_count} items")

# ─── 4. Final Filtering And Statistics ────────────────────────────────────

print("\n=== Final Master Table ===")
suitable = [i for i in master_items if i["suitable_for_probing"]]
print(f"  Total items (before filter): {len(master_items)}")
print(f"  Suitable for probing: {len(suitable)}")

ds_counts = Counter(i["dataset"] for i in suitable)
level_counts = Counter(i["target_level"] for i in suitable)
role_counts = Counter()
for i in suitable:
    for r in i["probe_roles"]:
        role_counts[r] += 1

print(f"  By dataset: {dict(ds_counts)}")
print(f"  By target level: {dict(level_counts)}")
print(f"  By probe role: {dict(role_counts)}")
print(f"  Total references available: {sum(i['n_references'] for i in suitable)}")

# Length statistics by dataset
for ds in ["TSAR2025", "OneStopEnglish", "ASSET"]:
    ds_items = [i for i in suitable if i["dataset"] == ds]
    if ds_items:
        lengths = [i["text_length_words"] for i in ds_items]
        print(f"  {ds} text length: "
              f"min={min(lengths)}, max={max(lengths)}, "
              f"mean={sum(lengths)/len(lengths):.0f}, "
              f"median={sorted(lengths)[len(lengths)//2]}")

# ─── 5. Comparison: v1 (excerpts) vs v2 (full articles) ──────────────────

print("\n=== v1 vs v2 Comparison (OneStopEnglish only) ===")
print(f"  v1 (excerpts): 118 items, avg ~65 words, "
      f"probe_roles=['directional'] only")
print(f"  v2 (full articles): {ose_count} items, "
      f"avg ~{sum(len(i['original'].split()) for i in suitable if i['dataset']=='OneStopEnglish')//ose_count} words, "
      f"probe_roles=['invariance','directional','shortcut']")
print(f"  Change: full articles enable ALL three probe families")
print(f"  Change: no alignment assumption at sentence level")
print(f"  Trade-off: items are paragraph-level (~600+ words), "
      f"noted as domain property in paper")

# ─── 6. Build Data Card ──────────────────────────────────────────────────

data_card = {
    "benchmark_name": "Simp-ProbeCore",
    "version": "v2",
    "change_from_v1": "OneStopEnglish uses full articles instead of "
                      "2-3 sentence excerpts. This avoids sentence-level "
                      "alignment assumptions and enables all three probe "
                      "families for OSE items.",
    "total_base_items": len(suitable),
    "by_dataset": dict(ds_counts),
    "by_target_level": dict(level_counts),
    "by_probe_role": dict(role_counts),
    "total_references": sum(i["n_references"] for i in suitable),
    "text_length_summary": {
        "TSAR2025": "paragraph-level, ~77 words avg",
        "OneStopEnglish": "article-level, ~600+ words avg",
        "ASSET": "sentence-level, ~22 words avg",
    },
    "source_corpora": [
        {
            "name": "TSAR 2025",
            "url": "https://huggingface.co/datasets/cardiffnlp/"
                   "TSAR2025_SharedTask_RCTS_Trial-Data",
            "role": "Main backbone: explicit CEFR targets (B2+ to A2/B1)",
            "items": ds_counts.get("TSAR2025", 0),
        },
        {
            "name": "OneStopEnglish",
            "url": "https://huggingface.co/datasets/SetFit/onestop_english",
            "role": "Full-article directional probes with aligned "
                    "difficulty levels, also supports invariance and shortcut",
            "items": ds_counts.get("OneStopEnglish", 0),
        },
        {
            "name": "ASSET",
            "url": "https://huggingface.co/datasets/facebook/asset",
            "role": "Sentence-level invariance and shortcut probes "
                    "with 10 reference simplifications per item",
            "items": ds_counts.get("ASSET", 0),
        },
    ],
    "filtering": {
        "min_words_asset": 12,
        "min_words_ose": 50,
        "min_words_tsar": 15,
        "removed_identical_simplifications": True,
    },
    "target_level_mapping": {
        "a2": "CEFR A2 (elementary)",
        "b1": "CEFR B1 (intermediate)",
        "simpler": "General simplification (no explicit CEFR target)",
    },
}

# ─── 7. Select 10 Transfer Examples ───────────────────────────────────────

print("\n=== Selecting 10 Transfer Examples ===")

examples = []

# 3 from TSAR (1 a2, 1 b1, 1 dual)
tsar_items = [i for i in suitable if i["dataset"] == "TSAR2025"]
tsar_a2 = [i for i in tsar_items if i["target_level"] == "a2"]
tsar_b1 = [i for i in tsar_items if i["target_level"] == "b1"]
if tsar_a2:
    examples.append({**tsar_a2[1], "example_role": "TSAR_a2_target"})
if tsar_b1:
    examples.append({**tsar_b1[1], "example_role": "TSAR_b1_target"})

# 1 from TSAR showing dual target (same original, two CEFR levels)
tsar_by_orig = defaultdict(list)
for i in tsar_items:
    tsar_by_orig[i["original"][:40]].append(i)
for key, items in tsar_by_orig.items():
    if len(items) == 2:
        examples.append({
            "example_role": "TSAR_dual_target_pair",
            "a2_item_id": items[0]["item_id"],
            "b1_item_id": items[1]["item_id"],
            "original_preview": items[0]["original"][:120],
            "a2_reference_preview": items[0]["references"][0][:120],
            "b1_reference_preview": items[1]["references"][0][:120],
        })
        break

# 3 from OneStopEnglish (1 adv→int, 1 adv→ele, 1 showing length)
ose_items = [i for i in suitable if i["dataset"] == "OneStopEnglish"]
ose_b1 = [i for i in ose_items if i["target_level"] == "b1"]
ose_a2 = [i for i in ose_items if i["target_level"] == "a2"]
if ose_b1:
    item = ose_b1[2]  # pick 3rd article for variety
    examples.append({
        "example_role": "OSE_adv_to_intermediate",
        "item_id": item["item_id"],
        "original_preview": item["original"][:200],
        "original_words": item["text_length_words"],
        "reference_preview": item["references"][0][:200],
        "reference_words": len(item["references"][0].split()),
    })
if ose_a2:
    item = ose_a2[2]
    examples.append({
        "example_role": "OSE_adv_to_elementary",
        "item_id": item["item_id"],
        "original_preview": item["original"][:200],
        "original_words": item["text_length_words"],
        "reference_preview": item["references"][0][:200],
        "reference_words": len(item["references"][0].split()),
    })

# 4 from ASSET (short, medium, long, multi-reference diversity)
asset_items = [i for i in suitable if i["dataset"] == "ASSET"
               and i["text_length_words"] >= 15]
asset_sorted = sorted(asset_items, key=lambda i: i["text_length_words"])
if len(asset_sorted) >= 4:
    q = len(asset_sorted) // 4
    for idx, label in [(0, "short"), (q, "medium"),
                       (2*q, "long"), (-1, "complex")]:
        item = asset_sorted[idx]
        examples.append({
            "example_role": f"ASSET_{label}_sentence",
            "item_id": item["item_id"],
            "original": item["original"],
            "text_length_words": item["text_length_words"],
            "n_references": item["n_references"],
            "reference_sample_1": item["references"][0],
            "reference_sample_2": item["references"][1]
                if len(item["references"]) > 1 else None,
        })

examples = examples[:10]
print(f"  Selected {len(examples)} examples")
for ex in examples:
    role = ex.get("example_role", "unknown")
    iid = ex.get("item_id", ex.get("a2_item_id", ""))
    preview = ex.get("original", ex.get("original_preview", ""))[:60]
    print(f"    [{role}] {iid}: {preview}...")

# ─── 8. Save Outputs ──────────────────────────────────────────────────────

master_path = f"{OUTPUT}/simp_master_table_v2.json"
with open(master_path, "w") as f:
    json.dump(suitable, f, indent=2)
print(f"\nSaved: {master_path} ({len(suitable)} items)")

card_path = f"{OUTPUT}/simp_data_card_v2.json"
with open(card_path, "w") as f:
    json.dump(data_card, f, indent=2)
print(f"Saved: {card_path}")

ex_path = f"{OUTPUT}/simp_transfer_examples_v2.json"
with open(ex_path, "w") as f:
    json.dump(examples, f, indent=2)
print(f"Saved: {ex_path} ({len(examples)} examples)")

print(f"\n=== Week 5 Deadline A (v2): COMPLETE ===")
