"""
SRS-ProbeCore Base Item Builder
================================
Week 2 — Deadline A + B

Takes parsed outputs from PURE, Großer, PROMISE_exp and:
  1. Normalizes all three into a unified schema
  2. Filters to high-quality, probe-suitable base items
  3. Balances across documents (no single doc dominates)
  4. Produces SRS-ProbeCore-v1 base items

Filtering criteria:
  - Word count: 8-60 words (single requirement range)
  - Must contain at least one modal verb
  - No pure section headers or table fragments
  - No duplicates (exact + near-duplicate)
  - Max 20 items per source document (balance)
  - Großer: all 249 aligned pairs kept (gold standard)
  - PROMISE: FR only, used for wording diversity + shortcut seeding

Run:
    python build_srs_probecore.py \
        --pure    /teamspace/studios/this_studio/parsed/pure_requirements.json \
        --grosser /teamspace/studios/this_studio/parsed/grosser_requirements.json \
        --promise /teamspace/studios/this_studio/parsed/promise_exp.json \
        --output-dir /teamspace/studios/this_studio/parsed/

Outputs:
    srs_probecore_v1.json        — final base items
    srs_probecore_v1_report.json — stats and audit info
    srs_probecore_v1_sample.jsonl — 50 random samples for manual audit
"""

import json
import re
import argparse
import random
from pathlib import Path
from collections import Counter, defaultdict

# ── Config ─────────────────────────────────────────────────────────────────
MIN_WORDS          = 8
MAX_WORDS          = 60
MAX_PER_DOC        = 20    # max base items from any single PURE document
MAX_PURE_TOTAL     = 600   # cap on PURE contribution
NEAR_DUP_CHARS     = 60    # first N chars used for near-dedup

MODAL_RE = re.compile(
    r'\b(shall|should|must|may|will|can|cannot|is required|are required)\b',
    re.IGNORECASE
)

# Patterns that indicate noise, not requirements
NOISE_PATTERNS = [
    re.compile(r'^\d+[\.\d]*\s+[A-Z]'),          # section header like "3.1 Overview"
    re.compile(r'^(table|figure|appendix)\s', re.IGNORECASE),
    re.compile(r'^(revision history|change log)', re.IGNORECASE),
    re.compile(r'^\w+\s*:\s*\w+'),                # key: value pairs
    re.compile(r'\|\s*\w+\s*\|'),                 # table rows
    re.compile(r'^(note|notes?|warning|caution)\s*:', re.IGNORECASE),
    re.compile(r'^\[SRSreq\s*\d+\]'),             # embedded req ID remnant
]

EARS_KEYWORDS = re.compile(
    r'\b(WHEN|WHILE|WHERE|IF\s+.*?\s+THEN|IF\s+.*?,\s+the)\b'
)

MODAL_STRENGTH = {
    "shall": 3, "must": 3,
    "should": 2, "will": 2,
    "may": 1, "can": 1,
}


# ── Helpers ────────────────────────────────────────────────────────────────

def get_modal(text: str) -> tuple[str, int]:
    """Return (strongest_modal, strength) found in text."""
    text_lower = text.lower()
    best_modal, best_strength = None, 0
    for modal, strength in MODAL_STRENGTH.items():
        if re.search(rf'\b{modal}\b', text_lower):
            if strength > best_strength:
                best_modal, best_strength = modal, strength
    return best_modal or "none", best_strength


def is_noise(text: str) -> bool:
    for pat in NOISE_PATTERNS:
        if pat.search(text):
            return True
    return False


def detect_ears_type(text: str) -> str:
    text_upper = text.upper()
    if text_upper.startswith("WHEN "):       return "EventDriven"
    if text_upper.startswith("WHILE "):      return "StateDriven"
    if text_upper.startswith("WHERE "):      return "OptionalFeatures"
    if text_upper.startswith("IF "):         return "UnwantedBehavior"
    if "SHALL" in text_upper or "SHOULD" in text_upper:
        return "Ubiquitous"
    return "unknown"


def near_dup_key(text: str) -> str:
    """Key for near-duplicate detection."""
    return re.sub(r'\s+', ' ', text.lower().strip())[:NEAR_DUP_CHARS]


# ── Source processors ──────────────────────────────────────────────────────

def process_pure(pure_path: Path) -> list[dict]:
    """
    Load PURE records, filter, balance across documents.
    Returns normalized base items.
    """
    raw = json.loads(pure_path.read_text(encoding='utf-8'))
    print(f"\n  PURE raw records: {len(raw)}")

    # Group by document
    by_doc = defaultdict(list)
    for r in raw:
        by_doc[r['document_id']].append(r)

    base_items = []
    seen_keys  = set()

    for doc_id, recs in sorted(by_doc.items()):
        doc_items = []

        for r in recs:
            text = r.get('requirement_text', '').strip()
            wc   = r.get('word_count', len(text.split()))

            # Word count filter
            if not (MIN_WORDS <= wc <= MAX_WORDS):
                continue

            # Must have modal
            if not MODAL_RE.search(text):
                continue

            # Noise filter
            if is_noise(text):
                continue

            # Near-dedup
            key = near_dup_key(text)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            modal, strength = get_modal(text)

            doc_items.append({
                "item_id":          f"PURE_{doc_id}_{r['req_id_raw']}",
                "source":           "PURE",
                "document_id":      doc_id,
                "requirement_text": text,
                "word_count":       wc,
                "modal":            modal,
                "modal_strength":   strength,
                "ears_type":        detect_ears_type(text),
                "priority":         r.get('priority'),
                "extraction_method":r.get('extraction_method', 'unknown'),
                # Probe fields (empty at base item stage)
                "target_norm":      "EARS",
                "reference_rewrite":None,
                "ears_template_label": None,
                "probe_neighborhoods": [],
            })

        # Balance: take top MAX_PER_DOC by word count preference (15-40 words ideal)
        doc_items.sort(key=lambda x: abs(x['word_count'] - 25))  # prefer ~25 word items
        base_items.extend(doc_items[:MAX_PER_DOC])

    # Global cap on PURE
    random.shuffle(base_items)
    base_items = base_items[:MAX_PURE_TOTAL]

    print(f"  PURE after filtering: {len(base_items)} base items")
    print(f"  Documents represented: {len(set(r['document_id'] for r in base_items))}")
    return base_items


def process_grosser(grosser_path: Path) -> list[dict]:
    """
    Load Großer aligned triples.
    All 249 fully-aligned items kept — these are gold standard.
    Returns normalized base items with reference rewrites.
    """
    raw = json.loads(grosser_path.read_text(encoding='utf-8'))
    print(f"\n  Großer raw records: {len(raw)}")

    base_items = []
    seen_keys  = set()

    for r in raw:
        free_text = str(r.get('free_text') or '').strip()
        ears_text = str(r.get('ears_text') or '').strip()
        wc        = r.get('free_word_count') or len(free_text.split())

        if not free_text or not ears_text:
            continue

        # Word count filter (more lenient for Großer — aerospace reqs can be longer)
        if not (MIN_WORDS <= wc <= 100):
            continue

        # Near-dedup
        key = near_dup_key(free_text)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        modal, strength = get_modal(free_text)
        has_ears = r.get('has_ears', False)
        has_master = r.get('has_master', False)

        base_items.append({
            "item_id":             f"GROSSER_{r['project']}_{r['req_id_raw']}",
            "source":              "Grosser",
            "document_id":         r['project'],
            "requirement_text":    free_text,
            "word_count":          wc,
            "modal":               modal,
            "modal_strength":      strength,
            "ears_type":           r.get('ears_template_label') or detect_ears_type(ears_text),
            "priority":            None,
            "extraction_method":   "aligned_pair",
            # Gold reference rewrites
            "target_norm":         "EARS",
            "reference_rewrite":   ears_text,
            "ears_template_label": r.get('ears_template_label'),
            "master_rewrite":      r.get('master_text'),
            "master_template_label": r.get('master_template_label'),
            "fully_aligned":       has_ears and has_master,
            "probe_neighborhoods": [],
        })

    print(f"  Großer after filtering: {len(base_items)} base items")
    print(f"  Projects represented: {sorted(set(r['document_id'] for r in base_items))}")
    return base_items


def process_promise(promise_path: Path) -> list[dict]:
    """
    Load PROMISE_exp.
    FR only — used for wording diversity and shortcut probe seeding.
    Keep max 150 items, balanced across projects.
    """
    raw = json.loads(promise_path.read_text(encoding='utf-8'))
    print(f"\n  PROMISE raw records: {len(raw)}")

    fr_recs = [r for r in raw if r.get('class_family') == 'FR']
    print(f"  PROMISE FR records: {len(fr_recs)}")

    base_items = []
    seen_keys  = set()
    by_project = defaultdict(list)

    for r in fr_recs:
        text = str(r.get('requirement_text', '')).strip()
        wc   = r.get('word_count', len(text.split()))

        if not (MIN_WORDS <= wc <= MAX_WORDS):
            continue
        if not MODAL_RE.search(text):
            continue
        if is_noise(text):
            continue

        key = near_dup_key(text)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        modal, strength = get_modal(text)
        by_project[r['project_id']].append({
            "item_id":          f"PROMISE_{r['project_id']}_{len(base_items):04d}",
            "source":           "PROMISE_exp",
            "document_id":      f"PROMISE_proj_{r['project_id']}",
            "requirement_text": text,
            "word_count":       wc,
            "modal":            modal,
            "modal_strength":   strength,
            "ears_type":        detect_ears_type(text),
            "priority":         None,
            "extraction_method":"fr_filter",
            "target_norm":      "EARS",
            "reference_rewrite":None,
            "ears_template_label": None,
            "req_class":        r.get('class', 'F'),
            "probe_neighborhoods": [],
        })

    # Balance across projects: max 5 per project, total 150
    for proj_items in by_project.values():
        random.shuffle(proj_items)
        base_items.extend(proj_items[:5])

    random.shuffle(base_items)
    base_items = base_items[:150]

    print(f"  PROMISE after filtering: {len(base_items)} base items")
    return base_items


# ── Quality report ─────────────────────────────────────────────────────────

def print_report(all_items: list[dict]):
    source_dist  = Counter(r['source'] for r in all_items)
    modal_dist   = Counter(r['modal'] for r in all_items)
    ears_dist    = Counter(r['ears_type'] for r in all_items)
    strength_dist= Counter(r['modal_strength'] for r in all_items)
    wc_all       = [r['word_count'] for r in all_items]

    has_ref      = sum(1 for r in all_items if r.get('reference_rewrite'))
    fully_aligned= sum(1 for r in all_items if r.get('fully_aligned'))

    print(f"\n{'='*65}")
    print(f"SRS-PROBECORE v1 — BASE ITEMS REPORT")
    print(f"{'='*65}")
    print(f"  Total base items      : {len(all_items)}")
    print(f"  With reference rewrite: {has_ref}")
    print(f"  Fully aligned (F+E+M) : {fully_aligned}")

    print(f"\n  Source distribution:")
    for src, cnt in sorted(source_dist.items(), key=lambda x: -x[1]):
        bar = "█" * (cnt * 30 // max(source_dist.values()))
        print(f"    {src:<15} {bar} {cnt}")

    print(f"\n  Modal distribution:")
    for modal, cnt in sorted(modal_dist.items(), key=lambda x: -x[1]):
        print(f"    {modal:<12}: {cnt}")

    print(f"\n  Modal strength (1=may, 2=should, 3=shall):")
    for s, cnt in sorted(strength_dist.items()):
        print(f"    strength {s}: {cnt}")

    print(f"\n  EARS type distribution:")
    for t, cnt in sorted(ears_dist.items(), key=lambda x: -x[1]):
        print(f"    {t:<20}: {cnt}")

    print(f"\n  Word count stats:")
    print(f"    Min   : {min(wc_all)}")
    print(f"    Max   : {max(wc_all)}")
    print(f"    Mean  : {sum(wc_all)/len(wc_all):.1f}")
    print(f"    Median: {sorted(wc_all)[len(wc_all)//2]}")

    print(f"\n  Sample base items:")
    samples = random.sample([r for r in all_items if 10 <= r['word_count'] <= 35], 
                             min(6, len(all_items)))
    for r in samples:
        ref = f" → {r['reference_rewrite'][:60]}..." if r.get('reference_rewrite') else ""
        print(f"    [{r['source']:<10}] [{r['modal']:<6}] {r['requirement_text'][:80]}")
        if ref:
            print(f"    {'':>12} {ref}")


# ── Main ───────────────────────────────────────────────────────────────────

def run(pure_path, grosser_path, promise_path, output_dir, seed=42):
    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"SRS-PROBECORE v1 BUILDER")
    print(f"{'='*65}")

    # Process each source
    pure_items    = process_pure(Path(pure_path))
    grosser_items = process_grosser(Path(grosser_path))
    promise_items = process_promise(Path(promise_path))

    # Combine — Großer first (highest quality), then PURE, then PROMISE
    all_items = grosser_items + pure_items + promise_items

    # Final global dedup pass
    seen = set()
    deduped = []
    for item in all_items:
        key = near_dup_key(item['requirement_text'])
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    print(f"\n  After global dedup: {len(deduped)} items "
          f"(removed {len(all_items)-len(deduped)} duplicates)")

    print_report(deduped)

    # Save
    out_path = output_dir / "srs_probecore_v1.json"
    out_path.write_text(json.dumps(deduped, indent=2, ensure_ascii=False))
    print(f"\n  Saved → {out_path}  ({len(deduped)} items)")

    # Report
    report = {
        "version":          "v1",
        "total_items":      len(deduped),
        "source_distribution": dict(Counter(r['source'] for r in deduped)),
        "modal_distribution":  dict(Counter(r['modal'] for r in deduped)),
        "ears_distribution":   dict(Counter(r['ears_type'] for r in deduped)),
        "with_reference_rewrite": sum(1 for r in deduped if r.get('reference_rewrite')),
        "fully_aligned":    sum(1 for r in deduped if r.get('fully_aligned')),
        "word_count_mean":  round(sum(r['word_count'] for r in deduped)/len(deduped), 2),
        "word_count_median":sorted(r['word_count'] for r in deduped)[len(deduped)//2],
    }
    report_path = output_dir / "srs_probecore_v1_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"  Saved → {report_path}")

    # 50-item sample for manual audit
    sample = random.sample(deduped, min(50, len(deduped)))
    sample_path = output_dir / "srs_probecore_v1_sample.jsonl"
    with open(sample_path, 'w', encoding='utf-8') as f:
        for item in sample:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  Saved → {sample_path}  (50-item manual audit sample)")

    return deduped


def main():
    parser = argparse.ArgumentParser(description="Build SRS-ProbeCore v1 base items")
    parser.add_argument("--pure",       required=True)
    parser.add_argument("--grosser",    required=True)
    parser.add_argument("--promise",    required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    run(args.pure, args.grosser, args.promise, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
