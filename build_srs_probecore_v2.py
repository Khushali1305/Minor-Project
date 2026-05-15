"""
SRS-ProbeCore Base Item Builder v2
====================================
Week 2 — Deadline B

Changes from v1:
  - 10 new filter rules derived from 50-item manual audit
  - FIX rules: auto-clean salvageable items
  - Stricter subject filter: reject pronoun-subject sentences
  - Reject context-dependent openers
  - Reject truncated list items
  - Reject glossary definitions
  - Reject deleted section markers
  - Reject external reference dependencies
  - Reject document header noise
  - Auto-fix: strip leading connective words
  - Auto-fix: strip OCR footnote artifacts

Run:
    python build_srs_probecore_v2.py \
        --pure    /teamspace/studios/this_studio/parsed/pure_requirements.json \
        --grosser /teamspace/studios/this_studio/parsed/grosser_requirements.json \
        --promise /teamspace/studios/this_studio/parsed/promise_exp.json \
        --output-dir /teamspace/studios/this_studio/parsed/
"""

import json
import re
import argparse
import random
from pathlib import Path
from collections import Counter, defaultdict

# ── Config ─────────────────────────────────────────────────────────────────
MIN_WORDS      = 8
MAX_WORDS      = 60
MAX_PER_DOC    = 20
MAX_PURE_TOTAL = 600
NEAR_DUP_CHARS = 60

MODAL_RE = re.compile(
    r'\b(shall|should|must|may|will|can|cannot|is required|are required)\b',
    re.IGNORECASE
)

MODAL_STRENGTH = {
    "shall": 3, "must": 3,
    "should": 2, "will": 2,
    "may": 1, "can": 1,
}

# ── Reject filters (audit-derived) ─────────────────────────────────────────

# 1. Context-dependent openers — sentence depends on prior text
CONTEXT_OPENERS = re.compile(
    r'^(for example|therefore|moreover|however|but if|additionally|'
    r'furthermore|in addition|as a result|consequently|thus|hence|'
    r'as mentioned|as described|as noted|as stated|as specified above|'
    r'as follows|in this case|in such cases|this means)\b',
    re.IGNORECASE
)

# 2. Pronoun subjects with no referent
PRONOUN_SUBJECT = re.compile(
    r'^(it |he |she |they |them |this |these |those |its )',
    re.IGNORECASE
)

# 3. Deleted section markers
DELETED_MARKER = re.compile(
    r'intentionally deleted|deliberately deleted|this (page|section) (is )?intentionally',
    re.IGNORECASE
)

# 4. Glossary / definition pattern: "Term – definition" or "Term: definition"
GLOSSARY_PATTERN = re.compile(
    r'^[A-Z][A-Za-z\s/\-]{2,40}[–—-]\s+[A-Z]',
)

# 5. Document header noise embedded in text
DOC_HEADER_NOISE = re.compile(
    r'\bSRS\s+\d+\b|\bpage\s+\d+\b|\bversion\s+\d+\.\d+\b',
    re.IGNORECASE
)

# 6. External reference dependencies
EXTERNAL_REF = re.compile(
    r'\[[\w\-\s]+\]|appendix\s+[A-Z]\b|section\s+\d+\.\d+|'
    r'per\s+\[|as\s+per\s+\[|see\s+\[|refer\s+to\s+\[',
    re.IGNORECASE
)

# 7. Truncated list items — ends with enumeration start
TRUNCATED_LIST = re.compile(
    r':\s*\d+\.\s*$|,\s*\d+\.\s*$|\(\s*[Mm]\s*\)\s*\d+\.',
)

# 8. ID prefix remnants at sentence start
ID_PREFIX = re.compile(
    r'^[A-Z0-9]{2,10}-[A-Z]{2,5}-[A-Z]{2,5}\d+\s+',  # C2C-IF-IS20
)

# 9. Community / process / non-system statements
NON_SYSTEM = re.compile(
    r'^(workshops? will|all people are|everyone (that|who)|'
    r'users? are (assumed|expected|required) to have|'
    r'this document (explains|describes|outlines)|'
    r'the (whole|entire) project is based)',
    re.IGNORECASE
)

# 10. OCR footnote artifact — ".NN " in middle of text
OCR_FOOTNOTE = re.compile(r'\.\d{1,3}\s+[A-Z]')

# ── Fix rules (auto-clean salvageable items) ───────────────────────────────

LEADING_CONNECTIVES = re.compile(
    r'^(therefore,?\s+|moreover,?\s+|however,?\s+|additionally,?\s+|'
    r'furthermore,?\s+|also,?\s+|note that,?\s+|requirement specification:\s+)',
    re.IGNORECASE
)

CURLY_QUOTES = str.maketrans('\u201c\u201d\u2018\u2019', '"\'"\'')


ID_PREFIX_STRIP = re.compile(r'^[A-Z0-9]{2,10}-[A-Z]{2,5}-[A-Z0-9]{2,5}\d*\s+')


def apply_fixes(text: str) -> str:
    """Auto-fix salvageable text issues."""
    # Fix curly quotes
    text = text.translate(CURLY_QUOTES)

    # Strip leading connective words
    text = LEADING_CONNECTIVES.sub('', text).strip()

    # Capitalise first letter after stripping
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    # Strip ID prefix remnants
    text = ID_PREFIX_STRIP.sub('', text).strip()

    # Fix OCR footnote artifacts — take only text before the artifact
    ocr_match = OCR_FOOTNOTE.search(text)
    if ocr_match:
        # Find the period before the footnote number
        cut = text.rfind('.', 0, ocr_match.start()) + 1
        if cut > MIN_WORDS:
            text = text[:cut].strip()

    return text


# ── Core reject check ──────────────────────────────────────────────────────

def should_reject(text: str) -> tuple[bool, str]:
    """
    Returns (reject: bool, reason: str).
    Apply AFTER fixes so we reject based on cleaned text.
    """
    if DELETED_MARKER.search(text):
        return True, "deleted_marker"

    if CONTEXT_OPENERS.match(text):
        return True, "context_opener"

    if PRONOUN_SUBJECT.match(text):
        return True, "pronoun_subject"

    if GLOSSARY_PATTERN.match(text):
        return True, "glossary_definition"

    if DOC_HEADER_NOISE.search(text):
        return True, "doc_header_noise"

    if TRUNCATED_LIST.search(text):
        return True, "truncated_list"

    if NON_SYSTEM.match(text):
        return True, "non_system_statement"

    # External refs: only reject for PURE (Großer items are expected to have some)
    # handled separately in process_grosser

    return False, ""


def get_modal(text: str) -> tuple[str, int]:
    text_lower = text.lower()
    best_modal, best_strength = None, 0
    for modal, strength in MODAL_STRENGTH.items():
        if re.search(rf'\b{modal}\b', text_lower):
            if strength > best_strength:
                best_modal, best_strength = modal, strength
    return best_modal or "none", best_strength


def detect_ears_type(text: str) -> str:
    t = text.upper()
    if t.startswith("WHEN "):   return "EventDriven"
    if t.startswith("WHILE "):  return "StateDriven"
    if t.startswith("WHERE "):  return "OptionalFeatures"
    if t.startswith("IF "):     return "UnwantedBehavior"
    if "SHALL" in t or "SHOULD" in t: return "Ubiquitous"
    return "unknown"


def near_dup_key(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower().strip())[:NEAR_DUP_CHARS]


# ── Source processors ──────────────────────────────────────────────────────

def process_pure(pure_path: Path, seen_keys: set) -> tuple[list, dict]:
    raw = json.loads(pure_path.read_text(encoding='utf-8'))
    print(f"\n  PURE raw records: {len(raw)}")

    by_doc = defaultdict(list)
    for r in raw:
        by_doc[r['document_id']].append(r)

    base_items = []
    reject_counts = Counter()

    for doc_id, recs in sorted(by_doc.items()):
        doc_items = []

        for r in recs:
            text = r.get('requirement_text', '').strip()
            wc   = r.get('word_count', len(text.split()))

            if not (MIN_WORDS <= wc <= MAX_WORDS):
                continue
            if not MODAL_RE.search(text):
                continue

            # Apply fixes first
            text = apply_fixes(text)
            wc   = len(text.split())

            if not (MIN_WORDS <= wc <= MAX_WORDS):
                reject_counts["word_count_after_fix"] += 1
                continue

            # Then reject check
            reject, reason = should_reject(text)
            if reject:
                reject_counts[reason] += 1
                continue

            # External ref check for PURE
            if EXTERNAL_REF.search(text):
                reject_counts["external_ref"] += 1
                continue

            key = near_dup_key(text)
            if key in seen_keys:
                reject_counts["duplicate"] += 1
                continue
            seen_keys.add(key)

            modal, strength = get_modal(text)

            doc_items.append({
                "item_id":           f"PURE_{doc_id}_{r['req_id_raw']}",
                "source":            "PURE",
                "document_id":       doc_id,
                "requirement_text":  text,
                "word_count":        len(text.split()),
                "modal":             modal,
                "modal_strength":    strength,
                "ears_type":         detect_ears_type(text),
                "priority":          r.get('priority'),
                "extraction_method": r.get('extraction_method', 'unknown'),
                "target_norm":       "EARS",
                "reference_rewrite": None,
                "ears_template_label": None,
                "probe_neighborhoods": [],
            })

        doc_items.sort(key=lambda x: abs(x['word_count'] - 25))
        base_items.extend(doc_items[:MAX_PER_DOC])

    random.shuffle(base_items)
    base_items = base_items[:MAX_PURE_TOTAL]

    print(f"  PURE after filtering : {len(base_items)} base items")
    print(f"  Rejection breakdown  : {dict(reject_counts)}")
    print(f"  Documents represented: {len(set(r['document_id'] for r in base_items))}")
    return base_items, reject_counts


def process_grosser(grosser_path: Path, seen_keys: set) -> tuple[list, dict]:
    raw = json.loads(grosser_path.read_text(encoding='utf-8'))
    print(f"\n  Großer raw records: {len(raw)}")

    base_items = []
    reject_counts = Counter()

    for r in raw:
        free_text = str(r.get('free_text') or '').strip()
        ears_text = str(r.get('ears_text') or '').strip()
        wc        = r.get('free_word_count') or len(free_text.split())

        if not free_text or not ears_text:
            reject_counts["missing_pair"] += 1
            continue

        if not (MIN_WORDS <= wc <= 100):
            reject_counts["word_count"] += 1
            continue

        # Apply fixes
        free_text = apply_fixes(free_text)

        # Reject deleted markers only — keep external refs for Großer
        # because aerospace reqs legitimately reference standards
        reject, reason = should_reject(free_text)
        if reject and reason not in ("external_ref",):
            reject_counts[reason] += 1
            continue

        key = near_dup_key(free_text)
        if key in seen_keys:
            reject_counts["duplicate"] += 1
            continue
        seen_keys.add(key)

        modal, strength = get_modal(free_text)

        base_items.append({
            "item_id":              f"GROSSER_{r['project']}_{r['req_id_raw']}",
            "source":               "Grosser",
            "document_id":          r['project'],
            "requirement_text":     free_text,
            "word_count":           len(free_text.split()),
            "modal":                modal,
            "modal_strength":       strength,
            "ears_type":            r.get('ears_template_label') or detect_ears_type(ears_text),
            "priority":             None,
            "extraction_method":    "aligned_pair",
            "target_norm":          "EARS",
            "reference_rewrite":    ears_text,
            "ears_template_label":  r.get('ears_template_label'),
            "master_rewrite":       r.get('master_text'),
            "master_template_label":r.get('master_template_label'),
            "fully_aligned":        r.get('has_ears') and r.get('has_master'),
            "probe_neighborhoods":  [],
        })

    print(f"  Großer after filtering : {len(base_items)} base items")
    print(f"  Rejection breakdown    : {dict(reject_counts)}")
    return base_items, reject_counts


def process_promise(promise_path: Path, seen_keys: set) -> tuple[list, dict]:
    raw = json.loads(promise_path.read_text(encoding='utf-8'))
    fr_recs = [r for r in raw if r.get('class_family') == 'FR']
    print(f"\n  PROMISE FR records: {len(fr_recs)}")

    base_items = []
    reject_counts = Counter()
    by_project = defaultdict(list)

    for r in fr_recs:
        text = str(r.get('requirement_text', '')).strip()
        wc   = r.get('word_count', len(text.split()))

        if not (MIN_WORDS <= wc <= MAX_WORDS):
            continue
        if not MODAL_RE.search(text):
            continue

        text = apply_fixes(text)
        wc   = len(text.split())

        if not (MIN_WORDS <= wc <= MAX_WORDS):
            reject_counts["word_count_after_fix"] += 1
            continue

        reject, reason = should_reject(text)
        if reject:
            reject_counts[reason] += 1
            continue

        key = near_dup_key(text)
        if key in seen_keys:
            reject_counts["duplicate"] += 1
            continue
        seen_keys.add(key)

        modal, strength = get_modal(text)

        by_project[r['project_id']].append({
            "item_id":           f"PROMISE_{r['project_id']}_{len(base_items):04d}",
            "source":            "PROMISE_exp",
            "document_id":       f"PROMISE_proj_{r['project_id']}",
            "requirement_text":  text,
            "word_count":        len(text.split()),
            "modal":             modal,
            "modal_strength":    strength,
            "ears_type":         detect_ears_type(text),
            "priority":          None,
            "extraction_method": "fr_filter",
            "target_norm":       "EARS",
            "reference_rewrite": None,
            "ears_template_label": None,
            "req_class":         r.get('class', 'F'),
            "probe_neighborhoods": [],
        })

    for proj_items in by_project.values():
        random.shuffle(proj_items)
        base_items.extend(proj_items[:5])

    random.shuffle(base_items)
    base_items = base_items[:150]

    print(f"  PROMISE after filtering: {len(base_items)} base items")
    print(f"  Rejection breakdown    : {dict(reject_counts)}")
    return base_items, reject_counts


# ── Report ─────────────────────────────────────────────────────────────────

def print_report(items: list):
    source_dist   = Counter(r['source'] for r in items)
    modal_dist    = Counter(r['modal'] for r in items)
    ears_dist     = Counter(r['ears_type'] for r in items)
    strength_dist = Counter(r['modal_strength'] for r in items)
    wc_all        = [r['word_count'] for r in items]
    has_ref       = sum(1 for r in items if r.get('reference_rewrite'))
    fully_aligned = sum(1 for r in items if r.get('fully_aligned'))

    print(f"\n{'='*65}")
    print(f"SRS-PROBECORE v2 — FINAL REPORT")
    print(f"{'='*65}")
    print(f"  Total base items      : {len(items)}")
    print(f"  With reference rewrite: {has_ref}")
    print(f"  Fully aligned (F+E+M) : {fully_aligned}")

    print(f"\n  Source distribution:")
    for src, cnt in sorted(source_dist.items(), key=lambda x: -x[1]):
        bar = "█" * (cnt * 25 // max(source_dist.values()))
        print(f"    {src:<15} {bar} {cnt}")

    print(f"\n  Modal distribution:")
    for modal, cnt in sorted(modal_dist.items(), key=lambda x: -x[1]):
        print(f"    {modal:<10}: {cnt}")

    print(f"\n  Modal strength (1=may/can, 2=should/will, 3=shall/must):")
    for s, cnt in sorted(strength_dist.items()):
        print(f"    strength {s}: {cnt}")

    print(f"\n  EARS type distribution:")
    for t, cnt in sorted(ears_dist.items(), key=lambda x: -x[1]):
        print(f"    {t:<25}: {cnt}")

    print(f"\n  Word count:")
    print(f"    Min   : {min(wc_all)}")
    print(f"    Max   : {max(wc_all)}")
    print(f"    Mean  : {sum(wc_all)/len(wc_all):.1f}")
    print(f"    Median: {sorted(wc_all)[len(wc_all)//2]}")

    print(f"\n  Samples (15-35 words, with reference rewrite):")
    samples = [r for r in items if 15 <= r['word_count'] <= 35
               and r.get('reference_rewrite')][:4]
    for r in samples:
        print(f"\n    [{r['source']:<10}] {r['requirement_text']}")
        print(f"    {'→ EARS':>12} {r['reference_rewrite']}")


# ── Main ───────────────────────────────────────────────────────────────────

def run(pure_path, grosser_path, promise_path, output_dir, seed=42):
    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"SRS-PROBECORE v2 BUILDER")
    print(f"{'='*65}")

    seen_keys = set()

    grosser_items, g_rejects = process_grosser(Path(grosser_path), seen_keys)
    pure_items,    p_rejects = process_pure(Path(pure_path), seen_keys)
    promise_items, r_rejects = process_promise(Path(promise_path), seen_keys)

    # Combine: Großer first (highest quality), then PURE, then PROMISE
    all_items = grosser_items + pure_items + promise_items

    print_report(all_items)

    # Save
    out_path = output_dir / "srs_probecore_v2.json"
    out_path.write_text(json.dumps(all_items, indent=2, ensure_ascii=False))
    print(f"\n  Saved → {out_path}  ({len(all_items)} items)")

    report = {
        "version":            "v2",
        "total_items":        len(all_items),
        "source_distribution":dict(Counter(r['source'] for r in all_items)),
        "modal_distribution": dict(Counter(r['modal'] for r in all_items)),
        "ears_distribution":  dict(Counter(r['ears_type'] for r in all_items)),
        "with_reference_rewrite": sum(1 for r in all_items if r.get('reference_rewrite')),
        "fully_aligned":      sum(1 for r in all_items if r.get('fully_aligned')),
        "word_count_mean":    round(sum(r['word_count'] for r in all_items)/len(all_items), 2),
        "word_count_median":  sorted(r['word_count'] for r in all_items)[len(all_items)//2],
        "audit_results": {
            "sample_size": 50,
            "keep": 38,
            "fix":  4,
            "reject": 8,
            "accept_rate_pct": 84,
        },
        "rejection_breakdown": {
            "pure":    dict(p_rejects),
            "grosser": dict(g_rejects),
            "promise": dict(r_rejects),
        }
    }

    report_path = output_dir / "srs_probecore_v2_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"  Saved → {report_path}")

    # 50-item sample for verification
    sample = random.sample(all_items, min(50, len(all_items)))
    sample_path = output_dir / "srs_probecore_v2_sample.jsonl"
    with open(sample_path, 'w', encoding='utf-8') as f:
        for item in sample:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  Saved → {sample_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pure",       required=True)
    parser.add_argument("--grosser",    required=True)
    parser.add_argument("--promise",    required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.pure, args.grosser, args.promise, args.output_dir, args.seed)


if __name__ == "__main__":
    main()