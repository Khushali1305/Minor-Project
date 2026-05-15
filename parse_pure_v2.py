"""
PURE Dataset Parser v2
=======================
Fixes in v2:
  - Two-pass strategy: detect document type before splitting
  - Requirement ID validation: must be sequential, small (< 5000), and dense
  - Fallback: sentence-level splitting with modal verb filter
  - Hard word count cap: discard records > 80 words (single requirements are short)
  - Year number exclusion: 1900-2099 range excluded from req ID candidates

Run:
    python parse_pure_v2.py \
        --input-dir  /teamspace/studios/this_studio/PURE/ \
        --output-dir /teamspace/studios/this_studio/parsed/
"""

import json
import re
import argparse
from pathlib import Path
from collections import Counter


# ── Constants ──────────────────────────────────────────────────────────────
MAX_WORD_COUNT   = 80    # single requirements should not exceed this
MIN_WORD_COUNT   = 5     # skip fragments shorter than this
MAX_REQ_ID       = 5000  # valid req IDs are small numbers (not years, not section refs)
MIN_REQ_DENSITY  = 3     # need at least this many valid IDs to trust ID-based splitting

# Modal verbs that signal requirement-like text
MODAL_RE = re.compile(
    r'\b(shall|should|must|may|will|can|cannot|can not|is required|are required)\b',
    re.IGNORECASE
)

# Sentence boundary pattern
SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

# 4-digit number pattern
FOUR_DIGIT_RE = re.compile(r'\b(\d{4})\b')

# Year pattern to exclude (1900-2099)
YEAR_RE = re.compile(r'\b(19\d{2}|20\d{2})\b')

# Priority pattern
PRIORITY_RE = re.compile(r'Priority\s+(\d)', re.IGNORECASE)


# ── OCR cleanup ────────────────────────────────────────────────────────────

def fix_ocr(text: str) -> str:
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Text extraction ────────────────────────────────────────────────────────

def extract_text_blocks(json_data: dict) -> list[str]:
    """Return list of text blocks, skipping page numbers and tiny fragments."""
    blocks = []
    for page in json_data.get('pages', []):
        for block in page.get('content', []):
            text = block.get('text', '').strip()
            if re.match(r'^\d{1,3}$', text):   # standalone page number
                continue
            if len(text.split()) < 3:
                continue
            blocks.append(fix_ocr(text))
    return blocks


# ── Document type detection ────────────────────────────────────────────────

def detect_req_ids(blocks: list[str]) -> list[int]:
    """
    Find all 4-digit numbers in the document that look like requirement IDs:
    - Not years (1900-2099)
    - Small (< MAX_REQ_ID)
    - Returns sorted unique list
    """
    years = set(YEAR_RE.findall(' '.join(blocks)))
    full_text = ' '.join(blocks)

    candidates = []
    for m in FOUR_DIGIT_RE.finditer(full_text):
        num_str = m.group(1)
        num = int(num_str)
        if num_str in years:
            continue
        if num >= MAX_REQ_ID:
            continue
        candidates.append(num)

    return sorted(set(candidates))


def is_id_based_document(req_ids: list[int]) -> bool:
    """
    Decide if a document uses sequential requirement IDs.
    Criteria:
      - At least MIN_REQ_DENSITY unique IDs
      - IDs form a roughly sequential series (gaps <= 50 between consecutive IDs)
      - Majority are multiples of 10 (common pattern: 0100, 0110, 0120...)
    """
    if len(req_ids) < MIN_REQ_DENSITY:
        return False

    # Check sequentiality — gaps between consecutive IDs
    gaps = [req_ids[i+1] - req_ids[i] for i in range(len(req_ids)-1)]
    if not gaps:
        return False

    median_gap = sorted(gaps)[len(gaps)//2]
    large_gaps = sum(1 for g in gaps if g > 100)

    # Reject if too many large gaps (document is not ID-based)
    if large_gaps > len(gaps) * 0.4:
        return False

    # Check if IDs are multiples of 10 (EARS numbering convention)
    multiples_of_10 = sum(1 for i in req_ids if i % 10 == 0)
    if multiples_of_10 > len(req_ids) * 0.5:
        return True

    # Accept if median gap is small and reasonable
    if median_gap <= 50 and len(req_ids) >= 5:
        return True

    return False


# ── Splitting strategies ───────────────────────────────────────────────────

def split_by_req_ids(full_text: str, doc_id: str) -> list[dict]:
    """
    Strategy A: Split on validated 4-digit requirement IDs.
    Used for documents like mashboot where reqs are numbered 0100, 0110...
    """
    years = set(YEAR_RE.findall(full_text))

    # Find all valid ID positions
    matches = []
    for m in FOUR_DIGIT_RE.finditer(full_text):
        num_str = m.group(1)
        num = int(num_str)
        if num_str in years:
            continue
        if num >= MAX_REQ_ID:
            continue
        matches.append(m)

    if not matches:
        return []

    records = []
    for i, match in enumerate(matches):
        req_id = match.group(1)
        start  = match.end()
        end    = matches[i+1].start() if i+1 < len(matches) else len(full_text)
        chunk  = full_text[start:end].strip()

        if not chunk:
            continue

        # Extract priority
        priority_m = PRIORITY_RE.search(chunk)
        priority   = int(priority_m.group(1)) if priority_m else None
        text       = PRIORITY_RE.sub('', chunk).strip()

        # Remove leading title fragment
        text = re.sub(
            r'^[A-Z][A-Za-z\s\-/]+?(?=\s+[A-Z][a-z]|\s+The\s|\s+If\s|\s+When\s|\s+WHEN|\s+WHERE)',
            '', text
        ).strip()
        text = re.sub(r'^[-–—]\s*', '', text).strip()
        text = re.sub(r'\s+', ' ', text).strip()

        wc = len(text.split())
        if wc < MIN_WORD_COUNT or wc > MAX_WORD_COUNT:
            continue

        records.append({
            "document_id":        doc_id,
            "requirement_id":     f"{doc_id}_{req_id}",
            "req_id_raw":         req_id,
            "requirement_text":   text,
            "priority":           priority,
            "is_requirement_like": bool(MODAL_RE.search(text)),
            "word_count":         wc,
            "extraction_method":  "id_split",
        })

    return records


def split_by_sentences(blocks: list[str], doc_id: str) -> list[dict]:
    """
    Strategy B: Sentence-level splitting with modal verb filter.
    Used for documents that do not have numeric requirement IDs.
    """
    records = []
    seen    = set()
    sent_idx = 0

    for block in blocks:
        # Split block into sentences
        sentences = SENT_SPLIT_RE.split(block)

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            wc = len(sent.split())
            if wc < MIN_WORD_COUNT or wc > MAX_WORD_COUNT:
                continue

            # Must contain a modal verb
            if not MODAL_RE.search(sent):
                continue

            # Deduplicate
            key = sent[:80].lower()
            if key in seen:
                continue
            seen.add(key)

            records.append({
                "document_id":        doc_id,
                "requirement_id":     f"{doc_id}_S{sent_idx:04d}",
                "req_id_raw":         f"S{sent_idx:04d}",
                "requirement_text":   sent,
                "priority":           None,
                "is_requirement_like": True,  # guaranteed by modal filter
                "word_count":         wc,
                "extraction_method":  "sentence_split",
            })
            sent_idx += 1

    return records


# ── Document parser ────────────────────────────────────────────────────────

def parse_document(json_path: Path) -> tuple[list[dict], dict]:
    doc_id = json_path.stem

    try:
        data = json.loads(json_path.read_text(encoding='utf-8', errors='replace'))
    except json.JSONDecodeError as e:
        return [], {"doc_id": doc_id, "error": str(e)}

    blocks    = extract_text_blocks(data)
    full_text = ' '.join(blocks)
    req_ids   = detect_req_ids(blocks)

    if is_id_based_document(req_ids):
        records = split_by_req_ids(full_text, doc_id)
        method  = "id_split"
    else:
        records = split_by_sentences(blocks, doc_id)
        method  = "sentence_split"

    # Final safety filter — discard anything still too long
    records = [r for r in records if r['word_count'] <= MAX_WORD_COUNT]

    stats = {
        "doc_id":           doc_id,
        "page_count":       len(data.get('pages', [])),
        "extraction_method":method,
        "req_ids_detected": len(req_ids),
        "records_extracted":len(records),
        "req_like_count":   sum(1 for r in records if r['is_requirement_like']),
        "word_count_mean":  round(sum(r['word_count'] for r in records)/max(len(records),1), 1),
    }

    return records, stats


# ── Main ───────────────────────────────────────────────────────────────────

def run(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(input_dir.glob("*.json"))

    print(f"\n{'='*65}")
    print(f"PURE PARSER v2")
    print(f"Input : {input_dir}  ({len(json_files)} files)")
    print(f"{'='*65}")

    all_records  = []
    all_stats    = []
    id_split_docs, sent_split_docs = 0, 0

    for json_path in json_files:
        records, stats = parse_document(json_path)
        if "error" in stats:
            print(f"  [FAIL] {json_path.name}: {stats['error']}")
            continue

        all_records.extend(records)
        all_stats.append(stats)

        method_tag = "[ID ]" if stats['extraction_method'] == 'id_split' else "[SEN]"
        if stats['extraction_method'] == 'id_split':
            id_split_docs += 1
        else:
            sent_split_docs += 1

        print(f"  {method_tag} {stats['doc_id']:<35} "
              f"{stats['records_extracted']:>4} records  "
              f"({stats['req_like_count']} req-like)  "
              f"mean_wc={stats['word_count_mean']}")

    # ── Summary ──
    wc_all      = [r['word_count'] for r in all_records]
    req_like    = [r for r in all_records if r['is_requirement_like']]
    method_dist = Counter(r['extraction_method'] for r in all_records)

    buckets = Counter()
    for w in wc_all:
        if w <= 20:    buckets['01-20'] += 1
        elif w <= 50:  buckets['21-50'] += 1
        elif w <= 80:  buckets['51-80'] += 1

    print(f"\n{'─'*65}")
    print(f"SUMMARY")
    print(f"{'─'*65}")
    print(f"  Documents         : {len(all_stats)}")
    print(f"  ID-split docs     : {id_split_docs}")
    print(f"  Sentence-split docs: {sent_split_docs}")
    print(f"  Total records     : {len(all_records)}")
    print(f"  Requirement-like  : {len(req_like)} ({100*len(req_like)//max(len(all_records),1)}%)")
    print(f"  Extraction methods: {dict(method_dist)}")
    print(f"\n  Word count distribution:")
    for k, v in sorted(buckets.items()):
        bar = "█" * (v * 30 // max(buckets.values(), default=1))
        print(f"    {k}: {bar} {v}")

    print(f"\n  Sample req-like records:")
    samples = [r for r in req_like if 10 <= r['word_count'] <= 40][:6]
    for r in samples:
        print(f"    [{r['extraction_method'][:2].upper()}] "
              f"wc={r['word_count']:>2} | {r['requirement_text'][:100]}")

    # ── Save ──
    out_reqs = output_dir / "pure_requirements.json"
    out_reqs.write_text(json.dumps(all_records, indent=2, ensure_ascii=False))
    print(f"\n  Saved → {out_reqs}  ({len(all_records)} records)")

    report = {
        "dataset":              "PURE",
        "version":              "v2",
        "documents_parsed":     len(all_stats),
        "id_split_docs":        id_split_docs,
        "sentence_split_docs":  sent_split_docs,
        "total_records":        len(all_records),
        "requirement_like":     len(req_like),
        "requirement_like_pct": round(100*len(req_like)/max(len(all_records),1), 1),
        "word_count_mean":      round(sum(wc_all)/max(len(wc_all),1), 2),
        "word_count_distribution": dict(buckets),
        "extraction_method_distribution": dict(method_dist),
        "per_document":         all_stats,
    }
    out_report = output_dir / "pure_report.json"
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"  Saved → {out_report}")


def main():
    parser = argparse.ArgumentParser(description="Parse PURE SRS dataset v2")
    parser.add_argument("--input-dir",  required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run(Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
