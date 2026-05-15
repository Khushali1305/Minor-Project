"""
PURE Dataset Parser
====================
Actual format (confirmed from file inspection):
  - PDF-to-JSON conversion output
  - Each page has content blocks of type "paragraph"
  - Requirements are embedded in blocks with pattern:
      DDDD [Title] Requirement text. Priority N
  - Multiple requirements concatenated per block
  - OCR artifacts: hyphenated line breaks, merged words

Strategy:
  1. Concatenate all text blocks across all pages per document
  2. Fix OCR hyphenation artifacts
  3. Split on 4-digit requirement IDs
  4. Extract requirement text and priority
  5. Apply requirement-like filter

Run:
    python parse_pure.py \
        --input-dir /teamspace/studios/this_studio/PURE/ \
        --output-dir /teamspace/studios/this_studio/parsed/

Outputs:
    pure_requirements.json   — all extracted requirements
    pure_report.json         — stats per document + overall
"""

import json
import re
import argparse
from pathlib import Path
from collections import Counter, defaultdict


# ── Requirement ID pattern: 4 digits at word boundary ──────────────────────
REQ_ID_PATTERN = re.compile(r'\b(\d{4})\b')

# ── Priority pattern at end of requirement ──────────────────────────────────
PRIORITY_PATTERN = re.compile(r'Priority\s+(\d)', re.IGNORECASE)

# ── Section header pattern — skip these as requirement text ────────────────
SECTION_HEADER_PATTERN = re.compile(
    r'^\d+(\.\d+)*\s+[A-Z][a-zA-Z\s]+$'
)

# ── Modal verbs that signal requirement-like text ──────────────────────────
MODALS = re.compile(
    r'\b(shall|should|must|may|will|can|cannot|can not|is required|are required)\b',
    re.IGNORECASE
)


def fix_ocr_artifacts(text: str) -> str:
    """
    Fix common OCR artifacts from PDF extraction:
    1. Hyphenated line breaks: 'require-\nments' → 'requirements'
    2. Missing spaces before capital letters in merged words
       (conservative — only split obvious CamelCase runs)
    """
    # Fix hyphenated line breaks
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'-\s{2,}', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_text_blocks(json_data: dict) -> str:
    """
    Pull all text blocks from all pages in order,
    skipping standalone page numbers and very short fragments.
    """
    blocks = []
    for page in json_data.get('pages', []):
        for block in page.get('content', []):
            text = block.get('text', '').strip()
            # Skip page numbers (standalone 1-3 digit strings)
            if re.match(r'^\d{1,3}$', text):
                continue
            # Skip very short fragments
            if len(text.split()) < 3:
                continue
            blocks.append(text)
    return ' '.join(blocks)


def split_into_requirement_chunks(full_text: str) -> list[tuple[str, str]]:
    """
    Split concatenated text into (req_id, chunk) pairs.
    Strategy: find all 4-digit IDs and use them as split points.

    Returns list of (req_id, raw_chunk) tuples.
    """
    # Find all positions of 4-digit IDs
    matches = list(REQ_ID_PATTERN.finditer(full_text))

    if not matches:
        return []

    chunks = []
    for i, match in enumerate(matches):
        req_id = match.group(1)
        start = match.end()
        # End is either the next 4-digit ID or end of text
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        chunk = full_text[start:end].strip()
        if chunk:
            chunks.append((req_id, chunk))

    return chunks


def parse_chunk(req_id: str, chunk: str, doc_id: str) -> dict | None:
    """
    Parse one (req_id, chunk) into a structured requirement record.
    
    Chunk format examples:
      "Purpose The external email system is to provide messaging. Priority 2"
      "User Account Creation - New user accounts can be created. Priority 1"
      "The system shall refresh the display every 60 seconds. Priority 1"
    """
    # Extract priority
    priority_match = PRIORITY_PATTERN.search(chunk)
    priority = int(priority_match.group(1)) if priority_match else None

    # Remove priority marker from text
    text = PRIORITY_PATTERN.sub('', chunk).strip()

    # Remove leading title-like fragment (Title Case phrase before first sentence)
    # Pattern: "User Account Creation - " or "Purpose " at start
    text = re.sub(r'^[A-Z][A-Za-z\s\-]+?(?=\s+[A-Z][a-z]|\s+The\s|\s+If\s|\s+When\s)', '', text).strip()
    text = re.sub(r'^[-–—]\s*', '', text).strip()

    # Clean up residual whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Skip if too short or clearly not a requirement
    if len(text.split()) < 4:
        return None

    # Skip pure section headers
    if SECTION_HEADER_PATTERN.match(text):
        return None

    # Determine if requirement-like
    is_req_like = bool(MODALS.search(text))

    return {
        "document_id":      doc_id,
        "requirement_id":   f"{doc_id}_{req_id}",
        "req_id_raw":       req_id,
        "requirement_text": text,
        "priority":         priority,
        "is_requirement_like": is_req_like,
        "word_count":       len(text.split()),
        "char_count":       len(text),
    }


def parse_document(json_path: Path) -> tuple[list[dict], dict]:
    """Parse one PURE JSON file into requirement records."""
    doc_id = json_path.stem  # e.g. "2010 - mashboot"

    try:
        data = json.loads(json_path.read_text(encoding='utf-8', errors='replace'))
    except json.JSONDecodeError as e:
        return [], {"doc_id": doc_id, "error": str(e)}

    # Step 1: Extract and clean full text
    raw_text = extract_text_blocks(data)
    clean_text = fix_ocr_artifacts(raw_text)

    # Step 2: Split on requirement IDs
    chunks = split_into_requirement_chunks(clean_text)

    if not chunks:
        return [], {
            "doc_id": doc_id,
            "total_blocks": sum(len(p.get('content',[])) for p in data.get('pages',[])),
            "req_chunks_found": 0,
            "warning": "No 4-digit requirement IDs found"
        }

    # Step 3: Parse each chunk
    records = []
    skipped = 0
    for req_id, chunk in chunks:
        record = parse_chunk(req_id, chunk, doc_id)
        if record:
            records.append(record)
        else:
            skipped += 1

    doc_stats = {
        "doc_id":           doc_id,
        "page_count":       len(data.get('pages', [])),
        "req_chunks_found": len(chunks),
        "records_parsed":   len(records),
        "records_skipped":  skipped,
        "req_like_count":   sum(1 for r in records if r['is_requirement_like']),
        "priority_dist":    dict(Counter(
            str(r['priority']) for r in records if r['priority']
        )),
    }

    return records, doc_stats


def run(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    print(f"\n{'='*60}")
    print(f"PURE DATASET PARSER")
    print(f"Input  : {input_dir}")
    print(f"Files  : {len(json_files)} JSON documents")
    print(f"{'='*60}")

    all_records = []
    all_doc_stats = []
    failed_docs = []

    for json_path in json_files:
        records, stats = parse_document(json_path)
        if "error" in stats:
            failed_docs.append(stats)
            print(f"  [FAIL] {json_path.name}: {stats['error']}")
        else:
            all_records.extend(records)
            all_doc_stats.append(stats)
            req_like = stats.get('req_like_count', 0)
            total    = stats.get('records_parsed', 0)
            print(f"  [OK] {json_path.stem:<35} "
                  f"{total:>4} records  "
                  f"({req_like} req-like)")

    # ── Overall stats ──────────────────────────────────────────
    word_counts  = [r['word_count'] for r in all_records]
    req_like_all = [r for r in all_records if r['is_requirement_like']]
    priority_all = Counter(str(r['priority']) for r in all_records if r['priority'])

    print(f"\n{'─'*60}")
    print(f"OVERALL SUMMARY")
    print(f"{'─'*60}")
    print(f"  Documents parsed    : {len(all_doc_stats)}")
    print(f"  Documents failed    : {len(failed_docs)}")
    print(f"  Total records       : {len(all_records)}")
    print(f"  Requirement-like    : {len(req_like_all)} "
          f"({100*len(req_like_all)//max(len(all_records),1)}%)")
    print(f"\n  Word count stats:")
    if word_counts:
        print(f"    Min    : {min(word_counts)}")
        print(f"    Max    : {max(word_counts)}")
        print(f"    Mean   : {sum(word_counts)/len(word_counts):.1f}")
        print(f"    Median : {sorted(word_counts)[len(word_counts)//2]}")
    print(f"\n  Priority distribution:")
    for p, cnt in sorted(priority_all.items()):
        print(f"    Priority {p}: {cnt}")

    print(f"\n  Sample requirement-like records:")
    for r in req_like_all[:5]:
        print(f"    [{r['req_id_raw']}] {r['requirement_text'][:100]}")

    # ── Save outputs ───────────────────────────────────────────
    out_reqs = output_dir / "pure_requirements.json"
    out_reqs.write_text(
        json.dumps(all_records, indent=2, ensure_ascii=False)
    )
    print(f"\n  Saved → {out_reqs}  ({len(all_records)} records)")

    report = {
        "dataset":              "PURE",
        "documents_parsed":     len(all_doc_stats),
        "documents_failed":     len(failed_docs),
        "total_records":        len(all_records),
        "requirement_like":     len(req_like_all),
        "requirement_like_pct": round(100*len(req_like_all)/max(len(all_records),1), 1),
        "word_count_mean":      round(sum(word_counts)/max(len(word_counts),1), 2),
        "word_count_median":    sorted(word_counts)[len(word_counts)//2] if word_counts else 0,
        "priority_distribution":dict(priority_all),
        "per_document":         all_doc_stats,
        "failed_documents":     failed_docs,
    }
    out_report = output_dir / "pure_report.json"
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"  Saved → {out_report}")


def main():
    parser = argparse.ArgumentParser(description="Parse PURE SRS dataset")
    parser.add_argument("--input-dir",  required=True,
                        help="Directory containing PURE *.json files")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for output files")
    args = parser.parse_args()

    run(Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
