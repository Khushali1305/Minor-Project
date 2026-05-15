"""
PROMISE_exp ARFF Parser
========================
Exact format (confirmed from file inspection):
  @ATTRIBUTE ProjectID {1,2,...,49}
  @ATTRIBUTE RequirementText string
  @ATTRIBUTE _class_ {F,A,L,LF,MN,O,PE,SC,SE,US,FT,PO}
  @DATA
  1,'The system shall refresh the display every 60 seconds.',PE

Output: promise_exp.json + promise_exp_report.json
"""

import json
import re
from pathlib import Path
from collections import Counter

NFR_CLASSES = {
    "F":  "Functional",
    "A":  "Availability",
    "L":  "Legal",
    "LF": "Look and Feel",
    "MN": "Maintainability",
    "O":  "Operational",
    "PE": "Performance",
    "SC": "Scalability",
    "SE": "Security",
    "US": "Usability",
    "FT": "Fault Tolerance",
    "PO": "Portability",
}


def parse_promise_arff(filepath: Path) -> list[dict]:
    """
    Parse PROMISE_exp.arff with the exact confirmed format:
        project_id,'requirement text',class_label
    Handles:
        - Single-quoted text fields
        - Commas inside quoted text
        - % comment lines
        - Blank lines
    """
    records = []
    in_data = False
    parse_errors = []

    lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()

        # Skip blanks and comments
        if not stripped or stripped.startswith("%"):
            continue

        # Detect @DATA marker
        if stripped.upper() == "@DATA":
            in_data = True
            continue

        # Skip @ATTRIBUTE and @RELATION lines
        if stripped.upper().startswith("@"):
            continue

        if not in_data:
            continue

        # --- Parse data line ---
        # Format: project_id,'requirement text',class
        # Use regex to handle quoted text robustly
        match = re.match(r"^(\d+),'(.*)',([A-Z]+)$", stripped)

        if match:
            project_id = int(match.group(1))
            req_text   = match.group(2).strip()
            cls        = match.group(3).strip()
        else:
            # Fallback: split on first comma, last comma
            # Handles edge cases like missing quotes
            parts = stripped.split(",", 1)
            if len(parts) < 2:
                parse_errors.append((line_num, stripped))
                continue
            try:
                project_id = int(parts[0].strip())
                rest = parts[1].strip()
                # Last comma separates text from class
                last_comma = rest.rfind(",")
                if last_comma == -1:
                    parse_errors.append((line_num, stripped))
                    continue
                req_text = rest[:last_comma].strip().strip("'")
                cls      = rest[last_comma+1:].strip()
            except (ValueError, IndexError):
                parse_errors.append((line_num, stripped))
                continue

        # Normalize class
        cls_upper = cls.upper()
        class_family = "FR" if cls_upper == "F" else "NFR"

        records.append({
            "project_id":       str(project_id),
            "requirement_text": req_text,
            "class":            cls_upper,
            "class_full":       NFR_CLASSES.get(cls_upper, cls_upper),
            "class_family":     class_family,
            "word_count":       len(req_text.split()),
        })

    return records, parse_errors


def print_report(records: list[dict], parse_errors: list):
    class_dist   = Counter(r["class"] for r in records)
    family_dist  = Counter(r["class_family"] for r in records)
    project_dist = Counter(r["project_id"] for r in records)
    word_counts  = [r["word_count"] for r in records]

    print(f"\n{'='*55}")
    print(f"PROMISE_exp — Parsed Results")
    print(f"{'='*55}")
    print(f"  Total records    : {len(records)}")
    print(f"  FR  (Functional) : {family_dist['FR']}")
    print(f"  NFR              : {family_dist['NFR']}")
    print(f"  Parse errors     : {len(parse_errors)}")

    print(f"\n  Class breakdown:")
    for cls, cnt in sorted(class_dist.items(), key=lambda x: -x[1]):
        label = NFR_CLASSES.get(cls, cls)
        bar   = "█" * (cnt * 20 // max(class_dist.values()))
        print(f"    {cls:>3} ({label:<16}) {bar} {cnt}")

    print(f"\n  Projects: {len(project_dist)} unique IDs")

    print(f"\n  Word count stats:")
    print(f"    Min    : {min(word_counts)}")
    print(f"    Max    : {max(word_counts)}")
    print(f"    Mean   : {sum(word_counts)/len(word_counts):.1f}")
    print(f"    Median : {sorted(word_counts)[len(word_counts)//2]}")

    print(f"\n  Sample records:")
    for r in records[:5]:
        print(f"    [{r['class']:>2}] {r['requirement_text'][:90]}")

    if parse_errors:
        print(f"\n  Parse errors (first 5):")
        for line_num, line in parse_errors[:5]:
            print(f"    Line {line_num}: {line[:80]}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      required=True, help="Path to PROMISE_exp.arff")
    parser.add_argument("--output-dir", default=".",   help="Directory for output files")
    args = parser.parse_args()

    arff_path  = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing: {arff_path}")
    records, parse_errors = parse_promise_arff(arff_path)

    print_report(records, parse_errors)

    # Save full JSON
    out_json = output_dir / "promise_exp.json"
    out_json.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"\n  Saved → {out_json}  ({len(records)} records)")

    # Save report
    report = {
        "dataset":          "PROMISE_exp",
        "total_records":    len(records),
        "fr_count":         sum(1 for r in records if r["class_family"] == "FR"),
        "nfr_count":        sum(1 for r in records if r["class_family"] == "NFR"),
        "class_distribution": dict(Counter(r["class"] for r in records)),
        "project_count":    len(set(r["project_id"] for r in records)),
        "word_count_mean":  round(sum(r["word_count"] for r in records)/len(records), 2),
        "word_count_median":sorted([r["word_count"] for r in records])[len(records)//2],
        "parse_errors":     len(parse_errors),
    }
    out_report = output_dir / "promise_exp_report.json"
    out_report.write_text(json.dumps(report, indent=2))
    print(f"  Saved → {out_report}")


if __name__ == "__main__":
    main()