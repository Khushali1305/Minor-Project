"""
SRS-ProbeCore Base Item Builder v3
====================================
Week 2 — Final Filter Pass

Fixes all 24 problems identified across v1 and v2 manual audits:

V1 fixes (10):
  1. Context-dependent openers
  2. Pronoun subjects with no referent
  3. Deleted section markers
  4. Glossary definitions
  5. Document header noise
  6. Truncated list items (ending with colon/enumeration)
  7. Non-system community statements
  8. OCR footnote artifacts mid-sentence
  9. External reference dependencies (PURE only)
  10. ID prefix remnants at sentence start

V2 new fixes (14):
  11. Modal strength 0 — no modal verb at all
  12. Use case / metadata template field dumps
  13. Assumption section entries
  14. Section header prefixes not stripped
  15. Sentences ending with colon
  16. OCR corruption detection (broken words, bullet chars)
  17. Document authoring instructions
  18. ABI/spec citation prose
  19. Use case scenario dumps
  20. Requirements database record dumps
  21. Document versioning statements
  22. Informal user intention statements ('may want to')
  23. Numeric section prefix not stripped from sentence start
  24. Multi-sentence merges with section tag contamination

Run:
    python build_srs_probecore_v3.py \
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

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════
# FIX RULES — applied before reject checks
# ══════════════════════════════════════════════════════════════════════════

CURLY_QUOTES = str.maketrans('\u201c\u201d\u2018\u2019', '"\'"\'')

# Leading connective words to strip
LEADING_CONNECTIVES = re.compile(
    r'^(therefore[,\s]+|moreover[,\s]+|however[,\s]+|additionally[,\s]+|'
    r'furthermore[,\s]+|also[,\s]+|note that[,\s]+|requirement specification:\s+|'
    r'with this in mind[,\s]+|as a result[,\s]+|consequently[,\s]+)',
    re.IGNORECASE
)

# ID prefix remnants at sentence start (fix 10, 23)
ID_PREFIX_STRIP = re.compile(
    r'^([A-Z0-9]{2,12}-[A-Z0-9]{2,8}-?[A-Z0-9]{0,8}\d*\s+)|'  # C2C-IF-IS20
    r'^(\d+(\.\d+){1,4}\s+[A-Z][a-z])',                          # 2.5.1 Incoming
    re.IGNORECASE
)

# Numeric section prefix: "14.5 Constraints 14.5.1 Operating:" at start
SECTION_PREFIX_STRIP = re.compile(
    r'^\d+(\.\d+)*\s+[A-Za-z\s]+:\s*[-–]?\s*',
)

# Bullet character at start
BULLET_STRIP = re.compile(r'^[•·▪▸►\-–—]\s+')

# OCR footnote artifact mid-sentence: ".27 The" or ".5 A"
OCR_FOOTNOTE = re.compile(r'\.\d{1,3}\s+[A-Z]')

# Trailing section references: "(4.3.4)" "(6.3.4.6)"
TRAILING_SECTION_REF = re.compile(r'\s*\(\s*\d+(\.\d+)+\s*\)\s*\.?$')

# Trailing "(O)" optionality markers
OPTIONALITY_MARKER = re.compile(r'\s*\([OMR]\)\s*')


def apply_fixes(text: str) -> str:
    """Apply all fix rules to clean salvageable text."""

    # Fix curly quotes
    text = text.translate(CURLY_QUOTES)

    # Strip bullet characters
    text = BULLET_STRIP.sub('', text).strip()

    # Strip leading connective words
    text = LEADING_CONNECTIVES.sub('', text).strip()

    # Strip ID prefix remnants
    m = ID_PREFIX_STRIP.match(text)
    if m:
        text = text[m.end():].strip()

    # Strip section header prefix
    m = SECTION_PREFIX_STRIP.match(text)
    if m and len(text[m.end():].split()) >= MIN_WORDS:
        text = text[m.end():].strip()

    # Strip trailing section cross-references "(4.3.4)"
    text = TRAILING_SECTION_REF.sub('.', text).strip()

    # Strip optionality markers "(O)" "(M)"
    text = OPTIONALITY_MARKER.sub(' ', text).strip()

    # Fix OCR footnote artifact — keep only text before it
    ocr_m = OCR_FOOTNOTE.search(text)
    if ocr_m:
        cut = text.rfind('.', 0, ocr_m.start())
        if cut > 0 and len(text[:cut+1].split()) >= MIN_WORDS:
            text = text[:cut+1].strip()

    # Normalise whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Capitalise first letter
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    return text


# ══════════════════════════════════════════════════════════════════════════
# REJECT RULES — applied after fixes
# Each returns (should_reject: bool, reason: str)
# ══════════════════════════════════════════════════════════════════════════

# Fix 11: Modal strength 0
def reject_no_modal(text: str) -> tuple:
    if not MODAL_RE.search(text):
        return True, "no_modal_verb"
    return False, ""


# Fix 1: Context-dependent openers
CONTEXT_OPENERS = re.compile(
    r'^(for example|therefore|moreover|however|but if|additionally|'
    r'furthermore|in addition|as a result|consequently|thus|hence|'
    r'as mentioned|as described|as noted|as stated|as specified above|'
    r'as follows|in this case|in such cases|this means|with this in mind)\b',
    re.IGNORECASE
)
def reject_context_opener(text: str) -> tuple:
    if CONTEXT_OPENERS.match(text):
        return True, "context_opener"
    return False, ""


# Fix 2: Pronoun subjects with no referent
PRONOUN_SUBJECT = re.compile(
    r'^(it |he |she |they |them |this |these |those |its )',
    re.IGNORECASE
)
def reject_pronoun_subject(text: str) -> tuple:
    if PRONOUN_SUBJECT.match(text):
        return True, "pronoun_subject"
    return False, ""


# Fix 3: Deleted section markers
DELETED_MARKER = re.compile(
    r'intentionally deleted|deliberately deleted|'
    r'this (page|section) (is )?intentionally',
    re.IGNORECASE
)
def reject_deleted_marker(text: str) -> tuple:
    if DELETED_MARKER.search(text):
        return True, "deleted_marker"
    return False, ""


# Fix 4: Glossary definitions
GLOSSARY_PATTERN = re.compile(r'^[A-Z][A-Za-z\s/\-]{2,40}[–—-]\s+[A-Z]')
def reject_glossary(text: str) -> tuple:
    if GLOSSARY_PATTERN.match(text):
        return True, "glossary_definition"
    return False, ""


# Fix 5: Document header noise
DOC_HEADER_NOISE = re.compile(
    r'\bSRS\s+\d+\b|\bpage\s+\d+\b|\bversion\s+\d+\.\d+\b',
    re.IGNORECASE
)
def reject_doc_header(text: str) -> tuple:
    if DOC_HEADER_NOISE.search(text):
        return True, "doc_header_noise"
    return False, ""


# Fix 6 + 15: Truncated list items — ends with colon or "1." enumeration
TRUNCATED_LIST = re.compile(
    r':\s*$|:\s*\d+\.\s*$|,\s*\d+\.\s*$|\(\s*[Mm]\s*\)\s*\d+\.',
)
def reject_truncated_list(text: str) -> tuple:
    if TRUNCATED_LIST.search(text):
        return True, "truncated_list"
    return False, ""


# Fix 7: Non-system community/process statements
NON_SYSTEM = re.compile(
    r'^(workshops? will\b|all people are\b|everyone (that|who)\b|'
    r'users? are (assumed|expected|required) to have\b|'
    r'this document (explains|describes|outlines)\b|'
    r'the (whole|entire) project is based\b|'
    r'this (srs|specification|document) (describes|explains|defines)\b)',
    re.IGNORECASE
)
def reject_non_system(text: str) -> tuple:
    if NON_SYSTEM.match(text):
        return True, "non_system_statement"
    return False, ""


# Fix 12: Use case / metadata template field dumps
METADATA_FIELDS = re.compile(
    r'\b(actors?|post conditions?|alternate course|exceptional course|'
    r'req id|origin:|priority:\s*\d|process source|related reqs?|'
    r'category:|systems?:)\b',
    re.IGNORECASE
)
def reject_metadata_dump(text: str) -> tuple:
    matches = METADATA_FIELDS.findall(text)
    if len(matches) >= 2:  # 2+ metadata fields = dump
        return True, "metadata_template_dump"
    return False, ""


# Fix 13: Assumption section entries
ASSUMPTION_ENTRY = re.compile(
    r'^it is (assumed|expected|understood|presupposed) that\b|'
    r'^(the system|this document) assumes?\b|'
    r'^assumptions?:\s',
    re.IGNORECASE
)
def reject_assumption(text: str) -> tuple:
    if ASSUMPTION_ENTRY.match(text):
        return True, "assumption_entry"
    return False, ""


# Fix 16: OCR corruption detection
OCR_CORRUPTION = re.compile(
    r'[a-z][A-Z]{2,}[a-z]|'        # random capitalisation mid-word
    r'\b[a-z]{1,2}[A-Z][a-z]{1,3}\b|'  # broken word boundary
    r'[^\x00-\x7F]{3,}|'            # 3+ consecutive non-ASCII
    r'\b\w*[^\w\s]\w*[^\w\s]\w*\b'  # multiple punctuation in single token
)
OBVIOUSLY_BROKEN = re.compile(
    r'\b[a-zA-Z]{1,3}\s[a-zA-Z]{1}\s[A-Z]\b|'  # "I W" "nteroperabi I W"
    r'[•·▪]{2,}'                                  # multiple bullets
)
def reject_ocr_corruption(text: str) -> tuple:
    if OBVIOUSLY_BROKEN.search(text):
        return True, "ocr_corruption"
    # Check ratio of non-ASCII chars
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii > len(text) * 0.05:  # more than 5% non-ASCII
        return True, "ocr_corruption"
    return False, ""


# Fix 17: Document authoring instructions
DOC_INSTRUCTIONS = re.compile(
    r'^(if this section is empty|fill in|insert here|'
    r'describe (here|below)|to be (filled|completed|written)|'
    r'add (here|description)|write (here|the))\b',
    re.IGNORECASE
)
def reject_doc_instruction(text: str) -> tuple:
    if DOC_INSTRUCTIONS.match(text):
        return True, "doc_authoring_instruction"
    return False, ""


# Fix 18: ABI/spec citation prose
CITATION_PROSE = re.compile(
    r'^[\d\w\s]+ (notes?|states?|specifies?|indicates?|requires?|says?) that\b',
    re.IGNORECASE
)
def reject_citation_prose(text: str) -> tuple:
    if CITATION_PROSE.match(text):
        return True, "citation_prose"
    return False, ""


# Fix 19: Use case scenario narrative
SCENARIO_NARRATIVE = re.compile(
    r'\b(after \w+ (discards?|clicks?|selects?|enters?|presses?))\b|'
    r'\b(bob|alice|sally|john|user1|actor)\b',
    re.IGNORECASE
)
def reject_scenario(text: str) -> tuple:
    if SCENARIO_NARRATIVE.search(text):
        return True, "use_case_scenario"
    return False, ""


# Fix 20: Requirements database record dumps
DB_RECORD = re.compile(
    r'(req id|name:|description:|related reqs?|process source).{0,30}'
    r'(req id|name:|description:|related reqs?|process source)',
    re.IGNORECASE
)
def reject_db_record(text: str) -> tuple:
    if DB_RECORD.search(text):
        return True, "db_record_dump"
    return False, ""


# Fix 21: Document versioning statements
VERSIONING = re.compile(
    r'\b(will be numbered|document will be (updated|revised|released|versioned)|'
    r'next (release|version) (of|will)|numbered \d+\.\d+)\b',
    re.IGNORECASE
)
def reject_versioning(text: str) -> tuple:
    if VERSIONING.search(text):
        return True, "versioning_statement"
    return False, ""


# Fix 22: Informal user intention statements
USER_INTENTION = re.compile(
    r'\b(may want to|might want to|could want to|'
    r'users? (may|might|could) (wish|want|like|prefer) to)\b',
    re.IGNORECASE
)
def reject_user_intention(text: str) -> tuple:
    if USER_INTENTION.search(text):
        return True, "user_intention_not_req"
    return False, ""


# Fix 24: Multi-sentence merges with section tag contamination
SECTION_TAG_MID = re.compile(
    r'\.\s+\d+\.\d+(\.\d+)*\s+[A-Z]'  # ". 2.2.3.1 Travel"
)
def reject_section_tag_merge(text: str) -> tuple:
    if SECTION_TAG_MID.search(text):
        return True, "section_tag_merge"
    return False, ""


# External references (PURE only)
EXTERNAL_REF = re.compile(
    r'\[[\w\-\s]+\]|appendix\s+[A-Z]\b|'
    r'per\s+\[|as\s+per\s+\[|see\s+\[|refer\s+to\s+\[',
    re.IGNORECASE
)


# ══════════════════════════════════════════════════════════════════════════
# MASTER REJECT PIPELINE
# ══════════════════════════════════════════════════════════════════════════

ALL_REJECT_RULES = [
    reject_no_modal,
    reject_context_opener,
    reject_pronoun_subject,
    reject_deleted_marker,
    reject_glossary,
    reject_doc_header,
    reject_truncated_list,
    reject_non_system,
    reject_metadata_dump,
    reject_assumption,
    reject_ocr_corruption,
    reject_doc_instruction,
    reject_citation_prose,
    reject_scenario,
    reject_db_record,
    reject_versioning,
    reject_user_intention,
    reject_section_tag_merge,
]


def should_reject(text: str, check_external_ref: bool = False) -> tuple:
    """Run all reject rules. Returns (reject: bool, reason: str)."""
    for rule in ALL_REJECT_RULES:
        reject, reason = rule(text)
        if reject:
            return True, reason

    if check_external_ref and EXTERNAL_REF.search(text):
        return True, "external_reference"

    return False, ""


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def get_modal(text: str) -> tuple:
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
    if "SHALL" in t or "SHOULD" in t or "MUST" in t:
        return "Ubiquitous"
    return "unknown"


def near_dup_key(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower().strip())[:NEAR_DUP_CHARS]


def make_base_item(item_id, source, doc_id, text, modal, strength,
                   extraction_method, extra=None) -> dict:
    item = {
        "item_id":            item_id,
        "source":             source,
        "document_id":        doc_id,
        "requirement_text":   text,
        "word_count":         len(text.split()),
        "modal":              modal,
        "modal_strength":     strength,
        "ears_type":          detect_ears_type(text),
        "priority":           None,
        "extraction_method":  extraction_method,
        "target_norm":        "EARS",
        "reference_rewrite":  None,
        "ears_template_label": None,
        "probe_neighborhoods": [],
    }
    if extra:
        item.update(extra)
    return item


# ══════════════════════════════════════════════════════════════════════════
# SOURCE PROCESSORS
# ══════════════════════════════════════════════════════════════════════════

def process_pure(pure_path: Path, seen_keys: set) -> tuple:
    raw = json.loads(pure_path.read_text(encoding='utf-8'))
    print(f"\n  PURE raw records     : {len(raw)}")

    by_doc = defaultdict(list)
    for r in raw:
        by_doc[r['document_id']].append(r)

    base_items    = []
    reject_counts = Counter()

    for doc_id, recs in sorted(by_doc.items()):
        doc_items = []

        for r in recs:
            text = r.get('requirement_text', '').strip()

            # Step 1: Apply fixes
            text = apply_fixes(text)
            wc   = len(text.split())

            # Step 2: Word count gate
            if not (MIN_WORDS <= wc <= MAX_WORDS):
                reject_counts["word_count"] += 1
                continue

            # Step 3: Modal presence gate (fast check before full pipeline)
            if not MODAL_RE.search(text):
                reject_counts["no_modal_verb"] += 1
                continue

            # Step 4: Full reject pipeline
            reject, reason = should_reject(text, check_external_ref=True)
            if reject:
                reject_counts[reason] += 1
                continue

            # Step 5: Near-dedup
            key = near_dup_key(text)
            if key in seen_keys:
                reject_counts["duplicate"] += 1
                continue
            seen_keys.add(key)

            modal, strength = get_modal(text)
            if strength == 0:
                reject_counts["modal_strength_0"] += 1
                continue

            doc_items.append(make_base_item(
                item_id=f"PURE_{doc_id}_{r['req_id_raw']}",
                source="PURE",
                doc_id=doc_id,
                text=text,
                modal=modal,
                strength=strength,
                extraction_method=r.get('extraction_method', 'unknown'),
                extra={"priority": r.get('priority')},
            ))

        # Balance: prefer items closest to 25 words (ideal probe length)
        doc_items.sort(key=lambda x: abs(x['word_count'] - 25))
        base_items.extend(doc_items[:MAX_PER_DOC])

    random.shuffle(base_items)
    base_items = base_items[:MAX_PURE_TOTAL]

    print(f"  PURE after filtering : {len(base_items)} base items")
    print(f"  Documents            : {len(set(r['document_id'] for r in base_items))}")
    print(f"  Top rejection causes : {dict(reject_counts.most_common(5))}")
    return base_items, reject_counts


def process_grosser(grosser_path: Path, seen_keys: set) -> tuple:
    raw = json.loads(grosser_path.read_text(encoding='utf-8'))
    print(f"\n  Großer raw records   : {len(raw)}")

    base_items    = []
    reject_counts = Counter()

    for r in raw:
        free_text = str(r.get('free_text') or '').strip()
        ears_text = str(r.get('ears_text') or '').strip()

        if not free_text or not ears_text:
            reject_counts["missing_pair"] += 1
            continue

        # Apply fixes
        free_text = apply_fixes(free_text)
        wc = len(free_text.split())

        # Großer items can be up to 100 words (aerospace reqs are longer)
        if not (MIN_WORDS <= wc <= 100):
            reject_counts["word_count"] += 1
            continue

        if not MODAL_RE.search(free_text):
            reject_counts["no_modal_verb"] += 1
            continue

        # Großer: run all rules EXCEPT external_ref
        # (aerospace reqs legitimately reference standards)
        reject, reason = should_reject(free_text, check_external_ref=False)
        if reject:
            reject_counts[reason] += 1
            continue

        key = near_dup_key(free_text)
        if key in seen_keys:
            reject_counts["duplicate"] += 1
            continue
        seen_keys.add(key)

        modal, strength = get_modal(free_text)
        if strength == 0:
            reject_counts["modal_strength_0"] += 1
            continue

        base_items.append(make_base_item(
            item_id=f"GROSSER_{r['project']}_{r['req_id_raw']}",
            source="Grosser",
            doc_id=r['project'],
            text=free_text,
            modal=modal,
            strength=strength,
            extraction_method="aligned_pair",
            extra={
                "reference_rewrite":    ears_text,
                "ears_template_label":  r.get('ears_template_label'),
                "master_rewrite":       r.get('master_text'),
                "master_template_label":r.get('master_template_label'),
                "fully_aligned":        r.get('has_ears') and r.get('has_master'),
            },
        ))

    print(f"  Großer after filter  : {len(base_items)} base items")
    print(f"  Projects             : {sorted(set(r['document_id'] for r in base_items))}")
    print(f"  Top rejection causes : {dict(reject_counts.most_common(5))}")
    return base_items, reject_counts


def process_promise(promise_path: Path, seen_keys: set) -> tuple:
    raw     = json.loads(promise_path.read_text(encoding='utf-8'))
    fr_recs = [r for r in raw if r.get('class_family') == 'FR']
    print(f"\n  PROMISE FR records   : {len(fr_recs)}")

    base_items    = []
    reject_counts = Counter()
    by_project    = defaultdict(list)

    for r in fr_recs:
        text = str(r.get('requirement_text', '')).strip()

        text = apply_fixes(text)
        wc   = len(text.split())

        if not (MIN_WORDS <= wc <= MAX_WORDS):
            reject_counts["word_count"] += 1
            continue

        if not MODAL_RE.search(text):
            reject_counts["no_modal_verb"] += 1
            continue

        reject, reason = should_reject(text, check_external_ref=True)
        if reject:
            reject_counts[reason] += 1
            continue

        key = near_dup_key(text)
        if key in seen_keys:
            reject_counts["duplicate"] += 1
            continue
        seen_keys.add(key)

        modal, strength = get_modal(text)
        if strength == 0:
            reject_counts["modal_strength_0"] += 1
            continue

        by_project[r['project_id']].append(make_base_item(
            item_id=f"PROMISE_{r['project_id']}_{len(base_items):04d}",
            source="PROMISE_exp",
            doc_id=f"PROMISE_proj_{r['project_id']}",
            text=text,
            modal=modal,
            strength=strength,
            extraction_method="fr_filter",
            extra={"req_class": r.get('class', 'F')},
        ))

    for proj_items in by_project.values():
        random.shuffle(proj_items)
        base_items.extend(proj_items[:5])

    random.shuffle(base_items)
    base_items = base_items[:150]

    print(f"  PROMISE after filter : {len(base_items)} base items")
    print(f"  Top rejection causes : {dict(reject_counts.most_common(5))}")
    return base_items, reject_counts


# ══════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════

def print_report(items: list):
    source_dist   = Counter(r['source'] for r in items)
    modal_dist    = Counter(r['modal'] for r in items)
    ears_dist     = Counter(r['ears_type'] for r in items)
    strength_dist = Counter(r['modal_strength'] for r in items)
    wc_all        = [r['word_count'] for r in items]
    has_ref       = sum(1 for r in items if r.get('reference_rewrite'))
    fully_aligned = sum(1 for r in items if r.get('fully_aligned'))

    print(f"\n{'='*65}")
    print(f"SRS-PROBECORE v3 — FINAL REPORT")
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

    print(f"\n  Obligation strength (1=may/can, 2=should/will, 3=shall/must):")
    for s, cnt in sorted(strength_dist.items()):
        bar = "█" * (cnt * 20 // max(strength_dist.values(), default=1))
        print(f"    strength {s}: {bar} {cnt}")

    print(f"\n  EARS type distribution:")
    for t, cnt in sorted(ears_dist.items(), key=lambda x: -x[1]):
        print(f"    {t:<25}: {cnt}")

    print(f"\n  Word count:")
    print(f"    Min   : {min(wc_all)}")
    print(f"    Max   : {max(wc_all)}")
    print(f"    Mean  : {sum(wc_all)/len(wc_all):.1f}")
    print(f"    Median: {sorted(wc_all)[len(wc_all)//2]}")

    print(f"\n  Samples (15-35 words):")
    samples = [r for r in items if 15 <= r['word_count'] <= 35]
    random.shuffle(samples)
    for r in samples[:5]:
        ref = f"\n    {'→ EARS':>12} {r['reference_rewrite']}" \
              if r.get('reference_rewrite') else ""
        print(f"\n    [{r['source']:<10}] [{r['modal']:<6}] "
              f"{r['requirement_text']}{ref}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def run(pure_path, grosser_path, promise_path, output_dir, seed=42):
    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"SRS-PROBECORE v3 BUILDER")
    print(f"Fixes: 24 filter rules (10 from v1 + 14 new from v2 audit)")
    print(f"{'='*65}")

    seen_keys = set()

    grosser_items, g_rej = process_grosser(Path(grosser_path), seen_keys)
    pure_items,    p_rej = process_pure(Path(pure_path), seen_keys)
    promise_items, r_rej = process_promise(Path(promise_path), seen_keys)

    all_items = grosser_items + pure_items + promise_items

    print_report(all_items)

    # Save main output
    out_path = output_dir / "srs_probecore_v3.json"
    out_path.write_text(json.dumps(all_items, indent=2, ensure_ascii=False))
    print(f"\n  Saved → {out_path}  ({len(all_items)} items)")

    # Save report
    report = {
        "version":                "v3",
        "filter_rules_applied":   24,
        "total_items":            len(all_items),
        "source_distribution":    dict(Counter(r['source'] for r in all_items)),
        "modal_distribution":     dict(Counter(r['modal'] for r in all_items)),
        "ears_distribution":      dict(Counter(r['ears_type'] for r in all_items)),
        "with_reference_rewrite": sum(1 for r in all_items if r.get('reference_rewrite')),
        "fully_aligned":          sum(1 for r in all_items if r.get('fully_aligned')),
        "word_count_mean":        round(sum(r['word_count'] for r in all_items)/len(all_items), 2),
        "word_count_median":      sorted(r['word_count'] for r in all_items)[len(all_items)//2],
        "audit_history": {
            "v1_accept_rate": "84%  (42/50 KEEP+FIX)",
            "v2_accept_rate": "~44% (22/50 KEEP+FIX)",
            "v3_target":      ">85% accept rate",
        },
        "rejection_breakdown": {
            "grosser": dict(g_rej),
            "pure":    dict(p_rej),
            "promise": dict(r_rej),
        }
    }
    report_path = output_dir / "srs_probecore_v3_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"  Saved → {report_path}")

    # 50-item audit sample
    sample = random.sample(all_items, min(50, len(all_items)))
    sample_path = output_dir / "srs_probecore_v3_sample.jsonl"
    with open(sample_path, 'w', encoding='utf-8') as f:
        for item in sample:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  Saved → {sample_path}  (50-item audit sample)")


def main():
    parser = argparse.ArgumentParser(
        description="Build SRS-ProbeCore v3 — all audit fixes applied"
    )
    parser.add_argument("--pure",       required=True)
    parser.add_argument("--grosser",    required=True)
    parser.add_argument("--promise",    required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.pure, args.grosser, args.promise, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
