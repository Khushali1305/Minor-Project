"""
SRS-ProbeCore Base Item Builder v4 — FINAL VERSION
=====================================================
Week 2 — Frozen Base Items

All gaps from v1, v2, v3 audits are closed.

New in v4 (3 gaps from v3 audit + 5 tightening rules):
  25. Tilde-delimited trailing metadata: "~ NESDIS Inspection ~"
  26. Source code blocks: file paths, function signatures, C comments
  27. Requirements table row dumps: column headers + cell values
  28. Second-person subject: "you can", "you should", "you must"
  29. Architectural description: "is responsible for", "is designed to"
  30. Document management statements: "will be reviewed", "will be updated"
  31. Cross-reference inline strip: "(3.3.2.40)" "(see 4.2)" mid-sentence
  32. Two-sentence merges starting with "Then,": split and keep first only

Run:
    python build_srs_probecore_v4.py \
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

LEADING_CONNECTIVES = re.compile(
    r'^(therefore[,\s]+|moreover[,\s]+|however[,\s]+|additionally[,\s]+|'
    r'furthermore[,\s]+|also[,\s]+|note that[,\s]+|requirement specification:\s+|'
    r'with this in mind[,\s]+|as a result[,\s]+|consequently[,\s]+)',
    re.IGNORECASE
)

ID_PREFIX_STRIP = re.compile(
    r'^([A-Z0-9]{2,12}-[A-Z0-9]{2,8}-?[A-Z0-9]{0,8}\d*\s+)|'
    r'^(\d+(\.\d+){1,4}\s+[A-Z][a-z])',
    re.IGNORECASE
)

SECTION_PREFIX_STRIP = re.compile(
    r'^\d+(\.\d+)*\s+[A-Za-z\s]{3,40}:\s*[-–]?\s*'
)

BULLET_STRIP = re.compile(r'^[•·▪▸►\-–—]\s+')

OCR_FOOTNOTE = re.compile(r'\.\d{1,3}\s+[A-Z]')

TRAILING_SECTION_REF = re.compile(r'\s*\(\s*\d+(\.\d+)+\s*\)\s*\.?$')

# Fix 31: Inline cross-references like "(3.3.2.40)" "(see 4.2)"
INLINE_CROSSREF = re.compile(r'\(\s*(?:see\s+)?\d+(\.\d+)+\s*\)')

OPTIONALITY_MARKER = re.compile(r'\s*\([OMR]\)\s*')

# Fix 25: Tilde-delimited trailing metadata "~ Text ~" or "~ Text"
TILDE_METADATA = re.compile(r'\s*~[^~]{0,80}~?\s*$')

# Fix 32: Two-sentence merge starting with "Then,"
THEN_SENTENCE = re.compile(r'\.\s+Then[,\s].*$', re.IGNORECASE)

# Requirements table header contamination at start
TABLE_HEADER_STRIP = re.compile(
    r'^[\w\s]{2,20}(req\s*id|requirement\s*(id|statement)|use\s*case|'
    r'id\s+requirement|statement\s+use)\s+\w{1,10}\d+\s+',
    re.IGNORECASE
)


def apply_fixes(text: str) -> str:
    """Apply all fix rules. Returns cleaned text."""

    # Fix curly quotes
    text = text.translate(CURLY_QUOTES)

    # Fix 25: Strip tilde-delimited trailing metadata
    text = TILDE_METADATA.sub('', text).strip()

    # Fix 32: Strip "Then, ..." second sentence
    text = THEN_SENTENCE.sub('.', text).strip()

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

    # Strip requirements table header contamination
    m = TABLE_HEADER_STRIP.match(text)
    if m and len(text[m.end():].split()) >= MIN_WORDS:
        text = text[m.end():].strip()

    # Fix 31: Strip inline cross-references "(3.3.2.40)"
    text = INLINE_CROSSREF.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Strip trailing section cross-references "(4.3.4)"
    text = TRAILING_SECTION_REF.sub('.', text).strip()

    # Strip optionality markers "(O)" "(M)"
    text = OPTIONALITY_MARKER.sub(' ', text).strip()

    # Fix OCR footnote artifact
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
# REJECT RULES — all 32 rules
# ══════════════════════════════════════════════════════════════════════════

# Rule 11: Modal strength 0
def reject_no_modal(text):
    if not MODAL_RE.search(text):
        return True, "no_modal_verb"
    return False, ""

# Rule 1: Context-dependent openers
CONTEXT_OPENERS = re.compile(
    r'^(for example|therefore|moreover|however|but if|additionally|'
    r'furthermore|in addition|as a result|consequently|thus|hence|'
    r'as mentioned|as described|as noted|as stated|as specified above|'
    r'as follows|in this case|in such cases|this means|with this in mind)\b',
    re.IGNORECASE
)
def reject_context_opener(text):
    if CONTEXT_OPENERS.match(text):
        return True, "context_opener"
    return False, ""

# Rule 2: Pronoun subjects
PRONOUN_SUBJECT = re.compile(
    r'^(it |he |she |they |them |this |these |those |its )',
    re.IGNORECASE
)
def reject_pronoun_subject(text):
    if PRONOUN_SUBJECT.match(text):
        return True, "pronoun_subject"
    return False, ""

# Rule 28: Second-person subject
SECOND_PERSON = re.compile(
    r'^(you (can|should|must|may|will|are|have)|'
    r'after you |once you |when you |if you )',
    re.IGNORECASE
)
def reject_second_person(text):
    if SECOND_PERSON.match(text):
        return True, "second_person_subject"
    return False, ""

# Rule 3: Deleted markers
DELETED_MARKER = re.compile(
    r'intentionally deleted|deliberately deleted|'
    r'this (page|section) (is )?intentionally',
    re.IGNORECASE
)
def reject_deleted_marker(text):
    if DELETED_MARKER.search(text):
        return True, "deleted_marker"
    return False, ""

# Rule 4: Glossary definitions
GLOSSARY_PATTERN = re.compile(r'^[A-Z][A-Za-z\s/\-]{2,40}[–—-]\s+[A-Z]')
def reject_glossary(text):
    if GLOSSARY_PATTERN.match(text):
        return True, "glossary_definition"
    return False, ""

# Rule 5: Document header noise
DOC_HEADER_NOISE = re.compile(
    r'\bSRS\s+\d+\b|\bpage\s+\d+\b|\bversion\s+\d+\.\d+\b',
    re.IGNORECASE
)
def reject_doc_header(text):
    if DOC_HEADER_NOISE.search(text):
        return True, "doc_header_noise"
    return False, ""

# Rule 6+15: Truncated list items
TRUNCATED_LIST = re.compile(
    r':\s*$|:\s*\d+\.\s*$|,\s*\d+\.\s*$|\(\s*[Mm]\s*\)\s*\d+\.'
)
def reject_truncated_list(text):
    if TRUNCATED_LIST.search(text):
        return True, "truncated_list"
    return False, ""

# Rule 7: Non-system statements
NON_SYSTEM = re.compile(
    r'^(workshops? will\b|all people are\b|everyone (that|who)\b|'
    r'users? are (assumed|expected|required) to have\b|'
    r'this document (explains|describes|outlines)\b|'
    r'the (whole|entire) project is based\b|'
    r'this (srs|specification|document) (describes|explains|defines)\b)',
    re.IGNORECASE
)
def reject_non_system(text):
    if NON_SYSTEM.match(text):
        return True, "non_system_statement"
    return False, ""

# Rule 12: Metadata template dumps
METADATA_FIELDS = re.compile(
    r'\b(actors?|post conditions?|alternate course|exceptional course|'
    r'req id|origin:|priority:\s*\d|process source|related reqs?|'
    r'category:|systems?:|users?:)\b',
    re.IGNORECASE
)
def reject_metadata_dump(text):
    if len(METADATA_FIELDS.findall(text)) >= 2:
        return True, "metadata_template_dump"
    return False, ""

# Rule 13: Assumption entries
ASSUMPTION_ENTRY = re.compile(
    r'^it is (assumed|expected|understood|presupposed) that\b|'
    r'^(the system|this document) assumes?\b|'
    r'^assumptions?:\s',
    re.IGNORECASE
)
def reject_assumption(text):
    if ASSUMPTION_ENTRY.match(text):
        return True, "assumption_entry"
    return False, ""

# Rule 16: OCR corruption
OBVIOUSLY_BROKEN = re.compile(
    r'\b[a-zA-Z]{1,3}\s[a-zA-Z]{1}\s[A-Z]\b|'
    r'[•·▪]{2,}|'
    r'\$[A-Z_]+/|'                    # file path like $OWROOT/bld
    r'extern\s+\w+\s+\w+\s*\(|'      # C function signature
    r'/\*[*\s]{5,}'                    # C comment block
)
def reject_ocr_corruption(text):
    if OBVIOUSLY_BROKEN.search(text):
        return True, "ocr_corruption"
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii > len(text) * 0.05:
        return True, "ocr_corruption"
    return False, ""

# Rule 26: Source code blocks
SOURCE_CODE = re.compile(
    r'(\$[A-Z_]+/[\w/\.]+)|'         # file paths $OWROOT/bld/...
    r'(extern\s+\w+\s+\w+\s*\()|'    # C extern declaration
    r'(/\*[\*\s]{3,})|'               # C comment /*****
    r'(\w+\.\w+\(\))|'               # method calls foo.bar()
    r'(#include\s*<)|'               # C include
    r'(\{\s*\/\*)',                   # { /* code block
    re.IGNORECASE
)
def reject_source_code(text):
    if SOURCE_CODE.search(text):
        return True, "source_code_block"
    return False, ""

# Rule 27: Requirements table row dumps
TABLE_ROW = re.compile(
    r'(req\s*id|use\s*case|req\s*statement).{0,60}'
    r'(req\s*id|use\s*case|must\s+be\s+able)',
    re.IGNORECASE
)
def reject_table_row(text):
    if TABLE_ROW.search(text):
        return True, "table_row_dump"
    return False, ""

# Rule 17: Document authoring instructions
DOC_INSTRUCTIONS = re.compile(
    r'^(if this section is empty|fill in|insert here|keep in mind that|'
    r'describe (here|below)|to be (filled|completed|written)|'
    r'add (here|description)|write (here|the))\b',
    re.IGNORECASE
)
def reject_doc_instruction(text):
    if DOC_INSTRUCTIONS.match(text):
        return True, "doc_authoring_instruction"
    return False, ""

# Rule 18: ABI/spec citation prose
CITATION_PROSE = re.compile(
    r'^[\d\w\s]+ (notes?|states?|specifies?|indicates?|requires?|says?) that\b',
    re.IGNORECASE
)
def reject_citation_prose(text):
    if CITATION_PROSE.match(text):
        return True, "citation_prose"
    return False, ""

# Rule 19: Use case scenario narratives
SCENARIO_NARRATIVE = re.compile(
    r'\b(after \w+ (discards?|clicks?|selects?|enters?|presses?))\b',
    re.IGNORECASE
)
def reject_scenario(text):
    if SCENARIO_NARRATIVE.search(text):
        return True, "use_case_scenario"
    return False, ""

# Rule 20: DB record dumps
DB_RECORD = re.compile(
    r'(req id|name:|description:|related reqs?|process source).{0,30}'
    r'(req id|name:|description:|related reqs?|process source)',
    re.IGNORECASE
)
def reject_db_record(text):
    if DB_RECORD.search(text):
        return True, "db_record_dump"
    return False, ""

# Rule 21: Document versioning
VERSIONING = re.compile(
    r'\b(will be numbered|document will be (updated|revised|released|versioned)|'
    r'next (release|version) (of|will)|numbered \d+\.\d+)\b',
    re.IGNORECASE
)
def reject_versioning(text):
    if VERSIONING.search(text):
        return True, "versioning_statement"
    return False, ""

# Rule 22: Informal user intention
USER_INTENTION = re.compile(
    r'\b(may want to|might want to|could want to|'
    r'users? (may|might|could) (wish|want|like|prefer) to)\b',
    re.IGNORECASE
)
def reject_user_intention(text):
    if USER_INTENTION.search(text):
        return True, "user_intention_not_req"
    return False, ""

# Rule 24: Section tag merge
SECTION_TAG_MID = re.compile(r'\.\s+\d+\.\d+(\.\d+)*\s+[A-Z]')
def reject_section_tag_merge(text):
    if SECTION_TAG_MID.search(text):
        return True, "section_tag_merge"
    return False, ""

# Rule 29: Architectural descriptions
ARCHITECTURAL = re.compile(
    r'^(the \w+ (is responsible for|is designed to|is intended to|'
    r'consists? of|is composed of|is made up of|is used to|'
    r'serves? as|acts? as|functions? as))',
    re.IGNORECASE
)
def reject_architectural(text):
    if ARCHITECTURAL.match(text):
        return True, "architectural_description"
    return False, ""

# Rule 30: Document management statements
DOC_MANAGEMENT = re.compile(
    r'\b(requirements? (will be|shall be) (reviewed|updated|refined|'
    r'added|removed|maintained|documented|tracked))\b|'
    r'\b(this document|this specification|this srs) (will|shall) be\b',
    re.IGNORECASE
)
def reject_doc_management(text):
    if DOC_MANAGEMENT.search(text):
        return True, "doc_management_statement"
    return False, ""

# External references (PURE/PROMISE only)
EXTERNAL_REF = re.compile(
    r'\[[\w\-\s]+\]|appendix\s+[A-Z]\b|'
    r'per\s+\[|as\s+per\s+\[|see\s+\[|refer\s+to\s+\[',
    re.IGNORECASE
)

# ── Master pipeline ────────────────────────────────────────────────────────

ALL_REJECT_RULES = [
    reject_no_modal,
    reject_context_opener,
    reject_pronoun_subject,
    reject_second_person,
    reject_deleted_marker,
    reject_glossary,
    reject_doc_header,
    reject_truncated_list,
    reject_non_system,
    reject_metadata_dump,
    reject_assumption,
    reject_ocr_corruption,
    reject_source_code,
    reject_table_row,
    reject_doc_instruction,
    reject_citation_prose,
    reject_scenario,
    reject_db_record,
    reject_versioning,
    reject_user_intention,
    reject_section_tag_merge,
    reject_architectural,
    reject_doc_management,
]


def should_reject(text: str, check_external_ref: bool = False) -> tuple:
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


def make_base_item(item_id, source, doc_id, text, modal,
                   strength, extraction_method, extra=None) -> dict:
    item = {
        "item_id":             item_id,
        "source":              source,
        "document_id":         doc_id,
        "requirement_text":    text,
        "word_count":          len(text.split()),
        "modal":               modal,
        "modal_strength":      strength,
        "ears_type":           detect_ears_type(text),
        "priority":            None,
        "extraction_method":   extraction_method,
        "target_norm":         "EARS",
        "reference_rewrite":   None,
        "ears_template_label": None,
        "probe_neighborhoods": [],
    }
    if extra:
        item.update(extra)
    return item


# ══════════════════════════════════════════════════════════════════════════
# SOURCE PROCESSORS
# ══════════════════════════════════════════════════════════════════════════

def process_source(records, source_name, seen_keys,
                   max_per_doc, max_total,
                   check_external_ref, max_words_override=None):
    """Generic processor — used by all three sources."""
    by_doc = defaultdict(list)
    for r in records:
        by_doc[r.get('document_id', 'unknown')].append(r)

    base_items    = []
    reject_counts = Counter()
    max_wc = max_words_override or MAX_WORDS

    for doc_id, recs in sorted(by_doc.items()):
        doc_items = []
        for r in recs:
            text = r.get('requirement_text', '').strip()

            # Apply fixes first
            text = apply_fixes(text)
            wc   = len(text.split())

            if not (MIN_WORDS <= wc <= max_wc):
                reject_counts["word_count"] += 1
                continue

            if not MODAL_RE.search(text):
                reject_counts["no_modal_verb"] += 1
                continue

            reject, reason = should_reject(text, check_external_ref)
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

            doc_items.append(
                make_base_item(
                    item_id=r['item_id'],
                    source=source_name,
                    doc_id=doc_id,
                    text=text,
                    modal=modal,
                    strength=strength,
                    extraction_method=r.get('extraction_method', 'unknown'),
                    extra=r.get('extra'),
                )
            )

        doc_items.sort(key=lambda x: abs(x['word_count'] - 25))
        base_items.extend(doc_items[:max_per_doc])

    if max_total:
        random.shuffle(base_items)
        base_items = base_items[:max_total]

    return base_items, reject_counts


def load_pure(pure_path: Path, seen_keys: set) -> tuple:
    raw = json.loads(pure_path.read_text(encoding='utf-8'))
    print(f"\n  PURE raw records     : {len(raw)}")

    records = []
    for r in raw:
        records.append({
            'item_id':           f"PURE_{r['document_id']}_{r['req_id_raw']}",
            'document_id':       r['document_id'],
            'requirement_text':  r.get('requirement_text', ''),
            'extraction_method': r.get('extraction_method', 'unknown'),
            'extra':             {'priority': r.get('priority')},
        })

    items, rejects = process_source(
        records, "PURE", seen_keys,
        max_per_doc=MAX_PER_DOC,
        max_total=MAX_PURE_TOTAL,
        check_external_ref=True,
    )
    print(f"  PURE after filtering : {len(items)} base items")
    print(f"  Documents            : {len(set(r['document_id'] for r in items))}")
    print(f"  Top rejection causes : {dict(rejects.most_common(5))}")
    return items, rejects


def load_grosser(grosser_path: Path, seen_keys: set) -> tuple:
    raw = json.loads(grosser_path.read_text(encoding='utf-8'))
    print(f"\n  Großer raw records   : {len(raw)}")

    records = []
    for r in raw:
        free_text = str(r.get('free_text') or '').strip()
        ears_text = str(r.get('ears_text') or '').strip()
        if not free_text or not ears_text:
            continue
        records.append({
            'item_id':           f"GROSSER_{r['project']}_{r['req_id_raw']}",
            'document_id':       r['project'],
            'requirement_text':  free_text,
            'extraction_method': 'aligned_pair',
            'extra': {
                'reference_rewrite':    ears_text,
                'ears_template_label':  r.get('ears_template_label'),
                'master_rewrite':       r.get('master_text'),
                'master_template_label':r.get('master_template_label'),
                'fully_aligned':        r.get('has_ears') and r.get('has_master'),
            },
        })

    items, rejects = process_source(
        records, "Grosser", seen_keys,
        max_per_doc=999,       # keep all Großer — it is gold standard
        max_total=None,
        check_external_ref=False,  # aerospace reqs legitimately ref standards
        max_words_override=100,    # aerospace reqs can be longer
    )
    print(f"  Großer after filter  : {len(items)} base items")
    print(f"  Projects             : {sorted(set(r['document_id'] for r in items))}")
    print(f"  Top rejection causes : {dict(rejects.most_common(5))}")
    return items, rejects


def load_promise(promise_path: Path, seen_keys: set) -> tuple:
    raw     = json.loads(promise_path.read_text(encoding='utf-8'))
    fr_recs = [r for r in raw if r.get('class_family') == 'FR']
    print(f"\n  PROMISE FR records   : {len(fr_recs)}")

    by_project = defaultdict(list)
    for r in fr_recs:
        by_project[r['project_id']].append(r)

    # Balance: max 5 per project before processing
    balanced = []
    for proj_items in by_project.values():
        random.shuffle(proj_items)
        balanced.extend(proj_items[:5])

    records = []
    for r in balanced:
        records.append({
            'item_id':           f"PROMISE_{r['project_id']}_{len(records):04d}",
            'document_id':       f"PROMISE_proj_{r['project_id']}",
            'requirement_text':  str(r.get('requirement_text', '')),
            'extraction_method': 'fr_filter',
            'extra':             {'req_class': r.get('class', 'F')},
        })

    items, rejects = process_source(
        records, "PROMISE_exp", seen_keys,
        max_per_doc=5,
        max_total=150,
        check_external_ref=True,
    )
    print(f"  PROMISE after filter : {len(items)} base items")
    print(f"  Top rejection causes : {dict(rejects.most_common(5))}")
    return items, rejects


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
    print(f"SRS-PROBECORE v4 — FINAL REPORT")
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

    print(f"\n  Obligation strength (1=may/can  2=should/will  3=shall/must):")
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

    print(f"\n  Samples (15-35 words, with reference where available):")
    samples = [r for r in items if 15 <= r['word_count'] <= 35]
    random.shuffle(samples)
    for r in samples[:5]:
        ref = (f"\n    {'→ EARS':>12} {r['reference_rewrite']}"
               if r.get('reference_rewrite') else "")
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
    print(f"SRS-PROBECORE v4 BUILDER — FINAL VERSION")
    print(f"32 filter rules  |  fix-before-reject pipeline")
    print(f"{'='*65}")

    seen_keys = set()

    grosser_items, g_rej = load_grosser(Path(grosser_path), seen_keys)
    pure_items,    p_rej = load_pure(Path(pure_path), seen_keys)
    promise_items, r_rej = load_promise(Path(promise_path), seen_keys)

    # Großer first (highest quality), then PURE, then PROMISE
    all_items = grosser_items + pure_items + promise_items

    print_report(all_items)

    # Save outputs
    out_path = output_dir / "srs_probecore_v4.json"
    out_path.write_text(json.dumps(all_items, indent=2, ensure_ascii=False))
    print(f"\n  Saved → {out_path}  ({len(all_items)} items)")

    report = {
        "version":                "v4_final",
        "filter_rules_applied":   32,
        "total_items":            len(all_items),
        "source_distribution":    dict(Counter(r['source'] for r in all_items)),
        "modal_distribution":     dict(Counter(r['modal'] for r in all_items)),
        "ears_distribution":      dict(Counter(r['ears_type'] for r in all_items)),
        "with_reference_rewrite": sum(1 for r in all_items if r.get('reference_rewrite')),
        "fully_aligned":          sum(1 for r in all_items if r.get('fully_aligned')),
        "word_count_mean":        round(
            sum(r['word_count'] for r in all_items) / len(all_items), 2),
        "word_count_median":      sorted(
            r['word_count'] for r in all_items)[len(all_items) // 2],
        "audit_history": {
            "v1_accept_rate": "84%",
            "v2_accept_rate": "44%",
            "v3_accept_rate": "~65%",
            "v4_target":      ">88%",
        },
        "rejection_breakdown": {
            "grosser": dict(g_rej),
            "pure":    dict(p_rej),
            "promise": dict(r_rej),
        },
    }
    report_path = output_dir / "srs_probecore_v4_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"  Saved → {report_path}")

    # 50-item audit sample
    sample = random.sample(all_items, min(50, len(all_items)))
    sample_path = output_dir / "srs_probecore_v4_sample.jsonl"
    with open(sample_path, 'w', encoding='utf-8') as f:
        for item in sample:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  Saved → {sample_path}  (50-item audit sample)")


def main():
    parser = argparse.ArgumentParser(
        description="SRS-ProbeCore v4 — Final base item builder"
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
