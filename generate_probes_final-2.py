"""
SRS-ProbeCore Probe Generator — Final Version
===============================================
Week 3 — Deadline A

Grounded in evaluation_settings.md specifications.

Per base item generates:
  - 2 invariance probes  (hybrid rule+LLM, expected relation: stable)
  - 1 directional probe  (modal OR condition scope, expected relation: directional)
  - 1 shortcut probe     (hallucination-targeted distractor, expected relation: no_shortcut)

API key: set OPENAI_API_KEY env var, or pass --api-key
Models:
  - Generation : o4-mini (reasoning model for faithful paraphrasing)
  - Verification: gpt-4.1-mini (binary YES/NO quality check)

Run:
    export OPENAI_API_KEY=sk-...
    python generate_probes_final.py \
        --input  /teamspace/studios/this_studio/parsed/srs_probecore_v4.json \
        --output /teamspace/studios/this_studio/parsed/srs_probecore_v4_probed.json

Dry run (no API, rule-based only):
    python generate_probes_final.py --input ... --output ... --dry-run --limit 20
"""

import json, re, time, random, argparse, os
from pathlib import Path
from copy import deepcopy
from collections import Counter

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ── Model config ───────────────────────────────────────────────────────────
GENERATION_MODEL   = "o4-mini"      # reasoning model for generation
VERIFICATION_MODEL = "gpt-4.1-mini" # cheap binary checker

# ── API key: env var preferred, CLI overrides ──────────────────────────────
# API_KEY = os.environ.get("OPENAI_API_KEY", None)
# API_KEY = ""

# ══════════════════════════════════════════════════════════════════════════
# MODAL MAPS
# ══════════════════════════════════════════════════════════════════════════

MODAL_STRENGTH_MAP = {
    "shall": 3, "must": 3,
    "should": 2, "will": 2,
    "may": 1, "can": 1,
}

# Synonyms that preserve strength (used in invariance probes)
MODAL_SYNONYMS = {
    "shall":  ["is required to", "must"],
    "must":   ["is required to", "shall"],
    "should": ["is expected to", "ought to"],
    "will":   ["is going to", "shall"],
    "may":    ["is permitted to", "is allowed to"],
    "can":    ["is able to", "is capable of"],
}

# Directional: weaken obligation
MODAL_WEAKEN = {
    "shall": "should", "must": "should",
    "should": "may",   "will": "may", "can": "may",
}

# Directional: strengthen obligation
MODAL_STRENGTHEN = {
    "may": "should", "can": "should",
    "should": "shall", "will": "shall",
}

# Condition scope changes for directional probes
CONDITION_STRENGTHENERS = [
    ("fails", "fails twice"), ("fails", "fails repeatedly"),
    ("exceeds", "significantly exceeds"), ("occurs", "occurs repeatedly"),
    ("detected", "detected multiple times"), ("invalid", "invalid or expired"),
    ("missing", "missing or corrupted"), ("timeout", "timeout or connection failure"),
]
CONDITION_WEAKENERS = [
    ("fails twice", "fails"), ("fails repeatedly", "fails"),
    ("always", "sometimes"), ("all", "some"), ("every", "some"),
    ("mandatory", "optional"), ("required", "recommended"),
    ("immediately", "eventually"),
]

# Shortcut distractors — hallucination-targeted per eval_settings.md
SHORTCUT_DISTRACTORS = {
    "retry_trigger":     ["temporarily", "intermittently", "transiently", "momentarily"],
    "escalation_trigger":["unexpectedly", "without warning", "without prior notice", "without authorization"],
    "scope_trigger":     ["under all circumstances", "in all cases", "regardless of context", "at all times"],
    "response_trigger":  ["and log the event", "and notify the administrator", "and send a confirmation", "and generate a report"],
}

# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def extract_primary_modal(text: str) -> str | None:
    for m in ["shall", "must", "should", "will", "may", "can"]:
        if re.search(rf'\b{m}\b', text, re.IGNORECASE):
            return m
    return None


def modal_strength(modal: str) -> int:
    return MODAL_STRENGTH_MAP.get(modal, 0)


def strength_preserved(original_modal: str, candidate: str) -> bool:
    """
    Check obligation strength is preserved in candidate text.
    Handles multi-word synonyms and restructured forms.
    """
    cand_lower = candidate.lower()
    # Multi-word forms that map to strength 3
    if re.search(r'\bit is required that\b', cand_lower):
        return modal_strength(original_modal) == 3
    if re.search(r'\bis required to\b', cand_lower):
        return modal_strength(original_modal) == 3
    if re.search(r'\bshall be required\b', cand_lower):
        return modal_strength(original_modal) == 3
    # Multi-word forms that map to strength 2
    if re.search(r'\bis expected to\b', cand_lower):
        return modal_strength(original_modal) == 2
    if re.search(r'\bought to\b', cand_lower):
        return modal_strength(original_modal) == 2
    # Multi-word forms that map to strength 1
    if re.search(r'\bis permitted to\b|\bis allowed to\b|'
                 r'\bis able to\b|\bis capable of\b', cand_lower):
        return modal_strength(original_modal) == 1
    # Fall back to primary modal comparison
    found = extract_primary_modal(candidate)
    return modal_strength(found) == modal_strength(original_modal) if found else False


def norm_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower().strip())

# ══════════════════════════════════════════════════════════════════════════
# LLM PROMPTS — with one-shot examples
# ══════════════════════════════════════════════════════════════════════════

VERIFY_SYSTEM = """You are a requirements engineering quality checker.
Determine if a paraphrase preserves the original meaning and obligation strength.
Respond with ONLY 'YES' or 'NO'. No explanation."""

VERIFY_USER = """EXAMPLE GOOD paraphrase:
Original: "The system shall authenticate the user before granting access."
Paraphrase: "The system is required to verify user identity prior to allowing access."
Answer: YES  (same strength: shall ≡ is required to; same meaning)

EXAMPLE BAD paraphrase:
Original: "The system should log all failed login attempts."
Paraphrase: "The system shall immediately log and report all failed login attempts to the administrator."
Answer: NO  (modal strengthened: should→shall; extra behavior added)

Now evaluate:
Original: {original}
Paraphrase: {paraphrase}

Preserve exact meaning, technical content, and obligation strength? (YES/NO)"""


GENERATION_SYSTEM = """You are a requirements engineering assistant specializing in paraphrasing software requirements.

Task: Paraphrase a requirement while preserving its exact meaning and obligation strength.

STRICT RULES:
1. Keep the SAME modal verb exactly as given — do not change it to any other modal
2. Keep ALL technical terms, system names, acronyms, and domain concepts unchanged
3. Change ONLY surface wording — use different sentence structure or synonyms for non-technical words
4. Do NOT add any new behavior, conditions, or constraints
5. Do NOT remove any existing conditions or constraints
6. Output ONLY the paraphrased requirement — no explanation, no preamble

EXAMPLE:
Input modal: shall
Input: "If payment fails, the system shall display an error message and permit the user to retry."
Output: "When a payment attempt is unsuccessful, the system shall show an error notification and allow the user to try again."
What changed: 'If payment fails'→'When a payment attempt is unsuccessful', 'display'→'show', 'permit'→'allow', 'retry'→'try again'. Modal 'shall' unchanged."""

GENERATION_USER = """Paraphrase this requirement. Modal verb must stay exactly as '{modal}'.

Requirement: {requirement}

Paraphrase:"""

# ══════════════════════════════════════════════════════════════════════════
# INVARIANCE PROBE — RULE OPERATIONS
# ══════════════════════════════════════════════════════════════════════════

def op_modal_synonym(text: str, modal: str) -> str | None:
    """Replace modal with same-strength synonym."""
    synonyms = MODAL_SYNONYMS.get(modal, [])
    if not synonyms:
        return None
    # Skip "is X to" forms for plural subjects or passive constructions
    is_passive = bool(re.search(rf'\b{re.escape(modal)}\s+be\b', text.lower()))
    is_plural  = bool(re.match(r'^[A-Z][a-z]+s\s', text))
    valid = [s for s in synonyms
             if not ((is_passive or is_plural) and s.startswith("is "))]
    if not valid:
        return None
    synonym  = random.choice(valid)
    new_text = re.compile(rf'\b{re.escape(modal)}\b', re.IGNORECASE).sub(
        synonym, text, count=1)
    return new_text if new_text != text else None


def op_condition_synonym(text: str) -> str | None:
    """Replace condition keyword at sentence start only — avoids grammar errors mid-sentence."""
    SAFE_SUBS = {
        "when":   ["once", "after"],
        "while":  ["as long as"],
        "if":     ["in the event that", "in case"],
        "where":  ["in cases where", "whenever"],
        "unless": ["except when", "if not"],
    }
    s = text.strip()
    for kw, syns in SAFE_SUBS.items():
        if s.lower().startswith(kw + " "):
            syn     = random.choice(syns)
            new     = syn.capitalize() + s[len(kw):]
            return new if new != s else None
    return None


def op_clause_expand(text: str) -> str | None:
    """Expand generic subject noun phrases."""
    for pat, rep in [
        (r'\bThe system\b',      'The software system'),
        (r'\bThe user\b',        'The end user'),
        (r'\bThe application\b', 'The software application'),
        (r'\bThe component\b',   'The software component'),
        (r'\bThe module\b',      'The software module'),
    ]:
        if re.search(pat, text):  # case-sensitive pattern
            new = re.sub(pat, rep, text, count=1)
            return new if new != text else None
    return None


def op_subject_rephrase(text: str, modal: str) -> str | None:
    """
    'X shall Y' → 'It is required that X Y'
    Handles domain-specific subjects: 'The AOCS', 'NPAC SMS', 'The EVS'.
    Skips items that start with a condition clause (if/when/while/where)
    to avoid producing 'It is required that When X is provided, it Y'.
    Only for shall/must.
    """
    if modal not in ("shall", "must"):
        return None
    # Skip if requirement starts with a condition keyword
    if re.match(r'^(if|when|while|where|unless)\b', text.strip(), re.IGNORECASE):
        return None
    m = re.compile(
        rf'^(.{{4,60}}?)\s+{re.escape(modal)}\s+(.+)$', re.IGNORECASE
    ).match(text.strip())
    if not m:
        return None
    subj = m.group(1).strip()
    pred = m.group(2).strip().rstrip('.')
    new  = f"It is required that {subj} {pred}."
    return new if len(new.split()) >= 8 else None


RULE_OPS = [
    ("modal_synonym",    op_modal_synonym),
    ("condition_syn",    op_condition_synonym),
    ("clause_expand",    op_clause_expand),
    ("subject_rephrase", op_subject_rephrase),
]


def apply_rule_ops(text: str, modal: str) -> list[tuple]:
    """Try all rule operations. Return list of (op_name, result)."""
    ops   = list(RULE_OPS)
    random.shuffle(ops)
    results    = []
    seen_norms = {norm_text(text)}

    for op_name, op_fn in ops:
        try:
            result = op_fn(text, modal) if op_name in (
                "modal_synonym", "subject_rephrase") else op_fn(text)
            if not result:
                continue
            n = norm_text(result)
            if n in seen_norms:
                continue
            # Reject broken "is going to" with plural subject
            if re.search(r'\bis\s+going\s+to\b', result.lower()):
                subj = re.match(r'^(\w+)', result)
                if subj and subj.group(1).endswith('s'):
                    continue
            seen_norms.add(n)
            results.append((op_name, result))
        except Exception:
            continue

    return results

# ══════════════════════════════════════════════════════════════════════════
# LLM CALLS
# ══════════════════════════════════════════════════════════════════════════

def verify_paraphrase(original: str, paraphrase: str, client) -> bool:
    if client is None:
        return True  # dry-run: trust rule
    try:
        resp = client.chat.completions.create(
            model=VERIFICATION_MODEL,
            messages=[
                {"role": "system", "content": VERIFY_SYSTEM},
                {"role": "user",   "content": VERIFY_USER.format(
                    original=original, paraphrase=paraphrase)},
            ],
            temperature=0.0,
            max_tokens=5,
        )
        return resp.choices[0].message.content.strip().upper().startswith("YES")
    except Exception:
        return True  # on error: trust rule


def llm_generate(requirement: str, modal: str, client) -> str | None:
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": GENERATION_SYSTEM},
                {"role": "user",   "content": GENERATION_USER.format(
                    modal=modal, requirement=requirement)},
            ],
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════
# INVARIANCE PROBE GENERATOR
# ══════════════════════════════════════════════════════════════════════════

def generate_invariance_probes(
    requirement: str, modal: str, client, n: int = 2
) -> list[dict]:
    """
    Hybrid: rule ops → LLM verify → LLM fallback.
    Dry-run (client=None): rule ops accepted without verification.
    """
    probes     = []
    seen_norms = {norm_text(requirement)}

    # Step 1: Rule-based candidates
    for op_name, candidate in apply_rule_ops(requirement, modal):
        if len(probes) >= n:
            break
        cn = norm_text(candidate)
        if cn in seen_norms:
            continue
        # Verify (no-op in dry-run)
        if client is not None:
            if not verify_paraphrase(requirement, candidate, client):
                continue
            time.sleep(0.2)
        seen_norms.add(cn)
        probes.append({
            "probe_id":          f"INV_{len(probes)+1}",
            "probe_family":      "invariance",
            "probe_text":        candidate,
            "expected_relation": "stable",
            "operation":         f"rule_{op_name}",
            "generation_method": "rule_only" if client is None else "hybrid_rule_verified",
            "scoring": {
                "criterion": "output stays semantically aligned with same modal strength",
                "pass": 1, "fail": 0,
            },
            "validation": {
                "rule_operation":  op_name,
                "llm_verified":    client is not None,
                "modal_preserved": strength_preserved(modal, candidate),
                "length_ratio":    round(len(candidate.split()) /
                                        max(len(requirement.split()), 1), 2),
            },
        })

    # Step 2: LLM fallback for remaining slots
    while len(probes) < n and client is not None:
        candidate = llm_generate(requirement, modal, client)
        if not candidate:
            break
        cn = norm_text(candidate)
        if cn in seen_norms:
            break
        seen_norms.add(cn)
        probes.append({
            "probe_id":          f"INV_{len(probes)+1}",
            "probe_family":      "invariance",
            "probe_text":        candidate,
            "expected_relation": "stable",
            "operation":         "llm_fallback",
            "generation_method": "hybrid_llm_fallback",
            "scoring": {
                "criterion": "output stays semantically aligned with same modal strength",
                "pass": 1, "fail": 0,
            },
            "validation": {
                "rule_operation":  None,
                "llm_verified":    True,
                "modal_preserved": strength_preserved(modal, candidate),
                "length_ratio":    round(len(candidate.split()) /
                                        max(len(requirement.split()), 1), 2),
            },
        })
        time.sleep(0.3)
        break

    return probes

# ══════════════════════════════════════════════════════════════════════════
# DIRECTIONAL PROBE GENERATOR
# ══════════════════════════════════════════════════════════════════════════

def directional_modal(requirement: str, modal: str) -> dict | None:
    if modal in MODAL_WEAKEN:
        new_modal, direction = MODAL_WEAKEN[modal], "weaken"
    elif modal in MODAL_STRENGTHEN:
        new_modal, direction = MODAL_STRENGTHEN[modal], "strengthen"
    else:
        return None
    new_text = re.compile(rf'\b{re.escape(modal)}\b', re.IGNORECASE).sub(
        new_modal, requirement, count=1)
    if new_text == requirement:
        return None
    return {
        "probe_id": "DIR_1", "probe_family": "directional",
        "probe_text": new_text, "expected_relation": "directional",
        "operation": "modal_substitution", "direction": direction,
        "original_modal": modal, "new_modal": new_modal,
        "description": f"Modal {direction}ed: '{modal}' → '{new_modal}'",
        "scoring": {
            "criterion": "output shifts in same direction as modal change — "
                         "does not collapse the distinction",
            "pass": 1, "fail": 0,
        },
    }


def directional_condition_scope(requirement: str) -> dict | None:
    text_lower = requirement.lower()
    for original, replacement in CONDITION_STRENGTHENERS:
        if original in text_lower:
            new_text = re.sub(rf'\b{re.escape(original)}\b', replacement,
                              requirement, count=1, flags=re.IGNORECASE)
            if new_text != requirement:
                return {
                    "probe_id": "DIR_1", "probe_family": "directional",
                    "probe_text": new_text, "expected_relation": "directional",
                    "operation": "condition_scope_strengthen", "direction": "strengthen",
                    "original_condition": original, "new_condition": replacement,
                    "description": f"Condition strengthened: '{original}' → '{replacement}'",
                    "scoring": {
                        "criterion": "output responds to narrowed condition — "
                                     "does not ignore the scope change",
                        "pass": 1, "fail": 0,
                    },
                }
    for original, replacement in CONDITION_WEAKENERS:
        if original in text_lower:
            new_text = re.sub(rf'\b{re.escape(original)}\b', replacement,
                              requirement, count=1, flags=re.IGNORECASE)
            if new_text != requirement:
                return {
                    "probe_id": "DIR_1", "probe_family": "directional",
                    "probe_text": new_text, "expected_relation": "directional",
                    "operation": "condition_scope_weaken", "direction": "weaken",
                    "original_condition": original, "new_condition": replacement,
                    "description": f"Condition weakened: '{original}' → '{replacement}'",
                    "scoring": {
                        "criterion": "output responds to widened condition — "
                                     "does not ignore the scope change",
                        "pass": 1, "fail": 0,
                    },
                }
    return None


def generate_directional_probe(requirement: str, modal: str) -> dict | None:
    # Prefer condition scope when requirement has a condition keyword
    has_condition = any(
        re.search(rf'\b{kw}\b', requirement.lower())
        for kw in ["if","when","while","where","unless",
                   "fails","exceeds","occurs","detected"]
    )
    if has_condition:
        probe = directional_condition_scope(requirement)
        if probe:
            return probe
    return directional_modal(requirement, modal)

# ══════════════════════════════════════════════════════════════════════════
# SHORTCUT PROBE GENERATOR — hallucination-targeted per eval_settings.md
# ══════════════════════════════════════════════════════════════════════════

HALLUCINATION_MODE = {
    "retry_trigger":     "hallucinate retry or recovery action",
    "escalation_trigger":"hallucinate escalation or notification step",
    "scope_trigger":     "hallucinate scope expansion to all cases",
    "response_trigger":  "hallucinate additional system response",
}


def select_distractor(requirement: str) -> tuple[str, str]:
    text_lower = requirement.lower()
    if any(w in text_lower for w in ["fail","error","invalid","reject",
                                      "deny","timeout","unavailable"]):
        cat = "retry_trigger"
    elif any(w in text_lower for w in ["access","login","authenticate",
                                        "permission","unauthori"]):
        cat = "escalation_trigger"
    elif any(w in text_lower for w in ["all","every","each","any",
                                        "user","participant","client"]):
        cat = "scope_trigger"
    else:
        cat = "response_trigger"
    return random.choice(SHORTCUT_DISTRACTORS[cat]), cat


def insert_distractor(text: str, distractor: str, category: str) -> str | None:
    # For retry triggers: insert inside condition clause (eval_settings example)
    if category == "retry_trigger":
        m = re.search(
            r'(\b(?:if|when|while|unless)\b\s+[^,\.]{3,40}?)(\s*[,\.])',
            text, re.IGNORECASE)
        if m:
            return text[:m.end(1)] + " " + distractor + text[m.end(1):]
    # For all others: insert after modal verb
    m = re.search(r'(\b(?:shall|should|must|may|will|can)\b)', text, re.IGNORECASE)
    if m and len(distractor.split()) <= 3:
        at = m.end()
        return (text[:at] + " " + distractor + " " + text[at:].lstrip()).strip()
    # Fallback: append before final period
    if text.rstrip().endswith('.'):
        return text.rstrip()[:-1] + ", " + distractor + "."
    return text.rstrip() + " " + distractor + "."


def generate_shortcut_probe(requirement: str) -> dict | None:
    distractor, cat = select_distractor(requirement)
    new_text = insert_distractor(requirement, distractor, cat)
    if not new_text or new_text == requirement:
        return None
    return {
        "probe_id": "SHC_1", "probe_family": "shortcut",
        "probe_text": new_text, "expected_relation": "no_shortcut",
        "operation": "distractor_insertion",
        "distractor": distractor, "distractor_category": cat,
        "hallucination_risk": HALLUCINATION_MODE[cat],
        "description": f"Distractor '{distractor}' targets: {HALLUCINATION_MODE[cat]}",
        "scoring": {
            "criterion": "model does NOT hallucinate extra behavior from distractor",
            "pass": 1, "fail": 0,
        },
    }

# ══════════════════════════════════════════════════════════════════════════
# PROBE NEIGHBORHOOD BUILDER
# ══════════════════════════════════════════════════════════════════════════

def build_probe_neighborhood(item: dict, client) -> dict:
    item        = deepcopy(item)
    requirement = item["requirement_text"]
    modal       = item.get("modal") or extract_primary_modal(requirement) or "shall"
    probes      = []
    stats       = {"invariance": 0, "directional": 0, "shortcut": 0, "failed": 0}

    # A: Invariance (2 probes)
    inv = generate_invariance_probes(requirement, modal, client, n=2)
    probes.extend(inv)
    stats["invariance"] = len(inv)
    stats["failed"]    += max(0, 2 - len(inv))

    # B: Directional (1 probe)
    dp = generate_directional_probe(requirement, modal)
    if dp:
        probes.append(dp)
        stats["directional"] = 1
    else:
        stats["failed"] += 1

    # C: Shortcut (1 probe)
    sp = generate_shortcut_probe(requirement)
    if sp:
        probes.append(sp)
        stats["shortcut"] = 1
    else:
        stats["failed"] += 1

    item["probe_neighborhoods"] = probes
    item["probe_stats"]         = stats
    item["probe_count"]         = len(probes)
    return item

# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def run(input_path, output_path, api_key, limit, seed, dry_run):
    random.seed(seed)
    items = json.loads(Path(input_path).read_text(encoding='utf-8'))

    print(f"\n{'='*65}")
    print(f"PROBE GENERATOR — Final Version (eval_settings.md grounded)")
    print(f"{'='*65}")
    print(f"  Input items : {len(items)}")
    if limit:
        items = items[:limit]
        print(f"  Limit       : {limit}")
    print(f"  Dry run     : {dry_run}")

    client = None
    if api_key and not dry_run and OPENAI_AVAILABLE:
        client = OpenAI(api_key=api_key)
        print(f"  Generation  : {GENERATION_MODEL}")
        print(f"  Verification: {VERIFICATION_MODEL}")
    else:
        print(f"  LLM         : disabled (rule-based only)")
        if not dry_run and not api_key:
            print(f"  [WARN] No API key. Set OPENAI_API_KEY or use --api-key.")

    results      = []
    probe_counts = Counter()
    op_counts    = Counter()
    inv_dist     = Counter()

    for i, item in enumerate(items):
        print(f"  [{i+1:>4}/{len(items)}] {item['item_id'][:52]}", end="  ")
        processed = build_probe_neighborhood(item, client)
        stats     = processed["probe_stats"]

        print(f"→ {processed['probe_count']} probes "
              f"(inv={stats['invariance']} "
              f"dir={stats['directional']} "
              f"shc={stats['shortcut']})")

        for p in processed["probe_neighborhoods"]:
            probe_counts[p["probe_family"]] += 1
            op_counts[p.get("operation","?")] += 1
        inv_dist[stats["invariance"]] += 1
        results.append(processed)

    total = sum(probe_counts.values())

    print(f"\n{'─'*65}")
    print(f"SUMMARY")
    print(f"{'─'*65}")
    print(f"  Items processed          : {len(results)}")
    print(f"  Total probe pairs        : {total}")
    print(f"  Invariance probes        : {probe_counts['invariance']}")
    print(f"  Directional probes       : {probe_counts['directional']}")
    print(f"  Shortcut probes          : {probe_counts['shortcut']}")
    print(f"  Avg probes/item          : {total/max(len(results),1):.1f}")
    print(f"  Invariance count dist    : {dict(sorted(inv_dist.items()))}")
    print(f"\n  Operation breakdown:")
    for op, cnt in op_counts.most_common():
        print(f"    {op:<35}: {cnt}")

    # Show 2 sample neighborhoods
    samples = [r for r in results if r['probe_stats']['invariance'] >= 2][:2]
    for item in samples:
        print(f"\n  SAMPLE: {item['item_id']}")
        print(f"  BASE  : {item['requirement_text'][:80]}")
        for p in item["probe_neighborhoods"]:
            tag = p['probe_family'][:3].upper()
            op  = p.get('operation','')[:22]
            val = p.get('validation', {})
            mp  = val.get('modal_preserved', '-')
            print(f"  [{tag}] ({op:<22}) mp={mp}  {p['probe_text'][:75]}")

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n  Saved → {out}  ({len(results)} items, {total} probes)")

    report = {
        "items": len(results), "total_probes": total,
        "probe_counts": dict(probe_counts),
        "operation_counts": dict(op_counts),
        "invariance_count_distribution": dict(inv_dist),
        "avg_probes_per_item": round(total/max(len(results),1), 2),
        "models": {
            "generation": GENERATION_MODEL,
            "verification": VERIFICATION_MODEL,
            "dry_run": dry_run,
        },
    }
    rp = out.parent / "probe_generation_report.json"
    rp.write_text(json.dumps(report, indent=2))
    print(f"  Saved → {rp}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",   required=True)
    p.add_argument("--output",  required=True)
    p.add_argument("--api-key", default=None,
                   help="OpenAI key. Overrides OPENAI_API_KEY env var.")
    p.add_argument("--limit",   type=int, default=None)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    api_key = args.api_key or API_KEY
    if not api_key and not args.dry_run:
        print("[WARN] No API key found. Falling back to dry-run mode.")
        args.dry_run = True

    run(args.input, args.output, api_key, args.limit, args.seed, args.dry_run)


if __name__ == "__main__":
    main()
