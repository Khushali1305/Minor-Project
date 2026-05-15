"""
SRS-ProbeCore Probe Generator — Final Version
===============================================
Week 3 — Deadline A

Grounded in evaluation_settings.md specifications.

Per base item generates:
  - 2 invariance probes  (hybrid rule+LLM, expected relation: stable)
  - 1 directional probe  (modal OR condition scope, expected relation: directional)
  - 1 shortcut probe     (hallucination-targeted distractor, expected relation: no_shortcut)

Key fixes from evaluation_settings.md review:
  - Directional: implements BOTH modal change AND condition scope change
  - Shortcut: distractors specifically designed to trigger hallucination of
    extra behavior (retry, escalation, recovery actions)
  - Source-family alignment: Großer → preferred for invariance+directional
    PURE/PROMISE → preferred for shortcut seeding
  - Scoring schema: 1/0 per probe pair, documented per family

Run:
    python generate_probes_final.py \
        --input   /teamspace/studios/this_studio/parsed/srs_probecore_v4.json \
        --output  /teamspace/studios/this_studio/parsed/srs_probecore_v4_probed.json \
        --api-key YOUR_OPENAI_KEY \
        [--dry-run]   # skip LLM, rule-based only
        [--limit 20]  # test on first N items
"""

import json
import re
import time
import random
import argparse
import os
from pathlib import Path
from copy import deepcopy
from collections import Counter

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ── Model config ───────────────────────────────────────────────────────────
# o4-mini: reasoning model — used for invariance paraphrase generation
#   reasoning ensures modal verb and technical terms are preserved faithfully
# gpt-4.1-mini: fast cheap model — binary YES/NO verification only
GENERATION_MODEL   = "o4-mini"
VERIFICATION_MODEL = "gpt-4.1-mini"

# ── API key ────────────────────────────────────────────────────────────────
# Priority: CLI --api-key > environment variable > None (dry-run mode)
# Set on Lightning AI: export OPENAI_API_KEY=sk-...
# API_KEY = os.environ.get("OPENAI_API_KEY", None)
# API_KEY = ""


# ══════════════════════════════════════════════════════════════════════════
# MODAL MAPS — grounded in evaluation_settings.md
# ══════════════════════════════════════════════════════════════════════════

# Directional probe: weaken obligation strength
# eval_settings example: "should warn" → "may warn"
MODAL_WEAKEN = {
    "shall":  "should",
    "must":   "should",
    "should": "may",
    "will":   "may",
    "can":    "may",
}

# Directional probe: strengthen obligation strength
# eval_settings example: "may retry" → "must retry"
MODAL_STRENGTHEN = {
    "may":    "should",
    "can":    "should",
    "should": "shall",
    "will":   "shall",
}

# Modal synonym map for invariance rule-based paraphrasing
# Same strength, different surface form
MODAL_SYNONYMS = {
    "shall":  ["is required to", "must"],
    "must":   ["is required to", "shall"],
    "should": ["is expected to", "ought to"],
    "will":   ["is going to", "shall"],
    "may":    ["is permitted to", "is allowed to"],
    "can":    ["is able to", "is capable of"],
}

# Condition keyword synonyms for invariance paraphrasing
CONDITION_SYNONYMS = {
    "when":   ["once", "after", "upon"],
    "while":  ["during the period that", "as long as"],
    "if":     ["in the event that", "in case"],
    "where":  ["in cases where", "whenever"],
    "unless": ["except when", "if not"],
}

# ══════════════════════════════════════════════════════════════════════════
# SHORTCUT DISTRACTORS — grounded in evaluation_settings.md
#
# The spec says shortcut probes must expose hallucination of extra behavior.
# eval_settings example: "temporarily" → model invents retry logic.
# All distractors below are chosen to tempt one of these failure modes:
#   - hallucinate retry/recovery action
#   - hallucinate escalation step
#   - hallucinate scope expansion
#   - hallucinate additional system response
# ══════════════════════════════════════════════════════════════════════════

SHORTCUT_DISTRACTORS = {
    # Tempt retry/recovery hallucination
    # eval_settings: "temporarily" → model invents retry
    "retry_trigger": [
        "temporarily",
        "intermittently",
        "transiently",
        "momentarily",
    ],

    # Tempt escalation hallucination
    # Model may add notification, alerting, or escalation step
    "escalation_trigger": [
        "unexpectedly",
        "without warning",
        "without prior notice",
        "without authorization",
    ],

    # Tempt scope expansion hallucination
    # Model may add "for all users" or "in all cases"
    "scope_trigger": [
        "under all circumstances",
        "in all cases",
        "regardless of context",
        "at all times",
    ],

    # Tempt additional response hallucination
    # Model may add logging, notification, or confirmation step
    "response_trigger": [
        "and log the event",
        "and notify the administrator",
        "and send a confirmation",
        "and generate a report",
    ],
}

# Insertion positions for shortcut distractors
INSERTION_AFTER_MODAL = "after_modal"
INSERTION_IN_CONDITION = "in_condition"
INSERTION_AT_END = "at_end"


# ══════════════════════════════════════════════════════════════════════════
# CONDITION SCOPE PATTERNS — for directional probes
# eval_settings: "if payment fails" → "if payment fails twice"
# ══════════════════════════════════════════════════════════════════════════

CONDITION_STRENGTHENERS = [
    ("fails", "fails twice"),
    ("fails", "fails repeatedly"),
    ("exceeds", "significantly exceeds"),
    ("occurs", "occurs repeatedly"),
    ("detected", "detected multiple times"),
    ("invalid", "invalid or expired"),
    ("missing", "missing or corrupted"),
    ("timeout", "timeout or connection failure"),
]

CONDITION_WEAKENERS = [
    ("fails twice", "fails"),
    ("fails repeatedly", "fails"),
    ("always", "sometimes"),
    ("all", "some"),
    ("every", "some"),
    ("mandatory", "optional"),
    ("required", "recommended"),
    ("immediately", "eventually"),
]


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def extract_primary_modal(text: str) -> str | None:
    priority = ["shall", "must", "should", "will", "may", "can"]
    text_lower = text.lower()
    for modal in priority:
        if re.search(rf'\b{modal}\b', text_lower):
            return modal
    return None


def get_modal_strength(modal: str) -> int:
    strength_map = {"shall": 3, "must": 3, "should": 2,
                    "will": 2, "may": 1, "can": 1}
    return strength_map.get(modal, 0)


# ══════════════════════════════════════════════════════════════════════════
# INVARIANCE PROBES — Hybrid rule-based + LLM verification
# ══════════════════════════════════════════════════════════════════════════

# ── Verification prompt — binary YES/NO with one-shot example ─────────────
VERIFY_SYSTEM = """You are a requirements engineering quality checker.
Determine if a paraphrase preserves the original meaning and obligation strength.
Respond with ONLY 'YES' or 'NO'. No explanation."""

VERIFY_USER = """Here is an example of a GOOD paraphrase:
Original: "The system shall authenticate the user before granting access."
Paraphrase: "The system is required to verify user identity prior to allowing access."
Answer: YES
Reason: Same modal strength (shall → is required to), same meaning, same technical scope.

Here is an example of a BAD paraphrase:
Original: "The system should log all failed login attempts."
Paraphrase: "The system shall immediately log and report all failed login attempts to the administrator."
Answer: NO
Reason: Modal strengthened (should → shall), extra behavior added (report to administrator).

Now evaluate this pair:
Original: {original}
Paraphrase: {paraphrase}

Does the paraphrase preserve the exact meaning, technical content, and obligation strength? (YES/NO)"""


# ── Generation prompt — one-shot example with reasoning model ─────────────
LLM_FALLBACK_SYSTEM = """You are a requirements engineering assistant specializing in paraphrasing software requirements.

Your task: Paraphrase a requirement while preserving its exact meaning and obligation strength.

STRICT RULES:
1. Keep the SAME modal verb — do not change 'shall' to 'must', 'should' to 'shall', etc.
2. Keep ALL technical terms, system names, and domain concepts unchanged
3. Change ONLY surface wording — use different sentence structure, synonyms for non-technical words
4. Do NOT add any new behavior, conditions, or constraints
5. Do NOT remove any existing conditions or constraints
6. Output ONLY the paraphrased requirement — no explanation, no preamble

EXAMPLE:
Input modal: shall
Input requirement: "If payment fails, the system shall display an error message and permit the user to retry."
Output: "When a payment attempt is unsuccessful, the system shall show an error notification and allow the user to try again."
Explanation of what changed: 'If payment fails' → 'When a payment attempt is unsuccessful', 'display' → 'show', 'error message' → 'error notification', 'permit' → 'allow', 'retry' → 'try again'. Modal 'shall' unchanged. No new behavior added."""

LLM_FALLBACK_USER = """Paraphrase this requirement. Modal verb '{modal}' must stay exactly as '{modal}'.

Requirement: {requirement}

Paraphrase:"""


def op_modal_synonym(text: str, modal: str) -> str | None:
    synonyms = MODAL_SYNONYMS.get(modal, [])
    if not synonyms:
        return None
    # Avoid synonyms that produce grammatically wrong sentences
    # e.g. "Customers will be billed" → "Customers is going to be billed" (wrong)
    # Skip multi-word synonyms when subject is plural or passive construction
    text_lower = text.lower()
    is_passive  = bool(re.search(rf'\b{re.escape(modal)}\s+be\b', text_lower))
    is_plural   = bool(re.match(r'^[A-Z][a-z]+s\s', text))  # "Customers will..."

    valid_synonyms = []
    for s in synonyms:
        # Skip "is going to" / "is permitted to" for plural or passive
        if (is_passive or is_plural) and s.startswith("is "):
            continue
        valid_synonyms.append(s)

    if not valid_synonyms:
        return None

    synonym = random.choice(valid_synonyms)
    pattern = re.compile(rf'\b{re.escape(modal)}\b', re.IGNORECASE)
    new_text = pattern.sub(synonym, text, count=1)
    return new_text if new_text != text else None


def op_condition_synonym(text: str) -> str | None:
    for keyword, synonyms in CONDITION_SYNONYMS.items():
        pattern = re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE)
        if pattern.search(text):
            synonym = random.choice(synonyms)
            new_text = pattern.sub(synonym, text, count=1)
            if new_text != text:
                # Fix capitalisation if keyword was at sentence start
                if text[0].isupper() and new_text[0].islower():
                    new_text = new_text[0].upper() + new_text[1:]
                return new_text
    return None


def op_clause_expand(text: str) -> str | None:
    for pattern_str, replacement in [
        (r'\bThe system\b', 'The software system'),
        (r'\bThe user\b', 'The end user'),
        (r'\bThe application\b', 'The software application'),
    ]:
        pattern = re.compile(pattern_str)  # case-sensitive — only expand when capitalised correctly
        if pattern.search(text):
            new_text = pattern.sub(replacement, text, count=1)
            if new_text != text:
                return new_text
    return None


# passive_active removed — produces broken grammar ("identifyed", "authenticateed")
# and is not needed since modal_synonym + condition_synonym + clause_expand
# give sufficient paraphrase variety


RULE_OPS = [
    ("modal_synonym",  op_modal_synonym),
    ("condition_syn",  op_condition_synonym),
    ("clause_expand",  op_clause_expand),
]


def apply_rule_ops(text: str, modal: str) -> list:
    ops = list(RULE_OPS)
    random.shuffle(ops)
    results = []
    seen_outputs = set()

    for op_name, op_fn in ops:
        try:
            if op_name == "modal_synonym":
                result = op_fn(text, modal)
            else:
                result = op_fn(text)

            if not result:
                continue

            # Near-dup check: compare full text, not just first 60 chars
            # (first 60 chars are often identical for small edits)
            result_norm = re.sub(r'\s+', ' ', result.lower().strip())
            text_norm   = re.sub(r'\s+', ' ', text.lower().strip())

            if result_norm == text_norm:
                continue  # identical after normalisation
            if result_norm in seen_outputs:
                continue  # duplicate of another candidate

            # Basic grammar sanity check
            # Reject if common modal replacement produced obviously broken grammar
            if re.search(r'\b(is|are|was|were)\s+going\s+to\b', result.lower()):
                # Only allow "is going to" if subject is singular non-passive
                subj_match = re.match(r'^(\w+)', result)
                if subj_match and result.lower().startswith(subj_match.group(1).lower()):
                    # Skip if plural subject (ends in 's') with "is going to"
                    if subj_match.group(1).endswith('s') and \
                       'is going to' in result.lower():
                        continue

            seen_outputs.add(result_norm)
            results.append((op_name, result))

        except Exception:
            continue

    return results


def verify_paraphrase(original: str, paraphrase: str, client) -> bool:
    if client is None:
        return True
    try:
        response = client.chat.completions.create(
            model=VERIFICATION_MODEL,
            messages=[
                {"role": "system", "content": VERIFY_SYSTEM},
                {"role": "user",   "content": VERIFY_USER.format(
                    original=original, paraphrase=paraphrase)},
            ],
            temperature=0.0,
            max_tokens=5,
        )
        return response.choices[0].message.content.strip().upper().startswith("YES")
    except Exception:
        return True  # trust rule on error


def llm_fallback(requirement: str, modal: str, client) -> str | None:
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": LLM_FALLBACK_SYSTEM},
                {"role": "user",   "content": LLM_FALLBACK_USER.format(
                    modal=modal, requirement=requirement)},
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


def generate_invariance_probes(
    requirement: str, modal: str, client, n: int = 2
) -> list:
    """
    Hybrid: rule-based generation → LLM verification → LLM fallback.

    In dry-run (client=None):
      - Rule-based candidates are accepted without LLM verification
      - LLM fallback is skipped
      - May produce fewer than n probes for items with no matching rules

    With LLM (client set):
      - Rule candidates verified by gpt-4.1-mini (binary YES/NO)
      - o4-mini fallback if rules produce < n verified probes
    """
    probes      = []
    seen_norms  = {re.sub(r'\s+', ' ', requirement.lower().strip())}

    # Step 1: Rule-based candidates
    rule_candidates = apply_rule_ops(requirement, modal)

    # Step 2: Accept or verify each candidate
    for op_name, candidate in rule_candidates:
        if len(probes) >= n:
            break

        cand_norm = re.sub(r'\s+', ' ', candidate.lower().strip())
        if cand_norm in seen_norms:
            continue

        # In dry-run: accept rule output directly (no LLM call)
        # With LLM: verify quality
        if client is not None:
            verified = verify_paraphrase(requirement, candidate, client)
            if not verified:
                continue
            time.sleep(0.2)
        # dry-run: trust the rule

        seen_norms.add(cand_norm)
        probes.append({
            "probe_id":           f"INV_{len(probes)+1}",
            "probe_family":       "invariance",
            "probe_text":         candidate,
            "expected_relation":  "stable",
            "operation":          f"rule_{op_name}",
            "generation_method":  "rule_only" if client is None
                                  else "hybrid_rule_verified",
            "scoring": {
                "criterion": "revised outputs remain semantically aligned "
                             "and preserve same modal strength",
                "pass": 1,
                "fail": 0,
            },
            "validation": {
                "rule_operation":  op_name,
                "llm_verified":    client is not None,
                "modal_preserved": extract_primary_modal(candidate) == modal,
                "length_ratio":    round(
                    len(candidate.split()) / max(len(requirement.split()), 1), 2),
            }
        })

    # Step 3: LLM fallback for remaining slots (only when client available)
    while len(probes) < n and client is not None:
        candidate = llm_fallback(requirement, modal, client)
        if not candidate:
            break

        cand_norm = re.sub(r'\s+', ' ', candidate.lower().strip())
        if cand_norm in seen_norms:
            break

        seen_norms.add(cand_norm)
        probes.append({
            "probe_id":           f"INV_{len(probes)+1}",
            "probe_family":       "invariance",
            "probe_text":         candidate,
            "expected_relation":  "stable",
            "operation":          "llm_fallback",
            "generation_method":  "hybrid_llm_fallback",
            "scoring": {
                "criterion": "revised outputs remain semantically aligned "
                             "and preserve same modal strength",
                "pass": 1,
                "fail": 0,
            },
            "validation": {
                "rule_operation":  None,
                "llm_verified":    True,
                "modal_preserved": extract_primary_modal(candidate) == modal,
                "length_ratio":    round(
                    len(candidate.split()) / max(len(requirement.split()), 1), 2),
            }
        })
        time.sleep(0.3)
        break

    return probes


# ══════════════════════════════════════════════════════════════════════════
# DIRECTIONAL PROBES — Rule-based
# eval_settings: modal change OR condition scope change
# ══════════════════════════════════════════════════════════════════════════

def directional_modal(requirement: str, modal: str) -> dict | None:
    """
    Change modal in predictable direction.
    eval_settings example: "should warn" → "may warn" (weaken)
                           "may retry"   → "must retry" (strengthen)
    """
    if modal in MODAL_WEAKEN:
        new_modal   = MODAL_WEAKEN[modal]
        direction   = "weaken"
        description = f"Modal weakened: '{modal}' → '{new_modal}'"
    elif modal in MODAL_STRENGTHEN:
        new_modal   = MODAL_STRENGTHEN[modal]
        direction   = "strengthen"
        description = f"Modal strengthened: '{modal}' → '{new_modal}'"
    else:
        return None

    pattern  = re.compile(rf'\b{re.escape(modal)}\b', re.IGNORECASE)
    new_text = pattern.sub(new_modal, requirement, count=1)
    if new_text == requirement:
        return None

    return {
        "probe_id":           "DIR_1",
        "probe_family":       "directional",
        "probe_text":         new_text,
        "expected_relation":  "directional",
        "operation":          "modal_substitution",
        "direction":          direction,
        "original_modal":     modal,
        "new_modal":          new_modal,
        "description":        description,
        "scoring": {
            "criterion": "revised output shifts in same direction as source edit — "
                         "does not collapse the modal distinction",
            "pass": 1,
            "fail": 0,
        },
    }


def directional_condition_scope(requirement: str) -> dict | None:
    """
    Change condition scope in predictable direction.
    eval_settings example: "if payment fails" → "if payment fails twice"
    Tries strengtheners first, then weakeners.
    """
    text_lower = requirement.lower()

    # Try strengtheners first
    for original, replacement in CONDITION_STRENGTHENERS:
        if original in text_lower:
            new_text = re.sub(
                rf'\b{re.escape(original)}\b', replacement,
                requirement, count=1, flags=re.IGNORECASE
            )
            if new_text != requirement:
                return {
                    "probe_id":           "DIR_1",
                    "probe_family":       "directional",
                    "probe_text":         new_text,
                    "expected_relation":  "directional",
                    "operation":          "condition_scope_strengthen",
                    "direction":          "strengthen",
                    "original_condition": original,
                    "new_condition":      replacement,
                    "description":        f"Condition strengthened: "
                                         f"'{original}' → '{replacement}'",
                    "scoring": {
                        "criterion": "revised output responds to narrowed/widened "
                                     "condition — does not ignore the scope change",
                        "pass": 1,
                        "fail": 0,
                    },
                }

    # Try weakeners
    for original, replacement in CONDITION_WEAKENERS:
        if original in text_lower:
            new_text = re.sub(
                rf'\b{re.escape(original)}\b', replacement,
                requirement, count=1, flags=re.IGNORECASE
            )
            if new_text != requirement:
                return {
                    "probe_id":           "DIR_1",
                    "probe_family":       "directional",
                    "probe_text":         new_text,
                    "expected_relation":  "directional",
                    "operation":          "condition_scope_weaken",
                    "direction":          "weaken",
                    "original_condition": original,
                    "new_condition":      replacement,
                    "description":        f"Condition weakened: "
                                         f"'{original}' → '{replacement}'",
                    "scoring": {
                        "criterion": "revised output responds to narrowed/widened "
                                     "condition — does not ignore the scope change",
                        "pass": 1,
                        "fail": 0,
                    },
                }

    return None


def generate_directional_probe(requirement: str, modal: str) -> dict | None:
    """
    Try condition scope change first (richer signal per eval_settings).
    Fall back to modal substitution.
    """
    # Prefer condition scope if requirement has a condition keyword
    text_lower = requirement.lower()
    has_condition = any(
        re.search(rf'\b{kw}\b', text_lower)
        for kw in ["if", "when", "while", "where", "unless",
                   "fails", "exceeds", "occurs", "detected"]
    )

    if has_condition:
        probe = directional_condition_scope(requirement)
        if probe:
            return probe

    # Fallback to modal substitution
    return directional_modal(requirement, modal)


# ══════════════════════════════════════════════════════════════════════════
# SHORTCUT PROBES — Rule-based, hallucination-targeted
# eval_settings: insert cue that tempts model to hallucinate extra behavior
# ══════════════════════════════════════════════════════════════════════════

def select_distractor(requirement: str) -> tuple:
    """
    Select distractor category based on requirement content.
    Each category is designed to trigger a specific hallucination failure mode.
    """
    text_lower = requirement.lower()

    # Retry/recovery hallucination: temporal triggers in failure contexts
    # eval_settings example: "temporarily" → model invents retry logic
    if any(w in text_lower for w in ["fail", "error", "invalid", "reject",
                                      "deny", "timeout", "unavailable"]):
        category = "retry_trigger"

    # Escalation hallucination: unexpected/unauthorized events
    elif any(w in text_lower for w in ["access", "login", "authenticate",
                                        "permission", "unauthori"]):
        category = "escalation_trigger"

    # Scope expansion hallucination: universal/all contexts
    elif any(w in text_lower for w in ["all", "every", "each", "any",
                                        "user", "participant", "client"]):
        category = "scope_trigger"

    # Additional response hallucination: action/processing requirements
    else:
        category = "response_trigger"

    distractor = random.choice(SHORTCUT_DISTRACTORS[category])
    return distractor, category


def insert_distractor(text: str, distractor: str, category: str) -> str | None:
    """
    Insert distractor at semantically appropriate position.

    For retry/escalation triggers: insert after condition keyword
    (most likely to trigger hallucination of recovery behavior)

    For scope/response triggers: insert after modal verb
    (most likely to trigger scope expansion)
    """
    # Strategy 1: Insert distractor word WITHIN the condition clause
    # eval_settings: "if payment fails" → "if payment fails temporarily"
    if category == "retry_trigger":
        condition_pattern = re.compile(
            r'(\b(?:if|when|while|unless)\b\s+[^,\.]{3,40}?)(\s*[,\.])',
            re.IGNORECASE
        )
        m = condition_pattern.search(text)
        if m:
            insert_at = m.end(1)
            new_text = text[:insert_at] + " " + distractor + text[insert_at:]
            return new_text.strip()

    # Strategy 2: Insert after modal verb
    modal_pattern = re.compile(
        r'(\b(?:shall|should|must|may|will|can)\b)',
        re.IGNORECASE
    )
    m = modal_pattern.search(text)
    if m:
        # Only insert if distractor is a word/short phrase (not a clause)
        if len(distractor.split()) <= 3:
            insert_at = m.end()
            new_text = (text[:insert_at] + " " + distractor +
                       " " + text[insert_at:].lstrip())
            return new_text.strip()

    # Strategy 3: Append before final period
    if text.rstrip().endswith('.'):
        new_text = text.rstrip()[:-1] + ", " + distractor + "."
        return new_text

    # Strategy 4: Append at end
    return text.rstrip() + " " + distractor + "."


def generate_shortcut_probe(requirement: str) -> dict | None:
    """
    Generate shortcut probe by inserting hallucination-targeted distractor.
    eval_settings: "if payment fails" → "if payment fails temporarily"
    Expected relation: no_shortcut
    Scoring: 1 if model does NOT hallucinate extra behavior, 0 if it does.
    """
    distractor, category = select_distractor(requirement)
    new_text = insert_distractor(requirement, distractor, category)

    if not new_text or new_text == requirement:
        return None

    # Map category to hallucination failure mode for scoring guidance
    failure_mode_map = {
        "retry_trigger":     "hallucinate retry or recovery action",
        "escalation_trigger":"hallucinate escalation or notification step",
        "scope_trigger":     "hallucinate scope expansion to all cases",
        "response_trigger":  "hallucinate additional system response",
    }

    return {
        "probe_id":            "SHC_1",
        "probe_family":        "shortcut",
        "probe_text":          new_text,
        "expected_relation":   "no_shortcut",
        "operation":           "distractor_insertion",
        "distractor":          distractor,
        "distractor_category": category,
        "hallucination_risk":  failure_mode_map[category],
        "description": (f"Distractor '{distractor}' inserted to trigger "
                       f"{failure_mode_map[category]}"),
        "scoring": {
            "criterion": "model does NOT hallucinate extra behavior "
                         "in response to distractor cue",
            "pass": 1,
            "fail": 0,
        },
    }


# ══════════════════════════════════════════════════════════════════════════
# PROBE NEIGHBORHOOD BUILDER
# ══════════════════════════════════════════════════════════════════════════

def build_probe_neighborhood(item: dict, client) -> dict:
    """
    Build complete probe neighborhood for one base item.
    Source-family alignment per eval_settings:
      Großer → invariance + directional (clean conformant starting points)
      PURE/PROMISE → all families, shortcut preferred
    """
    item        = deepcopy(item)
    requirement = item["requirement_text"]
    modal       = item.get("modal") or extract_primary_modal(requirement) or "shall"
    source      = item.get("source", "PURE")
    probes      = []
    stats       = {"invariance": 0, "directional": 0, "shortcut": 0, "failed": 0}

    # ── A. Invariance probes (hybrid) ─────────────────────────────────
    inv_probes = generate_invariance_probes(requirement, modal, client, n=2)
    probes.extend(inv_probes)
    stats["invariance"] = len(inv_probes)
    stats["failed"]    += max(0, 2 - len(inv_probes))

    # ── B. Directional probe ──────────────────────────────────────────
    dir_probe = generate_directional_probe(requirement, modal)
    if dir_probe:
        probes.append(dir_probe)
        stats["directional"] = 1
    else:
        stats["failed"] += 1

    # ── C. Shortcut probe ─────────────────────────────────────────────
    shc_probe = generate_shortcut_probe(requirement)
    if shc_probe:
        probes.append(shc_probe)
        stats["shortcut"] = 1
    else:
        stats["failed"] += 1

    item["probe_neighborhoods"] = probes
    item["probe_stats"]         = stats
    item["probe_count"]         = len(probes)
    item["probe_family_tags"]   = list({p["probe_family"] for p in probes})

    return item


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def run(input_path, output_path, api_key, limit, seed, dry_run):
    random.seed(seed)

    items = json.loads(Path(input_path).read_text(encoding='utf-8'))
    print(f"\n{'='*65}")
    print(f"PROBE GENERATOR — Final Version")
    print(f"Grounded in evaluation_settings.md")
    print(f"{'='*65}")
    print(f"  Input items : {len(items)}")
    print(f"  Limit       : {limit or 'all'}")
    print(f"  Dry run     : {dry_run}")

    if limit:
        items = items[:limit]

    # Init client
    client = None
    if api_key and not dry_run and OPENAI_AVAILABLE:
        client = OpenAI(api_key=api_key)
        print(f"  Generation  : {GENERATION_MODEL} (reasoning model)")
        print(f"  Verification: {VERIFICATION_MODEL} (binary YES/NO)")
    else:
        print(f"  LLM         : disabled — rule-based only")

    results      = []
    probe_counts = Counter()
    op_counts    = Counter()
    failed_items = []

    for i, item in enumerate(items):
        print(f"  [{i+1:>4}/{len(items)}] {item['item_id'][:55]}", end="  ")

        processed = build_probe_neighborhood(item, client)
        stats     = processed["probe_stats"]
        n_probes  = processed["probe_count"]

        print(f"→ {n_probes} probes "
              f"(inv={stats['invariance']} "
              f"dir={stats['directional']} "
              f"shc={stats['shortcut']} "
              f"fail={stats['failed']})")

        for p in processed["probe_neighborhoods"]:
            probe_counts[p["probe_family"]] += 1
            op_counts[p.get("operation", "unknown")] += 1

        if n_probes == 0:
            failed_items.append(item['item_id'])

        results.append(processed)

    # Summary
    total_probes = sum(probe_counts.values())
    print(f"\n{'─'*65}")
    print(f"SUMMARY")
    print(f"{'─'*65}")
    print(f"  Items processed    : {len(results)}")
    print(f"  Total probe pairs  : {total_probes}")
    print(f"  Invariance probes  : {probe_counts['invariance']}")
    print(f"  Directional probes : {probe_counts['directional']}")
    print(f"  Shortcut probes    : {probe_counts['shortcut']}")
    print(f"  Items with 0 probes: {len(failed_items)}")
    print(f"\n  Operation breakdown:")
    for op, cnt in op_counts.most_common():
        print(f"    {op:<35} : {cnt}")

    # Show sample probe neighborhoods
    sample_items = [r for r in results if r['probe_count'] >= 3][:2]
    for item in sample_items:
        print(f"\n  SAMPLE: {item['item_id']}")
        print(f"  BASE  : {item['requirement_text']}")
        for p in item['probe_neighborhoods']:
            tag = p['probe_family'][:3].upper()
            op  = p.get('operation','')[:20]
            print(f"  [{tag}] ({op:<20}) {p['probe_text'][:90]}")

    # Save
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n  Saved → {out_path}  ({len(results)} items, {total_probes} probes)")

    report = {
        "items_processed":     len(results),
        "total_probes":        total_probes,
        "probe_counts":        dict(probe_counts),
        "operation_counts":    dict(op_counts),
        "avg_probes_per_item": round(total_probes / max(len(results), 1), 2),
        "items_with_0_probes": len(failed_items),
        "failed_item_ids":     failed_items[:10],
        "grounded_in":         "evaluation_settings.md",
        "probe_families": {
            "invariance": {
                "expected_relation": "stable",
                "scoring": "1 if outputs remain semantically aligned "
                           "with same modal strength, 0 otherwise",
                "method": "hybrid rule-based + LLM verification",
            },
            "directional": {
                "expected_relation": "directional",
                "scoring": "1 if output shifts in same direction as source edit, "
                           "0 if it collapses the distinction",
                "operations": ["modal_substitution", "condition_scope_strengthen",
                               "condition_scope_weaken"],
            },
            "shortcut": {
                "expected_relation": "no_shortcut",
                "scoring": "1 if model does NOT hallucinate extra behavior, 0 if it does",
                "distractor_categories": list(SHORTCUT_DISTRACTORS.keys()),
                "hallucination_failure_modes": [
                    "hallucinate retry or recovery action",
                    "hallucinate escalation or notification step",
                    "hallucinate scope expansion to all cases",
                    "hallucinate additional system response",
                ],
            },
        },
    }

    report_path = out_path.parent / "probe_generation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"  Saved → {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate probe neighborhoods grounded in evaluation_settings.md"
    )
    parser.add_argument("--input",   required=True)
    parser.add_argument("--output",  required=True)
    parser.add_argument("--api-key", default=None,
                        help="OpenAI API key. Overrides OPENAI_API_KEY env var.")
    parser.add_argument("--limit",   type=int, default=None)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM — rule-based probes only")
    args = parser.parse_args()

    # API key priority: CLI > env var > None
    api_key = args.api_key or API_KEY
    if not api_key and not args.dry_run:
        print("[WARN] No API key found. Set OPENAI_API_KEY env var or use --api-key.")
        print("       Falling back to dry-run mode (rule-based only).")
        args.dry_run = True

    run(args.input, args.output, api_key,
        args.limit, args.seed, args.dry_run)


if __name__ == "__main__":
    main()
