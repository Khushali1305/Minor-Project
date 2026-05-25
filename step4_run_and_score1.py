#!/usr/bin/env python3
"""
Step 4: Run Models → Rule Score → LLM Judges → Comprehensive Results

KEY FIX (Priority 1):
  Semantic invariance probes are now scored JUDGE-ONLY.
  Word F1 is invalid for semantic paraphrases (different words, same meaning).
  rule_score = None for semantic probes → deferred entirely to judge verdict.
  Reliability calculations skip None scores (no deflation of numbers).

Scoring method per probe subtype:
  invariance + lexical/syntactic  →  rule_score (word F1 + readability) THEN judge
  invariance + semantic           →  judge ONLY  (rule_score = None)
  directional                     →  rule_score only (readability shift + overlap)
  shortcut                        →  rule_score only (critical detail preservation)

FIXES APPLIED:
  - --judge flag: run only one judge at a time (avoids shared rate-limit contention)
  - Per-judge thread count via JUDGES config "threads" key
  - Short exponential backoff (base=2s) for both Anthropic and Mistral
  - Retry logic moved into phase_judge directly (not api_client dependency)

Phases: run | score | judge | results | all
"""
import json, re, os, csv, sys, time, argparse, threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import (EXPERIMENTAL_MODELS, JUDGES, API_DELAY,
                    SIMPLIFY_SYSTEM, SIMPLIFY_USER_CEFR, SIMPLIFY_USER_GENERIC,
                    JUDGE_SYSTEM, JUDGE_USER, BERTSCORE_THRESHOLD,
                    READABILITY_DELTA_MAX, JUDGE_POLICY, API_THREADS)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(p):
    if not os.path.exists(p):
        return []
    with open(p, 'r', encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(d, p):
    with open(p, 'a', encoding='utf-8') as f:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')
        f.flush()


def save_jsonl(data, p):
    with open(p, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')


def save_csv(rows, path, fieldnames=None):
    if not rows:
        return
    if not fieldnames:
        fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def get_prompt(text, target):
    if target in ('a2', 'b1'):
        return SIMPLIFY_USER_CEFR.format(
            target={'a2': 'A2 (elementary)', 'b1': 'B1 (intermediate)'}[target],
            text=text)
    return SIMPLIFY_USER_GENERIC.format(text=text)


def is_local(prov):
    return prov in ("local", "hf")


def reliability(items, key='final_score'):
    """Mean of final_score, skipping None values (semantic probes pre-judge)."""
    valid = [i for i in items if i.get(key) is not None]
    return round(sum(i[key] for i in valid) / len(valid), 4) if valid else 0.0


def reliability_n(items, key='final_score'):
    """Count of probes with a valid (non-None) final score."""
    return sum(1 for i in items if i.get(key) is not None)


# ── rate-limit-aware API call with retries ────────────────────────────────────

def call_judge_api(provider, model_id, max_tokens, prompt, system):
    """
    Calls the judge API with exponential backoff on 429s.
    Base wait = 2s → 2, 4, 8, 16, 32, 64s across 6 attempts.
    Keeps api_client.py clean; all retry logic lives here.
    """
    from api_client import call_model

    last_exc = None
    for attempt in range(6):
        try:
            return call_model(prompt, system, provider, model_id, max_tokens)
        except Exception as e:
            msg = str(e)
            is_429 = ('429' in msg or
                      'rate_limit' in msg.lower() or
                      'rate limit' in msg.lower() or
                      'Rate limit' in msg)
            if is_429:
                wait = (2 ** attempt) * 2   # 2, 4, 8, 16, 32, 64 seconds
                print(f"  [{provider} 429] attempt {attempt+1}/6, sleeping {wait}s...",
                      flush=True)
                time.sleep(wait)
                last_exc = e
            else:
                raise  # non-rate-limit error → propagate immediately

    raise RuntimeError(
        f"{provider} rate limit: max retries exceeded. Last error: {last_exc}"
    )


# ═══════════════════════════════════════════════════════════
# PHASE 1: RUN
# ═══════════════════════════════════════════════════════════

def phase_run(args):
    from api_client import call_model
    items  = load_jsonl(f"{args.input_dir}/simp_master_table.jsonl")
    probes = load_jsonl(f"{args.input_dir}/simp_probecore_v1.jsonl")
    print(f"Base items: {len(items)}, Frozen probes: {len(probes)}", flush=True)

    base_queue = []
    for it in items:
        if it.get('ose_level') in (None, 'advanced') or it['dataset'] != 'OneStopEnglish':
            base_queue.append({
                'item_id': it['item_id'],
                'text': it['original'],
                'target': it.get('target_level', 'simpler'),
                'type': 'base',
                'dataset': it['dataset']
            })

    shared_probes = []
    semantic_by_model = defaultdict(list)
    for p in probes:
        if p.get('probe_subtype') == 'semantic':
            semantic_by_model[p.get('generator_model', '')].append(p)
        else:
            shared_probes.append(p)

    print(f"Base: {len(base_queue)}, Shared: {len(shared_probes)}", flush=True)
    print(f"Semantic by model: {dict((k, len(v)) for k, v in semantic_by_model.items())}", flush=True)

    models = EXPERIMENTAL_MODELS
    if args.models:
        names = set(args.models.split(','))
        models = [m for m in models if m['name'] in names]
    out_dir = f"{args.output_dir}/simp_model_outputs"
    os.makedirs(out_dir, exist_ok=True)

    for mcfg in models:
        mn = mcfg['name']
        out_path = f"{out_dir}/{mn}.jsonl"
        model_probes = shared_probes + semantic_by_model.get(mn, [])
        probe_queue = [{
            'item_id': p['probe_id'],
            'text': p.get('probe_text', ''),
            'target': p.get('target_level', 'simpler'),
            'type': 'probe',
            'probe_family': p.get('probe_family', ''),
            'base_item_id': p.get('base_item_id', ''),
            'dataset': p.get('dataset', '')
        } for p in model_probes]
        full_queue = base_queue + probe_queue

        done = set()
        if os.path.exists(out_path):
            for r in load_jsonl(out_path):
                done.add(r['item_id'])
        pending = [q for q in full_queue if q['item_id'] not in done]
        print(f"\n  {mn}: {len(pending)} pending (done={len(done)})", flush=True)
        if not pending:
            continue

        def _make_rec(q, output):
            rec = {
                'item_id': q['item_id'], 'model': mn, 'output': output,
                'output_words': len(output.split()), 'input_words': len(q['text'].split()),
                'type': q['type'], 'target': q['target'], 'dataset': q.get('dataset', '')
            }
            if q['type'] == 'probe':
                rec['probe_family'] = q.get('probe_family', '')
                rec['base_item_id'] = q.get('base_item_id', '')
            return rec

        errors = 0

        if is_local(mcfg['provider']):
            # Local/HF: sequential (GPU serialised anyway)
            for idx, q in enumerate(pending):
                try:
                    output = call_model(
                        get_prompt(q['text'], q['target']), SIMPLIFY_SYSTEM,
                        mcfg['provider'], mcfg['model_id'], mcfg.get('max_tokens', 2048)
                    ).strip()
                    append_jsonl(_make_rec(q, output), out_path)
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        print(f"    ERR {q['item_id']}: {e}", flush=True)
                if (idx + 1) % 50 == 0:
                    print(f"    [{idx+1}/{len(pending)}] errors={errors}", flush=True)
            try:
                from api_client import unload_hf_model
                unload_hf_model(mcfg['model_id'])
            except:
                pass
        else:
            # API providers: threaded I/O parallelism
            write_lock = threading.Lock()
            completed = 0
            n_threads = getattr(args, 'threads', None) or API_THREADS

            def _call_one_run(q):
                return q, call_model(
                    get_prompt(q['text'], q['target']), SIMPLIFY_SYSTEM,
                    mcfg['provider'], mcfg['model_id'], mcfg.get('max_tokens', 2048)
                ).strip()

            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                futures = {pool.submit(_call_one_run, q): q for q in pending}
                for fut in as_completed(futures):
                    nonlocal_q = futures[fut]
                    try:
                        q, output = fut.result()
                        with write_lock:
                            append_jsonl(_make_rec(q, output), out_path)
                    except Exception as e:
                        with write_lock:
                            errors += 1
                            if errors <= 5:
                                print(f"    ERR {nonlocal_q['item_id']}: {e}", flush=True)
                    with write_lock:
                        completed += 1
                        if completed % 50 == 0:
                            print(f"    [{completed}/{len(pending)}] errors={errors}", flush=True)

        print(f"  {mn}: done. errors={errors}", flush=True)


# ═══════════════════════════════════════════════════════════
# PHASE 2: RULE-BASED SCORING
# ═══════════════════════════════════════════════════════════

def phase_score(args):
    try:
        import textstat
        def readability(t):
            return textstat.flesch_kincaid_grade(t) if t.strip() else 0
    except ImportError:
        def readability(t):
            w = t.split()
            return sum(len(x) for x in w) / len(w) if w else 0

    def word_f1(ref, hyp):
        rw, hw = set(ref.lower().split()), set(hyp.lower().split())
        if not rw or not hw:
            return 0
        p = len(rw & hw) / len(hw)
        r = len(rw & hw) / len(rw)
        return 2 * p * r / (p + r) if (p + r) else 0

    probes = {p['probe_id']: p for p in load_jsonl(f"{args.input_dir}/simp_probecore_v1.jsonl")}
    out_dir = f"{args.input_dir}/simp_model_outputs"
    scores_dir = f"{args.output_dir}/simp_scores"
    os.makedirs(scores_dir, exist_ok=True)

    for mf in sorted(os.listdir(out_dir)):
        if not mf.endswith('.jsonl'):
            continue
        mn = mf.replace('.jsonl', '')
        outputs = {r['item_id']: r for r in load_jsonl(f"{out_dir}/{mf}")}
        base_out  = {k: v for k, v in outputs.items() if v.get('type') == 'base'}
        probe_out = {k: v for k, v in outputs.items() if v.get('type') == 'probe'}
        print(f"\n  Scoring {mn}: base={len(base_out)}, probe={len(probe_out)}", flush=True)

        scored = []
        sem_deferred = 0

        for pid, po in probe_out.items():
            pm = probes.get(pid)
            if not pm:
                continue
            fam     = pm.get('probe_family', '')
            subtype = pm.get('probe_subtype', '')
            bo      = base_out.get(pm.get('base_item_id', ''))
            if fam in ('invariance', 'directional') and not bo:
                continue

            base_txt  = bo.get('output', '') if bo else ''
            probe_txt = po.get('output', '')
            base_read = readability(base_txt)
            probe_read = readability(probe_txt)
            sim = word_f1(base_txt, probe_txt)

            row = {
                'probe_id':             pid,
                'model':                mn,
                'probe_family':         fam,
                'probe_subtype':        subtype,
                'tier':                 pm.get('tier', 1),
                'dataset':              pm.get('dataset', ''),
                'generator_model':      pm.get('generator_model', ''),
                'base_item_id':         pm.get('base_item_id', ''),
                'base_output_preview':  base_txt[:200],
                'probe_output_preview': probe_txt[:200],
                'similarity':           round(sim, 4),
                'base_readability':     round(base_read, 2),
                'probe_readability':    round(probe_read, 2),
                'read_delta':           round(abs(base_read - probe_read), 2),
            }

            if fam == 'invariance':
                if subtype == 'semantic':
                    # ── FIX: semantic probes deferred entirely to judge ──────
                    row['rule_score']     = None
                    row['scoring_method'] = 'judge_only'
                    row['rule_reason']    = (
                        'semantic_probe:word_f1_invalid_for_paraphrase'
                        ':deferred_to_judge'
                    )
                    sem_deferred += 1
                else:
                    # ── lexical / syntactic: word F1 valid ───────────────────
                    ok = (sim >= BERTSCORE_THRESHOLD and
                          abs(base_read - probe_read) <= READABILITY_DELTA_MAX)
                    row['rule_score']     = 1 if ok else 0
                    row['scoring_method'] = 'rule_then_judge'
                    row['rule_reason']    = (
                        f"sim={'PASS' if sim >= BERTSCORE_THRESHOLD else 'FAIL'}({sim:.3f}),"
                        f"read_delta={'PASS' if abs(base_read - probe_read) <= READABILITY_DELTA_MAX else 'FAIL'}"
                        f"({abs(base_read - probe_read):.1f})"
                    )

            elif fam == 'directional':
                overlap = word_f1(base_txt, probe_txt)
                ok = abs(base_read - probe_read) > 0.5 and overlap < 0.95
                row['rule_score']     = 1 if ok else 0
                row['scoring_method'] = 'rule_only'
                row['overlap']        = round(overlap, 4)
                row['rule_reason']    = (
                    f"read_shift={'PASS' if abs(base_read - probe_read) > 0.5 else 'FAIL'}"
                    f"({base_read - probe_read:.1f}),"
                    f"diff={'PASS' if overlap < 0.95 else 'FAIL'}({overlap:.3f})"
                )

            elif fam == 'shortcut':
                detail = pm.get('critical_detail', '')
                row['critical_detail'] = detail
                if detail:
                    dw = set(re.findall(r'[a-z]+', detail.lower())) - {
                        'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to',
                        'for', 'and', 'or', 'is', 'are'
                    }
                    ow = set(re.findall(r'[a-z]+', probe_txt.lower()))
                    preserved    = (len(dw & ow) / len(dw) >= 0.7) if dw else True
                    exact_match  = detail.lower() in probe_txt.lower()
                    row['detail_word_overlap'] = round(len(dw & ow) / len(dw), 3) if dw else 1.0
                else:
                    preserved = True
                    exact_match = True
                    row['detail_word_overlap'] = 1.0
                row['rule_score']         = 1 if preserved else 0
                row['scoring_method']     = 'rule_only'
                row['detail_exact_match'] = exact_match
                row['rule_reason']        = (
                    f"detail_preserved={'YES' if preserved else 'NO'},"
                    f"exact={'YES' if exact_match else 'NO'}"
                )

            scored.append(row)

        save_jsonl(scored, f"{scores_dir}/{mn}_rule_scores.jsonl")

        by_f = defaultdict(list)
        for s in scored:
            by_f[s['probe_family']].append(s)

        def rel_print(items):
            valid = [i for i in items if i.get('rule_score') is not None]
            r = sum(i['rule_score'] for i in valid) / len(valid) if valid else 0
            return f"{r:.1%}(n={len(valid)},deferred={len(items)-len(valid)})"

        print(
            f"    inv={rel_print(by_f['invariance'])} "
            f"dir={rel_print(by_f['directional'])} "
            f"sc={rel_print(by_f['shortcut'])} "
            f"| semantic_deferred={sem_deferred}",
            flush=True
        )


# ═══════════════════════════════════════════════════════════
# PHASE 3: LLM JUDGES
# ═══════════════════════════════════════════════════════════

def phase_judge(args):
    probes = {p['probe_id']: p for p in load_jsonl(f"{args.input_dir}/simp_probecore_v1.jsonl")}

    # Judge ALL invariance probes (lexical + syntactic + semantic)
    inv_probes = {k: v for k, v in probes.items() if v.get('probe_family') == 'invariance'}
    sem_count  = sum(1 for v in inv_probes.values() if v.get('probe_subtype') == 'semantic')
    lex_count  = sum(1 for v in inv_probes.values() if v.get('probe_subtype') != 'semantic')
    print(
        f"Invariance probes to judge: {len(inv_probes)} "
        f"(semantic={sem_count}, lexical/syntactic={lex_count})",
        flush=True
    )

    # ── Filter to a single judge if --judge flag is set ──────────────────────
    judges_to_run = JUDGES
    if getattr(args, 'judge', None):
        judges_to_run = [j for j in JUDGES if j['name'] == args.judge]
        if not judges_to_run:
            available = [j['name'] for j in JUDGES]
            raise ValueError(
                f"Judge '{args.judge}' not found in config. "
                f"Available: {available}"
            )
        print(f"Running single judge: {args.judge}", flush=True)

    judge_dir = f"{args.output_dir}/simp_judge_outputs"
    os.makedirs(judge_dir, exist_ok=True)

    def _run_judge(jcfg):
        """Run one judge over all its pending probes, threaded internally."""
        jn       = jcfg['name']
        out_path = f"{judge_dir}/{jn}_decisions.jsonl"
        done     = set()
        if os.path.exists(out_path):
            for r in load_jsonl(out_path):
                done.add(r['probe_id'])
        pending = [(pid, pm) for pid, pm in inv_probes.items() if pid not in done]
        print(f"\n  Judge {jn}: {len(pending)} pending (done={len(done)})", flush=True)
        if not pending:
            return

        write_lock = threading.Lock()
        errors    = 0
        completed = 0

        # Per-judge thread count: jcfg['threads'] → args.threads → API_THREADS
        n_threads = (
            jcfg.get('threads') or
            getattr(args, 'threads', None) or
            API_THREADS
        )
        print(f"  Judge {jn}: using {n_threads} thread(s)", flush=True)

        def _call_one_judge(pid_pm):
            pid, pm = pid_pm
            prompt = JUDGE_USER.format(
                source=pm.get('base_text', '')[:500],
                probe=pm.get('probe_text', '')[:500]
            )
            # Use rate-limit-aware wrapper instead of call_model directly
            resp = call_judge_api(
                jcfg['provider'], jcfg['model_id'],
                jcfg.get('max_tokens', 512),
                prompt, JUDGE_SYSTEM
            )
            resp = re.sub(r'^```json\s*', '', resp.strip())
            resp = re.sub(r'\s*```$', '', resp)
            v = json.loads(resp)
            return pid, pm, v

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = {pool.submit(_call_one_judge, item): item for item in pending}
            for fut in as_completed(futures):
                pid_pm = futures[fut]
                pid, pm = pid_pm
                try:
                    pid, pm, v = fut.result()
                    rec = {
                        'probe_id':           pid,
                        'judge':              jn,
                        'equivalent':         v.get('equivalent', False),
                        'confidence':         v.get('confidence', 'low'),
                        'failures':           v.get('failures', []),
                        'reason':             v.get('reason', ''),
                        'probe_subtype':      pm.get('probe_subtype', ''),
                        'generator_model':    pm.get('generator_model', ''),
                        'tier':               pm.get('tier', 1),
                        'dataset':            pm.get('dataset', ''),
                        'base_text_preview':  pm.get('base_text', '')[:200],
                        'probe_text_preview': pm.get('probe_text', '')[:200],
                    }
                except json.JSONDecodeError as e:
                    with write_lock:
                        errors += 1
                    rec = {'probe_id': pid, 'judge': jn, 'equivalent': None,
                           'confidence': 'parse_error', 'failures': ['JSON_PARSE'],
                           'reason': str(e)[:200]}
                except Exception as e:
                    with write_lock:
                        errors += 1
                        if errors <= 5:
                            print(f"    ERR {pid}: {e}", flush=True)
                    rec = {'probe_id': pid, 'judge': jn, 'equivalent': None,
                           'confidence': 'error', 'failures': ['API_ERROR'],
                           'reason': str(e)[:200]}
                with write_lock:
                    append_jsonl(rec, out_path)
                    completed += 1
                    if completed % 50 == 0:
                        print(f"    [{completed}/{len(pending)}] errors={errors}", flush=True)

        print(f"  {jn}: done. errors={errors}", flush=True)

    # Run judges — parallel if multiple, single thread if only one
    if len(judges_to_run) > 1:
        with ThreadPoolExecutor(max_workers=len(judges_to_run)) as judge_pool:
            list(judge_pool.map(_run_judge, judges_to_run))
    else:
        _run_judge(judges_to_run[0])

    # Merge into agreement file only when both judges have run
    all_judge_files = [
        f"{judge_dir}/{jcfg['name']}_decisions.jsonl"
        for jcfg in JUDGES
    ]
    if not all(os.path.exists(p) for p in all_judge_files):
        print(
            "\nSkipping agreement merge — not all judge output files exist yet.\n"
            "Run the other judge first, then re-run --phase judge (or --phase results).",
            flush=True
        )
        return

    print("\nMerging judge decisions...", flush=True)
    all_dec = defaultdict(dict)
    for jcfg in JUDGES:
        for d in load_jsonl(f"{judge_dir}/{jcfg['name']}_decisions.jsonl"):
            all_dec[d['probe_id']][d['judge']] = d

    agree = disagree = 0
    details = []
    for pid, judges in all_dec.items():
        if len(judges) < 2:
            continue
        j1, j2  = list(judges.values())[:2]
        agreed  = j1.get('equivalent') == j2.get('equivalent')
        if agreed:
            agree += 1
        else:
            disagree += 1
        details.append({
            'probe_id':  pid,
            'j1_name':   j1['judge'],
            'j1_equiv':  j1.get('equivalent'),
            'j2_name':   j2['judge'],
            'j2_equiv':  j2.get('equivalent'),
            'agreed':    agreed,
        })

    save_jsonl(details, f"{judge_dir}/judge_agreement.jsonl")
    save_jsonl(
        [d for d in details if not d['agreed']],
        f"{judge_dir}/judge_disagreements.jsonl"
    )
    total = agree + disagree
    if total:
        print(f"Agreement: {agree}/{total} ({agree/total:.1%})", flush=True)


# ═══════════════════════════════════════════════════════════
# PHASE 4: COMPREHENSIVE RESULTS
# ═══════════════════════════════════════════════════════════

def phase_results(args):
    scores_dir = f"{args.output_dir}/simp_scores"
    judge_dir  = f"{args.output_dir}/simp_judge_outputs"
    probes_all = {p['probe_id']: p for p in load_jsonl(f"{args.input_dir}/simp_probecore_v1.jsonl")}
    os.makedirs(scores_dir, exist_ok=True)

    # Load judge decisions
    judge_by_probe = defaultdict(dict)
    for jcfg in JUDGES:
        for d in load_jsonl(f"{judge_dir}/{jcfg['name']}_decisions.jsonl"):
            judge_by_probe[d['probe_id']][d['judge']] = d

    # Build judge agreement map
    agreement = {}
    if os.path.exists(f"{judge_dir}/judge_agreement.jsonl"):
        for d in load_jsonl(f"{judge_dir}/judge_agreement.jsonl"):
            if JUDGE_POLICY == "agree_only":
                agreement[d['probe_id']] = d.get('agreed', False) and bool(d.get('j1_equiv'))
            else:
                agreement[d['probe_id']] = bool(d.get('j1_equiv')) or bool(d.get('j2_equiv'))

    # Build instance-level rows
    all_instances = []
    model_scored  = {}

    for sf in sorted(os.listdir(scores_dir)):
        if not sf.endswith('_rule_scores.jsonl'):
            continue
        mn     = sf.replace('_rule_scores.jsonl', '')
        scored = load_jsonl(f"{scores_dir}/{sf}")
        rows   = []

        for s in scored:
            pid    = s['probe_id']
            jv     = agreement.get(pid)
            jd     = judge_by_probe.get(pid, {})
            jnames = list(jd.keys())
            method = s.get('scoring_method', 'rule_only')

            # ── Final score logic ────────────────────────────────────────────
            if s['probe_family'] == 'invariance':
                if jv is not None:
                    # Judge ran → use judge verdict for ALL invariance probes
                    final = 1 if jv else 0
                elif method == 'judge_only':
                    # Semantic probe, judge not yet run → exclude from reliability
                    final = None
                else:
                    # Lexical/syntactic, judge not yet run → fall back to rule
                    final = s.get('rule_score')
            else:
                # Directional / shortcut → rule score is final
                final = s.get('rule_score')

            row = {
                **s,
                'rule_score':         s.get('rule_score'),
                'judge_verified':     jv,
                'final_score':        final,
                'judge_1_name':       jnames[0] if jnames else '',
                'judge_1_equiv':      jd[jnames[0]].get('equivalent', '') if jnames else '',
                'judge_1_confidence': jd[jnames[0]].get('confidence', '') if jnames else '',
                'judge_2_name':       jnames[1] if len(jnames) > 1 else '',
                'judge_2_equiv':      jd[jnames[1]].get('equivalent', '') if len(jnames) > 1 else '',
                'judge_2_confidence': jd[jnames[1]].get('confidence', '') if len(jnames) > 1 else '',
            }
            rows.append(row)
            all_instances.append(row)
        model_scored[mn] = rows

    # ── Aggregated Output 1: Reliability Summary ──────────────────
    summaries = []
    for mn, scored in sorted(model_scored.items()):
        by_f  = defaultdict(list)
        by_ft = defaultdict(list)
        for s in scored:
            by_f[s['probe_family']].append(s)
            by_ft[(s['probe_family'], s.get('tier', 1))].append(s)
        row = {
            'model':       mn,
            'total':       len(scored),
            'inv_rel':     reliability(by_f['invariance']),
            'inv_n':       reliability_n(by_f['invariance']),
            'inv_n_total': len(by_f['invariance']),
            'dir_rel':     reliability(by_f['directional']),
            'dir_n':       len(by_f['directional']),
            'sc_rel':      reliability(by_f['shortcut']),
            'sc_n':        len(by_f['shortcut']),
            'global_rel':  reliability(scored),
            'global_n':    reliability_n(scored),
        }
        for fam in ['invariance', 'directional', 'shortcut']:
            for tier in [1, 2]:
                ti = by_ft.get((fam, tier), [])
                row[f'{fam}_t{tier}']   = reliability(ti)
                row[f'{fam}_t{tier}_n'] = reliability_n(ti)
        summaries.append(row)

    save_csv(summaries, f"{scores_dir}/reliability_summary.csv")
    with open(f"{scores_dir}/reliability_summary.json", 'w') as f:
        json.dump(summaries, f, indent=2)
    print(f"\n  ✓ reliability_summary.csv ({len(summaries)} models)", flush=True)

    # ── Aggregated Output 2: Per-Dataset ──────────────────────────
    ds_agg = defaultdict(lambda: defaultdict(list))
    for inst in all_instances:
        ds_agg[inst['model']][(inst['dataset'], inst['probe_family'])].append(inst)
    ds_rows = []
    for mn in sorted(ds_agg):
        for (ds, fam), items in sorted(ds_agg[mn].items()):
            ds_rows.append({
                'model':       mn,
                'dataset':     ds,
                'family':      fam,
                'reliability': reliability(items),
                'valid_n':     reliability_n(items),
                'total_n':     len(items),
            })
    save_csv(ds_rows, f"{scores_dir}/reliability_by_dataset.csv")
    print(f"  ✓ reliability_by_dataset.csv ({len(ds_rows)} rows)", flush=True)

    # ── Aggregated Output 3: Per-Tier ─────────────────────────────
    tier_rows = []
    for mn, scored in sorted(model_scored.items()):
        by_ft = defaultdict(list)
        for s in scored:
            by_ft[(s['probe_family'], s.get('tier', 1))].append(s)
        for fam in ['invariance', 'directional', 'shortcut']:
            t1 = by_ft.get((fam, 1), [])
            t2 = by_ft.get((fam, 2), [])
            r1 = reliability(t1)
            r2 = reliability(t2)
            tier_rows.append({
                'model':            mn,
                'family':           fam,
                'tier_1':           r1,
                'tier_1_n':         reliability_n(t1),
                'tier_2':           r2,
                'tier_2_n':         reliability_n(t2),
                'gap_t1_minus_t2':  round(r1 - r2, 4),
            })
    save_csv(tier_rows, f"{scores_dir}/reliability_by_tier.csv")
    print(f"  ✓ reliability_by_tier.csv ({len(tier_rows)} rows)", flush=True)

    # ── Aggregated Output 4: Per-Subtype ──────────────────────────
    sub_agg = defaultdict(lambda: defaultdict(list))
    for inst in all_instances:
        sub_agg[inst['model']][inst.get('probe_subtype', 'unknown')].append(inst)
    sub_rows = []
    for mn in sorted(sub_agg):
        for sub, items in sorted(sub_agg[mn].items()):
            fam = items[0]['probe_family'] if items else ''
            sub_rows.append({
                'model':         mn,
                'probe_subtype': sub,
                'family':        fam,
                'reliability':   reliability(items),
                'valid_n':       reliability_n(items),
                'total_n':       len(items),
            })
    save_csv(sub_rows, f"{scores_dir}/reliability_by_subtype.csv")
    print(f"  ✓ reliability_by_subtype.csv ({len(sub_rows)} rows)", flush=True)

    # ── Aggregated Output 5: Judge Agreement ──────────────────────
    all_agree = load_jsonl(f"{judge_dir}/judge_agreement.jsonl")
    total_j   = len(all_agree)
    n_agree   = sum(1 for d in all_agree if d.get('agreed'))
    cfreq     = Counter()
    for pid, judges in judge_by_probe.items():
        jl = list(judges.values())
        if len(jl) < 2:
            continue
        j1, j2 = jl[0], jl[1]
        if j1.get('equivalent') != j2.get('equivalent'):
            for jd in [j1, j2]:
                for f in (jd.get('failures') or []):
                    cfreq[f] += 1
    agree_rows = [
        {'metric': 'total_judged',    'value': total_j},
        {'metric': 'agreed',          'value': n_agree},
        {'metric': 'disagreed',       'value': total_j - n_agree},
        {'metric': 'agreement_rate',  'value': round(n_agree / total_j, 4) if total_j else 0},
    ]
    for crit, cnt in cfreq.most_common():
        agree_rows.append({'metric': f'disagree_{crit}', 'value': cnt})
    save_csv(agree_rows, f"{scores_dir}/judge_agreement_stats.csv")
    if total_j:
        print(f"  ✓ judge_agreement_stats.csv ({n_agree}/{total_j}={n_agree/total_j:.1%})", flush=True)

    # ── Paper Tables ──────────────────────────────────────────────
    save_csv(
        [{'Model': s['model'],
          'Invariance': f"{s['inv_rel']:.3f}", 'Inv_N': s['inv_n'],
          'Directional': f"{s['dir_rel']:.3f}", 'Dir_N': s['dir_n'],
          'Shortcut': f"{s['sc_rel']:.3f}", 'SC_N': s['sc_n'],
          'Global': f"{s['global_rel']:.3f}", 'Global_N': s['global_n']}
         for s in summaries],
        f"{scores_dir}/paper_table_1_main_results.csv"
    )

    datasets   = sorted(set(r['dataset'] for r in ds_rows))
    pt2_lookup = {(r['model'], r['dataset'], r['family']): r['reliability'] for r in ds_rows}
    pt2 = []
    for s in summaries:
        row = {'Model': s['model']}
        for ds in datasets:
            for fam in ['invariance', 'directional', 'shortcut']:
                row[f"{ds}_{fam[:3].upper()}"] = f"{pt2_lookup.get((s['model'], ds, fam), 0):.3f}"
        pt2.append(row)
    save_csv(pt2, f"{scores_dir}/paper_table_2_per_dataset.csv")

    save_csv(
        [{'Model': r['model'], 'Family': r['family'],
          'Tier_1': f"{r['tier_1']:.3f}", 'N_T1': r['tier_1_n'],
          'Tier_2': f"{r['tier_2']:.3f}", 'N_T2': r['tier_2_n'],
          'Gap': f"{r['gap_t1_minus_t2']:+.3f}"}
         for r in tier_rows],
        f"{scores_dir}/paper_table_3_tier_comparison.csv"
    )
    print(f"  ✓ paper_table_1/2/3 generated", flush=True)

    # ── Instance-level CSV ────────────────────────────────────────
    inst_fields = [
        'probe_id', 'model', 'probe_family', 'probe_subtype', 'tier', 'dataset',
        'generator_model', 'base_item_id', 'scoring_method', 'rule_score',
        'judge_verified', 'final_score', 'similarity', 'read_delta', 'overlap',
        'detail_found', 'base_readability', 'probe_readability', 'rule_reason',
        'critical_detail', 'detail_exact_match', 'detail_word_overlap',
        'judge_1_name', 'judge_1_equiv', 'judge_1_confidence',
        'judge_2_name', 'judge_2_equiv', 'judge_2_confidence',
        'base_output_preview', 'probe_output_preview',
    ]
    for r in all_instances:
        for f in inst_fields:
            if f not in r:
                r[f] = ''
    save_csv(all_instances, f"{scores_dir}/instance_level_results.csv", inst_fields)
    print(f"  ✓ instance_level_results.csv ({len(all_instances):,} rows)", flush=True)

    # Judge detail + disagreement CSVs
    jdd = []
    for jcfg in JUDGES:
        for d in load_jsonl(f"{judge_dir}/{jcfg['name']}_decisions.jsonl"):
            jdd.append({
                'probe_id':           d['probe_id'],
                'judge':              d['judge'],
                'equivalent':         d.get('equivalent'),
                'confidence':         d.get('confidence', ''),
                'failures':           str(d.get('failures', [])),
                'reason':             d.get('reason', ''),
                'probe_subtype':      d.get('probe_subtype', ''),
                'tier':               d.get('tier', ''),
                'dataset':            d.get('dataset', ''),
                'generator_model':    d.get('generator_model', ''),
                'base_text_preview':  d.get('base_text_preview', ''),
                'probe_text_preview': d.get('probe_text_preview', ''),
            })
    save_csv(jdd, f"{scores_dir}/judge_decisions_detail.csv")

    jda = []
    for pid, judges in judge_by_probe.items():
        jl = list(judges.values())
        if len(jl) < 2:
            continue
        j1, j2 = jl[0], jl[1]
        if j1.get('equivalent') == j2.get('equivalent'):
            continue
        pm = probes_all.get(pid, {})
        jda.append({
            'probe_id':           pid,
            'probe_subtype':      j1.get('probe_subtype', ''),
            'tier':               j1.get('tier', ''),
            'dataset':            j1.get('dataset', ''),
            'base_text_preview':  pm.get('base_text', '')[:200],
            'probe_text_preview': pm.get('probe_text', '')[:200],
            'j1_name':            j1['judge'],
            'j1_equivalent':      j1.get('equivalent'),
            'j1_confidence':      j1.get('confidence', ''),
            'j1_failures':        str(j1.get('failures', [])),
            'j1_reason':          j1.get('reason', ''),
            'j2_name':            j2['judge'],
            'j2_equivalent':      j2.get('equivalent'),
            'j2_confidence':      j2.get('confidence', ''),
            'j2_failures':        str(j2.get('failures', [])),
            'j2_reason':          j2.get('reason', ''),
        })
    save_csv(jda, f"{scores_dir}/judge_disagreement_analysis.csv")
    print(f"  ✓ judge_decisions_detail.csv ({len(jdd):,} rows)", flush=True)
    print(f"  ✓ judge_disagreement_analysis.csv ({len(jda):,} rows)", flush=True)

    # ── Final print ───────────────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print(f"  RELIABILITY TABLE  (None=awaiting judge verdict)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'Model':<22} {'Inv':>7} {'Dir':>7} {'SC':>7} {'Global':>8}", flush=True)
    print(f"  {'-'*55}", flush=True)
    for s in summaries:
        print(
            f"  {s['model']:<22} {s['inv_rel']:>7.1%} "
            f"{s['dir_rel']:>7.1%} {s['sc_rel']:>7.1%} "
            f"{s['global_rel']:>8.1%}",
            flush=True
        )
    print(f"\n  ✓ Step 4 complete.", flush=True)


def phase_summary(args):
    """Backward-compatible alias for phase_results."""
    phase_results(args)


# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Step 4: Run Models → Rule Score → LLM Judges → Results"
    )
    p.add_argument('--input-dir',  required=True,
                   help="Directory containing master table and probecore JSONL files")
    p.add_argument('--output-dir', required=True,
                   help="Directory for all output files")
    p.add_argument('--phase',
                   choices=['run', 'score', 'judge', 'results', 'summary', 'all'],
                   default='all',
                   help="Which phase(s) to execute (default: all)")
    p.add_argument('--models', default=None,
                   help="Comma-separated list of model names to run (default: all configured)")
    p.add_argument('--judge', default=None,
                   help="Run only this judge by name, e.g. claude-haiku or mistral-large-3")
    p.add_argument('--threads', type=int, default=None,
                   help="Worker threads for API calls (default: per-judge config or API_THREADS)")
    p.add_argument('--delay', type=float, default=API_DELAY,
                   help="Seconds to sleep between API calls (default: from config)")
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)

    if a.phase in ('run',                  'all'): phase_run(a)
    if a.phase in ('score',                'all'): phase_score(a)
    if a.phase in ('judge',                'all'): phase_judge(a)
    if a.phase in ('results', 'summary',   'all'): phase_results(a)


if __name__ == '__main__':
    main()