#!/usr/bin/env python3
"""
Step 4 — Run Models → Rule Score → LLM Judges → Comprehensive Results

SCORING ARCHITECTURE (consistent with SRS pipeline):
  1. Judges validate PROBE QUALITY: are the probe inputs semantically equivalent?
  2. Rule scores evaluate MODEL BEHAVIOR: does the model produce consistent outputs?
  3. Final score = rule_score for judge-validated probes; invalid probes are EXCLUDED

  Word-overlap F1 is used as a CONSERVATIVE LOWER BOUND on invariance similarity.
  Because valid simplifications may use different vocabulary while preserving meaning,
  this metric may underestimate true reliability. This is acknowledged in the paper.

Phases: run | score | judge | results | all
"""
import json, re, os, csv, sys, time, argparse, gc
from collections import Counter, defaultdict
from config import (EXPERIMENTAL_MODELS, JUDGES, API_DELAY,
                    SIMPLIFY_SYSTEM, SIMPLIFY_USER_CEFR, SIMPLIFY_USER_GENERIC,
                    JUDGE_SYSTEM, JUDGE_USER, BERTSCORE_THRESHOLD, READABILITY_DELTA_MAX, JUDGE_POLICY)

def load_jsonl(p):
    if not os.path.exists(p): return []
    with open(p,'r',encoding='utf-8') as f: return [json.loads(l) for l in f if l.strip()]
def append_jsonl(d, p):
    with open(p,'a',encoding='utf-8') as f: f.write(json.dumps(d,ensure_ascii=False)+'\n'); f.flush()
def save_jsonl(data, p):
    with open(p,'w',encoding='utf-8') as f:
        for d in data: f.write(json.dumps(d,ensure_ascii=False)+'\n')
def save_csv(rows, path, fieldnames=None):
    if not rows: return
    if not fieldnames: fieldnames = list(rows[0].keys())
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
def get_prompt(text, target):
    if target in ('a2','b1'):
        return SIMPLIFY_USER_CEFR.format(target={'a2':'A2 (elementary)','b1':'B1 (intermediate)'}[target], text=text)
    return SIMPLIFY_USER_GENERIC.format(text=text)
def is_local(prov): return prov in ("local","hf")
def estimate_tokens(text): return int(len(text.split()) * 1.3) + 10

# ═══════════════════════════════════════════════════════════
# CONSERVATIVE SIMILARITY METRIC
# ═══════════════════════════════════════════════════════════

def conservative_word_overlap_f1(ref, hyp):
    """
    Word-level F1 overlap between two texts.

    IMPORTANT: This is a CONSERVATIVE LOWER BOUND on semantic similarity.
    Two semantically identical texts with different vocabulary will score low.
    A high score reliably indicates similarity; a low score does NOT reliably
    indicate dissimilarity. This asymmetry is documented in the paper.

    Used for: invariance scoring (are model outputs consistent?)
    NOT used for: probe validation (that's the LLM judges' job)
    """
    ref_words = set(ref.lower().split())
    hyp_words = set(hyp.lower().split())
    if not ref_words or not hyp_words:
        return 0.0
    precision = len(ref_words & hyp_words) / len(hyp_words)
    recall = len(ref_words & hyp_words) / len(ref_words)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# ═══════════════════════════════════════════════════════════
# GPU BATCH ENGINE (for RTX 6000 Pro / H100)
# ═══════════════════════════════════════════════════════════

def _detect_type(mid):
    for n in ['t5','bart','mbart','pegasus','mt5']:
        if n in mid.lower(): return "seq2seq"
    return "causal"

def get_batch_size(model_name):
    if '70b' in model_name.lower(): return 6
    elif '32b' in model_name.lower(): return 12
    elif '27b' in model_name.lower(): return 12
    elif 'xl' in model_name.lower(): return 16
    elif 'large' in model_name.lower(): return 32
    elif '8b' in model_name.lower(): return 16
    else: return 8

class GPUBatchEngine:
    def __init__(self, model_id, quantize=None, model_type=None):
        import torch
        from transformers import AutoTokenizer
        self.model_id = model_id
        self.model_type = model_type or _detect_type(model_id)
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            try:
                from huggingface_hub import login; login(token=hf_token, add_to_git_credential=False)
            except: pass
        print(f"  [GPU] Loading {model_id} ({self.model_type}, q={quantize})...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        kw = {"trust_remote_code": True}
        if quantize == "4bit":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            kw["device_map"] = "auto"
        else:
            kw["torch_dtype"] = torch.float16; kw["device_map"] = "auto"
        if self.model_type == "seq2seq":
            from transformers import AutoModelForSeq2SeqLM
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **kw)
        else:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        self.model.eval()
        vram = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0
        print(f"  [GPU] Loaded | VRAM: {vram:.1f}GB", flush=True)

    def generate_batch(self, prompts, system, max_new_tokens=512, batch_size=8):
        import torch
        indexed = sorted(enumerate(prompts), key=lambda x: len(x[1].split()))
        results = [None]*len(prompts)
        total_in = total_out = 0; t0 = time.time()
        for bs in range(0, len(indexed), batch_size):
            batch = indexed[bs:bs+batch_size]
            bidx = [b[0] for b in batch]; bprompts = [b[1] for b in batch]
            if self.model_type == "seq2seq":
                texts = [f"{system}\n\n{p}" for p in bprompts]
                inp = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
                inp = {k:v.to(self.model.device) for k,v in inp.items()}
                total_in += inp['input_ids'].numel()
                with torch.no_grad(): outs = self.model.generate(**inp, max_new_tokens=max_new_tokens, num_beams=4)
                for idx, o in zip(bidx, outs):
                    results[idx] = self.tokenizer.decode(o, skip_special_tokens=True)
                    total_out += len(o)
            else:
                formatted = []
                for p in bprompts:
                    msgs = [{"role":"system","content":system},{"role":"user","content":p}]
                    if hasattr(self.tokenizer,'apply_chat_template'):
                        formatted.append(self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
                    else: formatted.append(f"{system}\n\n{p}")
                inp = self.tokenizer(formatted, return_tensors="pt", padding=True, truncation=True, max_length=4096)
                inp = {k:v.to(self.model.device) for k,v in inp.items()}
                inlen = inp['input_ids'].shape[1]; total_in += inp['input_ids'].numel()
                with torch.no_grad():
                    outs = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
                for idx, o in zip(bidx, outs):
                    new = o[inlen:]; results[idx] = self.tokenizer.decode(new, skip_special_tokens=True)
                    total_out += len(new)
            done = min(bs+batch_size, len(indexed))
            if (done//batch_size)%5==0 or done==len(indexed):
                el=time.time()-t0; rate=done/el if el else 0; eta=(len(indexed)-done)/rate if rate else 0
                print(f"    [BATCH] {done}/{len(indexed)} | {rate:.1f}/s | ETA:{eta/60:.0f}m | "
                      f"tok:{total_in//1000}K→{total_out//1000}K", flush=True)
            if bs%(batch_size*20)==0 and torch.cuda.is_available(): torch.cuda.empty_cache()
        el=time.time()-t0
        print(f"    [DONE] {len(prompts)} in {el/60:.1f}m | {len(prompts)/el:.1f}/s | "
              f"{total_in+total_out:,} tokens", flush=True)
        return results

    def unload(self):
        import torch; del self.model; del self.tokenizer; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"  [GPU] Unloaded {self.model_id}", flush=True)

# ═══════════════════════════════════════════════════════════
# PHASE 1: RUN (batch GPU + API)
# ═══════════════════════════════════════════════════════════
def phase_run(args):
    items = load_jsonl(f"{args.input_dir}/simp_master_table.jsonl")
    probes = load_jsonl(f"{args.input_dir}/simp_probecore_v1.jsonl")
    print(f"Base items: {len(items)}, Frozen probes: {len(probes)}", flush=True)
    base_queue = [{'item_id':it['item_id'],'text':it['original'],'target':it.get('target_level','simpler'),
        'type':'base','dataset':it['dataset']} for it in items
        if it.get('ose_level') in (None,'advanced') or it['dataset']!='OneStopEnglish']
    shared = []; sem_by_model = defaultdict(list)
    for p in probes:
        if p.get('probe_subtype')=='semantic': sem_by_model[p.get('generator_model','')].append(p)
        else: shared.append(p)
    print(f"Base: {len(base_queue)}, Shared: {len(shared)}, Semantic models: {list(sem_by_model.keys())}", flush=True)
    models = EXPERIMENTAL_MODELS
    if args.models: names=set(args.models.split(',')); models=[m for m in models if m['name'] in names]
    out_dir=f"{args.output_dir}/simp_model_outputs"; os.makedirs(out_dir, exist_ok=True)
    for mcfg in models:
        mn=mcfg['name']; out_path=f"{out_dir}/{mn}.jsonl"
        mprobes = shared + sem_by_model.get(mn,[])
        pqueue = [{'item_id':p['probe_id'],'text':p.get('probe_text',''),'target':p.get('target_level','simpler'),
            'type':'probe','probe_family':p.get('probe_family',''),'base_item_id':p.get('base_item_id',''),
            'dataset':p.get('dataset','')} for p in mprobes]
        full = base_queue + pqueue
        done=set()
        if os.path.exists(out_path):
            for r in load_jsonl(out_path): done.add(r['item_id'])
        pending=[q for q in full if q['item_id'] not in done]
        if not pending: print(f"\n  {mn}: done ({len(done)}), skip", flush=True); continue
        ptok=sum(estimate_tokens(q['text']) for q in pending)
        print(f"\n  {mn}: {len(pending)} pending (~{ptok//1000}K tokens)", flush=True)
        if is_local(mcfg['provider']):
            bs=get_batch_size(mn)
            eng=GPUBatchEngine(mcfg['model_id'], mcfg.get('quantize'), mcfg.get('type'))
            prompts=[get_prompt(q['text'],q['target']) for q in pending]
            results=eng.generate_batch(prompts, SIMPLIFY_SYSTEM, mcfg.get('max_tokens',512), bs)
            saved=0
            for q,out in zip(pending,results):
                if out is None or not out.strip(): continue
                out=out.strip()
                rec={'item_id':q['item_id'],'model':mn,'output':out,'output_words':len(out.split()),
                     'input_words':len(q['text'].split()),'type':q['type'],'target':q['target'],
                     'dataset':q.get('dataset','')}
                if q['type']=='probe': rec['probe_family']=q.get('probe_family',''); rec['base_item_id']=q.get('base_item_id','')
                append_jsonl(rec, out_path); saved+=1
            print(f"  {mn}: saved {saved}/{len(pending)}", flush=True)
            eng.unload()
        else:
            from api_client import call_model
            errors=0
            for idx,q in enumerate(pending):
                try:
                    out=call_model(get_prompt(q['text'],q['target']), SIMPLIFY_SYSTEM,
                                   mcfg['provider'], mcfg['model_id'], mcfg.get('max_tokens',2048)).strip()
                    rec={'item_id':q['item_id'],'model':mn,'output':out,'output_words':len(out.split()),
                         'input_words':len(q['text'].split()),'type':q['type'],'target':q['target'],
                         'dataset':q.get('dataset','')}
                    if q['type']=='probe': rec['probe_family']=q.get('probe_family',''); rec['base_item_id']=q.get('base_item_id','')
                    append_jsonl(rec, out_path)
                except Exception as e:
                    errors+=1
                    if errors<=5: print(f"    ERR {q['item_id']}: {e}", flush=True)
                if (idx+1)%50==0: print(f"    [{idx+1}/{len(pending)}] err={errors}", flush=True)
                time.sleep(args.delay)
            print(f"  {mn}: done. err={errors}", flush=True)

# ═══════════════════════════════════════════════════════════
# PHASE 2: RULE-BASED SCORING
# Uses conservative_word_overlap_f1 as LOWER BOUND similarity
# ═══════════════════════════════════════════════════════════
def phase_score(args):
    try:
        import textstat
        def readability(t): return textstat.flesch_kincaid_grade(t) if t.strip() else 0
    except ImportError:
        def readability(t):
            w=t.split(); return sum(len(x) for x in w)/len(w) if w else 0

    probes={p['probe_id']:p for p in load_jsonl(f"{args.input_dir}/simp_probecore_v1.jsonl")}
    out_dir=f"{args.input_dir}/simp_model_outputs"
    scores_dir=f"{args.output_dir}/simp_scores"; os.makedirs(scores_dir, exist_ok=True)
    for mf in sorted(os.listdir(out_dir)):
        if not mf.endswith('.jsonl'): continue
        mn=mf.replace('.jsonl','')
        outputs={r['item_id']:r for r in load_jsonl(f"{out_dir}/{mf}")}
        base_out={k:v for k,v in outputs.items() if v.get('type')=='base'}
        probe_out={k:v for k,v in outputs.items() if v.get('type')=='probe'}
        print(f"  Scoring {mn}: base={len(base_out)}, probe={len(probe_out)}", flush=True)
        scored=[]
        for pid, po in probe_out.items():
            pm=probes.get(pid); 
            if not pm: continue
            fam=pm.get('probe_family',''); bo=base_out.get(pm.get('base_item_id',''))
            if fam in ('invariance','directional') and not bo: continue
            base_txt=bo.get('output','') if bo else ''; probe_txt=po.get('output','')
            base_read=readability(base_txt); probe_read=readability(probe_txt)
            # Conservative word overlap (lower bound on semantic similarity)
            sim=conservative_word_overlap_f1(base_txt, probe_txt)
            row={'probe_id':pid,'model':mn,'probe_family':fam,'probe_subtype':pm.get('probe_subtype',''),
                 'tier':pm.get('tier',1),'dataset':pm.get('dataset',''),'generator_model':pm.get('generator_model',''),
                 'base_item_id':pm.get('base_item_id',''),
                 'conservative_similarity':round(sim,4),  # explicitly named conservative
                 'base_readability':round(base_read,2),'probe_readability':round(probe_read,2),
                 'readability_delta':round(abs(base_read-probe_read),2),
                 'base_output_preview':base_txt[:200],'probe_output_preview':probe_txt[:200]}
            if fam=='invariance':
                # Rule score: conservative check — outputs share vocabulary AND readability
                ok = sim >= BERTSCORE_THRESHOLD and abs(base_read-probe_read) <= READABILITY_DELTA_MAX
                row['rule_score'] = 1 if ok else 0
                row['rule_detail'] = (f"word_overlap={'PASS' if sim>=BERTSCORE_THRESHOLD else 'FAIL'}"
                    f"({sim:.3f}>={BERTSCORE_THRESHOLD}), "
                    f"read_delta={'PASS' if abs(base_read-probe_read)<=READABILITY_DELTA_MAX else 'FAIL'}"
                    f"({abs(base_read-probe_read):.1f}<={READABILITY_DELTA_MAX})")
                row['similarity_note'] = 'conservative_lower_bound: low score may reflect vocabulary variation, not semantic divergence'
            elif fam=='directional':
                ov=conservative_word_overlap_f1(base_txt, probe_txt)
                ok = abs(base_read-probe_read) > 0.5 and ov < 0.95
                row['rule_score'] = 1 if ok else 0
                row['output_overlap'] = round(ov,4)
                row['rule_detail'] = (f"read_shift={'PASS' if abs(base_read-probe_read)>0.5 else 'FAIL'}"
                    f"(delta={base_read-probe_read:.1f}), "
                    f"outputs_differ={'PASS' if ov<0.95 else 'FAIL'}(overlap={ov:.3f})")
            elif fam=='shortcut':
                detail=pm.get('critical_detail','')
                row['critical_detail']=detail
                if detail:
                    dw=set(re.findall(r'[a-z]+',detail.lower()))-{'the','a','an','of','in','on','at','to','for','and','or','is','are'}
                    ow=set(re.findall(r'[a-z]+',probe_txt.lower()))
                    preserved=(len(dw&ow)/len(dw)>=0.7) if dw else True
                    exact=detail.lower() in probe_txt.lower()
                else: preserved=True; exact=True
                row['rule_score']=1 if preserved else 0
                row['detail_exact_match']=exact
                row['detail_word_overlap']=round(len(dw&ow)/len(dw),3) if detail and dw else 1.0
                row['rule_detail']=f"preserved={'YES' if preserved else 'NO'}, exact={'YES' if exact else 'NO'}"
            scored.append(row)
        save_jsonl(scored, f"{scores_dir}/{mn}_rule_scores.jsonl")
        def rel(items): return sum(i['rule_score'] for i in items)/len(items) if items else 0
        by_f=defaultdict(list)
        for s in scored: by_f[s['probe_family']].append(s)
        print(f"    inv={rel(by_f['invariance']):.1%}({len(by_f['invariance'])}) "
              f"dir={rel(by_f['directional']):.1%}({len(by_f['directional'])}) "
              f"sc={rel(by_f['shortcut']):.1%}({len(by_f['shortcut'])})", flush=True)

# ═══════════════════════════════════════════════════════════
# PHASE 3: LLM JUDGES (validate probe quality, NOT model behavior)
# ═══════════════════════════════════════════════════════════
def phase_judge(args):
    from api_client import call_model
    probes={p['probe_id']:p for p in load_jsonl(f"{args.input_dir}/simp_probecore_v1.jsonl")}
    inv_probes={k:v for k,v in probes.items() if v.get('probe_family')=='invariance'}
    print(f"Invariance probes to judge: {len(inv_probes)}", flush=True)
    judge_dir=f"{args.output_dir}/simp_judge_outputs"; os.makedirs(judge_dir, exist_ok=True)
    for jcfg in JUDGES:
        jn=jcfg['name']; out_path=f"{judge_dir}/{jn}_decisions.jsonl"
        done=set()
        if os.path.exists(out_path):
            for r in load_jsonl(out_path): done.add(r['probe_id'])
        pending=[(pid,pm) for pid,pm in inv_probes.items() if pid not in done]
        print(f"\n  Judge {jn}: {len(pending)} pending (done={len(done)})", flush=True)
        if not pending: continue
        errors=0
        for idx,(pid,pm) in enumerate(pending):
            prompt=JUDGE_USER.format(source=pm.get('base_text','')[:500], probe=pm.get('probe_text','')[:500])
            try:
                resp=call_model(prompt, JUDGE_SYSTEM, jcfg['provider'], jcfg['model_id'], jcfg.get('max_tokens',512))
                resp=re.sub(r'^```json\s*','',resp.strip()); resp=re.sub(r'\s*```$','',resp)
                v=json.loads(resp)
                rec={'probe_id':pid,'judge':jn,'equivalent':v.get('equivalent',False),
                     'confidence':v.get('confidence','low'),'failures':v.get('failures',[]),
                     'reason':v.get('reason',''),'probe_subtype':pm.get('probe_subtype',''),
                     'generator_model':pm.get('generator_model',''),'tier':pm.get('tier',1),
                     'dataset':pm.get('dataset',''),'base_text_preview':pm.get('base_text','')[:200],
                     'probe_text_preview':pm.get('probe_text','')[:200]}
            except json.JSONDecodeError:
                errors+=1
                rec={'probe_id':pid,'judge':jn,'equivalent':None,'confidence':'parse_error',
                     'failures':['JSON_PARSE'],'reason':resp[:200] if 'resp' in dir() else ''}
            except Exception as e:
                errors+=1
                rec={'probe_id':pid,'judge':jn,'equivalent':None,'confidence':'error',
                     'failures':['API_ERROR'],'reason':str(e)[:200]}
            append_jsonl(rec, out_path)
            if (idx+1)%50==0: print(f"    [{idx+1}/{len(pending)}] err={errors}", flush=True)
            time.sleep(args.delay)
        print(f"  {jn}: done. err={errors}", flush=True)

# ═══════════════════════════════════════════════════════════
# PHASE 4: COMPREHENSIVE RESULTS
#
# CRITICAL SCORING LOGIC:
#   - Judges validate PROBE INPUTS (are they semantically equivalent?)
#   - Rule scores evaluate MODEL OUTPUTS (does model behave consistently?)
#   - If judges say probe valid   → final_score = rule_score (model behavior)
#   - If judges say probe invalid → EXCLUDE from reliability computation
#   - Directional/shortcut probes → final_score = rule_score (no judge needed)
# ═══════════════════════════════════════════════════════════
def phase_results(args):
    scores_dir=f"{args.output_dir}/simp_scores"
    judge_dir=f"{args.output_dir}/simp_judge_outputs"
    probes_all={p['probe_id']:p for p in load_jsonl(f"{args.input_dir}/simp_probecore_v1.jsonl")}
    os.makedirs(scores_dir, exist_ok=True)

    # ── Load judge decisions ──
    judge_by_probe=defaultdict(dict)
    for jcfg in JUDGES:
        for d in load_jsonl(f"{judge_dir}/{jcfg['name']}_decisions.jsonl"):
            judge_by_probe[d['probe_id']][d['judge']]=d

    # ── Compute probe validity from judges ──
    # probe_valid[pid] = True if both judges agree inputs are equivalent
    # probe_valid[pid] = False if judges say NOT equivalent → EXCLUDE from scoring
    # probe_valid[pid] absent → no judge data → use rule score only
    probe_valid = {}
    judge_detail_rows = []; disagreement_rows = []
    for pid, judges in judge_by_probe.items():
        jnames=list(judges.keys())
        for jn, jd in judges.items():
            judge_detail_rows.append({'probe_id':pid,'judge':jn,'equivalent':jd.get('equivalent'),
                'confidence':jd.get('confidence'),'failures':str(jd.get('failures',[])),
                'reason':jd.get('reason',''),'probe_subtype':jd.get('probe_subtype',''),
                'dataset':jd.get('dataset',''),'tier':jd.get('tier',''),
                'generator_model':jd.get('generator_model','')})
        if len(jnames)>=2:
            j1,j2=judges[jnames[0]],judges[jnames[1]]
            agreed=j1.get('equivalent')==j2.get('equivalent')
            if JUDGE_POLICY=="agree_only":
                probe_valid[pid] = agreed and j1.get('equivalent', False)
            else:
                probe_valid[pid] = j1.get('equivalent', False) or j2.get('equivalent', False)
            if not agreed:
                pm=probes_all.get(pid,{})
                disagreement_rows.append({'probe_id':pid,'probe_subtype':pm.get('probe_subtype',''),
                    'dataset':pm.get('dataset',''),'tier':pm.get('tier',''),
                    'base_text_preview':pm.get('base_text','')[:200],'probe_text_preview':pm.get('probe_text','')[:200],
                    'j1_name':j1['judge'],'j1_equivalent':j1.get('equivalent'),'j1_confidence':j1.get('confidence'),
                    'j1_failures':str(j1.get('failures',[])),'j1_reason':j1.get('reason',''),
                    'j2_name':j2['judge'],'j2_equivalent':j2.get('equivalent'),'j2_confidence':j2.get('confidence'),
                    'j2_failures':str(j2.get('failures',[])),'j2_reason':j2.get('reason','')})

    n_valid=sum(1 for v in probe_valid.values() if v)
    n_invalid=sum(1 for v in probe_valid.values() if not v)
    print(f"Judge results: {n_valid} valid probes, {n_invalid} invalid (excluded), "
          f"{len(disagreement_rows)} disagreements", flush=True)

    save_csv(judge_detail_rows, f"{scores_dir}/judge_decisions_detail.csv")
    save_csv(disagreement_rows, f"{scores_dir}/judge_disagreement_analysis.csv")

    # Peer review agreement by subtype
    agree_by=defaultdict(lambda:{'a':0,'d':0})
    for pid,judges in judge_by_probe.items():
        jn=list(judges.keys())
        if len(jn)<2: continue
        sub=probes_all.get(pid,{}).get('probe_subtype','?')
        ds=probes_all.get(pid,{}).get('dataset','?')
        if judges[jn[0]].get('equivalent')==judges[jn[1]].get('equivalent'): agree_by[(ds,sub)]['a']+=1
        else: agree_by[(ds,sub)]['d']+=1
    peer_rows=[{'dataset':k[0],'probe_subtype':k[1],'agreed':v['a'],'disagreed':v['d'],
        'total':v['a']+v['d'],'agreement_rate':round(v['a']/(v['a']+v['d']),4) if v['a']+v['d'] else 0}
        for k,v in sorted(agree_by.items())]
    save_csv(peer_rows, f"{scores_dir}/peer_review_agreement.csv")

    # Disagreement criteria frequency
    cfreq=Counter()
    for d in disagreement_rows:
        for jk in ['j1_failures','j2_failures']:
            try:
                for f in eval(d[jk]): cfreq[f]+=1
            except: pass
    save_csv([{'criterion':k,'count':v} for k,v in cfreq.most_common()],
             f"{scores_dir}/disagreement_criteria_frequency.csv")

    # ── Merge rule scores + judge validity → final scores ──
    all_inst=[]; model_summaries=[]
    excluded_count = 0

    for sf in sorted(os.listdir(scores_dir)):
        if not sf.endswith('_rule_scores.jsonl'): continue
        mn=sf.replace('_rule_scores.jsonl',''); scored=load_jsonl(f"{scores_dir}/{sf}")

        for s in scored:
            pid=s['probe_id']; jd=judge_by_probe.get(pid,{})
            jn=list(jd.keys())
            s['judge_1_name']=jn[0] if jn else ''
            s['judge_1_equiv']=jd[jn[0]].get('equivalent','') if jn else ''
            s['judge_2_name']=jn[1] if len(jn)>1 else ''
            s['judge_2_equiv']=jd[jn[1]].get('equivalent','') if len(jn)>1 else ''

            # ════════════════════════════════════════════════════
            # CRITICAL SCORING LOGIC (consistent with SRS pipeline):
            #   Judges validate PROBE QUALITY → filter valid probes
            #   Rule score evaluates MODEL BEHAVIOR → IS the final score
            # ════════════════════════════════════════════════════
            if s['probe_family'] == 'invariance' and pid in probe_valid:
                if probe_valid[pid]:
                    # Probe is VALID → model behavior determines score
                    s['probe_validity'] = 'valid'
                    s['final_score'] = s['rule_score']
                else:
                    # Probe is INVALID → EXCLUDE from reliability computation
                    s['probe_validity'] = 'invalid_excluded'
                    s['final_score'] = -1  # sentinel: excluded
                    excluded_count += 1
            else:
                # Directional/shortcut: no judge needed, rule score is final
                # Invariance without judge data: use rule score as-is
                s['probe_validity'] = 'no_judge' if s['probe_family']=='invariance' else 'not_applicable'
                s['final_score'] = s['rule_score']

            all_inst.append(s)

        # ── Per-model reliability (excluding invalid probes) ──
        valid_scored = [s for s in scored if s.get('final_score', -1) >= 0]
        def rel(items): return sum(i['final_score'] for i in items)/len(items) if items else 0
        by_f=defaultdict(list); by_ft=defaultdict(list)
        for s in valid_scored:
            by_f[s['probe_family']].append(s)
            by_ft[(s['probe_family'],s.get('tier',1))].append(s)
        row={'model':mn,'total_scored':len(valid_scored),
             'total_excluded':sum(1 for s in scored if s.get('final_score',-1)<0),
             'inv_reliability':round(rel(by_f['invariance']),4),'inv_n':len(by_f['invariance']),
             'dir_reliability':round(rel(by_f['directional']),4),'dir_n':len(by_f['directional']),
             'sc_reliability':round(rel(by_f['shortcut']),4),'sc_n':len(by_f['shortcut']),
             'global_reliability':round(rel(valid_scored),4)}
        for fam in ['invariance','directional','shortcut']:
            for tier in [1,2]:
                ti=by_ft.get((fam,tier),[])
                row[f'{fam}_t{tier}']=round(rel(ti),4); row[f'{fam}_t{tier}_n']=len(ti)
        model_summaries.append(row)

    print(f"\nTotal excluded (invalid probes): {excluded_count}", flush=True)

    # ── Save all result files ──
    inst_fields=['probe_id','model','probe_family','probe_subtype','tier','dataset','generator_model',
        'base_item_id','rule_score','probe_validity','final_score',
        'conservative_similarity','base_readability','probe_readability','readability_delta',
        'rule_detail','similarity_note','critical_detail','detail_exact_match','detail_word_overlap',
        'output_overlap','judge_1_name','judge_1_equiv','judge_2_name','judge_2_equiv',
        'base_output_preview','probe_output_preview']
    for r in all_inst:
        for f in inst_fields:
            if f not in r: r[f]=''
    save_csv(all_inst, f"{scores_dir}/instance_level_results.csv", inst_fields)

    if model_summaries:
        save_csv(model_summaries, f"{scores_dir}/reliability_summary.csv")
        with open(f"{scores_dir}/reliability_summary.json",'w') as f: json.dump(model_summaries,f,indent=2)

    # By dataset
    ds_agg=defaultdict(lambda:{'p':0,'t':0})
    for s in all_inst:
        if s.get('final_score',-1)<0: continue
        k=(s['model'],s['dataset'],s['probe_family']); ds_agg[k]['t']+=1; ds_agg[k]['p']+=s['final_score']
    save_csv([{'model':k[0],'dataset':k[1],'family':k[2],'reliability':round(v['p']/v['t'],4) if v['t'] else 0,
        'passed':v['p'],'total':v['t']} for k,v in sorted(ds_agg.items())],
        f"{scores_dir}/reliability_by_dataset.csv")

    # By subtype
    sub_agg=defaultdict(lambda:{'p':0,'t':0})
    for s in all_inst:
        if s.get('final_score',-1)<0: continue
        k=(s['model'],s['probe_subtype']); sub_agg[k]['t']+=1; sub_agg[k]['p']+=s['final_score']
    save_csv([{'model':k[0],'subtype':k[1],'reliability':round(v['p']/v['t'],4) if v['t'] else 0,
        'passed':v['p'],'total':v['t']} for k,v in sorted(sub_agg.items())],
        f"{scores_dir}/reliability_by_subtype.csv")

    # By tier
    tier_agg=defaultdict(lambda:{'p':0,'t':0})
    for s in all_inst:
        if s.get('final_score',-1)<0: continue
        k=(s['model'],s.get('tier',1),s['probe_family']); tier_agg[k]['t']+=1; tier_agg[k]['p']+=s['final_score']
    save_csv([{'model':k[0],'tier':k[1],'family':k[2],'reliability':round(v['p']/v['t'],4) if v['t'] else 0,
        'passed':v['p'],'total':v['t']} for k,v in sorted(tier_agg.items())],
        f"{scores_dir}/reliability_by_tier.csv")

    # Transform quality comparison
    tq=defaultdict(lambda:{'p':0,'t':0,'m':set()})
    for s in all_inst:
        if s.get('final_score',-1)<0: continue
        k=s['probe_subtype']; tq[k]['t']+=1; tq[k]['p']+=s['final_score']; tq[k]['m'].add(s['model'])
    save_csv([{'subtype':k,'avg_reliability':round(v['p']/v['t'],4) if v['t'] else 0,'total':v['t'],
        'n_models':len(v['m'])} for k,v in sorted(tq.items())],
        f"{scores_dir}/transform_quality_comparison.csv")

    # Pipeline overview
    all_gen=load_jsonl(f"{args.input_dir}/simp_probes_all.jsonl")
    valid_inst=[s for s in all_inst if s.get('final_score',-1)>=0]
    pipe=[{'stage':'probes_generated','count':len(all_gen)},
        {'stage':'probes_frozen','count':len(probes_all)},
        {'stage':'judge_validated_equivalent','count':n_valid},
        {'stage':'judge_validated_not_equivalent','count':n_invalid},
        {'stage':'judge_disagreements','count':len(disagreement_rows)},
        {'stage':'models_evaluated','count':len(model_summaries)},
        {'stage':'total_scored_instances','count':len(valid_inst)},
        {'stage':'total_excluded_instances','count':excluded_count}]
    save_csv(pipe, f"{scores_dir}/pipeline_overview.csv")

    # Paper table
    save_csv([{'Model':s['model'],'Invariance':f"{s['inv_reliability']:.3f}",
        'Directional':f"{s['dir_reliability']:.3f}",'Shortcut':f"{s['sc_reliability']:.3f}",
        'Global':f"{s['global_reliability']:.3f}",'N_scored':s['total_scored'],
        'N_excluded':s['total_excluded']} for s in model_summaries],
        f"{scores_dir}/paper_table_ready.csv")

    # Dataset characteristics
    items=load_jsonl(f"{args.input_dir}/simp_master_table.jsonl")
    dc=defaultdict(lambda:{'n':0,'w':[],'t1':0,'t2':0})
    for it in items: d=it['dataset']; dc[d]['n']+=1; dc[d]['w'].append(it.get('text_length_words',0)); \
        dc[d]['t1' if it.get('tier')==1 else 't2']+=1
    save_csv([{'dataset':d,'count':v['n'],'mean_words':round(sum(v['w'])/len(v['w']),1) if v['w'] else 0,
        'tier_1':v['t1'],'tier_2':v['t2']} for d,v in sorted(dc.items())],
        f"{scores_dir}/dataset_characteristics.csv")

    # ── Print ──
    print(f"\n{'='*70}\n  13 RESULT FILES GENERATED\n{'='*70}", flush=True)
    for f in ['instance_level_results.csv','reliability_summary.csv','reliability_by_dataset.csv',
              'reliability_by_subtype.csv','reliability_by_tier.csv','transform_quality_comparison.csv',
              'pipeline_overview.csv','paper_table_ready.csv','judge_decisions_detail.csv',
              'judge_disagreement_analysis.csv','peer_review_agreement.csv',
              'disagreement_criteria_frequency.csv','dataset_characteristics.csv']:
        fp=f"{scores_dir}/{f}"
        n=sum(1 for _ in open(fp))-1 if os.path.exists(fp) else 0
        print(f"  {f:<45} {n:>6} rows", flush=True)

    if model_summaries:
        print(f"\n  Scoring note: word-overlap F1 used as conservative lower bound.", flush=True)
        print(f"  Invalid probes (judge-excluded): {excluded_count}", flush=True)
        print(f"\n{'Model':<20} {'Inv':>6} {'Dir':>6} {'SC':>6} {'Global':>7} {'Excl':>5}", flush=True)
        print(f"{'-'*55}", flush=True)
        for s in model_summaries:
            print(f"{s['model']:<20} {s['inv_reliability']:>6.1%} {s['dir_reliability']:>6.1%} "
                  f"{s['sc_reliability']:>6.1%} {s['global_reliability']:>7.1%} {s['total_excluded']:>5}", flush=True)
    print(f"\n✓ Step 4 complete.", flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',required=True); p.add_argument('--output-dir',required=True)
    p.add_argument('--phase',choices=['run','score','judge','results','all'],default='all')
    p.add_argument('--models',default=None); p.add_argument('--delay',type=float,default=API_DELAY)
    a=p.parse_args(); os.makedirs(a.output_dir,exist_ok=True)
    if a.phase in ('run','all'):     phase_run(a)
    if a.phase in ('score','all'):   phase_score(a)
    if a.phase in ('judge','all'):   phase_judge(a)
    if a.phase in ('results','all'): phase_results(a)
if __name__=='__main__': main()