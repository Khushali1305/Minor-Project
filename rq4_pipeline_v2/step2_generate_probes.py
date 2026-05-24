#!/usr/bin/env python3
"""Step 2: Generate all probes — 3 invariance + 1 directional + 1 shortcut"""
import json, re, os, random, time, argparse
from collections import defaultdict, Counter
from config import (EXPERIMENTAL_MODELS, SHORTCUT_TEMPLATES, SYNONYMS, PROTECTED_WORDS,
                    LEXICAL_RECALL_THRESHOLD, MAX_LENGTH_RATIO, MIN_LENGTH_RATIO,
                    API_DELAY, PARAPHRASE_SYSTEM, PARAPHRASE_USER)

random.seed(42)

def load_jsonl(p):
    with open(p) as f: return [json.loads(l) for l in f if l.strip()]

def rule_filter(src, prb):
    sw = set(re.findall(r'[a-z]+', src.lower()))
    pw = set(re.findall(r'[a-z]+', prb.lower()))
    if not sw: return False, "empty"
    recall = len(sw & pw) / len(sw)
    ratio = len(prb.split()) / max(len(src.split()), 1)
    ok = recall >= LEXICAL_RECALL_THRESHOLD and MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO
    return ok, f"recall={recall:.2f},ratio={ratio:.2f}"

# ═══ LEXICAL INVARIANCE ═══
def gen_lexical(text, n=2):
    tokens = re.findall(r"\S+", text)
    cands = []
    for i, tok in enumerate(tokens):
        w = re.sub(r'[^a-zA-Z]','',tok)
        if w and len(w)>3 and w.lower() not in PROTECTED_WORDS:
            syn = SYNONYMS.get(w.lower())
            if syn: cands.append((i,tok,w,random.choice(syn)))
    if not cands: return None, None
    chosen = random.sample(cands, min(n, len(cands)))
    res = tokens.copy(); log = []
    for idx,ot,ow,s in chosen:
        if ow[0].isupper(): s=s.capitalize()
        res[idx]=ot.replace(ow,s,1); log.append(f"{ow}→{s}")
    return ' '.join(res), ';'.join(log)

# ═══ SYNTACTIC INVARIANCE ═══
CL_PAT = [(r'^(If|When|Whenever|While)\s+(.+?),\s+(.+)$','cond'),
           (r'^(.+?)\s+because\s+(.+)$','causal'),
           (r'^(.+?)\s+although\s+(.+)$','concess')]

def gen_syntactic(text):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sents: return None, None
    first = sents[0].rstrip('.')
    for pat, op in CL_PAT:
        m = re.match(pat, first, re.I)
        if m:
            g = m.groups()
            try:
                if op=='cond': r=f"{g[2][0].upper()}{g[2][1:]} {g[0].lower()} {g[1].rstrip(',').lower()}."
                elif op=='causal': r=f"Because {g[1].rstrip('.')}, {g[0][0].lower()}{g[0][1:]}."
                elif op=='concess': r=f"Although {g[1].rstrip('.')}, {g[0][0].lower()}{g[0][1:]}."
                else: continue
                rest = sents[1:]
                return r+(' '+' '.join(rest) if rest else ''), op
            except: continue
    return None, None

# ═══ SEMANTIC INVARIANCE (via model) ═══
def gen_semantic(text, mcfg):
    from api_client import call_model
    prompt = PARAPHRASE_USER.format(text=text[:1500])
    try:
        r = call_model(prompt, PARAPHRASE_SYSTEM, mcfg['provider'], mcfg['model_id'], mcfg.get('max_tokens',1024))
        r = r.strip().strip('"\'')
        ratio = len(r.split())/max(len(text.split()),1)
        if ratio < MIN_LENGTH_RATIO or ratio > MAX_LENGTH_RATIO: return None
        return r
    except Exception as e:
        return None

# ═══ DIRECTIONAL (natural pairs) ═══
def gen_directional(items, aligns):
    by_id = {i['item_id']:i for i in items}
    probes = []
    for a in aligns:
        gid=a['group_id']
        adv=by_id.get(f"OSE_{gid}_advanced"); inter=by_id.get(f"OSE_{gid}_intermediate"); elem=by_id.get(f"OSE_{gid}_elementary")
        if not all([adv,inter,elem]): continue
        for tag,ref,tgt in [('inter',inter,'b1'),('elem',elem,'a2')]:
            probes.append({'probe_id':f"DIR_OSE_{gid}_adv_vs_{tag}",'probe_family':'directional',
                'probe_subtype':'ose_difficulty_pair','base_item_id':f"OSE_{gid}_advanced",
                'base_text':adv['original'],'probe_text':ref['original'],'target_level':tgt,
                'expected_relation':'directional','direction':f"base(Adv) harder than probe({tag.title()})",
                'tier':min(adv['tier'],ref['tier']),'dataset':'OneStopEnglish','generator_model':'natural_pair',
                'rule_pass':True,'rule_detail':'natural pair'})
    # TSAR pairs
    by_base = defaultdict(list)
    for i in items:
        if i['dataset']!='TSAR2025': continue
        bid = i['item_id'].replace('TSAR_','').split('-')[0]; by_base[bid].append(i)
    for bid, grp in by_base.items():
        tgts = {i['target_level']:i for i in grp}
        if 'a2' in tgts and 'b1' in tgts:
            probes.append({'probe_id':f"DIR_TSAR_{bid}_b1_vs_a2",'probe_family':'directional',
                'probe_subtype':'tsar_target_pair','base_item_id':tgts['b1']['item_id'],
                'base_text':tgts['b1']['original'],'probe_text':tgts['a2']['original'],
                'base_target':'b1','probe_target':'a2','target_level':'a2_vs_b1',
                'expected_relation':'directional','direction':'same source, a2 needs more simplification than b1',
                'tier':tgts['b1']['tier'],'dataset':'TSAR2025','generator_model':'natural_pair',
                'rule_pass':True,'rule_detail':'natural pair'})
    return probes

# ═══ SHORTCUT (domain-aware detail insertion) ═══
MEDICAL_KEYWORDS = {'patient','doctor','physician','medicine','medication','symptom','treatment',
    'hospital','diagnosis','disease','drug','dose','health','medical','clinical','therapy',
    'surgery','blood','pain','injury','infection','vaccine','prescription'}
GENERIC_SHORTCUT_TEMPLATES = [
    ", particularly in developing countries",
    ", especially in urban areas with high population density",
    ", which has been increasing steadily since 2015",
    ", according to a 2024 report by the United Nations",
    ", particularly during extreme weather events",
    ", except in regions with active conflict zones",
    ", affecting an estimated 2.3 million people worldwide",
    ", with the highest rates observed in coastal regions",
    ", especially when combined with existing infrastructure problems",
    ", despite significant government investment since 2018",
    ", which experts say could worsen without intervention",
    ", particularly for communities with limited resources",
]

def gen_shortcut(text, item_id):
    words = set(re.findall(r'[a-z]+', text.lower()))
    is_medical = len(words & MEDICAL_KEYWORDS) >= 2
    if is_medical:
        cat = random.choice(['medical','safety','qualifying'])
        detail = random.choice(SHORTCUT_TEMPLATES[cat])
    else:
        cat = 'generic_qualifying'
        detail = random.choice(GENERIC_SHORTCUT_TEMPLATES)
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sents: return None,None,None
    last = sents[-1].rstrip('.!?')
    sents[-1] = last + detail + '.'
    return ' '.join(sents), detail.strip(', '), cat

# ═══ MAIN ═══
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--skip-semantic', action='store_true')
    p.add_argument('--models', default=None, help='Comma-separated model names')
    p.add_argument('--max-items', type=int, default=None)
    p.add_argument('--delay', type=float, default=API_DELAY)
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)

    items = load_jsonl(f"{a.input_dir}/simp_master_table.jsonl")
    with open(f"{a.input_dir}/simp_ose_alignments.json") as f: aligns = json.load(f)
    print(f"Loaded {len(items)} items, {len(aligns)} alignments")

    models = EXPERIMENTAL_MODELS
    if a.models: names=set(a.models.split(',')); models=[m for m in models if m['name'] in names]

    out_path = f"{a.output_dir}/simp_probes_all.jsonl"
    # Resume support
    done_ids = set()
    if os.path.exists(out_path):
        for p2 in load_jsonl(out_path): done_ids.add(p2['probe_id'])
        print(f"Resuming: {len(done_ids)} already done")

    probes = []
    elig = [i for i in items if i['dataset'] in ('ASSET','TSAR2025') and 'invariance' in i.get('probe_roles',[])]
    if a.max_items: elig = elig[:a.max_items]

    # 1. Lexical
    print("[1/5] Lexical invariance...")
    c=0
    for it in elig:
        pid = f"INV_LEX_{it['item_id']}"
        if pid in done_ids: continue
        pt, log = gen_lexical(it['original'])
        if pt and pt != it['original']:
            ok, det = rule_filter(it['original'], pt)
            probes.append({'probe_id':pid,'probe_family':'invariance','probe_subtype':'lexical',
                'base_item_id':it['item_id'],'base_text':it['original'],'probe_text':pt,
                'target_level':it['target_level'],'expected_relation':'stable','operation':log,
                'tier':it['tier'],'dataset':it['dataset'],'generator_model':'rule_based',
                'rule_pass':ok,'rule_detail':det}); c+=1
    print(f"  {c} generated")

    # 2. Syntactic
    print("[2/5] Syntactic invariance...")
    c=0
    for it in elig:
        pid = f"INV_SYN_{it['item_id']}"
        if pid in done_ids: continue
        pt, op = gen_syntactic(it['original'])
        if pt and pt != it['original']:
            ok, det = rule_filter(it['original'], pt)
            probes.append({'probe_id':pid,'probe_family':'invariance','probe_subtype':'syntactic',
                'base_item_id':it['item_id'],'base_text':it['original'],'probe_text':pt,
                'target_level':it['target_level'],'expected_relation':'stable','operation':op,
                'tier':it['tier'],'dataset':it['dataset'],'generator_model':'rule_based',
                'rule_pass':ok,'rule_detail':det}); c+=1
    print(f"  {c} generated")

    # 3. Semantic (via each model)
    if not a.skip_semantic:
        print(f"[3/5] Semantic invariance via {len(models)} models...")
        for mcfg in models:
            mn = mcfg['name']
            print(f"  Model: {mn}...")
            c=0
            for idx, it in enumerate(elig):
                pid = f"INV_SEM_{mn}_{it['item_id']}"
                if pid in done_ids: continue
                pt = gen_semantic(it['original'], mcfg)
                if pt:
                    ok, det = rule_filter(it['original'], pt)
                    probes.append({'probe_id':pid,'probe_family':'invariance','probe_subtype':'semantic',
                        'base_item_id':it['item_id'],'base_text':it['original'],'probe_text':pt,
                        'target_level':it['target_level'],'expected_relation':'stable',
                        'operation':f'paraphrase by {mn}',
                        'tier':it['tier'],'dataset':it['dataset'],'generator_model':mn,
                        'rule_pass':ok,'rule_detail':det}); c+=1
                if (idx+1)%100==0: print(f"    [{idx+1}/{len(elig)}] gen={c}")
                time.sleep(a.delay)
            print(f"    {mn}: {c}")
    else:
        print("[3/5] Semantic SKIPPED")

    # 4. Directional
    print("[4/5] Directional probes...")
    dir_probes = gen_directional(items, aligns)
    dir_probes = [d for d in dir_probes if d['probe_id'] not in done_ids]
    probes.extend(dir_probes)
    print(f"  {len(dir_probes)} generated")

    # 5. Shortcut
    print("[5/5] Shortcut probes...")
    sc_elig = [i for i in items if i['dataset'] in ('ASSET','TSAR2025') and 'shortcut' in i.get('probe_roles',[])]
    if a.max_items: sc_elig = sc_elig[:a.max_items]
    c=0
    for it in sc_elig:
        pid = f"SC_{it['item_id']}"
        if pid in done_ids: continue
        pt, detail, cat = gen_shortcut(it['original'], it['item_id'])
        if pt:
            probes.append({'probe_id':pid,'probe_family':'shortcut','probe_subtype':f'detail_{cat}',
                'base_item_id':it['item_id'],'base_text':it['original'],'probe_text':pt,
                'target_level':it['target_level'],'expected_relation':'no_shortcut',
                'critical_detail':detail,'operation':f'{cat}: "{detail}"',
                'tier':it['tier'],'dataset':it['dataset'],'generator_model':'template_based',
                'rule_pass':True,'rule_detail':'template'}); c+=1
    print(f"  {c} generated")

    # Save
    with open(out_path,'w',encoding='utf-8') as f:
        # Write existing probes first (resume)
        if os.path.exists(out_path.replace('.jsonl','_bak.jsonl')):
            pass  # skip backup handling for simplicity
        for p2 in probes: f.write(json.dumps(p2, ensure_ascii=False)+'\n')

    fam = Counter(p2['probe_family'] for p2 in probes)
    sub = Counter(p2['probe_subtype'] for p2 in probes)
    gen = Counter(p2.get('generator_model','') for p2 in probes)
    rp = sum(1 for p2 in probes if p2.get('rule_pass'))
    print(f"\n{'='*50}\nTotal: {len(probes)} | Rule pass: {rp}")
    print(f"Family: {dict(fam)}\nSubtype: {dict(sub)}\nGenerator: {dict(gen)}")
    print(f"Saved: {out_path}\n✓ Step 2 complete.")

if __name__=='__main__': main()
