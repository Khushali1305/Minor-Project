#!/usr/bin/env python3
"""Step 1: Ingest, normalize, align OSE, tier-tag → simp_master_table.jsonl"""
import json, re, os, argparse
from collections import Counter, defaultdict
from config import MIN_WORDS_ASSET, OSE_ALIGN_THRESHOLD

STOPS = frozenset('the a an in on at to of is was are were has have had be been and or but for with that this from by as it its not no they we he she i you if when where how what which who will would can could may might shall should do does did more most very also than then now just only all any some such each every much many own other new old first last long great little same about after before into over under between through during without because while their them these those there here both well still already often'.split())

ANAPHORIC = [re.compile(p, re.I) for p in [
    r'^\s*(They|He|She|It|We|These|Those|This|That)\s',
    r'^\s*(However|Furthermore|Moreover|Therefore|Thus|Hence|Meanwhile|Nevertheless)\b',
    r'^\s*(As mentioned|As noted|In addition|In contrast|On the other hand)\b',
]]

def classify_tier(text):
    first = re.split(r'(?<=[.!?])\s+', text.strip())[0] if text.strip() else ''
    for p in ANAPHORIC:
        if p.search(first): return 2
    if first and first[0].islower(): return 2
    return 1

def content_sig(text, n=20):
    words = re.findall(r'[a-z]+', text[:600].lower())
    return set(w for w in words if w not in STOPS and len(w)>3)

def align_ose(ose_all):
    by_label = defaultdict(list)
    for i, r in enumerate(ose_all): by_label[r['label_text']].append((i, content_sig(r['text'])))
    aligns, used_i, used_e = [], set(), set()
    for ai, asig in by_label['Advanced']:
        bi, bs = max(((ii,len(asig&isig)) for ii,isig in by_label['Intermediate'] if ii not in used_i), key=lambda x:x[1], default=(-1,0))
        be, bes = max(((ei,len(asig&esig)) for ei,esig in by_label['Elementary'] if ei not in used_e), key=lambda x:x[1], default=(-1,0))
        if bs >= OSE_ALIGN_THRESHOLD and bes >= OSE_ALIGN_THRESHOLD:
            gid = f"G{len(aligns):03d}"
            aligns.append({'group_id':gid,'adv':ai,'inter':bi,'elem':be})
            used_i.add(bi); used_e.add(be)
    return aligns

def smart_load(path):
    with open(path,'r',encoding='utf-8') as f:
        c = f.read(1)
    with open(path,'r',encoding='utf-8') as f:
        return json.load(f) if c == '[' else [json.loads(l) for l in f if l.strip()]

def clean_ose(text):
    text = re.sub(r'^\s*\d+\s+','',text.strip())
    return re.sub(r'^(Elementary|Intermediate|Advanced)\s*\n?','',text.strip(),flags=re.I).strip()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)

    tsar = smart_load(f"{a.data_dir}/tsar2025_train.json")
    ose = smart_load(f"{a.data_dir}/onestop_english_train.json") + smart_load(f"{a.data_dir}/onestop_english_test.json")
    at = smart_load(f"{a.data_dir}/asset_simplification_test.json")
    av = smart_load(f"{a.data_dir}/asset_simplification_validation.json")
    print(f"Raw: TSAR={len(tsar)} OSE={len(ose)} ASSET_t={len(at)} ASSET_v={len(av)}")

    aligns = align_ose(ose)
    print(f"OSE aligned triples: {len(aligns)}")
    with open(f"{a.output_dir}/simp_ose_alignments.json",'w') as f: json.dump(aligns,f,indent=2)

    idx_map = {}
    for al in aligns:
        for lvl in ('adv','inter','elem'): idx_map[al[lvl]] = (al['group_id'], lvl)
    cefr = {'advanced':'c1','intermediate':'b1','elementary':'a2'}

    items = []
    # TSAR
    for r in tsar:
        t = r['original'].strip(); tgt = r['target_cefr'].lower()
        items.append({'item_id':f"TSAR_{r['text_id']}",'dataset':'TSAR2025','original':t,
            'source_level':'b2+','target_level':tgt,'references':[r['reference'].strip()],
            'text_length_words':len(t.split()),'tier':classify_tier(t),
            'probe_roles':['invariance','directional','shortcut'],'ose_group_id':None})
    # OSE
    for i, r in enumerate(ose):
        if i not in idx_map: continue
        gid, lvl = idx_map[i]; label = r['label_text'].lower(); t = clean_ose(r['text'])
        pr = ['invariance','directional','shortcut'] if label=='advanced' else ['directional']
        items.append({'item_id':f"OSE_{gid}_{label}",'dataset':'OneStopEnglish','original':t,
            'source_level':cefr[label],'target_level':'b1' if label=='advanced' else None,
            'references':[],'text_length_words':len(t.split()),'tier':classify_tier(t),
            'probe_roles':pr,'ose_group_id':gid,'ose_level':label})
    # ASSET
    cnt = 0
    for split, recs in [('test',at),('val',av)]:
        for r in recs:
            t = r['original'].strip()
            if len(t.split()) < MIN_WORDS_ASSET: continue
            items.append({'item_id':f"ASSET_{split}_{cnt:04d}",'dataset':'ASSET','original':t,
                'source_level':'complex','target_level':'simpler',
                'references':list(set(s.strip() for s in r['simplifications'])),
                'text_length_words':len(t.split()),'tier':classify_tier(t),
                'probe_roles':['invariance','shortcut'],'ose_group_id':None})
            cnt += 1

    with open(f"{a.output_dir}/simp_master_table.jsonl",'w',encoding='utf-8') as f:
        for it in items: f.write(json.dumps(it,ensure_ascii=False)+'\n')

    # Data card
    card = {'total':len(items), 'by_dataset':dict(Counter(i['dataset'] for i in items)),
            'tier_1':sum(1 for i in items if i['tier']==1),
            'tier_2':sum(1 for i in items if i['tier']==2)}
    with open(f"{a.output_dir}/simp_data_card.json",'w') as f: json.dump(card,f,indent=2)
    print(f"Total: {card['total']} (T1={card['tier_1']}, T2={card['tier_2']})")
    print(f"By dataset: {card['by_dataset']}")
    print("✓ Step 1 complete.")

if __name__ == '__main__': main()
