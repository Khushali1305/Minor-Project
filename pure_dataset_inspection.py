import json
from pathlib import Path
from collections import Counter

pure = json.loads(Path('/teamspace/studios/this_studio/parse/pure_requirements.json').read_text())

# Word count distribution
wc = [r['word_count'] for r in pure]
buckets = Counter()
for w in wc:
    if w <= 20:    buckets['01-20'] += 1
    elif w <= 50:  buckets['21-50'] += 1
    elif w <= 100: buckets['51-100'] += 1
    elif w <= 200: buckets['101-200'] += 1
    else:          buckets['200+'] += 1

print('Word count distribution:')
for k,v in sorted(buckets.items()):
    print(f'  {k}: {v}')

print()
# Worst offenders
long_recs = sorted(pure, key=lambda r: r['word_count'], reverse=True)[:5]
print('5 longest records:')
for r in long_recs:
    doc = r['document_id'][:35]
    rid = r['req_id_raw']
    wc2 = r['word_count']
    text = r['requirement_text'][:200]
    print(f'  doc={doc} id={rid} wc={wc2}')
    print(f'  text[:200]: {text}')
    print()

# Check req_id 9241
id_9241 = [r for r in pure if r['req_id_raw'] == '9241']
doc_name = id_9241[0]['document_id'] if id_9241 else 'none'
sample = id_9241[0]['requirement_text'][:200] if id_9241 else ''
print(f'Records with req_id 9241: {len(id_9241)}')
print(f'  From document: {doc_name}')
print(f'  Sample text: {sample}')