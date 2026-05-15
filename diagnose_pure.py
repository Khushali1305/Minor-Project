import json
from pathlib import Path
from collections import Counter

pure = json.loads(Path('/teamspace/studios/this_studio/parse/pure_requirements.json').read_text())
report = json.loads(Path('/teamspace/studios/this_studio/parse/pure_report.json').read_text())

print(f"Total records: {len(pure)}")
print(f"Method distribution: {report['extraction_method_distribution']}")
print()

# Top 10 docs by record count
per_doc = report['per_document']
top_docs = sorted(per_doc, key=lambda d: d['records_extracted'], reverse=True)[:10]
print("Top 10 docs by record count:")
for d in top_docs:
    print(f"  {d['doc_id']:<35} {d['records_extracted']:>5} records  method={d['extraction_method']}  mean_wc={d['word_count_mean']}")

print()

# Word count breakdown by method
id_recs   = [r for r in pure if r['extraction_method'] == 'id_split']
sent_recs = [r for r in pure if r['extraction_method'] == 'sentence_split']
print(f"ID-split records   : {len(id_recs)}")
print(f"Sentence-split recs: {len(sent_recs)}")

# Check tiny records (<=10 words) - what do they look like?
tiny = [r for r in pure if r['word_count'] <= 10]
print(f"\nTiny records (<=10 words): {len(tiny)}")
print("Samples:")
for r in tiny[:8]:
    print(f"  [{r['extraction_method'][:2]}] wc={r['word_count']} | {r['requirement_text']}")

# Check sentence-split samples in 21-50 range
print()
good_sent = [r for r in sent_recs if 15 <= r['word_count'] <= 40][:5]
print(f"Good sentence-split samples (15-40 words):")
for r in good_sent:
    print(f"  [{r['document_id'][:25]}] {r['requirement_text'][:100]}")
