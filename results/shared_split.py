import json

probes = [json.loads(l) for l in open("simp_probecore_v1.jsonl")]
inv = [p for p in probes if p["probe_family"] == "invariance"]

chunk = len(inv) // 3
for i in range(3):
    shard = inv[i*chunk:(i+1)*chunk] if i < 2 else inv[i*chunk:]
    with open(f"probecore_shard{i+1}.jsonl", "w") as f:
        for p in shard:
            f.write(json.dumps(p) + "\n")