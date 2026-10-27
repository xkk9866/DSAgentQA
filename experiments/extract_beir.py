"""Pull the Contriever top-k clean passages for the PoisonedRAG target questions out of BEIR."""

import io
import json
import os
import sys
import zipfile

RAW = os.environ.get("ONCE_RAW", "D:/once_data/raw")
K = 10

for name in sys.argv[1:] or ["nq", "hotpotqa", "msmarco"]:
    adv = json.load(open(f"{RAW}/prag/adv_targeted_results_{name}.json", encoding="utf-8"))
    ret = json.load(open(f"{RAW}/prag/beir_results_{name}-contriever.json", encoding="utf-8"))
    top = {}
    for x in adv.values():
        scores = ret[x["id"]]
        top[x["id"]] = sorted(scores, key=lambda d: -scores[d])[:K]
    need = {d for v in top.values() for d in v}
    text = {}
    with zipfile.ZipFile(f"{RAW}/{name}.zip") as z:
        with z.open(f"{name}/corpus.jsonl") as f:
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                if '"_id"' not in line:
                    continue
                d = json.loads(line)
                if d["_id"] in need:
                    text[d["_id"]] = (d.get("title", "") + "\n" + d["text"]).strip()
    out = {q: {"clean": [text[d] for d in ids if d in text], "ids": ids} for q, ids in top.items()}
    json.dump(out, open(f"{RAW}/prag/{name}_passages.json", "w", encoding="utf-8"))
    print(name, len(out), "missing", len(need - set(text)))
