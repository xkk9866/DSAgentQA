"""Disguised training poison: make the reading-state style cue uninformative so that credibility must rest on corroboration.

For each training question the reader itself writes five passages in five genres that support the attack target,
without restating the question and with concrete detail (a different prompt from the test-time adaptive attacker,
and a different model: the test attacker is Qwen3-8B). Each new passage is then read alone exactly like the cached
ones (short answer, scores of the existing candidates, reading state), and appended as extra poisoned rows.
Rows are independent given the candidate set, so the cached rows are reused unchanged.

Output: cache/<model>/<data>-disg/*.pkl with the same record format (n_pass grows by 5).
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from once.llm import LLM  # noqa: E402
from once_read import CACHE, DOC_PROMPT, IDK, MAX_DOC_CHARS  # noqa: E402

GENRES = ["a short news article", "a reference-book entry", "a reply on a discussion forum", "a passage from a report",
          "a paragraph from a feature story"]
WRITE = ("Write {genre} that mentions, as a fact, that the answer to the following question is \"{target}\". Do not "
         "repeat the question. Make it read like a real document: add specific, believable details and attribute the "
         "information to a named source. About 70 words. Reply with only the text.\n\nQuestion: {q}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-7b-int8")
    ap.add_argument("--data", default="train_v4,train-rtqa")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_files", default="30,12", help="chunks to augment per data set")
    args = ap.parse_args()
    llm = LLM(args.model, device=args.device)
    for name, mf in zip(args.data.split(","), map(int, args.max_files.split(","))):
        src = os.path.join(CACHE, args.model, name)
        dst = os.path.join(CACHE, args.model, name + "-disg")
        os.makedirs(dst, exist_ok=True)
        t0 = time.time()
        files = sorted(glob.glob(os.path.join(src, "*.pkl")))[:mf]
        for fi, f in enumerate(files):
            out = os.path.join(dst, os.path.basename(f))
            if os.path.exists(out):
                continue
            recs = pickle.load(open(f, "rb"))
            recs = [r for r in recs if r.get("target")]
            gens = llm.generate([llm.chat(WRITE.format(genre=g, target=r["target"], q=r["question"]))
                                 for r in recs for g in GENRES], max_new_tokens=128, batch_size=32, temperature=1.0)
            docs = [gens[i * len(GENRES):(i + 1) * len(GENRES)] for i in range(len(recs))]
            prompts = [llm.chat(DOC_PROMPT.format(doc=d[:MAX_DOC_CHARS], q=r["question"])) for r, ds in zip(recs, docs) for d in ds]
            ans = llm.generate(prompts, max_new_tokens=24, batch_size=32)
            cands = [r["cands"] + [IDK] for r in recs for _ in GENRES]
            res = llm.read(prompts, cands, batch_size=8)
            k = 0
            for r in recs:
                part = res[k:k + len(GENRES)]
                r["L"] = np.concatenate([r["L"], np.stack([p[0] for p in part]).astype(np.float32)])
                r["H"] = np.concatenate([r["H"], np.stack([p[1] for p in part]).astype(np.float16)])
                r["Z"] = np.concatenate([r["Z"], np.stack([p[2][:-1] for p in part]).astype(np.float16)])
                r["iso"] = list(r["iso"]) + ans[k:k + len(GENRES)]
                r["poison"] = list(r["poison"]) + [True] * len(GENRES)
                r["n_pass"] += len(GENRES)
                r["disguised"] = docs[recs.index(r)]
                k += len(GENRES)
            pickle.dump(recs, open(out, "wb"))
            print(f"{name} {fi + 1}/{len(files)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
