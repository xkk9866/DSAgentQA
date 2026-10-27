"""Write the PubMedQA misinformation passages (see once.medical)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    from once.llm import LLM
    from once.medical import build
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    out = build(LLM("qwen2.5-7b", device=dev), SentenceTransformer("BAAI/bge-base-en-v1.5", device=dev))
    print("pubmedqa items", len(out))
