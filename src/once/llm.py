"""Frozen LLM reader: greedy generation, and prefix-shared candidate scoring with hidden states."""

from __future__ import annotations

import os

import torch
import transformers.modeling_utils as _mu
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# The loader pre-allocates one ~14 GB block before copying weights; on a shared card that block may
# not exist even with enough free memory in total. Load tensor by tensor instead.
_mu.caching_allocator_warmup = lambda *a, **k: None

MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-7b-int8": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-8b-int8": "Qwen/Qwen3-8B",
    "llama3.1-8b": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "mistral-7b-int8": "kaist-ai/mistral-orpo-beta",  # second reader from another model family
}
PROJ_DIM = 512


class LLM:
    def __init__(self, name: str, device: str = "cuda:0", dtype=torch.bfloat16, layer_frac: float = 0.75):
        path = os.environ.get("ONCE_MODEL_PATH") or MODELS.get(name, name)
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # device="auto" shards the model over all visible GPUs (for cards with little free memory)
        # "-int8" names load INT8 weights (bitsandbytes LLM.int8); compute stays in bf16
        quant = BitsAndBytesConfig(load_in_8bit=True) if name.endswith("-int8") else None
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map=device, quantization_config=quant)
        self.model.eval()
        self.device = self.model.get_input_embeddings().weight.device
        self.out_device = self.model.get_output_embeddings().weight.device
        self.eot = "<|eot_id|>" if "llama" in name else ("</s>" if "mistral" in name else "<|im_end|>")
        cfg = self.model.config
        self.layer = int(round(cfg.num_hidden_layers * layer_frac))
        g = torch.Generator().manual_seed(0)
        # fixed Johnson-Lindenstrauss projection so hidden states fit on disk
        self.proj = (torch.randn(cfg.hidden_size, PROJ_DIM, generator=g) / PROJ_DIM ** 0.5).to(self.out_device)

    def chat(self, user: str, system: str | None = None) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        kw = {"enable_thinking": False} if "qwen3" in self.name else {}
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)

    def _batches(self, order, prompts, batch_size, extra=0):
        """Length-sorted batches capped by count and by a token budget (ONCE_TOK_BUDGET)."""
        budget = int(os.environ.get("ONCE_TOK_BUDGET", "0"))
        i = 0
        while i < len(order):
            n = min(batch_size, len(order) - i)
            if budget:
                longest = len(prompts[order[i]]) // 3 + extra  # rough chars-per-token
                n = max(1, min(n, budget // max(1, longest)))
            yield order[i:i + n]
            i += n

    @torch.no_grad()
    def generate(self, prompts: list[str], max_new_tokens: int = 32, batch_size: int = 32,
                 temperature: float = 0.0, seed: int = 0) -> list[str]:
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        out = [None] * len(prompts)
        if temperature > 0:
            torch.manual_seed(seed)
        for idx in self._batches(order, prompts, batch_size, max_new_tokens):
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.device)
            kw = dict(do_sample=True, temperature=temperature, top_p=1.0) if temperature > 0 else dict(do_sample=False)
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id,
                                      top_k=None, **kw)
            texts = self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for i, t in zip(idx, texts):
                out[i] = t.strip()
        return out

    @torch.no_grad()
    def read(self, prefixes: list[str], cands: list[list[str]], batch_size: int = 8):
        """For each prefix and each of its candidates: log p(cand + end-of-turn | prefix).

        Returns per prefix (scores [m], prefix state [PROJ_DIM], candidate states [m, PROJ_DIM]).
        The prefix is encoded once and its KV cache is shared by all candidates.
        """
        order = sorted(range(len(prefixes)), key=lambda i: -len(prefixes[i]))
        out = [None] * len(prefixes)
        pad = self.tok.pad_token_id
        for idx in self._batches(order, prefixes, batch_size):
            B = len(idx)
            enc = self.tok([prefixes[i] for i in idx], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.device)
            mask = enc["attention_mask"]
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            o1 = self.model(input_ids=enc["input_ids"], attention_mask=mask, position_ids=pos,
                            use_cache=True, output_hidden_states=True)
            h_pre = o1.hidden_states[self.layer][:, -1].float().to(self.proj.device) @ self.proj
            first_lp = torch.log_softmax(o1.logits[:, -1].float(), -1)
            m = max(len(cands[i]) for i in idx)
            cids = [[self.tok(c + self.eot, add_special_tokens=False)["input_ids"] for c in cands[i]] for i in idx]
            T = max(len(x) for row in cids for x in row)
            ids = torch.full((B * m, T), pad, dtype=torch.long)
            cm = torch.zeros((B * m, T), dtype=torch.long)
            lens = torch.ones(B * m, dtype=torch.long)
            for b, row in enumerate(cids):
                for j, x in enumerate(row):
                    ids[b * m + j, :len(x)] = torch.tensor(x)
                    cm[b * m + j, :len(x)] = 1
                    lens[b * m + j] = len(x)
            ids, cm = ids.to(self.device), cm.to(self.device)
            cache = o1.past_key_values
            cache.batch_repeat_interleave(m)
            full_mask = torch.cat([mask.repeat_interleave(m, 0), cm], 1)
            start = mask.sum(-1).repeat_interleave(m)
            pos2 = start[:, None] + torch.arange(T, device=self.device)[None]
            o2 = self.model(input_ids=ids, attention_mask=full_mask, position_ids=pos2,
                            past_key_values=cache, output_hidden_states=True)
            od = o2.logits.device  # accelerate may return outputs on the input device, not the last shard
            ids, cm, lens = ids.to(od), cm.to(od), lens.to(od)
            lp = torch.log_softmax(o2.logits[:, :-1].float(), -1)
            tok_lp = torch.zeros((B * m, T), device=od)
            tok_lp[:, 0] = first_lp.to(od).repeat_interleave(m, 0).gather(-1, ids[:, :1]).squeeze(-1)
            if T > 1:
                tok_lp[:, 1:] = lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            tok_lp = tok_lp * cm
            sc = tok_lp.sum(-1)
            hc = o2.hidden_states[self.layer].float().to(od)
            h_end = hc[torch.arange(B * m, device=od), lens - 1] @ self.proj.to(od)
            for b, i in enumerate(idx):
                k = len(cands[i])
                out[i] = (sc[b * m:b * m + k].cpu().numpy(), h_pre[b].cpu().numpy(),
                          h_end[b * m:b * m + k].cpu().numpy())
            del o1, o2, cache, lp, hc
        return out
