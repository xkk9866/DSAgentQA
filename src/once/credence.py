"""CREDENCE: amortized truth discovery over a frozen LLM's readings.

Every source s -- the closed-book reading (the parametric memory), each retrieved passage read
alone, and optionally the LLM's joint reading of the whole context -- testifies a distribution
p_s(c) over the candidate answers. Testimony follows an epsilon-contamination model: with
probability a_s the source reports honestly, otherwise it emits an answer unrelated to the truth.
The log-likelihood of candidate c is then

    l_s(c) = log(a_s p_s(c) + (1 - a_s) / |C|) + log |C|,

which is exactly zero for a source with a_s = 0 and is bounded by log(1 + a_s |C| / (1 - a_s))
whatever the source asserts. The answer posterior is the product of these robust likelihoods.

a_s = r_s * rel_s, where rel_s = 1 - p_s(I don't know) is read off the LLM and r_s, the source's
credibility, is the latent quantity of truth discovery. Classical truth discovery estimates r_s
by iterating over many items; with one question at a time CREDENCE amortizes it: an axial set
transformer over the source x candidate grid predicts r_s from what every source says, and the
current posterior is fed back so that credibility and truth are refined jointly (unrolled EM).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e4


class MHA(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x, mask):  # x: (B, N, d), mask: (B, N) keys
        B, N, d = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        a = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
        a = a.masked_fill(~mask[:, None, None, :], NEG).softmax(-1)
        return self.o((a @ v).transpose(1, 2).reshape(B, N, d))


class AxialBlock(nn.Module):
    """Source-axis attention in every candidate column, candidate-axis attention in every row, FFN."""

    def __init__(self, d, heads, dropout):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.src, self.cand = MHA(d, heads), MHA(d, heads)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, rmask, cmask):
        B, R, C, d = x.shape
        xc = x.permute(0, 2, 1, 3).reshape(B * C, R, d)
        xc = xc + self.drop(self.src(self.ln1(xc), rmask[:, None].expand(B, C, R).reshape(B * C, R)))
        x = xc.view(B, C, R, d).permute(0, 2, 1, 3)
        xr = x.reshape(B * R, C, d)
        xr = xr + self.drop(self.cand(self.ln2(xr), cmask[:, None].expand(B, R, C).reshape(B * R, C)))
        x = xr.view(B, R, C, d)
        return x + self.drop(self.ff(self.ln3(x)))


PRIOR, PASSAGE, JOINT = 0, 1, 2


class Credence(nn.Module):
    N_SCALAR = 5

    def __init__(self, hdim=512, d=128, layers=2, heads=4, dropout=0.2, iters=2, robust=True,
                 learn_cred=True, use_joint=True, use_prior=True, residual=False, dep=True, use_z=True, use_h=True, direct=False):
        super().__init__()
        self.dep = dep
        self.use_z, self.use_h = use_z, use_h
        # direct: same input and encoder, but answer logits come straight from the network (no truth-discovery pool)
        self.direct = direct
        residual = residual or direct
        # source dependence: a learned kernel over source embeddings; copies of one source share one weight
        self.dproj = nn.Linear(2 * d, 32, bias=False)
        self.log_bw = nn.Parameter(torch.zeros(()))
        self.iters, self.robust, self.learn_cred = iters, robust, learn_cred
        self.use_joint, self.use_prior, self.residual = use_joint, use_prior, residual
        self.scalar = nn.Sequential(nn.Linear(self.N_SCALAR, d), nn.GELU(), nn.Linear(d, d))
        self.cell = nn.Sequential(nn.LayerNorm(hdim), nn.Linear(hdim, d))
        self.state = nn.Sequential(nn.LayerNorm(hdim), nn.Linear(hdim, d))
        self.type_emb = nn.Embedding(3, d)
        self.feedback = nn.Linear(3, d)  # posterior q(c), agreement of the row with q, the row's own likelihood
        nn.init.zeros_(self.feedback.weight)
        nn.init.zeros_(self.feedback.bias)
        self.blocks = nn.ModuleList(AxialBlock(d, heads, dropout) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.cred = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.log_tau = nn.Parameter(torch.zeros(3))  # testimony temperature per source type
        self.none = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.set_bias = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, 1))
        if residual:  # ablation: free-form per-candidate logits added to the pool
            self.res = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    # ---- inputs ------------------------------------------------------------------------------
    @staticmethod
    def readings(L, cmask):
        """L: (B, R, C+1) log-likelihoods, last column = "I don't know"."""
        C = L.shape[-1] - 1
        cm1 = torch.cat([cmask, torch.ones_like(cmask[:, :1])], 1)[:, None]
        full = L.masked_fill(~cm1, NEG).log_softmax(-1)
        lq = L[..., :C].masked_fill(~cmask[:, None], NEG).log_softmax(-1)
        rel = 1 - full[..., C].exp()
        return full, lq, rel

    def assemble(self, b):
        """Rows: [prior, passages..., joint]. Returns L (B,R,C+1), H (B,R,h), Z (B,R,C,h), types, rmask."""
        L = [b["L0"][:, None], b["L"]]
        H = [b["H0"][:, None], b["H"]]
        Z = [b["Z0"][:, None], b["Z"]]
        B, N = b["L"].shape[:2]
        dev = b["L"].device
        types = [torch.full((B, 1), PRIOR, device=dev), torch.full((B, N), PASSAGE, device=dev)]
        rmask = [torch.full((B, 1), self.use_prior, device=dev, dtype=torch.bool), b["pmask"]]
        if "LJ" in b:
            L.append(b["LJ"][:, None])
            H.append(b["HJ"][:, None])
            Z.append(b["ZJ"][:, None])
            types.append(torch.full((B, 1), JOINT, device=dev))
            jm = b.get("jmask", torch.ones(B, dtype=torch.bool, device=dev))
            rmask.append((jm & self.use_joint)[:, None])
        return (torch.cat(L, 1), torch.cat(H, 1), torch.cat(Z, 1), torch.cat(types, 1).long(),
                torch.cat(rmask, 1))

    def dependence(self, emb, rmask):
        """rho_s = sum_t kappa(s, t) over present sources; kappa(s, s) = 1, so rho_s >= 1."""
        u = F.normalize(self.dproj(emb), dim=-1)
        d2 = (u[:, :, None] - u[:, None]).pow(2).sum(-1)
        kap = torch.exp(-8.0 * self.log_bw.exp() * d2) * rmask[:, None, :].float()
        return kap.sum(-1).clamp(min=1.0)

    # ---- forward -----------------------------------------------------------------------------
    def forward(self, b, return_all=False):
        L, H, Z, types, rmask = self.assemble(b)
        cmask = b["cmask"]
        B, R, C1 = L.shape
        C = C1 - 1
        full, lq, rel = self.readings(L, cmask)
        top = (lq >= lq.max(-1, keepdim=True).values - 1e-6).float()
        s = torch.stack([lq.clamp(min=-20) / 10, full[..., :C].clamp(min=-20) / 10, rel[..., None].expand_as(lq),
                         top, (lq - lq[:, :1]).clamp(-20, 20) / 10], -1)
        x0 = self.scalar(s) + self.type_emb(types)[:, :, None]
        if self.use_z:
            x0 = x0 + self.cell(Z)
        if self.use_h:
            x0 = x0 + self.state(H)[:, :, None]
        ncand = cmask.sum(1).clamp(min=1).float()
        tau = self.log_tau.exp()[types]  # (B, R)
        p = (lq / tau[..., None]).masked_fill(~cmask[:, None], NEG).softmax(-1)  # testimony over candidates
        cm_f = cmask.float()
        q = cm_f / ncand[:, None]
        outs = []
        for it in range(self.iters):
            agree = (q[:, None] * lq.clamp(min=-20)).sum(-1, keepdim=True) / 10  # E_q log p_s
            fb = torch.stack([q[:, None].expand(B, R, C), agree.expand(B, R, C), p], -1)
            x = x0 + self.feedback(fb)
            for blk in self.blocks:
                x = blk(x, rmask, cmask)
            x = self.ln(x)
            w = cm_f[:, None, :, None]
            emb = torch.cat([(x * w).sum(2) / ncand[:, None, None], x.masked_fill(w == 0, NEG).max(2).values], -1)
            zc = self.cred(emb).squeeze(-1)  # credibility logit per source
            r = torch.sigmoid(zc) if self.learn_cred else torch.ones_like(zc)
            a = (r * rel).clamp(0, 1 - 1e-4) * rmask.float()
            if self.robust:
                ll = torch.log(a[..., None] * p + (1 - a[..., None]) / ncand[:, None, None]) + torch.log(ncand)[:, None, None]
            else:  # ablation: product of experts, credibility as an exponent (unbounded influence)
                ll = a[..., None] * torch.log(p.clamp(min=1e-6) * ncand[:, None, None])
            ll = ll.masked_fill(~cmask[:, None], 0) * rmask[..., None].float()
            if self.dep:
                omega = 1.0 / self.dependence(emb, rmask)
                ll = ll * omega[..., None]
            else:
                omega = torch.ones_like(a)
            score = ll.sum(1)
            ctx = torch.cat([(emb * rmask[..., None].float()).sum(1) / rmask.sum(1, keepdim=True).clamp(min=1),
                             emb.masked_fill(~rmask[..., None], NEG).max(1).values], -1)
            if self.residual:
                score = (0 if self.direct else score) + self.res((x * rmask[:, :, None, None].float()).sum(1)
                                         / rmask.sum(1)[:, None, None].clamp(min=1)).squeeze(-1)
            score = score.masked_fill(~cmask, NEG)
            logit = torch.cat([score, self.none(ctx)], -1)
            q = logit[:, :C].softmax(-1).detach() * cm_f
            outs.append((logit, zc, a, score, ctx))
        self.last_omega = omega
        logit, zc, a, score, ctx = outs[-1]
        self.last_binary = (score - self.set_bias(ctx)).masked_fill(~cmask, NEG)
        self.last_cred, self.last_types, self.last_rmask = zc, types, rmask
        self.last_iters = outs
        return (logit, outs) if return_all else logit
