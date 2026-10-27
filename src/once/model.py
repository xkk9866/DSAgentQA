"""ONCE-Net: an axial set transformer over a frozen LLM's isolated readings.

The input is a grid: rows are retrieved passages, columns are candidate answers. Cell (i, c)
holds how passage i, read alone, moves the LLM on candidate c relative to the closed-book
reading. Layers alternate two attentions:

* passage axis, redundancy-normalized attention (RNA): the logit of key j is lowered by
  log rho_j, where rho_j = sum_k kappa(j, k) is a learned kernel density over keys. A cluster
  of near-copies shares the attention one passage would get, however many copies there are.
* candidate axis, ordinary attention: answers compete inside each passage's reading.

The readout pools each column with RNA and scores it; a learned NONE token lets the model
abstain when no candidate is supported.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e4


class RNA(nn.Module):
    """Multi-head attention with keys reweighted by 1 / (learned kernel density)."""

    def __init__(self, d: int, heads: int = 4, kdim: int = 32, redundancy: bool = True):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.redundancy = redundancy
        self.kp = nn.Linear(d, heads * kdim, bias=False)
        self.kdim = kdim
        self.log_bw = nn.Parameter(torch.zeros(heads))  # kernel bandwidth per head

    def density(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # mask: keys that count as sources
        """log rho_j = log sum_k kappa(x_j, x_k); kappa(x, x) = 1 so rho_j >= 1."""
        B, N, _ = x.shape
        p = self.kp(x).view(B, N, self.h, self.kdim).transpose(1, 2)  # B h N k
        p = F.normalize(p, dim=-1)
        d2 = (p[:, :, :, None] - p[:, :, None]).pow(2).sum(-1)  # B h N N
        kap = torch.exp(-d2 * torch.exp(self.log_bw)[None, :, None, None] * 8.0)
        kap = kap * mask[:, None, None, :].float()
        return torch.log(kap.sum(-1).clamp(min=1.0))  # B h N

    def forward(self, xq, xkv, mask, return_attn=False, dmask=None):
        B, Nq, _ = xq.shape
        N = xkv.shape[1]
        q = self.q(xq).view(B, Nq, self.h, self.dk).transpose(1, 2)
        k = self.k(xkv).view(B, N, self.h, self.dk).transpose(1, 2)
        v = self.v(xkv).view(B, N, self.h, self.dk).transpose(1, 2)
        logit = q @ k.transpose(-1, -2) / math.sqrt(self.dk)
        if self.redundancy:
            logit = logit - self.density(xkv, mask if dmask is None else dmask)[:, :, None, :]
        logit = logit.masked_fill(~mask[:, None, None, :], NEG)
        a = logit.softmax(-1)
        out = (a @ v).transpose(1, 2).reshape(B, Nq, -1)
        return (self.o(out), a) if return_attn else self.o(out)


class Block(nn.Module):
    def __init__(self, d, heads, redundancy, dropout):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.pass_attn = RNA(d, heads, redundancy=redundancy)
        self.cand_attn = RNA(d, heads, redundancy=False)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pmask, cmask, smask=None):
        B, N, C, d = x.shape
        # passage axis: each candidate column is a set of passages (plus the joint verdict row)
        xc = x.permute(0, 2, 1, 3).reshape(B * C, N, d)
        pm = pmask[:, None].expand(B, C, N).reshape(B * C, N)
        sm = pm if smask is None else smask[:, None].expand(B, C, N).reshape(B * C, N)
        h = self.ln1(xc)
        xc = xc + self.drop(self.pass_attn(h, h, pm, dmask=sm))
        x = xc.view(B, C, N, d).permute(0, 2, 1, 3)
        # candidate axis: answers compete within each passage's reading
        xr = x.reshape(B * N, C, d)
        cm = cmask[:, None].expand(B, N, C).reshape(B * N, C)
        h = self.ln2(xr)
        xr = xr + self.drop(self.cand_attn(h, h, cm))
        x = xr.view(B, N, C, d)
        return x + self.drop(self.ff(self.ln3(x)))


class OnceNet(nn.Module):
    N_SCALAR = 6

    def __init__(self, hdim: int = 512, d: int = 192, layers: int = 2, heads: int = 4,
                 redundancy: bool = True, differential: bool = True, dropout: float = 0.1, neff: bool = False,
                 joint: bool = False, isolated: bool = True, residual: bool = False, prior_source: bool = False,
                 neutral_id: bool = False):
        super().__init__()
        self.neutral_id = neutral_id and prior_source
        # prior_source: the closed-book reading is one more source row, pooled and counted like a passage,
        # instead of an unbounded additive term in the readout
        self.prior_source = prior_source
        self.prior_type = nn.Parameter(torch.zeros(1, 1, 1, d))
        self.residual = residual and joint
        self.use_joint, self.use_isolated = joint, isolated
        self.differential = differential
        self.redundancy = redundancy
        self.use_neff = neff
        extra = (2 if neff else 1) + (d if joint else 0) - (1 if prior_source else 0)
        self.joint_type = nn.Parameter(torch.zeros(1, 1, 1, d))
        self.ln_joint = nn.LayerNorm(d)
        self.scalar = nn.Sequential(nn.Linear(self.N_SCALAR, d), nn.GELU(), nn.Linear(d, d))
        self.cell = nn.Sequential(nn.LayerNorm(hdim), nn.Linear(hdim, d))
        self.passage = nn.Sequential(nn.LayerNorm(hdim), nn.Linear(hdim, d))
        self.cand = nn.Sequential(nn.LayerNorm(hdim), nn.Linear(hdim, d))
        self.blocks = nn.ModuleList(Block(d, heads, redundancy, dropout) for _ in range(layers))
        self.pool_q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = RNA(d, heads, redundancy=redundancy)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d + extra, d), nn.GELU(), nn.Linear(d, 1))
        self.none = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.prior_w = nn.Parameter(torch.tensor(0.5))
        self.binary = nn.Linear(d + extra, 1)  # per-candidate "is a correct answer", for multi-answer sets
        if self.residual:
            # residual audit: the score starts at the LLM's joint verdict and the network learns corrections
            self.verdict_w = nn.Parameter(torch.tensor(1.0))
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def scalars(L, L0, pmask, cmask):
        """Per-cell reading statistics from candidate log-likelihoods.

        L: B N C+1 (last column = "I don't know"), L0: B C+1.
        """
        C = L.shape[-1] - 1
        cm1 = torch.cat([cmask, torch.ones_like(cmask[:, :1])], 1)
        full = (L.masked_fill(~cm1[:, None], NEG)).log_softmax(-1)
        lq = L[..., :C].masked_fill(~cmask[:, None], NEG).log_softmax(-1)
        lq0 = L0[:, :C].masked_fill(~cmask, NEG).log_softmax(-1)
        rel = 1 - full[..., C:].exp()
        top = (lq >= lq.max(-1, keepdim=True).values - 1e-6).float()
        s = torch.stack([lq.clamp(min=-20), (lq - lq0[:, None]).clamp(-20, 20), rel.expand_as(lq),
                         lq0[:, None].expand_as(lq).clamp(min=-20), top, full[..., :C].clamp(min=-20)], -1)
        return s / torch.tensor([10., 10., 1., 10., 1., 10.], device=s.device), lq0, full[..., :C].exp()

    def effective_sources(self, xc, pm, support, B, C, N):
        """N_eff(c) = sum_j p_j(c) / rho_j: how many independent passages back candidate c.

        rho_j is the readout's RNA kernel density, so a cluster of near-copies adds at most
        1/gamma however large it is, while dissimilar passages each add up to 1. Without RNA
        (rho = 1) this is the plain soft vote count.
        """
        if self.redundancy:
            rho = self.pool.density(xc, pm).exp().mean(1)  # (B*C, N), heads averaged
        else:
            rho = torch.ones(xc.shape[:2], device=xc.device)
        w = (support.permute(0, 2, 1).reshape(B * C, N) / rho) * pm.float()
        return w.sum(-1).view(B, C)

    def forward(self, batch, return_attn=False):
        L, L0, H, H0, Z, Z0 = (batch[k] for k in ("L", "L0", "H", "H0", "Z", "Z0"))
        pmask, cmask = batch["pmask"], batch["cmask"]
        s, lq0, support = self.scalars(L, L0, pmask, cmask)
        if self.prior_source:  # passage cells carry no copy of the prior; it enters only through its own row
            s = s * torch.tensor([1., 0., 1., 0., 1., 1.], device=s.device)
        if self.differential:
            zc, hp = Z - Z0[:, None], H - H0[:, None]
        else:
            zc, hp = Z, H
        if self.neutral_id:
            # candidate identity from the passages' readings, not the closed-book one, so no cell carries the prior
            w = pmask.float()[:, :, None, None]
            zid = (Z * w).sum(1) / w.sum(1).clamp(min=1)
        else:
            zid = Z0
        x = self.scalar(s) + self.cell(zc) + self.passage(hp)[:, :, None] + self.cand(zid)[:, None]
        if not self.use_isolated:  # ablation: the joint verdict and the prior only
            pmask = torch.zeros_like(pmask)
        if self.prior_source:
            s0, _, sup0 = self.scalars(L0[:, None], L0, None, cmask)
            x0 = self.scalar(s0) + self.cand(zid)[:, None] + self.prior_type
            if self.neutral_id:
                x0 = x0 + self.cell(Z0)[:, None]
            x = torch.cat([x0, x], 1)
            pmask = torch.cat([torch.ones_like(pmask[:, :1]), pmask], 1)
            support = torch.cat([sup0, support], 1)
        smask = pmask
        if self.use_joint:
            # the joint reading is one more row; it attends to the isolated testimony but is not a source
            sj, _, _ = self.scalars(batch["LJ"][:, None], L0, None, cmask)
            zj, hj = (batch["ZJ"] - Z0, batch["HJ"] - H0) if self.differential else (batch["ZJ"], batch["HJ"])
            xj = self.scalar(sj) + self.cell(zj[:, None]) + self.passage(hj)[:, None, None] + self.cand(zid)[:, None]
            x = torch.cat([x, xj + self.joint_type], 1)
            pmask_all = torch.cat([pmask, torch.ones_like(pmask[:, :1])], 1)
            smask = torch.cat([pmask, torch.zeros_like(pmask[:, :1])], 1)
        else:
            pmask_all = pmask
        for blk in self.blocks:
            x = blk(x, pmask_all, cmask, smask)
        xj_out = None
        if self.use_joint:
            x, xj_out = x[:, :-1], x[:, -1]
        B, N, C, d = x.shape
        xc = self.ln(x).permute(0, 2, 1, 3).reshape(B * C, N, d)
        pm = pmask[:, None].expand(B, C, N).reshape(B * C, N)
        q = self.pool_q.expand(B * C, 1, d)
        pooled, attn = self.pool(q, xc, pm, return_attn=True)
        pooled = pooled.view(B, C, d)
        prior = lq0.clamp(min=-20)[..., None]
        feats = [pooled] if self.prior_source else [pooled, prior / 10]
        if self.use_neff:
            feats.append(torch.log1p(self.effective_sources(xc, pm, support, B, C, N))[..., None])
        if self.use_joint:
            feats.append(self.ln_joint(xj_out))
        feats = torch.cat(feats, -1)
        logit = self.head(feats).squeeze(-1)
        if not self.prior_source:
            logit = logit + self.prior_w * prior.squeeze(-1)
        if self.residual:
            verdict = batch["LJ"][:, :C].masked_fill(~cmask, NEG).log_softmax(-1).clamp(min=-20)
            logit = logit + self.verdict_w * verdict
        logit = logit.masked_fill(~cmask, NEG)
        none = self.none(pooled.masked_fill(~cmask[..., None], 0).sum(1) / cmask.sum(1, keepdim=True).clamp(min=1))
        out = torch.cat([logit, none], -1)
        self.last_binary = self.binary(feats).squeeze(-1).masked_fill(~cmask, NEG)
        if return_attn:
            return out, attn.view(B, C, -1, N)
        return out
