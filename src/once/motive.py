"""MOTIVE: infer the answer together with the attacker's motive.

A poisoning attacker can make every injected passage look genuine, but it cannot hide what its passages agree on: to
succeed, all of them must assert one target answer. MOTIVE therefore reasons over hypotheses (a, t): a is the true
answer and t is the answer an attacker is pushing, or t = 0 (no attack). Each hypothesis explains every source:

  memory m          the closed-book reading; never attacker-controlled, honest under every hypothesis
  passage s         read alone; honest under (a, 0); under (a, t) the attacker chose which passages are its
                    own, so each passage is explained by its better role (profile likelihood over roles):
                    honest for a or written for t
  joint reader J    reads the whole context; honest under (a, 0); under (a, t) it has read the attacker's passages
                    and is treated exactly like a passage

An honest source with competence rho_s testifies through the bounded likelihood
    l_s(c) = log(rho_s p_s(c) + (1 - rho_s)/K) + log K        (Jeffrey's rule under epsilon-contamination)
and the hypothesis scores are
    S(a, 0) = g0 + l_m(a) + l_J(a) + sum_s l_s(a)
    S(a, t) = f_t + l_m(a) + sum_{s in passages + J} max(l_s(a), l_s(t)),     t != a.
Because the passage term is symmetric in (a, t), S(a, t) - S(t, a) = f_t - f_a + l_m(a) - l_m(t) exactly: no amount
of testimony can make a candidate the truth rather than the motive; only motive evidence f and memory can. Every
hypothesis that involves t gains the same amount from a passage that favours t, so a flood of such passages cannot
shift the posterior between them. f and g0 are bounded by a learned budget B, and the memory competence is computed
from the memory reading alone, so the attacker's total influence on the odds of t against a is bounded whatever the
number or the content of the injected passages.

A small permutation-equivariant network over the source x candidate grid supplies the quantities of this model:
rho_s per source, f_c per candidate (the motive score, pooled over the passages that back c), g0 (no attack) and a
none-of-the-candidates logit. Inputs are the reader's testimony p_s and its reading state h_s, nothing else.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .credence import NEG, AxialBlock

MEMORY, PASSAGE, JOINT = 0, 1, 2
LOG2 = math.log(2.0)


class Motive(nn.Module):
    def __init__(self, hdim=512, d=128, layers=2, heads=4, dropout=0.2, use_joint=True, use_memory=True,
                 use_h=True, symmetric=True, motive=True, direct=False, learn_rho=True, pool="motive",
                 j_suff=False, blocks=False, motive_prior=False):
        super().__init__()
        self.use_joint, self.use_memory, self.use_h = use_joint, use_memory, use_h
        # ablations: symmetric=False drops the role-swap (passages stay honest under attack hypotheses, i.e. the
        # attack only adds the motive prior); motive=False removes attack hypotheses altogether (product of robust
        # likelihoods, CREDENCE-style pooling without source credibility); direct=True reads answer logits from the
        # network (same inputs and encoder, no inference layer)
        self.symmetric, self.motive, self.direct, self.learn_rho = symmetric, motive, direct, learn_rho
        # matched pooling controls on the same encoder, inputs and training: "poe" (credibility as an exponent),
        # "trunc" (product of bounded contamination likelihoods = attack hypotheses removed), "mixture" (learned
        # linear mixture of testimonies), "vote" (learned credibility filter, then a vote among kept sources)
        self.pool = "trunc" if (pool == "motive" and not motive) else pool
        # j_suff: in an unattacked context the joint reading, which has read every passage, is a sufficient summary
        # of the context; the isolated passages enter the no-attack hypothesis only when no joint reading exists
        self.j_suff = j_suff
        # blocks: the attacked and the unattacked hypotheses are normalized separately and weighted by a bounded,
        # learned attack probability, so that the profile over roles cannot make attack hypotheses win by default
        self.use_blocks = blocks
        # motive_prior: the K-1 possible motives share the attack prior uniformly (log prior -log(K-1) per motive)
        self.motive_prior = motive_prior
        if blocks:
            self.top = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 3))  # no attack, attack, none
        self.cellf = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.state = nn.Sequential(nn.LayerNorm(hdim), nn.Linear(hdim, d))
        self.type_emb = nn.Embedding(3, d)
        self.blocks = nn.ModuleList(AxialBlock(d, heads, dropout) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.rho = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        # memory competence from the memory reading alone (attacker-independent)
        self.rho_mem = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.log_b = nn.Parameter(torch.tensor(math.log(math.expm1(4.0))))  # motive budget B = softplus(.) = 4
        self.att = nn.Linear(d, 1)
        self.fmot = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.g0 = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.none = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        if direct:
            self.out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    # ---- inputs --------------------------------------------------------------------------------
    def assemble(self, b):
        """Rows [memory, passages..., joint]: L (B,R,C+1) with an "I don't know" column, H (B,R,h), types, rmask."""
        B, N = b["L"].shape[:2]
        dev = b["L"].device
        L, H = [b["L0"][:, None], b["L"]], [b["H0"][:, None], b["H"]]
        types = [torch.full((B, 1), MEMORY, device=dev), torch.full((B, N), PASSAGE, device=dev)]
        rmask = [torch.full((B, 1), self.use_memory, device=dev, dtype=torch.bool), b["pmask"]]
        if "LJ" in b:
            L.append(b["LJ"][:, None])
            H.append(b["HJ"][:, None])
            types.append(torch.full((B, 1), JOINT, device=dev))
            jm = b.get("jmask", torch.ones(B, dtype=torch.bool, device=dev))
            rmask.append((jm & self.use_joint)[:, None])
        return torch.cat(L, 1), torch.cat(H, 1), torch.cat(types, 1).long(), torch.cat(rmask, 1)

    # ---- forward -------------------------------------------------------------------------------
    def forward(self, b):
        """Returns flat logits over hypotheses: (B, C*(C+1) + 1), index a*(C+1)+t for truth a and motive t (t = C is
        "no attack"), last entry = none of the candidates. Invalid pairs (t = a, masked candidates) are NEG."""
        L, H, types, rmask = self.assemble(b)
        cmask = b["cmask"]
        B, R, C1 = L.shape
        C = C1 - 1
        cm1 = torch.cat([cmask, torch.ones_like(cmask[:, :1])], 1)[:, None]
        full = L.masked_fill(~cm1, NEG).log_softmax(-1)
        lq = L[..., :C].masked_fill(~cmask[:, None], NEG).log_softmax(-1)
        rel = 1 - full[..., C].exp()
        feats = torch.stack([lq.clamp(min=-20) / 10, full[..., :C].clamp(min=-20) / 10, rel[..., None].expand_as(lq)], -1)
        x = self.cellf(feats) + self.type_emb(types)[:, :, None]
        if self.use_h:
            x = x + self.state(H)[:, :, None]
        cm0 = cmask[:, :, None].float()
        x_mem = x[:, 0]  # memory row before any interaction with the passages
        e_mem = torch.cat([(x_mem * cm0).sum(1) / cm0.sum(1).clamp(min=1),
                           x_mem.masked_fill(cm0 == 0, NEG).max(1).values], -1)
        for blk in self.blocks:
            x = blk(x, rmask, cmask)
        x = self.ln(x)
        cm_f = cmask.float()
        rm_f = rmask.float()
        ncand = cm_f.sum(1).clamp(min=1)
        w = cm_f[:, None, :, None]
        emb = torch.cat([(x * w).sum(2) / ncand[:, None, None], x.masked_fill(w == 0, NEG).max(2).values], -1)
        ctx = torch.cat([(emb[..., :x.shape[-1]] * rm_f[..., None]).sum(1) / rm_f.sum(1, keepdim=True).clamp(min=1),
                         x.masked_fill(~(rmask[:, :, None, None] & cmask[:, None, :, None]), NEG).amax((1, 2))], -1)
        none = self.none(ctx).squeeze(-1)
        self.last_rmask, self.last_types = rmask, types
        if self.direct:
            cand = torch.cat([(x * rm_f[:, :, None, None]).sum(1) / rm_f.sum(1)[:, None, None].clamp(min=1),
                              x.masked_fill(~rmask[:, :, None, None], NEG).max(1).values], -1)
            score = self.out(cand).squeeze(-1).masked_fill(~cmask, NEG)
            flat = torch.full((B, C, C + 1), NEG, device=x.device)
            flat[:, :, C] = score
            self.last_parts = None
            return torch.cat([flat.reshape(B, -1), none[:, None]], -1)

        # competence of every honest source; p_s over candidates
        zr = self.rho(emb).squeeze(-1)
        self.last_cred = zr
        if self.pool in ("poe", "mixture", "vote"):
            p = lq.exp() * cm_f[:, None]
            if self.pool == "poe":
                w_ = torch.sigmoid(zr) * rm_f
                score = (w_[..., None] * (torch.log(p.clamp(min=1e-6)) + torch.log(ncand)[:, None, None])).sum(1)
            elif self.pool == "mixture":
                w_ = zr.masked_fill(~rmask, NEG).softmax(1)
                score = torch.log((w_[..., None] * p).sum(1).clamp(min=1e-9))
            else:  # vote: soft weighted vote for training; hard filter-and-vote at inference
                c_ = torch.sigmoid(zr) * rm_f
                if self.training:
                    score = torch.log((c_[..., None] * p * rel[..., None]).sum(1).clamp(min=1e-9))
                else:
                    top = F.one_hot(lq.argmax(-1), C).float() * (rel > 0.5).float()[..., None]
                    keep = (c_ > 0.5).float()
                    votes = (keep[..., None] * top).sum(1)
                    soft = (c_[..., None] * p * rel[..., None]).sum(1)
                    score = votes + 1e-3 * soft / soft.sum(-1, keepdim=True).clamp(min=1e-9)
                    score = torch.log(score.clamp(min=1e-9))
            score = score.masked_fill(~cmask, NEG)
            flat = torch.full((B, C, C + 1), NEG, device=x.device)
            flat[:, :, C] = score
            self.last_parts = None
            return torch.cat([flat.reshape(B, -1), none[:, None]], -1)
        r = torch.sigmoid(zr) if self.learn_rho else torch.ones_like(rel)
        if self.learn_rho:
            r = torch.cat([torch.sigmoid(self.rho_mem(e_mem)), r[:, 1:]], 1)
        rho = (r * rel).clamp(0, 1 - 1e-4) * rm_f
        p = lq.exp() * cm_f[:, None]
        ll = torch.log(rho[..., None] * p + (1 - rho[..., None]) / ncand[:, None, None]) + torch.log(ncand)[:, None, None]
        ll = ll.masked_fill(~cmask[:, None], 0) * rm_f[..., None]  # (B, R, C)
        is_mem = (types == MEMORY).float() * rm_f
        is_pas = (types == PASSAGE).float() * rm_f
        is_jnt = (types == JOINT).float() * rm_f
        l_mem = (ll * is_mem[..., None]).sum(1)  # (B, C)
        l_honest = (ll * (is_pas + is_jnt)[..., None]).sum(1)

        # motive score per candidate: attention over the passages that back it, plus their mean
        pw = (p * rel[..., None]) * is_pas[..., None]  # soft membership of passage s in the block of c
        a_logit = torch.log(pw.clamp(min=1e-6)) + self.att(x).squeeze(-1)
        a_logit = a_logit.masked_fill(~(is_pas.bool()[..., None] & cmask[:, None]), NEG)
        alpha = a_logit.softmax(1)  # over sources, per candidate
        b_att = (alpha[..., None] * x).sum(1)
        b_mean = (pw[..., None] * x).sum(1) / pw.sum(1)[..., None].clamp(min=1e-6)
        Bm = F.softplus(self.log_b)
        f = Bm * torch.tanh(self.fmot(torch.cat([b_att, b_mean, ctx[:, None, :x.shape[-1]].expand(-1, C, -1)], -1))
                            .squeeze(-1) / Bm)
        g0 = Bm * torch.tanh(self.g0(ctx).squeeze(-1) / Bm)

        S = torch.full((B, C, C + 1), NEG, device=x.device)
        if self.j_suff:
            has_j = is_jnt.sum(1, keepdim=True) > 0  # (B, 1)
            l_jnt = (ll * is_jnt[..., None]).sum(1)
            l_pas = (ll * is_pas[..., None]).sum(1)
            S[:, :, C] = g0[:, None] + l_mem + torch.where(has_j, l_jnt, l_pas)
        else:
            S[:, :, C] = g0[:, None] + l_mem + l_honest
        if self.pool == "motive":
            src = (is_pas + is_jnt)[..., None, None]
            la, lt = ll[:, :, :, None], ll[:, :, None, :]
            if self.symmetric:
                pair = torch.maximum(la, lt) * src  # (B, R, C_a, C_t)
            else:
                pair = la.expand(-1, -1, -1, C) * src
            S[:, :, :C] = f[:, None, :] + l_mem[..., None] + pair.sum(1)
            if self.motive_prior:
                S[:, :, :C] = S[:, :, :C] - torch.log((ncand - 1).clamp(min=1))[:, None, None]
            valid = cmask[:, :, None] & cmask[:, None, :] & ~torch.eye(C, dtype=torch.bool, device=x.device)[None]
            S[:, :, :C] = S[:, :, :C].masked_fill(~valid, NEG)
        S = S.masked_fill(~cmask[:, :, None], NEG)
        self.last_parts = {"rho": rho, "f": f, "g0": g0, "l_mem": l_mem, "B": Bm}
        if self.use_blocks and self.pool == "motive":
            top = (Bm * torch.tanh(self.top(ctx) / Bm)).log_softmax(-1)  # (B, 3)
            s0 = (S[:, :, C] - g0[:, None]).masked_fill(~cmask, NEG).log_softmax(1)  # P(a | no attack)
            sa = S[:, :, :C].reshape(B, -1).log_softmax(1).view(B, C, C)  # P(a, t | attack)
            out = torch.full_like(S, NEG)
            out[:, :, C] = (top[:, :1] + s0).masked_fill(~cmask, NEG)
            out[:, :, :C] = torch.where(S[:, :, :C] > NEG / 2, top[:, 1:2, None] + sa, torch.full_like(sa, NEG))
            self.last_parts["top"] = top
            return torch.cat([out.reshape(B, -1), top[:, 2:3]], -1)
        return torch.cat([S.reshape(B, -1), none[:, None]], -1)

    @staticmethod
    def split(flat, C):
        """Flat hypothesis logits -> answer logits (B, C+1; last = none) and motive log-posterior (B, C+1)."""
        B = flat.shape[0]
        S = flat[:, :-1].view(B, C, C + 1)
        ans = torch.cat([torch.logsumexp(S, 2), flat[:, -1:]], 1)
        mot = torch.logsumexp(S, 1)
        return ans, mot
