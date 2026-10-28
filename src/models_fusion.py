"""Corrected gated cross-modal attention + all ablation variants (PyTorch).

Fixes relative to the original 08_fusion_improved.py
  1. Cross-attention is real scaled dot-product attention of ONE tabular query over the 7x7=49
     spatial image tokens (the original code multiplied two global vectors elementwise and
     softmaxed over feature channels - a channel gate, not spatial attention).
  2. The gate sees BOTH streams: f = g*f_attn + (1-g)*f_tab with g = sigmoid(W[f_attn;f_tab]).
     (The original called gate(attn_out, attn_out), i.e. g*x + (1-g)*x == x, a no-op.)
  3. Optional gradient-reversal adversary that removes redshift information from the fused
     representation (novel selection-shortcut regulariser).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x, lam):
    return _GRL.apply(x, lam)


class TabEncoder(nn.Module):
    def __init__(self, d_in, d=128, p=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(128, d), nn.LayerNorm(d), nn.GELU())

    def forward(self, x):
        return self.net(x)


class SpatialCrossAttention(nn.Module):
    """softmax(Q K^T / sqrt(d)) V with Q from the tabular embedding, K,V from image tokens."""
    def __init__(self, d_model, d_attn=128, n_tokens=49):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(d_model, d_attn), nn.Linear(d_model, d_attn), nn.Linear(d_model, d_attn)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))   # learned 7x7 position embedding
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.out = nn.Linear(d_attn, d_model)
        self.scale = 1.0 / math.sqrt(d_attn)

    def forward(self, f_tab, tokens):                      # (B,D), (B,N,D)
        t = tokens + self.pos
        Q = self.q(f_tab).unsqueeze(1)                      # (B,1,da)
        K, V = self.k(t), self.v(t)                         # (B,N,da)
        w = torch.softmax((Q @ K.transpose(1, 2)) * self.scale, dim=-1)   # (B,1,N)
        return self.out((w @ V).squeeze(1)), w.squeeze(1)


class LearnedQueryPool(nn.Module):
    """Image-only spatial control: attention pooling with a learned (non-tabular) query."""
    def __init__(self, d_model, d_attn=128, n_tokens=49):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_attn) * 0.02)
        self.k, self.v = nn.Linear(d_model, d_attn), nn.Linear(d_model, d_attn)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.out = nn.Linear(d_attn, d_model)
        self.scale = 1.0 / math.sqrt(d_attn)

    def forward(self, tokens):
        t = tokens + self.pos
        K, V = self.k(t), self.v(t)
        w = torch.softmax((self.query.expand(len(t), -1, -1) @ K.transpose(1, 2)) * self.scale, dim=-1)
        return self.out((w @ V).squeeze(1)), w.squeeze(1)


MODES = ["img_gap", "img_spatial", "tab", "concat", "gate_global", "attn_nogate", "attn_gate", "attn_gate_grl"]


class FusionNet(nn.Module):
    """mode:
      img_gap      image only, global-average-pooled 1280-d feature -> MLP head (EfficientNet-style head)
      img_spatial  image only, learned-query attention pooling over the 49 tokens
      tab          tabular only
      concat       [GAP-projection ; tab embedding] -> head
      gate_global  tab-query attention on a single global token + learned gate (paper's 'global features' ablation)
      attn_nogate  spatial cross-attention, fixed 0.5/0.5 mixing (no learned gate)
      attn_gate    spatial cross-attention + learned sigmoid gate  (proposed)
      attn_gate_grl  attn_gate + gradient-reversal redshift adversary (novel)
    """
    def __init__(self, mode, tab_dim, d=128, d_tok_in=128, d_gap_in=1280, p=0.3, n_tokens=49):
        super().__init__()
        assert mode in MODES
        self.mode, self.d = mode, d
        self.gap_proj = nn.Sequential(nn.Dropout(0.2), nn.Linear(d_gap_in, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(p))
        self.tok_proj = nn.Sequential(nn.Linear(d_tok_in, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(p))
        self.tab_enc = TabEncoder(tab_dim, d, p) if mode not in ("img_gap", "img_spatial") else None
        if mode == "img_spatial":
            self.pool = LearnedQueryPool(d, n_tokens=n_tokens)
        if mode in ("attn_nogate", "attn_gate", "attn_gate_grl"):
            self.xattn = SpatialCrossAttention(d, n_tokens=n_tokens)
        if mode == "gate_global":
            self.xattn = SpatialCrossAttention(d, n_tokens=1)
        if mode in ("gate_global", "attn_gate", "attn_gate_grl"):
            self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())
        d_head_in = 2 * d if mode == "concat" else d
        self.head = nn.Sequential(nn.Linear(d_head_in, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1))
        if mode == "attn_gate_grl":
            self.adv = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))

    def encode(self, gap, tok, tab):
        m, aux = self.mode, {}
        if m == "img_gap":
            return self.gap_proj(gap), aux
        if m == "img_spatial":
            z, w = self.pool(self.tok_proj(tok)); aux["attn"] = w
            return z, aux
        ft = self.tab_enc(tab)
        if m == "tab":
            return ft, aux
        if m == "concat":
            return torch.cat([self.gap_proj(gap), ft], 1), aux
        if m == "gate_global":
            fa, w = self.xattn(ft, self.gap_proj(gap).unsqueeze(1))
        else:
            fa, w = self.xattn(ft, self.tok_proj(tok)); aux["attn"] = w
        if m == "attn_nogate":
            return 0.5 * fa + 0.5 * ft, aux
        g = self.gate(torch.cat([fa, ft], 1)); aux["gate"] = g
        return g * fa + (1 - g) * ft, aux

    def forward(self, gap, tok, tab, grl_lambda=0.0):
        h, aux = self.encode(gap, tok, tab)
        logit = self.head(h).squeeze(1)
        adv = None
        if self.mode == "attn_gate_grl":
            adv = self.adv(grad_reverse(h, grl_lambda)).squeeze(1)
        return logit, adv, h, aux
