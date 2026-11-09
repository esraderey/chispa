"""Modelo B20 y variantes de CHISPA-lab.

B20: d=320, 12 capas, 5 cabezas de 64, SwiGLU 864, RMSNorm, RoPE, vocabulario 16,384 atado, contexto 1,024 (~20.1M).
Memoria de definiciones (E5): W_q, W_k (d×d cada una) y claves por símbolo: modo "estado" (media de los estados del
propio modelo sobre la cabecera) o "chars" (suma de cubetas de tripletas de caracteres del nombre + MLP).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 16384
    d: int = 320
    n_layer: int = 12
    n_head: int = 5
    ffn: int = 864
    ctx: int = 1024
    memoria: str = ""          # "", "estado" o "chars"
    char_buckets: int = 4096   # solo memoria "chars"


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.w


def rope(T, hd, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device).float() / hd))
    t = torch.arange(T, device=device).float()
    f = torch.outer(t, inv)
    return f.cos(), f.sin()


def aplicar_rope(x, cos, sin):  # x: B,H,T,hd
    x1, x2 = x[..., ::2], x[..., 1::2]
    c, s = cos[None, None], sin[None, None]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1).flatten(-2)


class Bloque(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.h = c.n_head; self.hd = c.d // c.n_head
        self.n1 = RMSNorm(c.d); self.qkv = nn.Linear(c.d, 3 * c.d, bias=False); self.o = nn.Linear(c.d, c.d, bias=False)
        self.n2 = RMSNorm(c.d); self.w12 = nn.Linear(c.d, 2 * c.ffn, bias=False); self.w3 = nn.Linear(c.ffn, c.d, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k = aplicar_rope(q, cos, sin), aplicar_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(y.transpose(1, 2).reshape(B, T, D))
        a, b = self.w12(self.n2(x)).chunk(2, -1)
        return x + self.w3(F.silu(a) * b)


class GPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d)
        self.blocks = nn.ModuleList(Bloque(c) for _ in range(c.n_layer))
        self.nf = RMSNorm(c.d)
        if c.memoria:
            self.wq = nn.Linear(c.d, c.d, bias=False)
            self.wk = nn.Linear(c.d, c.d, bias=False)
            if c.memoria == "chars":
                self.cub = nn.EmbeddingBag(c.char_buckets, c.d, mode="sum")
        self.apply(self._init)
        for b in self.blocks:  # escala de salida de las residuales
            nn.init.normal_(b.o.weight, std=0.02 / math.sqrt(2 * c.n_layer))
            nn.init.normal_(b.w3.weight, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding, nn.EmbeddingBag)):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def hidden(self, idx):
        T = idx.shape[1]
        cos, sin = rope(T, self.c.d // self.c.n_head, idx.device)
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, cos.to(x.dtype), sin.to(x.dtype))
        return self.nf(x)

    def logits(self, h):
        return h @ self.emb.weight.t()

    def forward(self, idx):
        return self.hidden(idx)

    # ---- memoria de definiciones
    def claves_estado(self, cab_ids, cab_mask):
        """cab_ids: S×L cabeceras tokenizadas; cab_mask: S×L (1 = token real). Devuelve S×d."""
        h = self.hidden(cab_ids)
        m = cab_mask.unsqueeze(-1).to(h.dtype)
        return self.wk((h * m).sum(1) / m.sum(1).clamp(min=1))

    def claves_chars(self, tri_idx, tri_off):
        """tri_idx/tri_off: formato EmbeddingBag (cubetas de tripletas por símbolo). Devuelve S×d."""
        return self.wk(self.cub(tri_idx, tri_off))

    def consultas(self, h):
        return self.wq(h)


def ce_por_trozos(h, w, y, trozo=4096, ignore=-100):
    """Entropía cruzada con la cabeza atada, por trozos de posiciones para no materializar todos los logits."""
    h = h.reshape(-1, h.shape[-1]); y = y.reshape(-1)
    tot = h.new_zeros((), dtype=torch.float32); n = (y != ignore).sum()
    for i in range(0, h.shape[0], trozo):
        lg = (h[i:i + trozo] @ w.t()).float()
        tot = tot + F.cross_entropy(lg, y[i:i + trozo], ignore_index=ignore, reduction="sum")
    return tot / n.clamp(min=1)


def guardar(model, path, extra=None):
    torch.save({"config": asdict(model.c), "model": model.state_dict(), **(extra or {})}, path)


def cargar_modelo(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = GPT(Config(**ck["config"])); m.load_state_dict(ck["model"]); return m, ck
