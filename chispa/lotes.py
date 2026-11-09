"""Fuentes de lotes y pérdidas por modo de entrenamiento."""
from __future__ import annotations

from pathlib import Path

import torch

from .model import ce_por_trozos


class FuenteBase:
    def __init__(self, datos: Path, ctx: int, seed: int):
        from .train import Flujo
        self.f = Flujo(str(datos / "tok" / "base-*.bin"), ctx, seed)

    def lote(self, B):
        return self.f.lote(B)

    def saltar(self, B):
        self.f.lote(B)


def fuente(modo, datos, ctx, seed, a):
    if modo == "base":
        return FuenteBase(Path(datos), ctx, seed)
    if modo == "ul":
        from .oraculo import FuenteUL
        return FuenteUL(Path(datos), ctx, seed)
    if modo in ("memA", "memB", "memC"):
        from .memdata import FuenteMem
        return FuenteMem(Path(datos), ctx, seed, modo, a)
    raise ValueError(modo)


def perdida(modo, fwd, model, lote, dev, a):
    if modo == "base":
        x, y = (t.to(dev, non_blocking=True) for t in lote)
        h = fwd(x)
        loss = ce_por_trozos(h, model.emb.weight, y)
        return loss, {"loss": loss.item()}
    if modo == "ul":
        from .oraculo import perdida_ul
        return perdida_ul(fwd, model, lote, dev, a)
    if modo in ("memA", "memB", "memC"):
        from .memdata import perdida_mem
        return perdida_mem(modo, fwd, model, lote, dev, a)
    raise ValueError(modo)
