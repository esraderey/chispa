"""E4 — Monitor amortizado: datos anotados con el oráculo de nombres visibles y pérdida L_UL.

Construcción: python -m chispa.oraculo --datos DATOS --tokens 1.2e8 --seed 2024 [--procs 16]
  -> DATOS/ul/ul-XXX.bin (uint16 tokens), ul-XXX.ann (int32: id de conjunto o -1), ul-XXX.sets.npz (conjuntos)
Un documento con `import *`, `exec(` o `globals()` va sin oráculo (todo -1). Solo los documentos sin FIM llevan oráculo.
"""
from __future__ import annotations

import argparse, ast, bisect, json, random, re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import ce_por_trozos
from .symbols import parse

_TK = None; _FIRST = {}


def _init(datos):
    global _TK
    from .tok import cargar
    _TK = cargar(datos)


def _primeras(v):
    if v not in _FIRST:
        a = _TK.encode(v, add_special_tokens=False).ids; b = _TK.encode(" " + v, add_special_tokens=False).ids
        _FIRST[v] = {a[0], b[0]} if a and b else set()
    return _FIRST[v]


def _doc(args):
    text, use_oracle = args
    enc = _TK.encode(text, add_special_tokens=False)
    ids = enc.ids; ann = [-1] * len(ids); sets = []
    if not use_oracle or re.search(r"import \*|\bexec\(|\bglobals\(\)", text):
        return ids, ann, sets
    tree = parse(text)
    if tree is None:
        return ids, ann, sets
    from .e3 import visibles
    from .e1 import _char_offsets
    conv = _char_offsets(text)
    starts = [a for a, _ in enc.offsets]
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))]
    cache = {}
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)):
            continue
        c = conv(n.lineno, n.col_offset)
        t = bisect.bisect_left(starts, c)
        if t < len(starts) and starts[t] == c:
            pass                                             # pieza que empieza en el nombre
        elif t > 0 and starts[t - 1] == c - 1 and text[c - 1] == " ":
            t -= 1                                           # pieza « nombre» con el espacio delante
        else:
            continue
        if t == 0:
            continue
        key = tuple(id(f) for f in funcs if f.lineno <= n.lineno <= (f.end_lineno or f.lineno))
        if key not in cache:
            vis = visibles(tree, n.lineno)
            allowed = set()
            for v in vis:
                allowed |= _primeras(v)
            sets.append(sorted(allowed)); cache[key] = len(sets) - 1
        ann[t] = cache[key]
    return ids, ann, sets


def construir(datos: Path, tokens: float, seed: int, procs: int, shard=25_000_000):
    from .data import repos, doc_entrenamiento
    out = datos / "ul"; out.mkdir(exist_ok=True)
    rng = random.Random(seed); eot = None

    def docs():
        for r in repos(datos, "train"):
            if rng.random() > 0.25:  # submuestra para no recorrer todo el corpus
                continue
            for f in r["files"]:
                t, isf = doc_entrenamiento(f["content"], rng)
                yield (t, not isf)

    _init(str(datos)); eot = _TK.token_to_id("<|endoftext|>")
    toks, anns, sets, k, total = [], [], [], 0, 0
    def flush(final=False):
        nonlocal toks, anns, sets, k
        if not toks or (len(toks) < shard and not final):
            return
        np.array(toks, dtype=np.uint16).tofile(out / f"ul-{k:03d}.bin")
        np.array(anns, dtype=np.int32).tofile(out / f"ul-{k:03d}.ann")
        lens = np.array([len(s) for s in sets], dtype=np.int64)
        np.savez(out / f"ul-{k:03d}.sets.npz", ids=np.array([i for s in sets for i in s], dtype=np.int32),
                 off=np.r_[0, np.cumsum(lens)])
        k += 1; toks, anns, sets = [], [], []
    with Pool(procs, initializer=_init, initargs=(str(datos),)) as p:
        for ids, ann, ss in p.imap(_doc, docs(), chunksize=64):
            base = len(sets)
            toks += ids + [eot]; anns += [a + base if a >= 0 else -1 for a in ann] + [-1]; sets += ss
            total += len(ids) + 1
            flush()
            if total >= tokens:
                break
    flush(final=True)
    nann = sum(int((np.fromfile(f, dtype=np.int32) >= 0).sum()) for f in out.glob("ul-*.ann"))
    json.dump({"tokens": total, "posiciones_oraculo": nann, "seed": seed, "shards": k}, open(out / "ul.json", "w"))
    print(f"ul: {total} tokens, {nann} posiciones con oráculo, {k} fragmentos", flush=True)


class FuenteUL:
    def __init__(self, datos: Path, ctx: int, seed: int):
        from .tok import cargar
        tk = cargar(datos)
        self.first = torch.zeros(tk.get_vocab_size(), dtype=torch.bool)
        for t, i in tk.get_vocab().items():
            if re.match(r"^Ġ?[A-Za-z_]", t):
                self.first[i] = True
        fs = sorted((datos / "ul").glob("ul-*.bin"))
        self.toks = [np.memmap(f, dtype=np.uint16, mode="r") for f in fs]
        self.anns = [np.memmap(f.with_suffix(".ann"), dtype=np.int32, mode="r") for f in fs]
        self.sets = [np.load(str(f).replace(".bin", ".sets.npz")) for f in fs]
        self.lens = np.array([len(t) - ctx - 1 for t in self.toks]); self.ctx = ctx
        self.rng = np.random.default_rng(seed)

    def lote(self, B):
        fi = self.rng.choice(len(self.toks), size=B, p=self.lens / self.lens.sum())
        xs, conj = [], []
        for b, f in enumerate(fi):
            i = int(self.rng.integers(0, self.lens[f]))
            xs.append(torch.from_numpy(self.toks[f][i:i + self.ctx + 1].astype(np.int64)))
            a = self.anns[f][i + 1:i + self.ctx + 1]
            for p in np.nonzero(a >= 0)[0]:
                s = self.sets[f]; o = s["off"]; sid = int(a[p])
                conj.append((b, int(p), s["ids"][o[sid]:o[sid + 1]]))
        x = torch.stack(xs)
        return x[:, :-1], x[:, 1:], conj, self.first

    def saltar(self, B):
        self.lote(B)


def perdida_ul(fwd, model, lote, dev, a):
    x, y, conj, first = lote
    x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
    h = fwd(x)
    ce = ce_por_trozos(h, model.emb.weight, y)
    parts = {"loss": ce.item(), "n_ul": float(len(conj))}
    if not conj or a.alpha == 0:
        return ce, {**parts, "ul": 0.0}
    bi = torch.tensor([c[0] for c in conj], device=dev); pi = torch.tensor([c[1] for c in conj], device=dev)
    p = F.softmax(model.logits(h[bi, pi]).float(), -1)
    allowed = torch.zeros_like(p, dtype=torch.bool)
    for r, (_, _, ids) in enumerate(conj):
        allowed[r, torch.from_numpy(np.asarray(ids, dtype=np.int64)).to(dev)] = True
    fm = first.to(dev)
    m = (p * (fm & ~allowed)).sum(-1)               # masa prohibida: inicios de nombre no visibles
    ul = -torch.log((1 - m).clamp(min=1e-6)).mean()
    return ce + a.alpha * ul, {**parts, "ul": ul.item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True); ap.add_argument("--tokens", type=float, default=1.2e8)
    ap.add_argument("--seed", type=int, default=2024); ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    construir(Path(a.datos), a.tokens, a.seed, a.procs)


if __name__ == "__main__":
    main()
