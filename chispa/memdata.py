"""E5 — Memoria de definiciones: datos, lotes y pérdidas de los brazos A, B y C. Ver PREREGISTRO.md y DESVIACIONES.md.

Secuencia = ventana de W=768 tokens de un archivo (solo izquierda a derecha). Objetivos = posiciones donde empieza un
nombre activo en E (definido en el archivo fuera de la ventana, importado del repositorio o de un paquete, o miembro de
una clase del repositorio) que no aparece antes en la ventana. Hasta 4 símbolos objetivo por secuencia llevan cabecera
(≤64 tokens) para B/C; el brazo A recibe 256 tokens de firmas de lo importado, elegidas SIN mirar la ventana.

Construcción: python -m chispa.memdata --datos D --split train --tokens 4e8 --seed 99 [--procs 16]
              python -m chispa.memdata --datos D --split heldout --tokens 2e7 --seed 5 (evaluación)
"""
from __future__ import annotations

import argparse, ast, hashlib, json, random, re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import ce_por_trozos
from .symbols import BUILTINS, Repo, activos, parse

W, PREF, NSYM, HL, KMAX, NTRI = 768, 256, 4, 64, 32, 32
_TK = None


def _init(datos):
    global _TK
    from .tok import cargar
    _TK = cargar(datos)


def trigramas(name, buckets=4096):
    s = f"#{name}#"
    return [int(hashlib.md5(s[i:i + 3].encode()).hexdigest()[:8], 16) % buckets for i in range(len(s) - 2)][:NTRI]


def _entradas(nombres, modulos, miembros):
    """Entradas de E por nombre emitido -> cabecera (prioridad: nombres > atributos de módulo > miembros)."""
    e = {}
    for k, v in miembros.items():
        e[k] = v
    for al, d in modulos.items():
        for k, v in d.items():
            e[k] = f"{al}.{v}" if not str(v).startswith(al) else v
    for k, (_, v) in nombres.items():
        e[k] = v
    return e


def _repo(args):
    r, seed = args
    repo = Repo(r["files"]); rng = random.Random(seed)
    pad = _TK.token_to_id("<pad>"); out = []
    for f in r["files"]:
        src = f["content"]; tree = parse(src)
        if tree is None:
            continue
        nombres, modulos, miembros = activos(repo, f["path"], src)
        E = _entradas(nombres, modulos, miembros)
        if not E:
            continue
        # prefijo del brazo A: cabeceras de lo importado en orden de importación (sin mirar la ventana)
        imp = [v for k, (o, v) in nombres.items() if o in ("repo", "paquete")]
        for al, d in modulos.items():
            imp += list(d.values())[:8]
        pref_ids = _TK.encode("\n".join(f"# {h}" for h in imp) + "\n", add_special_tokens=False).ids[:PREF] if imp else []
        enc = _TK.encode(src, add_special_tokens=False)
        ids, offs = enc.ids, [a for a, _ in enc.offsets]
        # posiciones de inicio de nombre (Name Load y atributos) con su nombre
        from .e1 import _char_offsets
        conv = _char_offsets(src); occ = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in BUILTINS:
                occ[conv(n.lineno, n.col_offset)] = n.id
            elif isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load) and getattr(n, "end_lineno", None):
                occ[conv(n.end_lineno, n.end_col_offset) - len(n.attr)] = n.attr
        tok_at = {}
        for t, a in enumerate(offs):
            if a in occ:
                tok_at[t] = occ[a]
            elif a + 1 in occ and src[a] == " ":
                tok_at[t] = occ[a + 1]
        for w0 in range(0, max(1, len(ids) - 32), W):
            win = ids[w0:w0 + W]
            if len(win) < 64:
                break
            text0 = offs[w0]
            tg = []
            for t in range(w0 + 1, w0 + len(win)):
                nm = tok_at.get(t)
                if nm and nm in E and not re.search(r"(?<!\w)" + re.escape(nm) + r"(?!\w)", src[text0:offs[t]]):
                    tg.append((t - w0, nm))
            if not tg:
                continue
            tg = tg[:KMAX]
            syms = []
            for _, nm in tg:
                if nm not in syms:
                    syms.append(nm)
            rng.shuffle(syms); syms = syms[:NSYM]
            main = np.full(W, pad, np.uint16); main[:len(win)] = win
            pref = np.full(PREF, pad, np.uint16); pref[PREF - len(pref_ids):] = pref_ids if pref_ids else []
            hdr = np.full((NSYM, HL), pad, np.uint16); tri = np.full((NSYM, NTRI), -1, np.int16)
            for j, nm in enumerate(syms):
                h = _TK.encode(str(E[nm]), add_special_tokens=False).ids[:HL]; hdr[j, :len(h)] = h
                tr = trigramas(nm); tri[j, :len(tr)] = tr
            pos = np.full(KMAX, -1, np.int16); sym = np.full(KMAX, -1, np.int8)
            for k, (p, nm) in enumerate(tg):
                pos[k] = p; sym[k] = syms.index(nm) if nm in syms else -1
            names = [nm for _, nm in tg]
            out.append((main, pref, hdr, tri, pos, sym, len(win), names, len(E), f["path"]))
    return out


def construir(datos: Path, split: str, tokens: float, seed: int, procs: int):
    from .data import repos
    od = datos / "mem"; od.mkdir(exist_ok=True)
    rs = [r for r in repos(datos, split) if len(r["files"]) >= 2]
    random.Random(seed).shuffle(rs)
    cols = {k: [] for k in ("main", "pref", "hdr", "tri", "pos", "sym", "n", "E")}; names = []; total = 0
    with Pool(procs, initializer=_init, initargs=(str(datos),)) as p:
        for rows in p.imap_unordered(_repo, [(r, seed + i) for i, r in enumerate(rs)], chunksize=2):
            for main, pref, hdr, tri, pos, sym, n, nm, nE, _path in rows:
                for k, v in zip(("main", "pref", "hdr", "tri", "pos", "sym", "n", "E"), (main, pref, hdr, tri, pos, sym, n, nE)):
                    cols[k].append(v)
                names.append(nm); total += n
            if total >= tokens:
                break
    np.savez(od / f"{split}.npz", **{k: np.stack(v) if k not in ("n", "E") else np.array(v) for k, v in cols.items()})
    json.dump(names, open(od / f"{split}_nombres.json", "w"))
    meta = {"secuencias": len(names), "tokens": total, "objetivos": int((np.stack(cols["pos"]) >= 0).sum()),
            "con_cabecera": int((np.stack(cols["sym"]) >= 0).sum()), "E_p95": float(np.percentile(cols["E"], 95)), "seed": seed}
    json.dump(meta, open(od / f"{split}.json", "w")); print(split, meta, flush=True)


class FuenteMem:
    def __init__(self, datos: Path, ctx: int, seed: int, modo: str, a):
        d = np.load(datos / "mem" / "train.npz")
        self.d = {k: d[k] for k in d.files}; self.N = len(self.d["n"]); self.modo = modo
        self.rng = np.random.default_rng(seed)
        from .tok import cargar
        self.pad = cargar(datos).token_to_id("<pad>")

    def lote(self, B):
        i = self.rng.integers(0, self.N, B)
        g = {k: torch.from_numpy(v[i].astype(np.int64)) for k, v in self.d.items() if k not in ("E",)}
        return g, self.pad, self.modo

    def saltar(self, B):
        self.lote(B)


def perdida_mem(modo, fwd, model, lote, dev, a):
    g, pad, _ = lote
    g = {k: v.to(dev, non_blocking=True) for k, v in g.items()}
    main = g["main"]
    if modo == "memA":
        x = torch.cat([g["pref"], main], 1)
        h = fwd(x[:, :-1]); y = x[:, 1:].clone()
        y[:, :PREF - 1] = -100                       # sin pérdida sobre el prefijo
        y[y == pad] = -100
        loss = ce_por_trozos(h, model.emb.weight, y)
        return loss, {"loss": loss.item(), "mem": 0.0}
    h = fwd(main[:, :-1]); y = main[:, 1:].clone(); y[y == pad] = -100
    lm = ce_por_trozos(h, model.emb.weight, y)
    # memoria: claves de los ≤4 símbolos de cada secuencia; negativos = los del lote
    B = main.shape[0]
    if modo == "memB":
        hdr = g["hdr"].reshape(B * NSYM, HL); mask = (hdr != pad)
        keys = model.claves_estado(hdr, mask)
    else:
        tri = g["tri"].reshape(B * NSYM, NTRI); valid = tri >= 0
        flat = tri[valid]; lens = valid.sum(1)
        off = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), lens.cumsum(0)[:-1]])
        keys = model.claves_chars(flat, off)
    pos, sym = g["pos"], g["sym"]
    sel = (pos > 0) & (sym >= 0)
    if sel.sum() == 0:
        return lm, {"loss": lm.item(), "mem": 0.0}
    bi = torch.arange(B, device=dev)[:, None].expand_as(pos)[sel]
    pi = pos[sel] - 1                                # el estado que predice el token en pos
    tgt = bi * NSYM + sym[sel]
    hq = h[bi, pi]
    lv = model.logits(hq).float(); lmem = (model.consultas(hq) @ keys.t()).float()
    vacio = (g["hdr"].reshape(B * NSYM, HL)[:, 0] == pad) if modo == "memB" else ~(g["tri"].reshape(B * NSYM, NTRI)[:, 0] >= 0)
    lmem = lmem.masked_fill(vacio[None, :], -1e4)
    mem = F.cross_entropy(torch.cat([lv, lmem], 1), lv.shape[1] + tgt)
    return lm + mem, {"loss": lm.item(), "mem": mem.item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True); ap.add_argument("--split", default="train")
    ap.add_argument("--tokens", type=float, default=4e8); ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    construir(Path(a.datos), a.split, a.tokens, a.seed, a.procs)


if __name__ == "__main__":
    main()
