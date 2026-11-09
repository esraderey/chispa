"""E1 — Oportunidad de la memoria de definiciones (CPU). Ver PREREGISTRO.md.

Uso: python -m chispa.e1 --datos DATOS --out RES/e1 [--repos 500] [--seed 1337] [--procs 8]
Salida cruda: RES/e1/detalle.jsonl (por archivo); resumen: RES/e1/resumen.json (recalculable con --solo-analisis).
"""
from __future__ import annotations

import argparse, ast, bisect, json, random, re
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .symbols import BUILTINS, Repo, activos, parse

VENT = 1024
_TK = None


def _init(datos):
    global _TK
    from .tok import cargar
    _TK = cargar(datos)


def _char_offsets(src):
    """Mapa (línea, col_bytes_utf8) -> índice de carácter."""
    starts, acc = [], 0
    lines = src.splitlines(keepends=True)
    for l in lines:
        starts.append(acc); acc += len(l)
    def conv(lineno, col):
        if lineno - 1 >= len(lines):
            return acc
        l = lines[lineno - 1]
        return starts[lineno - 1] + len(l.encode("utf-8")[:col].decode("utf-8", "ignore"))
    return conv


def archivo(repo: Repo, f):
    src = f["content"]
    tree = parse(src)
    if tree is None:
        return None
    nombres, modulos, miembros = activos(repo, f["path"], src)
    nombres_repo = {k for k, (o, _) in nombres.items() if o != "paquete"}
    modulos_repo = {a for a in modulos if repo.resolver(a) is not None}
    E = len(nombres) + sum(len(v) for v in modulos.values()) + len(miembros)
    E_repo = len(nombres_repo) + sum(len(modulos[a]) for a in modulos_repo) + len(miembros)
    enc = _TK.encode(src, add_special_tokens=False)
    starts = [a for a, _ in enc.offsets]
    conv = _char_offsets(src)
    tot = op = op_repo = 0
    origen = {"archivo": 0, "repo": 0, "paquete": 0, "modulo": 0, "miembro": 0}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            name, c = n.id, conv(n.lineno, n.col_offset)
            if name in BUILTINS:
                continue
            cand = name in nombres; cand_r = name in nombres_repo; org = nombres[name][0] if cand else None
        elif isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load) and getattr(n, "end_lineno", None):
            name = n.attr
            c = conv(n.end_lineno, n.end_col_offset) - len(name)
            if isinstance(n.value, ast.Name) and n.value.id in modulos:
                cand = name in modulos[n.value.id]; cand_r = cand and n.value.id in modulos_repo; org = "modulo"
            else:
                cand = name in miembros; cand_r = cand; org = "miembro"
        else:
            continue
        tot += 1
        if not cand:
            continue
        i = bisect.bisect_left(starts, c)
        ws = starts[max(0, i - VENT)] if starts else 0
        if re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", src[ws:c]):
            continue
        op += 1; op_repo += cand_r; origen[org] += 1
    return {"path": f["path"], "ocurrencias": tot, "oportunidad": op, "oportunidad_repo": op_repo,
            "E": E, "E_repo": E_repo, "origen": origen}


def _repo(r):
    repo = Repo(r["files"])
    out = []
    for f in r["files"]:
        d = archivo(repo, f)
        if d:
            d["repo"] = r["repo"]; out.append(d)
    return out


def analisis(out: Path, seed=1337):
    rows = [json.loads(l) for l in open(out / "detalle.jsonl", encoding="utf-8")]
    por_repo = {}
    for r in rows:
        a = por_repo.setdefault(r["repo"], [0, 0, 0]); a[0] += r["ocurrencias"]; a[1] += r["oportunidad"]; a[2] += r["oportunidad_repo"]
    keys = sorted(por_repo); M = np.array([por_repo[k] for k in keys], dtype=float)
    frac = M[:, 1].sum() / M[:, 0].sum(); frac_r = M[:, 2].sum() / M[:, 0].sum()
    rng = np.random.default_rng(seed); bs = []
    for _ in range(2000):
        s = M[rng.integers(0, len(M), len(M))]; bs.append(s[:, 1].sum() / max(1, s[:, 0].sum()))
    E = np.repeat([r["E"] for r in rows], [max(1, r["ocurrencias"]) for r in rows])
    origen = {}
    for r in rows:
        for k, v in r["origen"].items():
            origen[k] = origen.get(k, 0) + v
    res = {"repos": len(keys), "archivos": len(rows), "ocurrencias": int(M[:, 0].sum()),
           "oportunidad": round(float(frac), 4), "ic90": [round(float(np.percentile(bs, 5)), 4), round(float(np.percentile(bs, 95)), 4)],
           "oportunidad_solo_repo": round(float(frac_r), 4), "E_p50": float(np.percentile(E, 50)), "E_p95": float(np.percentile(E, 95)),
           "origen": origen,
           "veredicto": "INCONCLUSO" if len(keys) < 100 else ("MUERE" if (frac < 0.05 or np.percentile(E, 95) > 20000) else "SOBREVIVE")}
    json.dump(res, open(out / "resumen.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--repos", type=int, default=500); ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--procs", type=int, default=8); ap.add_argument("--solo-analisis", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if not a.solo_analisis:
        from .data import repos
        todos = [r for r in repos(Path(a.datos), "heldout") if len(r["files"]) >= 2]
        random.Random(a.seed).shuffle(todos)
        sel = todos[:a.repos]
        json.dump({"repos": [r["repo"] for r in sel], "seed": a.seed, "disponibles": len(todos)},
                  open(out / "muestra.json", "w"))
        with Pool(a.procs, initializer=_init, initargs=(a.datos,)) as p, open(out / "detalle.jsonl", "w", encoding="utf-8") as fo:
            for rows in p.imap_unordered(_repo, sel, chunksize=4):
                for r in rows:
                    fo.write(json.dumps(r) + "\n")
    analisis(out, a.seed)


if __name__ == "__main__":
    main()
