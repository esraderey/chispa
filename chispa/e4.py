"""E4 — Monitor amortizado: control de sesgo −β y comparación entre brazos. Ver PREREGISTRO.md.

  python -m chispa.e4 dev       --datos D --cursores RES/cursores.jsonl --out RES/cursores_dev.jsonl   # 2,000 disjuntos
  python -m chispa.e4 beta      --datos D --modelo MLE/final.pt --dev RES/cursores_dev.jsonl --out RES/e4/beta
  python -m chispa.e4 sesgado   --datos D --modelo MLE/final.pt --cursores RES/cursores.jsonl --beta B --out RES/e4/sesgo
  python -m chispa.e4 comparar  --mle RES/e4/mle_s1 RES/e4/mle_s2 --ul RES/e4/ul_s1 RES/e4/ul_s2 --sesgo RES/e4/sesgo --out RES/e4
"""
from __future__ import annotations

import argparse, builtins, json, random, re
from pathlib import Path

import numpy as np
import torch

BETAS = (1.0, 2.0, 4.0)


def hacer_sesgo(beta):
    cache = {}
    def fn(g, prompts):
        if "first" not in cache:
            V = g.tk.get_vocab_size(); first = torch.zeros(V, dtype=torch.bool)
            for t, i in g.tk.get_vocab().items():
                if re.match(r"^Ġ?[A-Za-z_]", t):
                    first[i] = True
            bi = set()
            for b in dir(builtins):
                bi |= set(g.ids(b)[:1]) | set(g.ids(" " + b)[:1])
            cache["first"] = first.to(g.dev); cache["bi"] = torch.tensor(sorted(bi), device=g.dev)
        first = cache["first"]
        out = torch.zeros(len(prompts), first.numel(), device=g.dev)
        for i, p in enumerate(prompts):
            ok = torch.zeros_like(first); ok[torch.tensor(p, device=g.dev)] = True; ok[cache["bi"]] = True
            out[i][first & ~ok] = -beta
        return out
    return fn


def dev(datos, cur_main, out, n=2000, seed=7):
    from .gen import cursores
    usados = {(c["repo"], c["path"], c["line"]) for c in map(json.loads, open(cur_main, encoding="utf-8"))}
    tmp = Path(out).with_suffix(".tmp.jsonl")
    cursores(Path(datos), tmp, n * 4, seed)
    sel = [c for c in map(json.loads, open(tmp, encoding="utf-8")) if (c["repo"], c["path"], c["line"]) not in usados][:n]
    with open(out, "w", encoding="utf-8") as fo:
        for c in sel:
            fo.write(json.dumps(c, ensure_ascii=False) + "\n")
    tmp.unlink()
    print(f"dev: {len(sel)} cursores disjuntos", flush=True)


def correr(datos, modelo, cur, out, beta):
    from .gen import completar
    from .e3 import clasificar
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    completar(datos, modelo, cur, out / "compl.jsonl", bias_fn=hacer_sesgo(beta) if beta else None)
    return clasificar(datos, cur, out / "compl.jsonl", out)


def beta(datos, modelo, cur_dev, out):
    out = Path(out); res = {0.0: correr(datos, modelo, cur_dev, out / "b0", 0.0)}
    for b in BETAS:
        res[b] = correr(datos, modelo, cur_dev, out / f"b{b:g}", b)
    base = res[0.0]
    validos = [b for b in BETAS if base["em_primer_id"] - res[b]["em_primer_id"] <= 0.01]
    elegido = min(validos, key=lambda b: res[b]["tasa_undefined"]) if validos else min(BETAS)
    json.dump({"elegido": elegido, "dev": {str(k): {"tasa": v["tasa_undefined"], "em": v["em_primer_id"]} for k, v in res.items()}},
              open(out / "beta.json", "w"), indent=1)
    print("beta elegido:", elegido, flush=True)


def _por_repo(dirs):
    acc = {}
    for d in dirs:
        for r in map(json.loads, open(Path(d) / "indef.jsonl", encoding="utf-8")):
            if not r.get("parse"):
                continue
            a = acc.setdefault(r["repo"], [0, 0, 0, 0])
            a[0] += 1; a[1] += bool(r["nuevos"])
            if r["em_id"] is not None:
                a[2] += 1; a[3] += r["em_id"]
    return acc


def comparar(mle, ul, sesgo, out, seed=1337):
    A, B, S = _por_repo(mle), _por_repo(ul), _por_repo([sesgo])
    reps = sorted(set(A) & set(B) & set(S))
    M = np.array([A[r] for r in reps], float); U = np.array([B[r] for r in reps], float); Z = np.array([S[r] for r in reps], float)
    tasa = lambda X: X[:, 1].sum() / X[:, 0].sum()
    em = lambda X: X[:, 3].sum() / max(1, X[:, 2].sum())
    red = lambda m, u: 1 - tasa(u) / tasa(m)
    rng = np.random.default_rng(seed); bs = []
    for _ in range(2000):
        i = rng.integers(0, len(reps), len(reps)); bs.append(red(M[i], U[i]))
    r_ul, r_sesgo = red(M, U), red(M, Z)
    ic = [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]
    caida_em = em(M) - em(U)
    muere = r_ul < 0.25 or ic[0] <= 0 or caida_em > 0.01 or (r_ul > 0 and r_sesgo >= 0.8 * r_ul)
    res = {"repos": len(reps), "tasa_mle": round(tasa(M), 4), "tasa_ul": round(tasa(U), 4), "tasa_sesgo": round(tasa(Z), 4),
           "reduccion_ul": round(r_ul, 4), "reduccion_ul_ic90": [round(x, 4) for x in ic], "reduccion_sesgo": round(r_sesgo, 4),
           "em_mle": round(em(M), 4), "em_ul": round(em(U), 4), "caida_em": round(caida_em, 4),
           "veredicto": "MUERE" if muere else "SOBREVIVE"}
    json.dump(res, open(Path(out) / "resumen.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--datos"); ap.add_argument("--modelo"); ap.add_argument("--cursores")
    ap.add_argument("--dev"); ap.add_argument("--out", required=True); ap.add_argument("--beta", type=float)
    ap.add_argument("--mle", nargs="+"); ap.add_argument("--ul", nargs="+"); ap.add_argument("--sesgo")
    a = ap.parse_args()
    if a.cmd == "dev":
        dev(a.datos, a.cursores, a.out)
    elif a.cmd == "beta":
        beta(a.datos, a.modelo, a.dev, a.out)
    elif a.cmd == "sesgado":
        correr(a.datos, a.modelo, a.cursores, a.out, a.beta)
    elif a.cmd == "comparar":
        comparar(a.mle, a.ul, a.sesgo, a.out)


if __name__ == "__main__":
    main()
