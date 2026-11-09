"""E2 — Prueba del sufijo offline. Ver PREREGISTRO.md.

Uso: python -m chispa.e2 --datos DATOS --modelo RUN/final.pt --cursores RES/cursores.jsonl --compl RES/base/compl.jsonl --out RES/e2
Salida cruda: RES/e2/delta.jsonl (por cursor), RES/e2/latencia.jsonl; resumen: RES/e2/resumen.json
"""
from __future__ import annotations

import argparse, json, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

S_TOK, P_TOK = 16, 760


def lev_sim(a: str, b: str) -> float:
    a, b = a.strip(), b.strip()
    if not a and not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1 - prev[-1] / max(len(a), len(b))


@torch.no_grad()
def logp_tramos(m, seqs, tramos, dev):
    """Suma de log p de los tokens seqs[i][a:b] (tramos[i]=(a,b)), por lotes con relleno a la derecha."""
    B = len(seqs); L = max(len(s) for s in seqs)
    x = torch.zeros((B, L), dtype=torch.long, device=dev)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s, device=dev)
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
        h = m(x)
    out = []
    for i, (a, b) in enumerate(tramos):
        lg = m.logits(h[i, a - 1:b - 1]).float()
        out.append(float(F.log_softmax(lg, -1).gather(1, x[i, a:b, None]).sum()))
    return out


def deltas(m, tk, curs, comps, dev, lote=32, otro=None):
    ids = lambda s: tk.encode(s, add_special_tokens=False).ids
    res = []
    for i in range(0, len(curs), lote):
        sq1, tr1, sq2, tr2 = [], [], [], []
        for k in range(i, min(i + lote, len(curs))):
            c, g = curs[k], comps[k]
            P = ids(c["prefix"])[-P_TOK:]
            suf = (otro[k] if otro else c)["suffix"]
            S = ids(suf)[:S_TOK]; M = ids(g["compl"]) if g["compl"] else []
            sq1.append(P + M + S); tr1.append((len(P) + len(M), len(P) + len(M) + len(S)))
            sq2.append(P + S); tr2.append((len(P), len(P) + len(S)))
        a = logp_tramos(m, sq1, tr1, dev); b = logp_tramos(m, sq2, tr2, dev)
        res += [x - y for x, y in zip(a, b)]
    return res


def logistico(X, y, iters=50, l2=1e-3):
    X = np.c_[np.ones(len(X)), X]; w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w)); W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + l2 * np.eye(X.shape[1]); g = X.T @ (p - y) + l2 * w
        w -= np.linalg.solve(H, g)
    return w


def auroc(s, y):
    s, y = np.asarray(s), np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def oof(F_, y, grupos, k=5, seed=1337):
    """Predicciones fuera de partición con validación cruzada agrupada por repositorio."""
    gs = sorted(set(grupos)); random.Random(seed).shuffle(gs)
    fold = {g: i % k for i, g in enumerate(gs)}; f = np.array([fold[g] for g in grupos])
    pred = np.zeros(len(y))
    for j in range(k):
        tr, te = f != j, f == j
        mu, sd = F_[tr].mean(0), F_[tr].std(0) + 1e-9
        w = logistico((F_[tr] - mu) / sd, y[tr])
        pred[te] = np.c_[np.ones(te.sum()), (F_[te] - mu) / sd] @ w
    return pred


@torch.no_grad()
def latencia(m_cpu, tk, curs, comps, n=500, hilos=(1, 4)):
    ids = lambda s: tk.encode(s, add_special_tokens=False).ids
    out = []
    for th in hilos:
        torch.set_num_threads(th)
        for c, g in list(zip(curs, comps))[:n]:
            P = torch.tensor([ids(c["prefix"])[-P_TOK:]]); S = ids(c["suffix"])[:S_TOK]; M = ids(g["compl"] or " ")
            caches = [[None, None] for _ in m_cpu.blocks]
            m_cpu.hidden_kv(P, caches, 0)                       # el prefijo ya está cacheado al generar
            base = [[k.clone(), v.clone()] for k, v in caches]
            t0 = time.perf_counter()
            c1 = [[k, v] for k, v in base]; m_cpu.hidden_kv(torch.tensor([M + S]), c1, P.shape[1])
            c2 = [[k, v] for k, v in base]; m_cpu.hidden_kv(torch.tensor([S]), c2, P.shape[1])
            out.append({"hilos": th, "ms": (time.perf_counter() - t0) * 1000, "m": len(M)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True); ap.add_argument("--modelo", required=True)
    ap.add_argument("--cursores", required=True); ap.add_argument("--compl", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1337); ap.add_argument("--n-lat", type=int, default=500)
    a = ap.parse_args()
    from .tok import cargar
    from .model import cargar_modelo
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tk = cargar(a.datos); m, _ = cargar_modelo(a.modelo, dev); m.eval()
    curs = [json.loads(l) for l in open(a.cursores, encoding="utf-8")]
    comps = [json.loads(l) for l in open(a.compl, encoding="utf-8")]
    perm = list(range(len(curs))); rng = random.Random(a.seed)
    while True:  # sufijo de otro archivo (control)
        rng.shuffle(perm)
        if all(curs[i]["path"] != curs[p]["path"] or curs[i]["repo"] != curs[p]["repo"] for i, p in enumerate(perm)):
            break
    d = deltas(m, tk, curs, comps, dev); d_ctl = deltas(m, tk, curs, comps, dev, otro=[curs[p] for p in perm])
    with open(out / "delta.jsonl", "w", encoding="utf-8") as fo:
        for c, g, x, xc in zip(curs, comps, d, d_ctl):
            fo.write(json.dumps({"repo": c["repo"], "delta": x, "delta_ctl": xc, "lp_medio": g["lp_medio"], "n": g["n"],
                                 "sim": lev_sim(g["compl"], c["truth"])}) + "\n")
    if a.n_lat > 0:
        medir_latencia(a.modelo, tk, curs, comps, out, a.n_lat)
    analisis(out, a.seed)


def medir_latencia(modelo, tk, curs, comps, out, n):
    from .model import cargar_modelo
    m_cpu, _ = cargar_modelo(modelo, "cpu"); m_cpu.eval()
    m_cpu = torch.ao.quantization.quantize_dynamic(m_cpu, {torch.nn.Linear}, dtype=torch.qint8)
    lat = latencia(m_cpu, tk, curs, comps, n)
    with open(Path(out) / "latencia.jsonl", "w") as fo:
        for r in lat:
            fo.write(json.dumps(r) + "\n")


def analisis(out, seed=1337):
    out = Path(out)
    rows = [json.loads(l) for l in open(out / "delta.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["lp_medio"] is not None]
    y = np.array([r["sim"] >= 0.8 for r in rows], dtype=float)
    g = [r["repo"] for r in rows]
    base = np.array([[r["lp_medio"], r["n"]] for r in rows]); dl = np.array([[r["delta"]] for r in rows]); dc = np.array([[r["delta_ctl"]] for r in rows])
    p0 = oof(base, y, g, seed=seed); p1 = oof(np.c_[dl, base], y, g, seed=seed); pc = oof(np.c_[dc, base], y, g, seed=seed)
    a0, a1, ac = auroc(p0, y), auroc(p1, y), auroc(pc, y)
    reps = sorted(set(g)); idx = {r: [i for i, x in enumerate(g) if x == r] for r in reps}
    rng = np.random.default_rng(seed); diffs = []
    for _ in range(1000):
        sel = np.concatenate([idx[r] for r in rng.choice(reps, len(reps))])
        diffs.append(auroc(p1[sel], y[sel]) - auroc(p0[sel], y[sel]))
    lf = out / "latencia.jsonl"
    lat = [json.loads(l) for l in open(lf)] if lf.exists() else []
    p95 = {th: float(np.percentile([r["ms"] for r in lat if r["hilos"] == th], 95)) for th in sorted({r["hilos"] for r in lat})}
    mejora = a1 - a0
    res = {"cursores": len(rows), "aceptables": int(y.sum()), "auroc_base": round(a0, 4), "auroc_delta": round(a1, 4),
           "auroc_control_otro_sufijo": round(ac, 4), "mejora": round(mejora, 4),
           "mejora_ic90": [round(float(np.nanpercentile(diffs, 5)), 4), round(float(np.nanpercentile(diffs, 95)), 4)],
           "latencia_extra_p95_ms": p95,
           "veredicto": "INCONCLUSO" if (np.isnan(mejora) or y.sum() < 50 or (1 - y).sum() < 50) else
                        ("MUERE" if mejora < 0.02 else ("PENDIENTE_LATENCIA" if not p95 else
                        ("MUERE" if min(p95.values()) > 50 else "SOBREVIVE")))}
    json.dump(res, open(out / "resumen.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
