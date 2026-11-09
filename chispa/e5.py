"""E5 — Evaluación de la memoria de definiciones. Ver PREREGISTRO.md.

  python -m chispa.e5 objetivos --datos D --out RES/e5/objetivos.jsonl [--n 20000 --seed 5]
  python -m chispa.e5 evaluar   --datos D --modelo RUN/final.pt --brazo A|B|C --objetivos RES/e5/objetivos.jsonl --out RES/e5/A_s1
  python -m chispa.e5 latencia  --datos D --modelo RUN/final.pt --brazo A|B|C --objetivos ... --out ...   (en el portátil)
  python -m chispa.e5 comparar  --A RES/e5/A_s1 RES/e5/A_s2 --B RES/e5/B_s1 RES/e5/B_s2 --C RES/e5/C_s1 --out RES/e5
Salida cruda: pred.jsonl (por objetivo); resumen: resumen.json.
"""
from __future__ import annotations

import argparse, json, random, re, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .memdata import W, PREF, HL, trigramas, _entradas, _init
from . import memdata

CONT = re.compile(r"^[A-Za-z0-9_]+$")


def objetivos(datos, out, n, seed):
    """Ventanas de repositorios retenidos con sus objetivos, el prefijo del brazo A y TODO E del archivo."""
    from .data import repos
    from .symbols import Repo, activos
    _init(datos)
    rs = [r for r in repos(Path(datos), "heldout") if len(r["files"]) >= 2]
    random.Random(seed).shuffle(rs)
    total = 0
    with open(out, "w", encoding="utf-8") as fo:
        for r in rs:
            repo = Repo(r["files"]); src = {f["path"]: f["content"] for f in r["files"]}
            for main, pref, hdr, tri, pos, sym, nw, names, nE, path in memdata._repo((r, seed)):
                E = _entradas(*activos(repo, path, src[path]))
                fo.write(json.dumps({"repo": r["repo"], "path": path, "main": main[:nw].tolist(), "pref": pref.tolist(),
                                     "pos": [int(p) for p in pos if p >= 0], "nombres": names,
                                     "E": {k: str(v) for k, v in E.items()}}, ensure_ascii=False) + "\n")
                total += len(names)
            if total >= n:
                break
    print(f"objetivos: {total}", flush=True)


def _decodifica(m, tk, prompts, dev, max_steps=6):
    """Continúa cada prompt con piezas de identificador hasta que deje de serlo; devuelve el texto emitido."""
    outs = [""] * len(prompts); vivos = list(range(len(prompts))); seqs = [list(p) for p in prompts]
    for step in range(max_steps):
        if not vivos:
            break
        L = max(len(seqs[i]) for i in vivos)
        x = torch.zeros((len(vivos), L), dtype=torch.long, device=dev)
        for j, i in enumerate(vivos):
            x[j, :len(seqs[i])] = torch.tensor(seqs[i], device=dev)
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            h = m(x)
        nxt = m.logits(h[torch.arange(len(vivos)), torch.tensor([len(seqs[i]) - 1 for i in vivos])]).float().argmax(-1)
        nuevos = []
        for j, i in enumerate(vivos):
            piece = tk.decode([int(nxt[j])])
            txt = piece.lstrip(" ") if step == 0 else piece
            if (step == 0 and re.match(r"^ ?[A-Za-z_][A-Za-z0-9_]*$", piece)) or (step > 0 and CONT.match(piece)):
                outs[i] += txt; seqs[i].append(int(nxt[j])); nuevos.append(i)
        vivos = nuevos
    return outs


@torch.no_grad()
def evaluar(datos, modelo, brazo, obj_path, out, dev=None, lote=48):
    from .tok import cargar
    from .model import cargar_modelo
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    tk = cargar(datos); m, _ = cargar_modelo(modelo, dev); m.eval()
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(obj_path, encoding="utf-8")]
    pad = tk.token_to_id("<pad>")
    with open(out / "pred.jsonl", "w", encoding="utf-8") as fo:
        for r in rows:
            pref = list(r["pref"]) if brazo == "A" else []  # 256 posiciones con relleno, como en el entrenamiento
            prompts = [pref + r["main"][:p] for p in r["pos"]]
            names = r["nombres"]
            mem_pred = [None] * len(prompts)
            if brazo in ("B", "C") and r["E"]:
                ent = list(r["E"].keys())
                if brazo == "B":
                    hdr = [tk.encode(r["E"][k], add_special_tokens=False).ids[:HL] for k in ent]
                    H = torch.full((len(ent), HL), pad, dtype=torch.long, device=dev)
                    for i, h in enumerate(hdr):
                        H[i, :len(h)] = torch.tensor(h, device=dev)
                    keys = torch.cat([m.claves_estado(H[i:i + 256], H[i:i + 256] != pad) for i in range(0, len(ent), 256)])
                else:
                    tri = [trigramas(k) for k in ent]
                    flat = torch.tensor([t for tr in tri for t in tr], device=dev)
                    off = torch.tensor(np.r_[0, np.cumsum([len(t) for t in tri])[:-1]], device=dev)
                    keys = m.claves_chars(flat, off)
                for i in range(0, len(prompts), lote):
                    ps = prompts[i:i + lote]; L = max(len(p) for p in ps)
                    x = torch.zeros((len(ps), L), dtype=torch.long, device=dev)
                    for j, p in enumerate(ps):
                        x[j, :len(p)] = torch.tensor(p, device=dev)
                    h = m(x)[torch.arange(len(ps)), torch.tensor([len(p) - 1 for p in ps])].float()
                    lv = m.logits(h); lm = m.consultas(h) @ keys.float().t()
                    best = torch.cat([lv, lm], 1).argmax(-1)
                    for j, b in enumerate(best.tolist()):
                        if b >= lv.shape[1]:
                            mem_pred[i + j] = ent[b - lv.shape[1]]
            pend = [i for i, mp in enumerate(mem_pred) if mp is None]
            dec = {}
            for i in range(0, len(pend), lote):
                ch = pend[i:i + lote]
                for k, o in zip(ch, _decodifica(m, tk, [prompts[q] for q in ch], dev)):
                    dec[k] = o
            for k, nm in enumerate(names):
                pred = mem_pred[k] if mem_pred[k] is not None else dec.get(k, "")
                fo.write(json.dumps({"repo": r["repo"], "nombre": nm, "pred": pred, "via": "memoria" if mem_pred[k] else "bpe",
                                     "acierto": pred == nm}) + "\n")
    return analisis([out])


@torch.no_grad()
def latencia(datos, modelo, brazo, obj_path, out, n=200):
    from .tok import cargar
    from .model import cargar_modelo
    tk = cargar(datos); m, _ = cargar_modelo(modelo, "cpu"); m.eval()
    m = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
    pad = tk.token_to_id("<pad>"); rows = [json.loads(l) for l in open(obj_path, encoding="utf-8")][:n]
    res = []
    for th in (1, 4):
        torch.set_num_threads(th)
        for r in rows:
            pref = list(r["pref"]) if brazo == "A" else []  # 256 posiciones con relleno, como en el entrenamiento
            x = torch.tensor([pref + r["main"]])
            K = torch.randn(max(1, len(r["E"])), m.c.d) if brazo != "A" else None
            t0 = time.perf_counter()
            h = m(x)
            if K is not None:
                _ = m.consultas(h[:, -1]) @ K.t()
            res.append({"hilos": th, "ms": (time.perf_counter() - t0) * 1000, "E": len(r["E"])})
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(Path(out) / "latencia.jsonl", "w") as fo:
        for x in res:
            fo.write(json.dumps(x) + "\n")
    p = {th: float(np.percentile([x["ms"] for x in res if x["hilos"] == th], 50)) for th in (1, 4)}
    print(brazo, "latencia p50 ms:", p, flush=True)


def analisis(dirs, seed=1337):
    acc = {}
    for d in dirs:
        for r in map(json.loads, open(Path(d) / "pred.jsonl", encoding="utf-8")):
            a = acc.setdefault(r["repo"], [0, 0]); a[0] += 1; a[1] += r["acierto"]
    M = np.array(list(acc.values()), float)
    em = M[:, 1].sum() / M[:, 0].sum()
    rng = np.random.default_rng(seed)
    bs = [(lambda s: s[:, 1].sum() / s[:, 0].sum())(M[rng.integers(0, len(M), len(M))]) for _ in range(2000)]
    res = {"objetivos": int(M[:, 0].sum()), "repos": len(M), "em": round(float(em), 4),
           "em_ic90": [round(float(np.percentile(bs, 5)), 4), round(float(np.percentile(bs, 95)), 4)]}
    json.dump(res, open(Path(dirs[0]) / "resumen.json", "w"), indent=1)
    print(json.dumps(res)); return res


def comparar(A, B, C, out):
    ra, rb, rc = analisis(A), analisis(B), analisis(C)
    lat = {}
    for k, ds in (("A", A), ("B", B), ("C", C)):
        f = Path(ds[0]) / "latencia.jsonl"
        if f.exists():
            xs = [json.loads(l) for l in open(f)]
            lat[k] = float(np.percentile([x["ms"] for x in xs if x["hilos"] == 4], 50))
    dif_ba, dif_bc = rb["em"] - ra["em"], rb["em"] - rc["em"]
    lat_ok = (lat.get("B", 0) <= lat.get("A", float("inf"))) if lat else None
    res = {"em_A": ra["em"], "em_B": rb["em"], "em_C": rc["em"], "B_menos_A": round(dif_ba, 4), "B_menos_C": round(dif_bc, 4),
           "latencia_p50_ms_4hilos": lat,
           "veredicto_P1": "MUERE" if (dif_ba < 0.03 or lat_ok is False) else ("PENDIENTE_LATENCIA" if lat_ok is None else "SOBREVIVE"),
           "semantica": "SOBRA" if dif_bc < 0.01 else "IMPORTA"}
    json.dump(res, open(Path(out) / "resumen.json", "w"), indent=1); print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--datos"); ap.add_argument("--modelo"); ap.add_argument("--brazo")
    ap.add_argument("--objetivos"); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000); ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--A", nargs="+"); ap.add_argument("--B", nargs="+"); ap.add_argument("--C", nargs="+")
    a = ap.parse_args()
    if a.cmd == "objetivos":
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); objetivos(a.datos, a.out, a.n, a.seed)
    elif a.cmd == "evaluar":
        evaluar(a.datos, a.modelo, a.brazo, a.objetivos, a.out)
    elif a.cmd == "latencia":
        latencia(a.datos, a.modelo, a.brazo, a.objetivos, a.out)
    elif a.cmd == "comparar":
        comparar(a.A, a.B, a.C, a.out)


if __name__ == "__main__":
    main()
