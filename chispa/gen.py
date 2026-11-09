"""Cursores de evaluación y generación FIM voraz por lotes (E2, E3, E4).

Cursor = inicio del contenido de una línea no vacía dentro de un cuerpo de función, en repositorios retenidos.
Prefijo = texto hasta el cursor (con la sangría); verdad = resto de la línea; sufijo = texto tras el salto de línea.

Uso:
  python -m chispa.gen cursores --datos DATOS --out RES/cursores.jsonl --n 20000 --seed 1337
  python -m chispa.gen completar --datos DATOS --modelo RUN/final.pt --cursores RES/cursores.jsonl --out RES/x/compl.jsonl
"""
from __future__ import annotations

import argparse, ast, json, random
from pathlib import Path

import torch
import torch.nn.functional as F

from .symbols import parse

PRE_TOK, SUF_TOK, MID_MAX = 760, 200, 48


def cursores(datos: Path, out: Path, n: int, seed: int):
    from .data import repos
    rng = random.Random(seed)
    todos = list(repos(datos, "heldout"))
    rng.shuffle(todos)
    res = []
    for r in todos:
        for f in r["files"]:
            src = f["content"]; tree = parse(src)
            if tree is None:
                continue
            lines = src.split("\n")
            cand = []
            for nd in ast.walk(tree):
                if isinstance(nd, (ast.FunctionDef, ast.AsyncFunctionDef)) and nd.body:
                    a = nd.body[0].lineno - 1; b = (nd.end_lineno or nd.lineno) - 1
                    for i in range(a, b + 1):
                        s = lines[i].strip() if i < len(lines) else ""
                        if len(s) >= 4 and not s.startswith(("#", '"""', "'''")):
                            cand.append(i)
            for i in rng.sample(cand, min(2, len(cand))):
                ind = len(lines[i]) - len(lines[i].lstrip())
                pre = "\n".join(lines[:i]) + ("\n" if i else "") + lines[i][:ind]
                res.append({"repo": r["repo"], "path": f["path"], "line": i, "prefix": pre,
                            "truth": lines[i][ind:], "suffix": "\n" + "\n".join(lines[i + 1:])})
            if len(res) >= n:
                break
        if len(res) >= n:
            break
    with open(out, "w", encoding="utf-8") as fo:
        for c in res[:n]:
            fo.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"cursores: {min(n, len(res))} de {len(todos)} repos", flush=True)


class Gen:
    def __init__(self, model, tk, dev):
        self.m, self.tk, self.dev = model.to(dev).eval(), tk, dev
        self.eot = tk.token_to_id("<|endoftext|>")
        self.fp, self.fs, self.fm = (tk.token_to_id(t) for t in ("<fim_prefix>", "<fim_suffix>", "<fim_middle>"))
        self.nl = {i for t, i in tk.get_vocab().items() if "Ċ" in t}  # piezas con salto de línea

    def ids(self, s):
        return self.tk.encode(s, add_special_tokens=False).ids

    def prompt_fim(self, prefix, suffix):
        p = self.ids(prefix)[-PRE_TOK:]; s = self.ids(suffix)[:SUF_TOK]
        return [self.fp] + p + [self.fs] + s + [self.fm]

    @torch.no_grad()
    def voraz(self, prompts, max_new=MID_MAX, bias=None):
        """Generación voraz por lotes con relleno a la derecha (válido por la máscara causal).
        bias: función (lista de prompts) -> tensor B×V aditivo sobre los logits, opcional (control de E4)."""
        B = len(prompts); L = max(len(p) for p in prompts) + max_new
        x = torch.full((B, L), self.eot, dtype=torch.long, device=self.dev)
        lens = torch.tensor([len(p) for p in prompts], device=self.dev)
        for i, p in enumerate(prompts):
            x[i, :len(p)] = torch.tensor(p, device=self.dev)
        out = [[] for _ in range(B)]; lps = [[] for _ in range(B)]; vivo = [True] * B
        b_add = bias(prompts) if bias else None
        for _ in range(max_new):
            T = int(lens.max())
            with torch.autocast(self.dev, dtype=torch.bfloat16, enabled=(self.dev == "cuda")):
                h = self.m(x[:, :T])
            hl = h[torch.arange(B, device=self.dev), lens - 1]
            lg = self.m.logits(hl).float()
            if b_add is not None:
                lg = lg + b_add
            lp = F.log_softmax(lg, -1); nx = lp.argmax(-1)
            for i in range(B):
                if not vivo[i]:
                    continue
                t = int(nx[i])
                if t == self.eot or t in self.nl or t in (self.fp, self.fs, self.fm):
                    vivo[i] = False; continue
                out[i].append(t); lps[i].append(float(lp[i, t]))
                x[i, lens[i]] = t; lens[i] += 1
            if not any(vivo):
                break
        return out, lps


def completar(datos, modelo, cur_path, out, lote=32, dev=None, bias_fn=None):
    from .tok import cargar
    from .model import cargar_modelo
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    tk = cargar(datos); m, _ = cargar_modelo(modelo, dev); g = Gen(m, tk, dev)
    curs = [json.loads(l) for l in open(cur_path, encoding="utf-8")]
    with open(out, "w", encoding="utf-8") as fo:
        for i in range(0, len(curs), lote):
            ch = curs[i:i + lote]
            prompts = [g.prompt_fim(c["prefix"], c["suffix"]) for c in ch]
            outs, lps = g.voraz(prompts, bias=(lambda P: bias_fn(g, P)) if bias_fn else None)
            for c, o, lp in zip(ch, outs, lps):
                fo.write(json.dumps({"repo": c["repo"], "path": c["path"], "line": c["line"],
                                     "compl": tk.decode(o), "lp_medio": (sum(lp) / len(lp)) if lp else None,
                                     "n": len(o)}, ensure_ascii=False) + "\n")
            if i % (lote * 20) == 0:
                print(f"completados {i + len(ch)}/{len(curs)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--datos", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000); ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--modelo"); ap.add_argument("--cursores"); ap.add_argument("--lote", type=int, default=32)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.cmd == "cursores":
        cursores(Path(a.datos), Path(a.out), a.n, a.seed)
    elif a.cmd == "completar":
        completar(a.datos, a.modelo, a.cursores, a.out, a.lote)


if __name__ == "__main__":
    main()
