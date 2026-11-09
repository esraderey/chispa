"""E3 — análisis de sensibilidad (exploratorio, posterior a ver el resultado preregistrado; no cambia el veredicto).

(a) detector léxico excluyendo cursores dentro de cadenas de varias líneas (docstrings);
(b) solo pyflakes, en los completados donde el archivo resultante parsea, con la misma clasificación.
Uso: python -m chispa.e3_sensibilidad --datos D --res RES/base --cursores RES/cursores.jsonl
"""
from __future__ import annotations

import argparse, io, json, tokenize
from pathlib import Path

from .data import repos
from .e3 import locales_hermanos
from .symbols import Repo, parse


def en_cadena(src, ln):
    try:
        for t in tokenize.generate_tokens(io.StringIO(src).readline):
            if t.type == tokenize.STRING and t.start[0] < ln <= t.end[0]:
                return True
            if t.start[0] > ln:
                break
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return False
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True); ap.add_argument("--res", required=True); ap.add_argument("--cursores", required=True)
    a = ap.parse_args()
    cur = [json.loads(l) for l in open(a.cursores, encoding="utf-8")]
    rows = [json.loads(l) for l in open(Path(a.res) / "indef.jsonl", encoding="utf-8")]
    dentro = [en_cadena(c["prefix"] + c["truth"] + c["suffix"], c["line"] + 1) for c in cur]
    sel = [r for r, d in zip(rows, dentro) if not d and r.get("parse")]
    inv = sum(1 for r in sel for x in r["nuevos"] if x["clase"] == "invencion"); tot = sum(len(r["nuevos"]) for r in sel)
    res = {"cursores_en_cadenas": sum(dentro),
           "a_sin_cadenas": {"completados": len(sel), "tasa": round(sum(1 for r in sel if r["nuevos"]) / len(sel), 4),
                             "invenciones": inv, "nuevos": tot, "frac_invencion": round(inv / max(1, tot), 4)}}
    need = {c["repo"] for c in cur}
    idx = {r["repo"]: Repo(r["files"]) for r in repos(Path(a.datos), "heldout") if r["repo"] in need}
    cls = {"invencion": 0, "local_hermano": 0, "otro_archivo": 0}; n = con = 0
    for c, r in zip(cur, rows):
        if "pyflakes_nuevos" not in r:
            continue
        n += 1; con += bool(r["pyflakes_nuevos"])
        tree = parse(c["prefix"] + c["truth"] + c["suffix"]); herm = locales_hermanos(tree, c["line"] + 1) if tree else set()
        rp = idx[c["repo"]]
        otros = {k for k, (p, _) in rp.top_all.items() if p != c["path"]} | {k for k, (p, _) in rp.mem_all.items() if p != c["path"]}
        for nm in r["pyflakes_nuevos"]:
            cls["local_hermano" if nm in herm else ("otro_archivo" if nm in otros else "invencion")] += 1
    t = sum(cls.values())
    res["b_pyflakes"] = {"completados_que_parsean": n, "tasa": round(con / max(1, n), 4), "clases": cls,
                         "frac_invencion": round(cls["invencion"] / max(1, t), 4)}
    json.dump(res, open(Path(a.res) / "sensibilidad.json", "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
