"""E3 (y métricas de E4): nombres indefinidos introducidos por los completados. Ver PREREGISTRO.md.

Uso:
  python -m chispa.e3 clasificar --datos DATOS --cursores RES/cursores.jsonl --compl RES/X/compl.jsonl --out RES/X
  python -m chispa.e3 sesgo ...   (ver chispa.e4)
Salida cruda: RES/X/indef.jsonl (por completado); resumen: RES/X/resumen_e3.json
"""
from __future__ import annotations

import argparse, ast, json, re
from pathlib import Path

import numpy as np

from .symbols import Repo, parse

ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def indefinidos(src: str) -> set:
    """(nombre, línea) de los Undefined Name de pyflakes."""
    from pyflakes import checker, messages
    tree = parse(src)
    if tree is None:
        return None
    w = checker.Checker(tree, filename="x.py")
    return {(m.message_args[0], m.lineno) for m in w.messages if isinstance(m, messages.UndefinedName)}


def visibles(tree, linea: int) -> set:
    """Nombres visibles en la línea según el árbol del archivo original (aproximación por ámbito léxico)."""
    import builtins as _b
    vis = set(dir(_b))
    def ligados(nodes):
        out = set()
        for s in nodes:
            for n in ast.walk(s):
                if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                    out.add(n.id)
                elif isinstance(n, (ast.Import, ast.ImportFrom)):
                    for al in n.names:
                        out.add((al.asname or al.name).split(".")[0])
                elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    out.add(n.name)
                elif isinstance(n, ast.arg):
                    out.add(n.arg)
                elif isinstance(n, ast.ExceptHandler) and n.name:
                    out.add(n.name)
                elif isinstance(n, (ast.Global, ast.Nonlocal)):
                    out.update(n.names)
        return out
    # nivel de módulo: sentencias de primer nivel, sin entrar en cuerpos de función o clase
    for s in tree.body:
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            vis.add(s.name)
        else:
            vis |= ligados([s])
    vis |= {"__file__", "__name__", "__doc__", "__builtins__"}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and \
                n.lineno <= linea <= (n.end_lineno or n.lineno):
            vis |= ligados([n.args]) | ligados(n.body if isinstance(n.body, list) else [n.body])
        elif isinstance(n, ast.ClassDef) and n.lineno <= linea <= (n.end_lineno or n.lineno):
            for s in n.body:  # ligaduras directas del cuerpo de la clase (visibles en el propio cuerpo)
                if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    vis.add(s.name)
                elif not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    vis |= ligados([s])
    return vis


def usados_linea(texto: str) -> set:
    """Nombres leídos en una línea (tokens NAME que no son atributos, kwargs, palabras clave ni ligados en la línea)."""
    import io, keyword, tokenize
    toks = []
    try:
        for t in tokenize.generate_tokens(io.StringIO(texto + "\n").readline):
            if t.type in (tokenize.NAME, tokenize.OP):
                toks.append((t.type, t.string))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        toks = [(tokenize.NAME if ID.fullmatch(w) else tokenize.OP, w) for w in re.findall(r"[A-Za-z_]\w*|==|!=|<=|>=|\S", texto)]
    ligados = set()
    m = re.match(r"^\s*([A-Za-z_][\w\s,\[\]\(\)\*]*?)\s*(?<![=!<>])=(?!=)", texto)
    if m and not texto.lstrip().startswith(("if ", "elif ", "while ", "return ", "assert ")):
        ligados |= set(ID.findall(m.group(1)))
    for f in re.finditer(r"\bfor\s+([\w\s,\(\)]+?)\s+in\b", texto):
        ligados |= set(ID.findall(f.group(1)))
    for f in re.finditer(r"\blambda\s+([^:]*):", texto):
        ligados |= set(ID.findall(f.group(1)))
    for f in re.finditer(r"\bas\s+([A-Za-z_]\w*)", texto):
        ligados.add(f.group(1))
    out, prof = set(), 0
    for i, (ty, s) in enumerate(toks):
        if ty != tokenize.NAME:
            prof += s in "([{"; prof -= s in ")]}"
            continue
        if keyword.iskeyword(s) or keyword.issoftkeyword(s) or s in ligados:
            continue
        prev = toks[i - 1][1] if i else ""
        nxt = toks[i + 1][1] if i + 1 < len(toks) else ""
        if prev == "." or (prof > 0 and nxt == "=") or prev in ("def", "class"):
            continue
        out.add(s)
    return out


def indef_lexico(tree, linea: int, texto: str) -> set:
    return usados_linea(texto) - visibles(tree, linea)


def locales_hermanos(tree, linea: int) -> set:
    """Nombres ligados en funciones que NO contienen la línea del cursor."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if n.lineno <= linea <= (n.end_lineno or n.lineno):
                continue
            for a in n.args.args + n.args.kwonlyargs + n.args.posonlyargs:
                out.add(a.arg)
            for s in ast.walk(n):
                if isinstance(s, ast.Name) and isinstance(s.ctx, ast.Store):
                    out.add(s.id)
    return out


def clasificar(datos, cur_path, compl_path, out):
    from .data import repos
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    curs = [json.loads(l) for l in open(cur_path, encoding="utf-8")]
    comps = [json.loads(l) for l in open(compl_path, encoding="utf-8")]
    need = {c["repo"] for c in curs}
    idx = {r["repo"]: Repo(r["files"]) for r in repos(Path(datos), "heldout") if r["repo"] in need}
    with open(out / "indef.jsonl", "w", encoding="utf-8") as fo:
        for c, g in zip(curs, comps):
            assert (c["repo"], c["path"], c["line"]) == (g["repo"], g["path"], g["line"])
            ln = c["line"] + 1
            orig = c["prefix"] + c["truth"] + c["suffix"]
            new = c["prefix"] + g["compl"] + c["suffix"]
            rec = {"repo": c["repo"], "path": c["path"], "line": c["line"]}
            tid = ID.match(c["truth"]); gid = ID.match(g["compl"] or "")
            rec["em_id"] = bool(tid and gid and tid.group(0) == gid.group(0)) if tid else None
            rec["em_linea"] = g["compl"].strip() == c["truth"].strip()
            tree = parse(orig)
            if tree is None:
                rec["parse"] = False
            else:
                rec["parse"] = True
                nuevos = sorted(indef_lexico(tree, ln, g["compl"]) - indef_lexico(tree, ln, c["truth"]))
                u0, u1 = indefinidos(orig), indefinidos(new)  # validación cruzada donde parsea
                if u1 is not None:
                    rec["pyflakes_nuevos"] = sorted({n for n, l in u1 if l == ln} - {n for n, l in u0 if l == ln})
                repo = idx[c["repo"]]
                herm = locales_hermanos(tree, ln)
                otros = {k for k, (p, _) in repo.top_all.items() if p != c["path"]} | \
                        {k for k, (p, _) in repo.mem_all.items() if p != c["path"]}
                rec["nuevos"] = [{"nombre": n, "clase": "local_hermano" if n in herm else
                                  ("otro_archivo" if n in otros else "invencion")} for n in nuevos]
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return analisis(out)


def analisis(out, seed=1337):
    out = Path(out)
    rows = [json.loads(l) for l in open(out / "indef.jsonl", encoding="utf-8")]
    ok = [r for r in rows if r.get("parse")]
    por_repo = {}
    for r in ok:
        a = por_repo.setdefault(r["repo"], [0, 0, 0, 0])
        a[0] += 1; a[1] += bool(r["nuevos"])
        a[2] += sum(1 for n in r["nuevos"] if n["clase"] == "invencion"); a[3] += len(r["nuevos"])
    M = np.array(list(por_repo.values()), dtype=float)
    rng = np.random.default_rng(seed)
    def boot(f):
        v = [f(M[rng.integers(0, len(M), len(M))]) for _ in range(2000)]
        return [round(float(np.percentile(v, 5)), 4), round(float(np.percentile(v, 95)), 4)]
    tasa = M[:, 1].sum() / M[:, 0].sum()
    frac_inv = M[:, 2].sum() / max(1, M[:, 3].sum())
    clases = {}
    for r in ok:
        for n in r["nuevos"]:
            clases[n["clase"]] = clases.get(n["clase"], 0) + 1
    em = [r["em_id"] for r in rows if r["em_id"] is not None]
    res = {"completados": len(rows), "analizables": len(ok), "repos": len(M),
           "tasa_undefined": round(float(tasa), 4), "tasa_ic90": boot(lambda s: s[:, 1].sum() / max(1, s[:, 0].sum())),
           "nuevos_total": int(M[:, 3].sum()), "clases": clases,
           "frac_invencion": round(float(frac_inv), 4), "frac_invencion_ic90": boot(lambda s: s[:, 2].sum() / max(1, s[:, 3].sum())),
           "em_primer_id": round(float(np.mean(em)), 4) if em else None,
           "em_linea": round(float(np.mean([r["em_linea"] for r in rows])), 4),
           "veredicto_E3": "INCONCLUSO" if M[:, 3].sum() < 30 else ("MUERE" if frac_inv < 0.30 else "SOBREVIVE")}
    json.dump(res, open(out / "resumen_e3.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--datos"); ap.add_argument("--cursores"); ap.add_argument("--compl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "clasificar":
        clasificar(a.datos, a.cursores, a.compl, a.out)
    elif a.cmd == "analisis":
        analisis(a.out)


if __name__ == "__main__":
    main()
