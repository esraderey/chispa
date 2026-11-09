"""Índice de símbolos estático (sin ejecutar código): definiciones del repositorio con cabecera, importaciones con alias y
exportaciones de módulos de la biblioteca estándar o de paquetes instalados (leídos como fuente, nunca importados)."""
from __future__ import annotations

import ast, builtins, keyword, os, sys, sysconfig
from functools import lru_cache
from pathlib import Path

BUILTINS = set(dir(builtins)) | set(keyword.kwlist) | {"self", "cls"}
STDLIB = set(getattr(sys, "stdlib_module_names", ()))


def parse(src: str):
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ast.parse(src)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


def cabecera(src_lines, node, maxc=240) -> str:
    """Firma (líneas de la definición hasta ':') + primera línea del docstring."""
    a = node.lineno - 1
    b = getattr(node.body[0], "lineno", node.lineno) - 1 if getattr(node, "body", None) else a
    sig = " ".join(l.strip() for l in src_lines[a:max(a + 1, b)])[:maxc]
    doc = ast.get_docstring(node) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else None
    if doc and doc.strip():
        sig += "  # " + doc.strip().splitlines()[0][:120]
    return sig


def definiciones(src: str) -> tuple[dict, dict]:
    """Devuelve (nivel_superior: nombre -> cabecera, miembros: nombre -> cabecera) de un archivo."""
    tree = parse(src)
    top, mem = {}, {}
    if tree is None:
        return top, mem
    lines = src.splitlines()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top[n.name] = cabecera(lines, n)
            if isinstance(n, ast.ClassDef):
                for m in n.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        mem.setdefault(m.name, f"{n.name}.{cabecera(lines, m)}")
                    elif isinstance(m, ast.Assign):
                        for t in m.targets:
                            if isinstance(t, ast.Name):
                                mem.setdefault(t.id, f"{n.name}.{t.id} = ...")
                for s in ast.walk(n):
                    if isinstance(s, ast.Attribute) and isinstance(s.ctx, ast.Store) and \
                            isinstance(s.value, ast.Name) and s.value.id == "self":
                        mem.setdefault(s.attr, f"{n.name}.self.{s.attr}")
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            ts = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in ts:
                if isinstance(t, ast.Name):
                    top[t.id] = lines[n.lineno - 1].strip()[:200] if n.lineno - 1 < len(lines) else t.id
    return top, mem


def modulo_de_ruta(path: str) -> str:
    p = path[:-3] if path.endswith(".py") else path
    parts = [x for x in p.replace("\\", "/").split("/") if x]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


class Repo:
    """Índice de un repositorio: módulos (por todos sus sufijos), definiciones y miembros."""

    def __init__(self, files):
        self.files = files
        self.defs = {}     # ruta -> (top, mem)
        self.mods = {}     # sufijo de módulo -> ruta
        for f in files:
            self.defs[f["path"]] = definiciones(f["content"])
            parts = modulo_de_ruta(f["path"]).split(".")
            for i in range(len(parts)):
                self.mods.setdefault(".".join(parts[i:]), f["path"])
        self.top_all, self.mem_all = {}, {}
        for p, (t, m) in self.defs.items():
            for k, v in t.items():
                self.top_all.setdefault(k, (p, v))
            for k, v in m.items():
                self.mem_all.setdefault(k, (p, v))

    def resolver(self, mod: str):
        return self.mods.get(mod)


@lru_cache(maxsize=None)
def raices() -> tuple:
    rs = [sysconfig.get_paths()["stdlib"]]
    extra = os.environ.get("CHISPA_SITEPKGS")
    if extra:
        rs.append(extra)
    return tuple(rs)


@lru_cache(maxsize=20000)
def exportaciones_paquete(mod: str, prof: int = 0) -> tuple:
    """Exportaciones estáticas de un módulo instalado: __all__ literal o nombres públicos de nivel superior,
    más re-exportaciones de un nivel (from .x import y / *). Devuelve tupla de (nombre, cabecera)."""
    rel = mod.replace(".", "/")
    for r in raices():
        for cand in (Path(r) / f"{rel}.py", Path(r) / rel / "__init__.py"):
            if cand.is_file():
                try:
                    src = cand.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                tree = parse(src)
                if tree is None:
                    return ()
                top, _ = definiciones(src)
                out = {k: v for k, v in top.items() if not k.startswith("_")}
                for n in tree.body:
                    if isinstance(n, ast.ImportFrom) and prof < 1:
                        base = mod if cand.name == "__init__.py" else mod.rsplit(".", 1)[0]
                        sub = (n.module or "")
                        full = sub if n.level == 0 else ".".join(x for x in [
                            base.rsplit(".", n.level - 1)[0] if n.level > 1 else base, sub] if x)
                        for al in n.names:
                            if al.name == "*":
                                out.update(dict(exportaciones_paquete(full, prof + 1)))
                            elif not (al.asname or al.name).startswith("_"):
                                out.setdefault(al.asname or al.name, f"from {full} import {al.name}")
                    if isinstance(n, ast.Assign):
                        for t in n.targets:
                            if isinstance(t, ast.Name) and t.id == "__all__" and isinstance(n.value, (ast.List, ast.Tuple)):
                                names = [e.value for e in n.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
                                out = {k: out.get(k, k) for k in names}
                return tuple(out.items())
    return ()


def importaciones(tree, path: str):
    """Lista de (alias_local, modulo, nombre|None). nombre None = se importa el módulo."""
    out = []
    if tree is None:
        return out
    pkg = modulo_de_ruta(path).rsplit(".", 1)[0] if "." in modulo_de_ruta(path) else ""
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for al in n.names:
                if al.asname:
                    out.append((al.asname, al.name, None))
                else:
                    out.append((al.name.split(".")[0], al.name.split(".")[0], None))
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ""
            if n.level:
                up = pkg.split(".") if pkg else []
                up = up[:len(up) - (n.level - 1)] if n.level > 1 else up
                base = ".".join(x for x in up + ([n.module] if n.module else []) if x)
            for al in n.names:
                if al.name != "*":
                    out.append((al.asname or al.name, base, al.name))
                else:
                    out.append(("*", base, "*"))
    return out


def activos(repo: Repo, path: str, src: str):
    """Conjunto activo E de un archivo.
    Devuelve (nombres: nombre -> (origen, cabecera), modulos: alias -> {attr: cabecera}, miembros: nombre -> cabecera)."""
    tree = parse(src)
    nombres, modulos, miembros = {}, {}, {}
    top, mem = repo.defs.get(path, ({}, {}))
    for k, v in top.items():
        nombres[k] = ("archivo", v)
    for k, (p, v) in repo.mem_all.items():
        miembros.setdefault(k, v)
    for alias, mod, name in importaciones(tree, path):
        rp = repo.resolver(mod) if mod else None
        if rp is not None:
            t2, m2 = repo.defs[rp]
            if name is None:
                modulos[alias] = dict(t2)
            elif name == "*":
                for k, v in t2.items():
                    nombres.setdefault(k, ("repo", v))
            elif name in t2:
                nombres[alias] = ("repo", t2[name])
            else:  # submódulo del repo
                sp = repo.resolver(f"{mod}.{name}")
                if sp is not None:
                    modulos[alias] = dict(repo.defs[sp][0])
            continue
        if name is None:
            ex = exportaciones_paquete(mod)
            if ex:
                modulos[alias] = dict(ex)
        elif name == "*":
            for k, v in exportaciones_paquete(mod):
                nombres.setdefault(k, ("paquete", v))
        else:
            ex = dict(exportaciones_paquete(mod))
            if name in ex:
                nombres[alias] = ("paquete", ex[name])
            else:
                sub = exportaciones_paquete(f"{mod}.{name}")
                if sub:
                    modulos[alias] = dict(sub)
    return nombres, modulos, miembros
