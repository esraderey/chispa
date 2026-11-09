"""Instala sin dependencias los N paquetes de terceros más importados del corpus, solo para leer su fuente (E1, E5).

Uso: python -m chispa.paquetes --datos D --destino /workspace/sitepkgs --n 20
"""
from __future__ import annotations

import argparse, ast, collections, json, subprocess, sys
from pathlib import Path

from .symbols import STDLIB, parse

PIP = {"sklearn": "scikit-learn", "yaml": "pyyaml", "PIL": "pillow", "cv2": "opencv-python", "bs4": "beautifulsoup4",
       "dateutil": "python-dateutil", "google": "protobuf", "jwt": "pyjwt", "Crypto": "pycryptodome", "MySQLdb": "mysqlclient"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True); ap.add_argument("--destino", required=True)
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--muestra", type=int, default=20000)
    a = ap.parse_args()
    from .data import repos
    cnt = collections.Counter(); k = 0
    for r in repos(Path(a.datos), "train"):
        locales = {Path(f["path"]).parts[0].removesuffix(".py") for f in r["files"]} | \
                  {Path(f["path"]).stem for f in r["files"]}
        mods = set()
        for f in r["files"]:
            t = parse(f["content"])
            if t is None:
                continue
            for n in ast.walk(t):
                if isinstance(n, ast.Import):
                    mods |= {al.name.split(".")[0] for al in n.names}
                elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                    mods.add(n.module.split(".")[0])
        cnt.update(m for m in mods if m not in STDLIB and m not in locales and m != "__future__")
        k += 1
        if k >= a.muestra:
            break
    top = [m for m, _ in cnt.most_common(a.n)]
    ok = []
    for m in top:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", a.destino, PIP.get(m, m)],
                           capture_output=True, text=True)
        ok.append({"modulo": m, "pip": PIP.get(m, m), "repos": cnt[m], "instalado": r.returncode == 0})
    json.dump(ok, open(Path(a.destino) / "_chispa_paquetes.json", "w"), indent=1)
    print(json.dumps(ok, indent=1))


if __name__ == "__main__":
    main()
