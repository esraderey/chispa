"""Entrenamiento de CHISPA-lab.

Modos:
  base : B20 sobre DATOS/tok/base-*.bin
  ul   : continúa un checkpoint con L_FIM + alpha·L_UL (E4); necesita --oraculo (máscaras precalculadas por chispa.oraculo)
  memA / memB / memC : brazos de E5 sobre DATOS/mem/*.pt (chispa.memdata)

Artefactos en --out: config.json, log.jsonl, ckpt.pt (cada --ckpt-min minutos y al final), final.pt.
Uso: python -m chispa.train --modo base --datos DATOS --out RUN --tokens 5e8 --seed 1337
"""
from __future__ import annotations

import argparse, glob, json, math, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import GPT, Config, ce_por_trozos, guardar, cargar_modelo


def commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "sin-git"


class Flujo:
    """Muestreo determinista de ventanas de ctx+1 tokens sobre fragmentos uint16."""

    def __init__(self, pattern, ctx, seed):
        self.files = sorted(glob.glob(pattern))
        assert self.files, f"sin datos: {pattern}"
        self.arrs = [np.memmap(f, dtype=np.uint16, mode="r") for f in self.files]
        self.lens = np.array([len(a) - ctx - 1 for a in self.arrs])
        self.ctx = ctx
        self.rng = np.random.default_rng(seed)

    def lote(self, B):
        fi = self.rng.choice(len(self.arrs), size=B, p=self.lens / self.lens.sum())
        xs = []
        for f in fi:
            i = self.rng.integers(0, self.lens[f])
            xs.append(torch.from_numpy(self.arrs[f][i:i + self.ctx + 1].astype(np.int64)))
        x = torch.stack(xs)
        return x[:, :-1], x[:, 1:]


def lr_wsd(step, total, peak, warm=0.02, decay=0.2):
    w = max(1, int(total * warm)); d0 = int(total * (1 - decay))
    if step < w:
        return peak * (step + 1) / w
    if step < d0:
        return peak
    return peak * max(0.0, (total - step) / max(1, total - d0))


def entrenar(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    datos = Path(a.datos)
    if a.init:
        model, ck0 = cargar_modelo(a.init, "cpu")
    else:
        memoria = {"memB": "estado", "memC": "chars"}.get(a.modo, "")
        model = GPT(Config(n_layer=a.layers, d=a.d, n_head=a.d // 64, ffn=a.ffn, ctx=a.ctx, memoria=memoria))
    model.to(dev)
    B = a.micro; accum = max(1, a.batch_tokens // (B * a.ctx))
    steps = int(a.tokens // (B * a.ctx * accum))
    cfg = {**vars(a), "commit": commit(), "params": model.n_params(), "steps": steps, "accum": accum,
           "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "torch": torch.__version__}
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    print(json.dumps({k: cfg[k] for k in ("params", "steps", "accum", "device")}), flush=True)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "emb" in n else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=(dev == "cuda"))
    fwd = model
    if a.compile and dev == "cuda":
        try:
            fwd = torch.compile(model)
        except Exception as e:  # Windows sin triton
            print("sin compile:", e, flush=True)

    start = 0
    ckp = out / "ckpt.pt"
    if ckp.exists():  # reanudación determinista
        ck = torch.load(ckp, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start = ck["step"] + 1
        print(f"reanudo en el paso {start}", flush=True)

    from . import lotes
    fuente = lotes.fuente(a.modo, datos, a.ctx, a.seed, a)
    for _ in range(start * accum):  # avanza el generador para reanudar con el mismo orden
        fuente.saltar(B)

    freno = None
    if a.freno:
        sys.path.insert(0, a.freno)
        from frenotermico import FrenoEnProceso
        freno = FrenoEnProceso(traza=str(out / "freno.csv")); freno.monitor.iniciar()

    log = open(out / "log.jsonl", "a")
    t0 = time.time(); tlast = time.time(); toks = 0
    model.train()
    for step in range(start, steps):
        lr = lr_wsd(step, steps, a.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        tot = {}
        for _ in range(accum):
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
                loss, parts = lotes.perdida(a.modo, fwd, model, fuente.lote(B), dev, a)
            (loss / accum).backward()
            for k, v in parts.items():
                tot[k] = tot.get(k, 0.0) + v / accum
            toks += B * a.ctx
            if freno:
                freno.paso()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % a.log_every == 0 or step == steps - 1:
            dt = time.time() - tlast; tlast = time.time()
            rec = {"step": step, "lr": lr, **{k: round(v, 5) for k, v in tot.items()},
                   "tok_s": round(toks / max(dt, 1e-9)), "t": round(time.time() - t0, 1)}
            toks = 0
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
        if a.ckpt_min and (time.time() - getattr(entrenar, "_last", t0)) > a.ckpt_min * 60:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, ckp)
            entrenar._last = time.time()
        if a.save_at and step == int(steps * a.save_at):
            guardar(model, out / f"at{int(a.save_at*100)}.pt", {"step": step})
    guardar(model, out / "final.pt", {"step": steps})
    if ckp.exists():
        ckp.unlink()
    print("fin", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", default="base"); ap.add_argument("--datos", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=float, default=5e8); ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--init"); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--d", type=int, default=320); ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--ffn", type=int, default=864); ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--micro", type=int, default=32); ap.add_argument("--batch-tokens", type=int, default=131072)
    ap.add_argument("--compile", action="store_true"); ap.add_argument("--freno", default="")
    ap.add_argument("--log-every", type=int, default=20); ap.add_argument("--ckpt-min", type=float, default=20)
    ap.add_argument("--save-at", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=0.3)      # E4
    ap.add_argument("--prefijo", type=int, default=256)      # E5 brazo A
    entrenar(ap.parse_args())


if __name__ == "__main__":
    main()
