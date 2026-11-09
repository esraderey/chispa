"""Genera la ficha del modelo de Hugging Face (README.md) desde los artefactos reales, sin cifras escritas a mano.

Uso: python -m chispa.tarjeta --hf hf/chispa-20m-python --res RES --run RUNS/B20 --datos DATOS
"""
from __future__ import annotations

import argparse, json
from pathlib import Path


def _j(p):
    p = Path(p)
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True); ap.add_argument("--res", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--datos", required=True); ap.add_argument("--repo", default="esraderey/chispa")
    a = ap.parse_args()
    cfg = _j(Path(a.run) / "config.json"); base = _j(Path(a.datos) / "tok" / "base.json") or {}
    stats = _j(Path(a.datos) / "repos" / "stats.json") or {}; e3 = _j(Path(a.res) / "base" / "resumen_e3.json")
    exp = _j(Path(a.hf) / "chispa_export.json") or {}
    log = [json.loads(l) for l in open(Path(a.run) / "log.jsonl")] if (Path(a.run) / "log.jsonl").exists() else []
    loss_final = log[-1]["loss"] if log else None
    met = ""
    if e3:
        met = (f"| Coincidencia exacta del primer identificador de la línea | {e3['em_primer_id']:.1%} |\n"
               f"| Coincidencia exacta de la línea completa | {e3['em_linea']:.1%} |\n"
               f"| Completados que introducen un nombre indefinido | {e3['tasa_undefined']:.1%} "
               f"(IC90 {e3['tasa_ic90'][0]:.1%}–{e3['tasa_ic90'][1]:.1%}) |\n")
    txt = f"""---
license: mit
language: [code]
library_name: transformers
pipeline_tag: text-generation
tags: [code, python, code-completion, fill-in-the-middle, tiny, llama]
datasets: [codeparrot/codeparrot-clean]
---

# CHISPA-20M-python

Modelo diminuto ({cfg['params']/1e6:.1f}M parámetros) de completado de línea de Python, entrenado desde cero como
laboratorio de la serie de RFC CHISPA. **Es un artefacto de investigación, no un asistente de código**: completa
el resto de una línea; no escribe funciones a partir de un docstring.

*A tiny (~{cfg['params']/1e6:.0f}M) Python line-completion model trained from scratch with fill-in-the-middle. Research
artifact, not a coding assistant. Code, preregistration and experiments: https://github.com/{a.repo}*

## Arquitectura

LLaMA: d=320, 12 capas, 5 cabezas de 64, SwiGLU 864, RMSNorm, RoPE, embeddings atados, contexto 1,024.
Tokenizador BPE a nivel de byte de 16,384 piezas con pretokenización alineada a lexemas de Python (identificador,
número, puntuación, salto de línea con sangría). Convertido desde el formato del laboratorio; diferencia máxima de
logits en la conversión: {exp.get('dif_max_logits', float('nan')):.1e}.

## Datos y entrenamiento

- `codeparrot/codeparrot-clean`, solo archivos con licencia permisiva (MIT, Apache-2.0, BSD-2/3, ISC, Unlicense),
  {stats.get('train', ['?', '?'])[1]} archivos de {stats.get('train', ['?'])[0]} repositorios de entrenamiento. El 3 % de los
  repositorios (por hash del nombre) se reservó para evaluación y no se vio al entrenar.
- {base.get('tokens', 0)/1e6:.0f}M tokens, el 50 % de los documentos en formato FIM (PSM); la mitad de ellos con el hueco al
  final de una línea.
- AdamW, pico 3e-3, calendario WSD, lote de 131,072 tokens, bf16, una H100. Pérdida final de entrenamiento: {loss_final}.

## Uso

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tk = AutoTokenizer.from_pretrained("{a.repo.split('/')[0]}/chispa-20m-python")
m = AutoModelForCausalLM.from_pretrained("{a.repo.split('/')[0]}/chispa-20m-python")
eos, pad = tk.convert_tokens_to_ids("<|endoftext|>"), tk.convert_tokens_to_ids("<pad>")
# FIM en formato PSM: el modelo rellena el hueco entre prefijo y sufijo
prompt = "<fim_prefix>def read_lines(path):\\n    with open(path) as f:\\n        <fim_suffix>\\n<fim_middle>"
ids = tk(prompt, return_tensors="pt").input_ids
out = m.generate(ids, max_new_tokens=16, do_sample=False, eos_token_id=eos, pad_token_id=pad)
linea = tk.decode(out[0][ids.shape[1]:], skip_special_tokens=True).split("\\n")[0]
print(linea.lstrip())   # -> return f.readlines()
```

Si el prefijo termina justo después de la sangría, el modelo tiende a empezar con espacios de más: recorta el inicio
de la línea (`lstrip`) o prohíbe en el primer paso las piezas que empiezan por espacio, que es lo que se hizo en los
experimentos.

## Evaluación (repositorios retenidos, 20,000 cursores dentro de cuerpos de función)

| Métrica | Valor |
|---|---|
{met}
Detalle, intervalos y experimentos E1–E5: https://github.com/{a.repo}

## Limitaciones

Modelo de 20M: inventa nombres, no conoce APIs poco comunes y su ventana es de 1,024 tokens. Solo Python. Las métricas
son offline y no se validaron con usuarios. El corpus procede de GitHub y puede contener código con errores o inseguro.
"""
    Path(a.hf, "README.md").write_text(txt, encoding="utf-8"); print(txt)


if __name__ == "__main__":
    main()
