#!/usr/bin/env bash
# Rehace los brazos B y C de E5 tras corregir la escala de la memoria (DESVIACIONES 2026-11-09). A se conserva.
cd /workspace/chispa-lab
D=/workspace/datos; R=/workspace/res; RUNS=/workspace/runs
export CHISPA_SITEPKGS=/workspace/sitepkgs
{
set -euo pipefail
for run in B:1337 B:1338 C:1337; do
  arm=${run%%:*}; s=${run##*:}
  [ -f "$R/e5/${arm}_$s/resumen.json" ] && continue
  rm -rf "$RUNS/E5_${arm}_$s"
  python -m chispa.train --modo mem$arm --datos "$D" --out "$RUNS/E5_${arm}_$s" --tokens 4e8 --seed $s --ctx 768          --compile --micro 64 --lam-mem 0.3                                               > logs/19_E5_${arm}_$s.log 2>&1
  python -m chispa.e5 evaluar --datos "$D" --modelo "$RUNS/E5_${arm}_$s/final.pt" --brazo $arm          --objetivos "$R/e5/objetivos.jsonl" --out "$R/e5/${arm}_$s"                      >> logs/19_E5_${arm}_$s.log 2>&1
done
python -m chispa.e5 comparar --A "$R/e5/A_1337" "$R/e5/A_1338" --B "$R/e5/B_1337" "$R/e5/B_1338" --C "$R/e5/C_1337"        --out "$R/e5"                                                                      > logs/20_e5.log 2>&1
echo E5_OK
} > logs/e5_rehacer.log 2>&1
