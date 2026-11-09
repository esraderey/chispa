#!/usr/bin/env bash
# Fase 3 en el servidor: E5 (datos de memoria, cinco modelos y evaluación). La latencia se mide luego en el portátil.
set -euo pipefail
cd /workspace/chispa-lab
D=/workspace/datos; R=/workspace/res; RUNS=/workspace/runs
export CHISPA_SITEPKGS=/workspace/sitepkgs
NP=24
python -m chispa.memdata --datos "$D" --split train --tokens 4.2e8 --seed 99 --procs "$NP"  > logs/16_memdata.log 2>&1
python -m chispa.e5 objetivos --datos "$D" --out "$R/e5/objetivos.jsonl" --n 20000 --seed 5 >> logs/16_memdata.log 2>&1
for run in A:1337 A:1338 B:1337 B:1338 C:1337; do
  arm=${run%%:*}; s=${run##*:}
  python -m chispa.train --modo mem$arm --datos "$D" --out "$RUNS/E5_${arm}_$s" --tokens 4e8 --seed $s --ctx 768 \
         --compile --micro 64                                                             > logs/17_E5_${arm}_$s.log 2>&1
  python -m chispa.e5 evaluar --datos "$D" --modelo "$RUNS/E5_${arm}_$s/final.pt" --brazo $arm \
         --objetivos "$R/e5/objetivos.jsonl" --out "$R/e5/${arm}_$s"                      >> logs/17_E5_${arm}_$s.log 2>&1
done
python -m chispa.e5 comparar --A "$R/e5/A_1337" "$R/e5/A_1338" --B "$R/e5/B_1337" "$R/e5/B_1338" --C "$R/e5/C_1337" \
       --out "$R/e5"                                                                      > logs/18_e5.log 2>&1
echo FASE3_OK > logs/fase3.done
