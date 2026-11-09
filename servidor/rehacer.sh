#!/usr/bin/env bash
# Rehace E2-E4 tras corregir el FIM a nivel de segmento (DESVIACIONES 2026-11-09). E1 y los cursores se conservan.
cd /workspace/chispa-lab
D=/workspace/datos; R=/workspace/res; RUNS=/workspace/runs
{
set -euo pipefail
mkdir -p /workspace/res_fim_doc
[ -d "$R/base" ] && mv "$R/base" /workspace/res_fim_doc/
[ -d "$R/e2" ] && mv "$R/e2" /workspace/res_fim_doc/
[ -d "$RUNS/B20" ] && mv "$RUNS/B20" /workspace/res_fim_doc/B20
rm -rf "$D/ul"; rm -f "$D"/tok/base-*.bin
python -m chispa.data base --datos "$D" --tokens 5.2e8 --seed 1337 > logs/r4_base_datos.log 2>&1
python -m chispa.train --modo base --datos "$D" --out "$RUNS/B20" --tokens 5e8 --seed 1337 --compile --micro 64 --save-at 0.8        > logs/r5_B20.log 2>&1
bash servidor/fase2.sh
bash servidor/fase3.sh
echo REHACER_OK
} > logs/rehacer.log 2>&1
