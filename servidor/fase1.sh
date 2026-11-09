#!/usr/bin/env bash
# Fase 1 en el servidor: datos completos + B20 base (con checkpoint al 80% para E4).
set -euo pipefail
cd /workspace/chispa-lab
mkdir -p logs
D=/workspace/datos
pip install -q tokenizers huggingface_hub pyflakes > logs/pip.log 2>&1
nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv > logs/gpu.txt
python -m chispa.data descargar   --out "$D"                        > logs/1_descargar.log 2>&1
python -m chispa.data agrupar     --datos "$D"                      > logs/2_agrupar.log 2>&1
python -m chispa.data tokenizador --datos "$D" --chars 5e7          > logs/3_tok.log 2>&1
python -m chispa.data base        --datos "$D" --tokens 5.2e8 --seed 1337 > logs/4_base_datos.log 2>&1
python -m chispa.train --modo base --datos "$D" --out /workspace/runs/B20 --tokens 5e8 --seed 1337 \
       --compile --micro 64 --save-at 0.8                           > logs/5_B20.log 2>&1
echo FASE1_OK > logs/fase1.done
