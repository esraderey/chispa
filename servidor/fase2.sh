#!/usr/bin/env bash
# Fase 2 en el servidor: E1, E3, E2 (sin latencia) y E4. Requiere fase1 terminada.
set -euo pipefail
cd /workspace/chispa-lab
D=/workspace/datos; R=/workspace/res; RUNS=/workspace/runs
mkdir -p logs "$R"
export CHISPA_SITEPKGS=/workspace/sitepkgs
NP=$(nproc)
python -m chispa.paquetes --datos "$D" --destino /workspace/sitepkgs --n 20            > logs/6_paquetes.log 2>&1
python -m chispa.e1 --datos "$D" --out "$R/e1" --repos 500 --procs "$NP"                > logs/7_e1.log 2>&1
python -m chispa.gen cursores --datos "$D" --out "$R/cursores.jsonl" --n 20000 --seed 1337 > logs/8_cursores.log 2>&1
python -m chispa.e4 dev --datos "$D" --cursores "$R/cursores.jsonl" --out "$R/cursores_dev.jsonl" >> logs/8_cursores.log 2>&1
# E3: completados del B20 final
python -m chispa.gen completar --datos "$D" --modelo "$RUNS/B20/final.pt" --cursores "$R/cursores.jsonl" \
       --out "$R/base/compl.jsonl" --lote 64                                             > logs/9_compl_base.log 2>&1
python -m chispa.e3 clasificar --datos "$D" --cursores "$R/cursores.jsonl" --compl "$R/base/compl.jsonl" --out "$R/base" \
                                                                                         > logs/10_e3.log 2>&1
# E2: Δ en GPU; la latencia se mide luego en el portátil
python -m chispa.e2 --datos "$D" --modelo "$RUNS/B20/final.pt" --cursores "$R/cursores.jsonl" \
       --compl "$R/base/compl.jsonl" --out "$R/e2" --n-lat 0                             > logs/11_e2.log 2>&1
# E4: datos con oráculo y cuatro continuaciones desde el 80 %
python -m chispa.oraculo --datos "$D" --tokens 1.2e8 --seed 2024 --procs "$NP"          > logs/12_oraculo.log 2>&1
for s in 1337 1338; do
  for arm in mle ul; do
    al=0; [ "$arm" = ul ] && al=0.3
    python -m chispa.train --modo ul --alpha $al --init "$RUNS/B20/at80.pt" --solo-decay --datos "$D" \
           --out "$RUNS/E4_${arm}_$s" --tokens 1e8 --seed $s --compile --micro 64       > logs/13_E4_${arm}_$s.log 2>&1
    python -m chispa.gen completar --datos "$D" --modelo "$RUNS/E4_${arm}_$s/final.pt" --cursores "$R/cursores.jsonl" \
           --out "$R/e4/${arm}_$s/compl.jsonl" --lote 64                                 >> logs/13_E4_${arm}_$s.log 2>&1
    python -m chispa.e3 clasificar --datos "$D" --cursores "$R/cursores.jsonl" --compl "$R/e4/${arm}_$s/compl.jsonl" \
           --out "$R/e4/${arm}_$s"                                                       >> logs/13_E4_${arm}_$s.log 2>&1
  done
done
python -m chispa.e4 beta --datos "$D" --modelo "$RUNS/E4_mle_1337/final.pt" --dev "$R/cursores_dev.jsonl" --out "$R/e4/beta" \
                                                                                         > logs/14_beta.log 2>&1
B=$(python -c "import json;print(json.load(open('$R/e4/beta/beta.json'))['elegido'])")
python -m chispa.e4 sesgado --datos "$D" --modelo "$RUNS/E4_mle_1337/final.pt" --cursores "$R/cursores.jsonl" --beta "$B" \
       --out "$R/e4/sesgo"                                                               >> logs/14_beta.log 2>&1
python -m chispa.e4 comparar --mle "$R/e4/mle_1337" "$R/e4/mle_1338" --ul "$R/e4/ul_1337" "$R/e4/ul_1338" \
       --sesgo "$R/e4/sesgo" --out "$R/e4"                                               > logs/15_e4.log 2>&1
echo FASE2_OK > logs/fase2.done
