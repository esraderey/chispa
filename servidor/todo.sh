#!/usr/bin/env bash
# Corre las tres fases sin vigilancia; cada fase deja logs/faseN.done al terminar.
cd /workspace/chispa-lab
mkdir -p logs
{ bash servidor/fase1.sh && bash servidor/fase2.sh && bash servidor/fase3.sh && echo TODO_OK; } > logs/todo.log 2>&1
