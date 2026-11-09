# CHISPA-lab

Laboratorio reproducible para un **modelo local de completado de Python de 20M parámetros** y para los experimentos
E1–E5 de RFC-CHISPA-3: memoria de definiciones, prueba del sufijo y monitor amortizado.

*Reproducible lab for a 20M-parameter local Python completion model, with preregistered experiments (E1–E5) on three
ideas: a definition memory, a suffix test for abstention, and a training-time name monitor. Results, positive or
negative, are generated from raw artifacts. Code in Python + PyTorch; the base model exports to LLaMA format.*

## Qué hay aquí

| Ruta | Qué es |
|---|---|
| `PREREGISTRO.md` | Hipótesis, métricas, umbrales de muerte y controles, fijados antes de ver datos |
| `DESVIACIONES.md` | Cambios respecto al preregistro, cada uno anotado antes de ver la métrica afectada |
| `PREDICCIONES.md` | Predicciones del autor, fechadas en git antes de los resultados |
| `RESULTADOS.md` | Tabla de resultados generada por `python -m chispa.informe` desde `res/` |
| `chispa/` | Datos, tokenizador, modelo, entrenamiento, E1–E5, exportación a Hugging Face |
| `servidor/` | Las tres fases para una GPU alquilada (`todo.sh` las encadena) |

## Los experimentos

| Exp. | Pregunta | Muere si |
|---|---|---|
| E1 | ¿Cuántos identificadores apuntan a un símbolo del proyecto o de un paquete que no está en la ventana? | < 5 % o \|E\| p95 > 20,000 |
| E2 | ¿Comparar la verosimilitud del código de abajo con y sin la sugerencia ayuda a decidir si mostrarla? | AUROC + < 0.02 o latencia > 50 ms |
| E3 | ¿Los nombres indefinidos de un modelo de 20M son invenciones? | invenciones < 30 % |
| E4 | ¿Castigar nombres no visibles al entrenar reduce los nombres indefinidos mejor que un sesgo sin entrenar? | reducción < 25 %, IC toca 0, EM cae > 1 pt o el sesgo logra ≥ 80 % |
| E5 | ¿Una memoria cuyas filas son el estado del modelo al leer cada definición gana a meter las firmas en el prompt? | B − A < 3 pt o B más lento que A |

## Resultados

Ver **[RESULTADOS.md](RESULTADOS.md)**, generado desde `res/`. En resumen: la prueba del sufijo (P2) se valida; la memoria
de definiciones (P1) y el monitor amortizado (P3) no alcanzan sus umbrales y se retiran. Durante la ejecución se
detectaron y corrigieron tres defectos que afectaban a los resultados (FIM por documento, sangría inicial y escala de la
memoria en E5), documentados en [DESVIACIONES.md](DESVIACIONES.md) con lo que se vio antes de corregirlos, y un fallo de
memoria RAM en E4 que solo detenía el proceso.

- `res/`: salidas crudas por ejemplo y resúmenes de E1–E5 (los cursores de evaluación se regeneran con
  `python -m chispa.gen cursores --seed 1337` y los objetivos de E5 con `python -m chispa.e5 objetivos --seed 5`).
- `runs_config/`: configuración y registro de pérdida de cada entrenamiento; `logs_servidor/`: registros de la H100.
- `docs/rfc/`: los tres RFC de la serie CHISPA.
- Modelo base en formato LLaMA de Hugging Face: `python -m chispa.export_hf` (se publica como `chispa-20m-python`).

## Reproducir

```bash
# prueba local en pequeño (CPU o GPU de consumo)
python -m chispa.data parcial --out datos --mb 25
python -m chispa.data agrupar --datos datos
python -m chispa.data tokenizador --datos datos --chars 20e6
python -m chispa.data base --datos datos --tokens 3e6
python -m chispa.train --modo base --datos datos --out runs/humo --tokens 2e6 --micro 8 --batch-tokens 16384
```

```bash
# escala completa en una GPU alquilada (H100: ~1.5–2 h)
bash servidor/todo.sh
```

Después, la latencia en la CPU del editor (`python -m chispa.e2 ... --n-lat 500` y `python -m chispa.e5 latencia ...`) y
el informe (`python -m chispa.informe --res res --out RESULTADOS.md`).

## Datos y licencias

Código: MIT. El modelo se entrena solo con archivos de licencia permisiva (MIT, Apache-2.0, BSD-2/3, ISC, Unlicense)
de `codeparrot/codeparrot-clean`. El 3 % de los repositorios, elegido por hash del nombre, queda fuera del entrenamiento
y se usa solo para evaluar.

## Contexto

Serie de RFC CHISPA (firmada ESRADEREY): RFC-CHISPA-1 propuso el diseño de 20M, RFC-CHISPA-2 lo contrastó con el arte
previo y RFC-CHISPA-3 retiró sus propuestas y planteó las tres que aquí se ponen a prueba.
