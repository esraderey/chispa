# DESVIACIONES respecto a PREREGISTRO.md

Cada entrada lleva fecha y se escribe ANTES de ver la métrica afectada.

## 2026-11-09 — E3/E4: detector de nombres indefinidos
- **Preregistrado:** pyflakes sobre el archivo con la línea completada.
- **Problema (visto en la prueba de humo con un modelo sin entrenar, sin datos reales):** pyflakes exige que el archivo
  parsee; un completado que rompe la sintaxis queda fuera, y eso sesgaría la tasa hacia los completados buenos.
- **Cambio:** detector léxico por ámbito para TODOS los completados: nombres visibles = ligados a nivel de módulo,
  importados, parámetros y locales de las funciones que contienen la línea (del árbol del archivo original), builtins y
  los ligados en la propia línea; nombres usados = tokens NAME de la línea que no son atributos (tras «.»), ni argumentos
  con nombre, ni palabras clave. Pyflakes se mantiene en el subconjunto que parsea y se reporta su acuerdo con el
  detector léxico.
- **Métricas y umbrales:** sin cambio.

## 2026-11-09 — E4: pérdida solo en el primer subtoken del nombre
- **Preregistrado:** «en cada posición que inicia o continúa un nombre».
- **Cambio:** la pérdida L_UL se aplica solo en la posición que inicia un nombre (primer subtoken), con A_t = primeros
  subtokens de los nombres visibles. Motivo: simplicidad y coste; las continuaciones quedan sin penalizar.
- **Umbrales:** sin cambio.

## 2026-11-09 — E5: tope de símbolos con cabecera
- **Preregistrado:** ≤8 símbolos con cabecera por secuencia.
- **Cambio:** ≤4 símbolos × 64 tokens = 256 tokens de cabecera por secuencia, para igualar el cómputo extra del brazo A
  (256 tokens de prefijo). Sin esto, B y C gastarían el doble de cómputo extra que A y la comparación no sería justa.
- **Umbrales:** sin cambio.
