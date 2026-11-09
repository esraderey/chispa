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

## 2026-11-09 — E2 y E5: dónde y cómo se mide la latencia
- **Preregistrado:** latencia extra en CPU (E2) y latencia por paso en CPU (E5), sin fijar la máquina.
- **Cambio:** la latencia se mide en el portátil del usuario (la CPU del editor), no en el servidor, con int8 dinámico y
  1 y 4 hilos. En E2, la extra de la prueba del sufijo con el prefijo ya en caché KV. En E5, el prefill de la ventana
  (A: 256 de prefijo + 768; B/C: 768 + producto con las claves de E), que es la parte que difiere entre brazos.
- **Umbrales:** sin cambio.

## 2026-11-09 — FIM a nivel de segmento (defecto de implementación; se rehacen E2, E3 y E4)
- **Preregistrado:** FIM en el 50 % de los documentos (la mitad con hueco = resto de una línea), formato PSM.
- **Defecto detectado:** el FIM se aplicaba al archivo entero, pero los archivos miden ~2,700 tokens de media y el
  entrenamiento ve ventanas de 1,024. El modelo casi nunca vio prefijo, sufijo y medio en la misma ventana, y en la
  evaluación continuaba el sufijo en vez de rellenar el hueco. Se detectó inspeccionando ejemplos de E3 (0.8 % de
  coincidencia del primer identificador con una pérdida final de 1.48, implausible), antes de mirar E4 y E5.
- **Cambio:** con probabilidad 0.5, el documento de entrenamiento es un segmento de líneas consecutivas de ≤2,400
  caracteres (≈800 tokens) transformado con FIM (mismas reglas: mitad con hueco = resto de una línea); si no, el
  archivo entero sin FIM. En la evaluación, prefijo y sufijo se recortan en saltos de línea. Se reentrena el B20 y se
  rehacen E2, E3 y E4 desde cero. Los artefactos de la corrida defectuosa se conservan en `res_fim_doc/` y no se usan.
- **Vistos antes del cambio (no válidos):** E3 con invenciones 83.7 % y E2 con mejora de AUROC +0.029 sobre 113
  sugerencias aceptables. Los umbrales no cambian; E1 (sin FIM) se conserva.

## 2026-11-09 — Decodificación: sin espacio inicial en el primer paso
- **Preregistrado:** completado voraz FIM hasta fin de línea.
- **Problema (inspección cualitativa de los primeros 15,000 completados del B20 reentrenado, antes de calcular E2–E4):**
  la mayoría empezaba con espacios de sangría extra. En el entrenamiento FIM el corte cae a menudo dentro de la sangría,
  así que el modelo duda de si la sangría del prefijo está completa.
- **Cambio:** en el primer paso se prohíben las piezas que empiezan por espacio o tabulador. Es una restricción de la
  tarea: el cursor está al inicio del contenido de la línea y la verdad nunca empieza por espacio. Se aplica igual en
  E2, E3 y E4 (todos los brazos y el control de sesgo). Los completados parciales sin este cambio se descartan.
- **Umbrales:** sin cambio.

## 2026-11-09 — E5: escala de los logits de memoria y peso de su pérdida (defecto; se rehacen B y C)
- **Preregistrado:** la pérdida de memoria se suma a la de lenguaje, sin peso explícito.
- **Defecto detectado:** la pérdida de lenguaje del brazo B se estancó en ~4.4 frente a ~1.5 del brazo A. Medido en
  local: al inicio, la norma del gradiente de la memoria era 542 frente a 9.5 del lenguaje; con el recorte global a
  1.0, el modelo de lenguaje casi no aprendía. Se detectó por la curva de pérdida, antes de evaluar B.
- **Cambio:** logits de memoria escalados por 1/√d (como la atención) en entrenamiento y evaluación, y peso λ = 0.3 para
  la pérdida de memoria en B y C (gradientes medidos: B ~9 frente a 9.5; C ~0.5 frente a 8.5). Se descarta B_1337 (en
  `res_e5_defecto/`) y se reentrenan B (2 semillas) y C. El brazo A no usa memoria y no cambia.
- **Umbrales:** sin cambio.
