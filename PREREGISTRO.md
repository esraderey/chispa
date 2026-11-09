# PREREGISTRO — E1–E5 de RFC-CHISPA-3

**Fecha:** 2026-11-09, ANTES de construir el arnés y de ver ningún dato. Fuente de las hipótesis y umbrales: §3 y §5 de
`D:\rfc-chispa\chispa3\RFC-CHISPA-3.md` (firmado ESRADEREY). Los umbrales no se cambian después de ver resultados; cualquier
cambio se anota en `DESVIACIONES.md` con fecha y motivo, antes de mirar la métrica afectada.

## Material común (fijado)

- **Datos:** `codeparrot/codeparrot-clean` (Hugging Face, Python de GitHub con `repo_name` y `path`), los 54 fragmentos,
  reagrupados por repositorio. Licencias permisivas (mit, apache-2.0, bsd-2-clause, bsd-3-clause, isc, unlicense).
  Partición por repositorio con `sha1(repo_name) % 100`: 0–2 → retenidos (evaluación), resto → entrenamiento. Ningún archivo
  de un repositorio retenido entra en entrenamiento.
- **Tokenizador:** BPE a nivel de byte de 16,384 piezas con pretokenización alineada a lexemas de Python (identificador,
  número, corrida de puntuación, salto de línea más sangría), entrenado en 50M caracteres de entrenamiento. Especiales:
  `<|endoftext|>`, `<fim_prefix>`, `<fim_suffix>`, `<fim_middle>`, `<pad>`.
- **Modelo base B20:** decodificador d=320, 12 capas, 5 cabezas de 64, SwiGLU 864, RMSNorm, RoPE, vocabulario atado, contexto
  1,024; ≈20.1M parámetros. AdamW (β 0.9/0.95, wd 0.1), pico 3e-3, WSD (calentamiento 2%, decaimiento el 20% final), lote
  de 131,072 tokens, bf16. FIM en el 50% de los documentos (la mitad de ellos con hueco = resto de una línea), formato PSM.
- **Semillas:** base 1337; réplicas 1338, 1339. Orden de datos determinista por semilla.
- **Hardware:** RTX 4090 alquilada. El arnés se prueba antes en la RTX 4060 local a escala diminuta; esas pruebas no
  producen resultados reportables.
- **Artefactos obligatorios por corrida:** `config.json`, commit del arnés, `log.jsonl` (paso, pérdida, tok/s, hora),
  checkpoint final, y la salida cruda de cada evaluación (`*.jsonl` por ejemplo). Toda cifra del informe se recalcula con
  `analiza.py` desde esos archivos; ninguna se transcribe a mano.

## E1 — Oportunidad de la memoria de definiciones (CPU)

- **Hipótesis (P1, S1 de su oportunidad):** en repositorios retenidos, una fracción no trivial de las apariciones de
  identificadores refiere a un símbolo definido en el repositorio o en paquetes importados y ausente de los 1,024 tokens previos.
- **Métrica:** sobre 500 repositorios retenidos (sorteo con semilla 1337), fracción de apariciones de nombres en contexto
  Load (incluido `a.b`) cuyo nombre completo está en E y no en la ventana; y percentil 95 de |E| por posición.
  E = definiciones de nivel superior, clases, métodos y atributos `self.x` del repositorio + exportaciones estáticas de los
  módulos importados (biblioteca estándar y los 20 paquetes más importados del corpus, instalados sin dependencias solo
  para leer su fuente), activadas por las importaciones del archivo con sus alias.
- **Muere la Propuesta 1 si** la oportunidad es < 5 % o si |E| p95 > 20,000.
- **Control:** misma medida con E restringido al repositorio (sin paquetes), para atribuir la oportunidad.

## E2 — Prueba del sufijo offline (GPU ligera + CPU)

- **Hipótesis (P2):** la señal Δ = log p(s | prefijo, m) − log p(s | prefijo), con s = 16 primeros tokens del sufijo,
  añade información sobre si una sugerencia es aceptable más allá del log-prob medio de m.
- **Métrica:** 20,000 cursores en archivos retenidos (inicio de línea no vacía dentro de un cuerpo de función, semilla
  1337); m = completado voraz FIM del B20 hasta fin de línea; etiqueta «aceptable» = similitud de edición normalizada
  ≥ 0.8 con la línea real. Logístico sobre (Δ, logprob medio de m, longitud) frente a logístico sobre (logprob medio,
  longitud), validación cruzada de 5 particiones por repositorio. AUROC con IC del 90 % por bootstrap de repositorios.
  Latencia extra en CPU (int8 dinámico, 1 hilo y 4 hilos) en 500 cursores.
- **Muere la Propuesta 2 si** la mejora de AUROC < 0.02, o si la latencia extra p95 > 50 ms.
- **Control:** el mismo logístico con Δ calculado sobre un sufijo de OTRO archivo (Δ barajado); si iguala a Δ real, la
  señal no viene del código de abajo.

## E3 — Origen de los nombres indefinidos (CPU)

- **Hipótesis (P3, S1):** la mayoría de los nombres indefinidos que introduce un modelo de 20M son invenciones.
- **Métrica:** 20,000 completados de línea del B20 en repositorios retenidos; pyflakes sobre el archivo con la línea
  completada, por ámbito; cada Undefined Name nuevo (ausente en el archivo original) se clasifica en: invención (no existe
  en el repositorio), otro archivo del repositorio, local de un cuerpo hermano del mismo archivo.
- **Muere la Propuesta 3 si** las invenciones son < 30 % de los Undefined Name nuevos.
- **Descriptivo adicional:** tasa de Undefined Name por completado del B20 (comparable a [R32]: ~4.5 % a 350M).

## E4 — Monitor amortizado (GPU)

- **Hipótesis (P3):** añadir en el último 20 % del entrenamiento la pérdida L_UL = media[−log(1 − m_t)] con α = 0.3
  reduce los Undefined Name sin degradar la coincidencia de identificadores.
- **Diseño:** desde el checkpoint del B20 al 80 % (antes del decaimiento), dos brazos que terminan el entrenamiento
  (el 20 % restante, ~100M tokens a escala completa): MLE y MLE + α·L_UL, con el mismo orden de datos; 2 semillas de orden
  por brazo. Control: el brazo MLE con un sesgo −β en inferencia sobre primeras piezas de nombres ausentes del contexto y
  de los builtins, β ∈ {1, 2, 4} elegido en un conjunto de desarrollo disjunto.
- **Métrica:** ≥20,000 completados de cuerpos de función retenidos (los cursores de E3); tasa de Undefined Name nuevos con
  IC del 90 % por bootstrap; coincidencia exacta del primer identificador de la línea.
- **Muere la Propuesta 3 si** la reducción relativa de Undefined Name es < 25 % o su IC toca 0; o si la coincidencia exacta
  de identificadores cae > 1 punto; o si el sesgo sin entrenar logra ≥ 80 % de la reducción.

## E5 — Memoria de definiciones (GPU)

- **Hipótesis (P1):** con las mismas condiciones, la memoria de definiciones (B) acierta más identificadores definidos en
  otro archivo que meter las firmas como texto en el prompt (A), y la semántica de la definición importa frente a claves
  desde los caracteres del nombre (C).
- **Diseño:** tres modelos B20 entrenados desde cero con los mismos datos de repositorios (≥2 archivos) y el mismo número
  de tokens de texto principal (0.4B a escala completa): A = 256 tokens de firmas importadas al inicio del contexto;
  B = memoria (claves = W_k·media de los estados del propio modelo sobre la cabecera, ≤8 símbolos con cabecera por
  secuencia, negativos dentro del lote); C = memoria con claves desde las tripletas de caracteres del nombre.
  Semillas: A y B con 2 semillas, C con 1.
- **Métrica:** coincidencia exacta del identificador completo en posiciones de repositorios retenidos donde el nombre está
  definido en otro archivo del repositorio y no aparece en la ventana; latencia por paso en CPU.
- **Muere la Propuesta 1 si** B no supera a A en ≥ 3 puntos o si su latencia en CPU es mayor que la de A; la semántica sobra
  si B no supera a C en ≥ 1 punto.

## Lo que este preregistro NO permite

- Elegir otra métrica, otro umbral u otra partición después de ver resultados.
- Reportar solo la mejor semilla.
- Declarar viva una propuesta con un experimento inconcluso: inconcluso se reporta como inconcluso.
