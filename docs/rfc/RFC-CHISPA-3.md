# RFC-CHISPA-3 — Memoria de definiciones, prueba del sufijo y monitor amortizado para un completador local de ≤100M

- Estado: PROPUESTO
- Fecha: 2026-11-09
- Firmado: ESRADEREY
- Expedición: ¿qué hay que cambiar en el diseño CHISPA (modelo local de completado de Python de 20M, ampliable hasta 100M parámetros, entrenable en una RTX 4060 de portátil o en una RTX 4090 alquilada y ejecutable en la CPU del editor) a la luz del arte previo completo, y qué otras palancas de ayuda útil a ≤100M no se han considerado?
- Método: expedición grande; 6 exploradores Sonnet, 1 lector de ola 2 y 1 explorador de novedad; 129 anclas en 57 obras, todas leídas por extracto; verificación ciega por fuente con 2 verificadores y escalada (fuentes defectuosas: C 0/12, F 1/6, R 0/20; cotas al 90 %: 17.5 %, 51.0 % y 10.9 %); canarios de verificación 4/4; sondeo sellado con recall 13/17; 3 lotes de propuestas (los supervivientes de RFC-CHISPA-2 congelados antes del barrido y 2 innovadores Opus), 9 propuestas, 48 objeciones ancladas (0 inválidas; incoherencias sembradas detectadas 3/3); 3 supervivientes; 0 experimentos ejecutados.

## 1. Resumen

Este RFC contrasta CHISPA con un mapa de 57 obras y retira las tres propuestas de RFC-CHISPA-2. El protocolo de laboratorio era muTransfer sin muP, la tabla viva era la CNN Softmax con otro conjunto de candidatos, y la puerta de resolución solo podría filtrar en torno al 6% de las sugerencias erróneas. Propone tres palancas nuevas para un completador de ≤100M. Primera: una memoria de definiciones cuyas filas de salida son el estado del propio modelo al leer cada definición del repositorio o de los paquetes importados. Segunda: una prueba del sufijo que compara la verosimilitud del código de abajo con y sin la sugerencia para decidir si mostrarla. Tercera: un monitor amortizado que castiga nombres indefinidos solo al entrenar. Cinco experimentos, ≤2.7 h de GPU cada uno, pueden matarlas.

## 2. Estado del arte

Calcular las filas de salida desde caracteres ya existe: la CNN Softmax las genera con una CNN de caracteres y las precalcula sin coste de inferencia [R01]. Un término de corrección por palabra de 128 dimensiones solo reduce a la mitad su brecha con la softmax regular, de 39.8 a 35.8 de perplejidad [R01]. HashFormers representa los embeddings con 99.1K parámetros [R06]. Copiar identificadores con un puntero da 13 veces más acierto en identificadores de Python [R13], aunque en esa misma tabla la atención de ventana más larga (30.2%) supera al puntero (27.3%) [R13]. El vocabulario dinámico admite tramos arbitrarios durante la generación, con un codificador de frases inicializado con GPT-2 de 124M y sin tocar el modelo base [R20]. VocabTailor sube el TTFT de 1.74 ms a 3.71 ms [R21].

El contexto del repositorio mejora el completado: RepoCoder más del 10% [R24] y DraCo 3.43% en coincidencia exacta [R29]. MGD eleva la tasa de compilación de SantaCoder-1.1B de 58.77 a 72.09 [R33], y ToolGen sube la validez estática a costa de aproximadamente el doble de tiempo [R35]. Undefined Name es de los errores estáticos más comunes y apenas cambia con el tamaño: 4.472% de los completados a 350M y 4.323% a 16B [R32]. Un trabajo ajusta modelos de 7B y 3B con pares cuya opción rechazada inventa nombres del proyecto y sube +18.8 points el exact match en Delulu [R57].

Para decidir qué mostrar, un modelo de aceptación reduce los falsos positivos de 55.09% a 17.44% [R48], y el filtro CDHF oculta el 25% de las sugerencias garantizando que el 95% habrían sido rechazadas [R50]. Un modelo local de 121M alcanza el 73.8% del rendimiento de uno de 7B en una cascada [R55], y un borrador entrenado con FastDraft logra hasta 3x de aceleración en completado de código [R53]. Muon alcanza la pérdida objetivo con 48-52% del cómputo de AdamW entre 30M y 200M [R47], y en muP muchos hiperparámetros óptimos se mantienen al cambiar el tamaño [R40].

{obs} Ninguna obra localizada mide estas palancas en un modelo de código de ≤100M. {obs} Los huecos que este RFC ataca tienen grado G2 en el mapa y se redactan como cota del barrido, no como inexistencia:

H4 — Copia de identificadores completos desde una tabla del repositorio o de los paquetes como filas de salida calculadas en cada paso.

H8 — Tablas de símbolos o firmas del repositorio y documentación de terceros como contexto.

H10 — Comprobación de nombres contra una tabla en un modelo de código de ≤100M.

H11 — Coste en CPU de la restricción por gramática o por nombres en un modelo pequeño.

H16 — Abstención por señal propia de un modelo local de ≤100M en CPU.

## 3. Propuesta(s)

### Cambios respecto a RFC-CHISPA-2 (regla propuesta por el director)

- Propuesta A de RFC-CHISPA-2 (protocolo de laboratorio a producto): cae como propuesta. Decidir en pequeño y transferir es la premisa de muTransfer [R40], y el protocolo no usaba muP. Residuo: la receta del producto de 100M usa muP [R40], Muon [R47], AST-FIM [R45] y Horizon-Length Prediction [R43] (regla propuesta).
- Propuesta B (tabla viva): queda absorbida por la Propuesta 1. Las filas desde caracteres son la CNN Softmax [R01] y solo sobreviven como brazo de control.
- Propuesta C (puerta de resolución): cae. Con Undefined Name en 4.472% de los completados [R32] y una aceptación del 21.4% [R50], solo `{der: 4.472% / (1 − 0.214) ≈ 5.7%}` de las sugerencias erróneas tendrían un nombre indefinido, y `ast` no parsea un archivo a medio escribir [R36]. Residuo: una búsqueda O(1) en la Propuesta 3.
- Supuestos eje:
  - S1: las palancas nuevas aportan más que las retiradas — indicador de cambio: NKS de cada superviviente frente al modelo base — evidencia discriminante: E3, E4 y E5.
- Estado: PROPUESTA.

### Propuesta 1 (I2-P2 + residuo de I1-P2) — Memoria de definiciones: filas de salida leídas del propio modelo

- Casilla: H4, H8.
- Mecanismo: un indexador en segundo plano recorre el repositorio y la fuente de los paquetes instalados. Por cada def, class o atributo público pasa por el modelo su cabecera (firma y docstring, ≤64 tokens) y guarda k = W_k·media(h) en int8 junto al nombre completo. El conjunto activo E sale de las importaciones del archivo con sus alias (`import numpy as np` activa `np.<nombre>`; tras `obj.` solo miembros). Los logits son el vocabulario BPE más {q_t·k_e : e∈E}, con q_t = W_q·h_t; elegir e emite el identificador entero. Reparto (regla propuesta): base B20 de d=320, 12 capas y vocabulario de 16,384 atado más W_q y W_k, `{der: 12·320²·12 + 16,384·320 + 2·320² = 20,193,280}`. Entrenamiento: en cada inicio de identificador presente en E, entropía cruzada sobre el vocabulario más los símbolos usados en el archivo y 256 negativos del lote. Correcciones de la crítica: como máximo 8 símbolos con cabecera y gradiente por secuencia, con claves cacheadas por lote (objeción I2-P2/1 de C3), y corrección logQ de los negativos muestreados para calibrar frente a |E| en inferencia (objeción I2-P2/2 de C3). Control (residuo de RFC-CHISPA-2): claves calculadas desde los caracteres del nombre, al estilo de [R01].
- Novedad frente al mapa: [R20] usa un codificador de frases aparte de 124M, [R01] calcula las filas desde caracteres y los punteros copian de la ventana [R13]. Aquí cada fila es el estado del propio modelo al leer la definición, y la importación con alias decide cuáles compiten.
- Coste (estimación): +30% de cómputo de entrenamiento con el tope de 8 símbolos; índice de 100k símbolos `{der: 1e5 × 320 B = 32 MB}`; en CPU, `{der: 5,000 × 320 = 1.6M MAC por paso}` con |E| = 5,000.
- Riesgos: (1) el doble camino (piezas BPE o entrada de E) descalibra las probabilidades; (2) tras `obj.` sin tipos, E puede ser grande; (3) las claves quedan obsoletas tras editar una firma hasta reindexar.
- Falsación: E1 y E5.
- Supuestos eje:
  - S1: en ≤100M el fallo de identificadores es de acceso y no de elección — indicador de cambio: fracción de errores del base cuyo nombre correcto ya estaba en la ventana — evidencia discriminante: >60% (umbral propuesto) apunta a elección y resta valor a la memoria.
- Estado: RETIRADO. Resultado: ../../RESULTADOS.md (E5 no alcanza el umbral de B − A; E1 sobrevive en el filo).

### Propuesta 2 (I2-P3, corregida) — Prueba del sufijo: el código de abajo como verificador del propio modelo

- Casilla: H16, H11.
- Mecanismo: se cachea la KV del prefijo. La rama FIM genera el medio m y, desde esa caché, dos prefills izquierda-a-derecha puntúan los primeros 16 tokens del sufijo s, con m (re-codificado sin el sufijo) y sin m (corrección de la crítica, objeción I2-P3/1 de C3). Señal Δ = log p(s | prefijo, m) − log p(s | prefijo). Se muestra la sugerencia principal si un logístico de 4 pesos sobre (Δ, log-prob medio de m, longitud) supera un τ calibrado en archivos retenidos. Corrección de la crítica (objeciones I2-P3/1 de C1 y I2-P3/3 de C3): Δ solo decide si se muestra el candidato principal; para reordenar candidatos basta log p(s | prefijo, m), porque el término sin m se cancela. Sin parámetros nuevos y sin entrenamiento extra.
- Novedad frente al mapa: los filtros del mapa aprenden de telemetría o con modelos aparte ([R48] [R50]), y la restricción por token de ToolGen dobla el tiempo [R35]. Aquí verifica el mismo modelo, con una sola comprobación por sugerencia, y el oráculo es el código que ya está bajo el cursor.
- Coste (estimación): `{der: (16 + 32 + 16) · 2 · 20e6 ≈ 2.6e9 FLOP}` extra por sugerencia con |m| = 32.
- Riesgos: (1) al final de una función el sufijo es independiente y Δ ≈ 0; (2) repetir la primera línea del sufijo sube Δ por copia (se veta por igualdad de líneas); (3) la etiqueta de calibración es un proxy offline (H18).
- Falsación: E2.
- Supuestos eje:
  - S1: los fallos de un modelo de 20M rompen la continuidad con el código de abajo — indicador de cambio: separación de Δ entre completados aceptables y no aceptables — evidencia discriminante: KS < 0.1 (umbral propuesto) la refuta.
- Estado: VALIDADO. Resultado: ../../RESULTADOS.md (E2 supera el umbral de AUROC y el de latencia con 4 hilos).

### Propuesta 3 (I3-P3 + residuo de I1-P3) — Monitor amortizado: la tabla de símbolos actúa al entrenar, no al decodificar

- Casilla: H10, H11.
- Mecanismo: sin parámetros nuevos. Offline, por ámbito y con `symtable` (corrección de la crítica, objeción I3-P3/2 de C2), D = nombres ligados visibles ∪ builtins ∪ importados ∪ exportaciones de los módulos importados del repositorio y de la biblioteca estándar. Los atributos de paquetes de terceros quedan sin penalizar (corrección de la crítica, objeción I3-P3/2 de C1), y los archivos con `import *`, `exec` o `globals()` van sin oráculo. D tokenizado forma un trie. En cada posición que inicia o continúa un nombre en contexto Load, con prefijo π, la masa prohibida es m_t = p(tokens de identificador) − p(hijos de π en el trie). Objetivo: L = L_FIM + α·media_t[−log(1 − m_t)], con α = 0.3 (umbral propuesto), en el último 20% de los tokens. Inferencia: modelo intacto; como residuo de RFC-CHISPA-2, una búsqueda O(1) del identificador cerrado en la tabla para filtrar lo mostrado.
- Novedad frente al mapa: MGD, ToolGen y la decodificación con tipos restringen en inferencia ([R33] [R35] [R38]), y [R57] penaliza nombres inventados con pares de preferencia en modelos de 3B y 7B ya entrenados. Aquí el oráculo es una tabla de símbolos durante el preentrenamiento de un modelo ≤100M, y el coste por token en CPU es nulo.
- Coste (estimación): trie por archivo en el cargador de CPU; techo declarado: con [R32] el error que ataca ronda el 4.5% de los completados a 350M.
- Riesgos: (1) conservadurismo ante funciones aún no escritas; (2) desajuste entre el trie y las piezas BPE; (3) el unlikelihood demasiado fuerte puede disparar la perplejidad (V0-conjetura).
- Falsación: E3 y E4.
- Supuestos eje:
  - S1: los nombres indefinidos de un modelo de 20M son invenciones y no referencias a otros archivos — indicador de cambio: fracción de esos nombres que existen en otro archivo del repositorio — evidencia discriminante: si es mayoría (umbral propuesto), el remedio es la Propuesta 1, no la pérdida.
- Estado: RETIRADO. Resultado: ../../RESULTADOS.md (E4 no alcanza la reducción preregistrada y el sesgo sin entrenar la iguala; E3 sobrevive sin robustez).

## 4. Alternativas

- I1-P1 «Protocolo de laboratorio a producto» (RFC-CHISPA-2 A): cayó por refrito de muTransfer [R40] y de MobileLLM [R07] y por no tener falsación en el paquete; residuo: receta del producto con muP, Muon, AST-FIM y HLP.
- I1-P2 «Tabla viva» (RFC-CHISPA-2 B): cayó por refrito de la CNN Softmax [R01]; residuo: brazo de control de la Propuesta 1.
- I1-P3 «Puerta de resolución» (RFC-CHISPA-2 C): cayó por un techo de ~5.7% de sugerencias filtrables y por no poder analizar un archivo roto; residuo: búsqueda O(1) en la Propuesta 3.
- I2-P1 «Atención por árbol de ámbitos»: cayó por una caché que se invalida al editar un cuerpo hermano y por no medir la latencia que reclama; residuo: su diagnóstico de nombres locales de cuerpos hermanos entra en E2.
- I3-P1 «Memoria de patrones con renombrado»: no se adopta por cupo; su novedad se limita al renombrado frente a CoG [R19] y le falta un brazo de recuperación [R24]. Queda como pregunta abierta.
- I3-P2 «El repositorio en los pesos»: no se adopta por cupo; su compuerta no se mide tras cuantizar a int8. Queda como pregunta abierta, porque es el único canal sin coste de contexto en CPU.

## 5. Riesgos, costes y preguntas abiertas

Experimentos asesinos, en orden ascendente de coste dentro de cada unidad. Ninguno se ejecutó en esta corrida. E2–E5 usan un modelo base B20 (d=320, 12 capas, vocabulario de 16,384, 0.5B tokens), cuyo entrenamiento cuesta ≈0.8 h de RTX 4090 (estimación) y se comparte.

- E1 — Oportunidad de la memoria, en CPU sobre 500 repositorios retenidos: fracción de apariciones de identificadores definidos en el repositorio o en paquetes importados y ausentes de la ventana, y tamaño de E por posición. Mata la Propuesta 1 si la oportunidad es <5% o si |E| supera 20,000 en el percentil 95 (umbrales propuestos). Coste: 0.5 h CPU.
- E2 — Prueba del sufijo offline: con el B20 y 20,000 cursores retenidos, AUROC del logístico para predecir una sugerencia aceptable frente al log-prob medio solo, y latencia en CPU con int8. Mata la Propuesta 2 si el AUROC no mejora ≥0.02 o si la latencia extra p95 pasa de 50 ms (umbrales propuestos). Coste: 0.5 h GPU.
- E3 — Origen de los nombres indefinidos, en CPU: 20,000 completados del B20 sobre repositorios retenidos, pyflakes por ámbito y clasificación de cada Undefined Name en invención, nombre de otro archivo o local de un cuerpo hermano. Mata la Propuesta 3 si las invenciones son <30% (umbral propuesto); en ese caso el remedio es la Propuesta 1. Coste: 1 h CPU.
- E4 — Monitor amortizado: desde el checkpoint del B20 al 80%, dos brazos de 300M tokens (MLE y MLE con la pérdida, α = 0.3), ≥20,000 completados de cuerpos de función con intervalo de confianza, y un control de sesgo −β en inferencia sin entrenar. Mata la Propuesta 3 si Undefined Name baja <25% relativo con el intervalo tocando 0, si la coincidencia exacta de identificadores cae >1 punto o si el sesgo sin entrenar logra ≥80% de la reducción (umbrales propuestos). Coste: 1 h GPU.
- E5 — Memoria de definiciones: tres B20 a 0.4B tokens (A: firmas importadas como texto en 256 tokens de prompt; B: memoria; C: memoria con claves desde caracteres). Mata la Propuesta 1 si B no supera a A en ≥3 puntos de coincidencia exacta en identificadores definidos en otro archivo sin más latencia en CPU que A; si B no supera a C en ≥1 punto, la semántica de la definición sobra (umbrales propuestos). Coste: 2.7 h GPU.

Costes y riesgos globales:
- GPU: `{der: 0.8 + 0.5 + 1 + 2.7 = 5.0 h}` de RTX 4090, ~2–5 USD (estimación), y 1.5 h de CPU. En la RTX 4060 de portátil, multiplicar por su velocidad relativa y por el factor térmico de E1 de RFC-CHISPA-1.
- Producto (regla propuesta, residuo de I1-P1): solo con las palancas que sobrevivan, un modelo de ≤100M con forma profunda y estrecha al estilo de [R07], hiperparámetros transferidos con muP desde el B20 [R40], Muon [R47], AST-FIM [R45] y HLP [R43]; su coste depende del rendimiento real de la GPU (objeción I1-P1/4 de C3).
- Toda la evidencia del mapa viene de resúmenes y tablas leídas por extracto; ninguna cita se cotejó contra texto crudo. {obs} La cota de F es ancha (51.0%) porque el estrato es pequeño.

Preguntas abiertas: la memoria de patrones con renombrado (I3-P1) con un brazo de recuperación; el adaptador por repositorio entrenado en reposo (I3-P2) con la compuerta medida tras cuantizar; un borrador de ~20M para decodificación especulativa de un modelo de 100M en CPU (H17); la latencia real en CPU con cuantización a 20–100M (H19); destilación o sintéticos hacia ≤100M (H14).

## 6. Referencias

Consultadas el 2026-11-08 y 2026-11-09.

R01 Jozefowicz et al., Exploring the Limits of Language Modeling, 2016-02 — https://ar5iv.labs.arxiv.org/html/1602.02410
R06 Xue y Aletras, HashFormers: Towards Vocabulary-independent Pre-trained Transformers, 2022-10 — https://arxiv.org/abs/2210.07904
R07 Liu et al., MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases, 2024-02 — https://arxiv.org/abs/2402.14905
R13 Bhoopchand et al., Learning Python Code Suggestion with a Sparse Pointer Network, 2016-11 — https://arxiv.org/abs/1611.08307
R19 Lan et al., Copy Is All You Need, 2023-07 — https://arxiv.org/abs/2307.06962
R20 Liu et al., Generation with Dynamic Vocabulary, 2024-10 — https://arxiv.org/html/2410.08481v1
R21 VocabTailor: vocabulario dinámico para modelos pequeños, 2025-08 — https://arxiv.org/html/2508.15229v3
R24 Zhang et al., RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation, 2023-03 — https://arxiv.org/abs/2303.12570
R29 DraCo: completado de repositorio guiado por flujo de datos, 2024-05 — https://arxiv.org/abs/2405.19782
R32 Ding et al., A Static Evaluation of Code Completion by Large Language Models, 2023-06 — https://arxiv.org/abs/2306.03203
R33 Agrawal et al., Guiding Language Models of Code with Global Context using Monitors, 2023-06 — https://arxiv.org/abs/2306.10763
R35 ToolGen: generación de código con herramientas de autocompletado, 2024-01 — https://arxiv.org/html/2401.06391v3
R36 Decodificación restringida para FIM con cocientes de gramáticas, 2024-02 — https://arxiv.org/abs/2402.17988
R38 Mündler et al., Type-Constrained Code Generation with Language Models, 2025-04 — https://arxiv.org/abs/2504.09246
R40 Yang et al., Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer, 2022-03 — https://arxiv.org/abs/2203.03466
R43 Horizon-Length Prediction para FIM, 2024-10 — https://arxiv.org/abs/2410.03103
R45 AST-FIM: preentrenamiento FIM consciente de la estructura, 2025-05 — https://arxiv.org/abs/2506.00204
R47 Estudio de Muon en decoders de 30M a 200M, 2025-09 — https://arxiv.org/abs/2509.24406
R48 Toward Less Hidden Cost of Code Completion with Acceptance and Ranking Models, 2021-06 — https://arxiv.org/abs/2106.13928
R50 Mozannar et al., When to Show a Suggestion? Integrating Human Feedback in AI-Assisted Programming, 2023-06 — https://arxiv.org/html/2306.04930v3
R53 FastDraft: How to Train Your Draft, 2024-11 — https://arxiv.org/abs/2411.11055
R55 MCCom: cascada de modelo local y modelo en la nube para completado, 2026-03 — https://arxiv.org/abs/2603.05974
R57 Negativos duros de alucinación para completado FIM de código (SFT frente a DPO/ORPO), 2026-06 — https://arxiv.org/abs/2606.03130
