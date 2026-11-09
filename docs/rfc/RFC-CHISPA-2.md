# RFC-CHISPA-2 — Qué cambiar en CHISPA: tabla viva de nombres, puerta de resolución y protocolo de laboratorio a producto

- Estado: PROPUESTO
- Fecha: 2026-11-08
- Firmado: ESRADEREY
- Expedición: ¿qué hay que cambiar en el diseño CHISPA (modelo local de completado de Python de 20M, ampliable hasta 100M parámetros, entrenable en una RTX 4060 de portátil o en una GPU alquilada) a la luz del arte previo y de la evidencia sobre sus ideas?
- Método: expedición pequeña; 3 exploradores Sonnet y 1 lector de ola 2; 52 anclas en 27 obras, todas leídas por extracto; verificación ciega por fuente en 2 pasadas (fuentes defectuosas: C 0/8, F 2/12, R 0/9; cotas al 90 %: 25.0 %, 38.6 % y 22.6 %); canarios de verificación 4/4; sondeo sellado con recall 3/11; diseño de partida del director congelado antes del barrido más 1 innovador Sonnet, 5 propuestas, 12 objeciones ancladas (0 inválidas; incoherencias sembradas detectadas 2/2); 3 supervivientes, dos de ellos por fusión; 0 experimentos ejecutados.

## 1. Resumen

RFC-CHISPA-1 reclamaba novedad en dos ideas que ya tienen arte previo: generar embeddings desde caracteres y copiar identificadores con un puntero. Este RFC las rebaja y propone tres cambios. Primero, una tabla viva de nombres: los identificadores del repositorio y de los paquetes instalados entran en la salida como filas calculadas desde sus tripletas de caracteres, incluidos los que quedan fuera de la ventana o nunca se vieron. Segundo, una puerta sin parámetros que, antes de mostrar una sugerencia, comprueba con `ast` y `symtable` que todos sus nombres existen. Tercero, un protocolo: decidir con un laboratorio de 20M igualado por cómputo y entrenar una sola vez el producto de 99.8M en una RTX 4090 alquilada. Nada está ejecutado: seis experimentos de ≤3.6 h pueden matarlo, empezando por dos pruebas en CPU.

## 2. Estado del arte

Representar las palabras desde sus caracteres tiene arte previo directo: un modelo de 2015 usa solo caracteres como entrada e iguala el estado del arte en Penn Treebank con 60% menos parámetros [R03]. T-FREE embebe palabras con activaciones dispersas sobre tripletas de caracteres y reporta más de 85% de reducción en las capas de embedding y cabeza [R09]. A escala sub-billón, MobileLLM combina profundidad y estrechez con embeddings compartidos y mejora 2.7%/4.3% frente a los modelos previos de 125M/350M [R01]. En LLM de código, el tamaño, la regex de pretokenización y los datos del tokenizador afectan significativamente el rendimiento [R06].

Copiar identificadores también tiene arte previo: una red de puntero dispersa para Python da 13 times more accurate prediction of identifiers [R18], y una red de mezcla con puntero copia palabras fuera de vocabulario desde el contexto cercano [R11]. Añadir contexto del repositorio mejora el completado: RepoCoder supera en más del 10% a la línea base de solo el archivo [R13], y ContextModule añade definiciones de símbolos y mejora la aceptación [R16]. Los modelos de completado actuales se apoyan sobre todo en el contexto inmediato del archivo [R16]. Las APIs de bibliotecas privadas se abordan recuperando su documentación [R17].

Para filtrar o restringir lo que se sugiere, un modelo de aceptación reduce los falsos positivos de 55.09% a 17.44% [R19]. La guía por monitores de análisis estático mejora la compilación de SantaCoder-1.1B [R12], y la restricción por tipos reduce los errores de compilación en más de la mitad en TypeScript [R15]. En CodeCompose las sugerencias multilínea casi duplicaron las pulsaciones ahorradas, de 9% a 17% [R20].

En entrenamiento, la ventaja de los optimizadores matriciales sobre AdamW baja de 1.4x a 0.1B a 1.1x a 1.2B [R21], y Muon escalado reporta ~2× de eficiencia frente a AdamW [R26]. El currículo FIM da ganancias más pronunciadas en los modelos pequeños [R23], y el ajuste supervisado mejora FIM sin posprocesado cuando el hueco son líneas completas [R24].

{obs} Ninguna obra localizada mide estas ideas en un modelo de código de ≤100M. {obs} Los huecos que este RFC ataca tienen grado G2 en el mapa y se redactan como cota del barrido, no como inexistencia:

H1 — Softmax de salida compuesto desde caracteres en un modelo de código de 50-150M.

H3 — Profundo frente a ancho a 50-150M en código.

H4 — Copia de identificadores desde una tabla del repositorio en un modelo de ≤100M.

H5 — Análisis estático como filtro de sugerencias con modelos ≤100M.

H6 — Nombres de bibliotecas de terceros en modelos de completado pequeños.

H7 — Muon frente a AdamW a ~0.1B en código.

## 3. Propuesta(s)

### Cambios respecto a RFC-CHISPA-1 (regla propuesta por el director)

- P1 (tabla generada desde bytes) deja de ser apuesta: tiene arte previo directo ([R09] [R03]) y a 100M una tabla libre de 16,384 es `{der: 16,384 × 576 / 97,940,160 ≈ 9.6%}` del presupuesto. Su residuo vive en la Propuesta B.
- P2 (cabeza de copia) se sustituye por la Propuesta B. Su marginal exacta por tramo y la prohibición de seguir con pieza de identificador tras copiar se conservan.
- P3 (banco NKS y abstención) se mantiene como métrica principal; la Propuesta C se pone delante de la abstención.
- Plan experimental: un solo esqueleto base por escala, brazos igualados por FLOP fijados de antemano, entropía cruzada por trozos, checkpoints reanudables cada 15–30 min y horas en RTX 4090 alquilada. En la RTX 4060 de portátil, todas las horas se multiplican por el factor térmico que mida E1 de RFC-CHISPA-1.
- Supuestos eje:
  - S1: retirar P1 no pierde calidad — indicador de cambio: el brazo con tabla libre iguala o supera al de filas calculadas fuera de los nombres del repositorio — evidencia discriminante: NLL fuera de tramos en E4.
- Estado: PROPUESTA.

### Propuesta A (I1-P3, enmendada) — Protocolo: laboratorio de 20M, producto de 99.8M

- Casilla: H3, H7.
- Mecanismo: las decisiones se toman en un laboratorio de 20M con brazos igualados por FLOP, y solo lo que depende de escala se repite en una corrida corta de 100M antes de la corrida única del producto. Enmienda de la crítica (objeción I1-P3/1): la forma profunda frente a ancha también se repite a 100M. Laboratorio (regla propuesta): 10 bloques d=384 (6 cabezas de 64, 2 KV, SwiGLU 1,024), tabla libre de 8,192 y tabla viva de 4,096 cubetas, `{der: 10 × 1,573,632 + 8,192 × 384 + 4,096 × 384 + 2 × (384² + 384) + 384 = 20,750,976}`. Producto (regla propuesta; residuo de I1-P1): 24 bloques d=576 (9 cabezas de 64, 3 KV, SwiGLU 1,536), tabla libre de 16,384 atada, tabla viva de 8,192 cubetas, contexto 1,536 y cabeza de abstención, `{der: 24 × 3,540,096 + 16,384 × 576 + 8,192 × 576 + 2 × (576² + 576) + 576 + 37,249 = 99,820,609}`. Datos: Python filtrado y separado por repositorio, 6B tokens (estimación). Optimizador: Muon en las matrices de los bloques y AdamW en embeddings y normas; corrección de la crítica (objeción I1-P1/2): la ventaja medida de los optimizadores matriciales encoge con el tamaño y es mayor a 0.1B que a 1.2B, no «empieza» en 0.1B.
- Coste (estimación): `{der: 6 × 85.6e6 + 6 × 16,384 × 576 + 12 × 24 × 768 × 576 ≈ 6.98e8 FLOP/token}`; producto `{der: 6e9 × 6.98e8 / 7e13 / 3,600 ≈ 16.6 h}` de RTX 4090 a 7e13 FLOP/s efectivos (V0-conjetura), unos 20 h facturables y ~7 USD a 0.34 USD/h (estimación); laboratorio E3–E6 ≈ 8 h, ~3 USD.
- Riesgos: (1) lo que gana a 20M puede no transferir a 100M; (2) igualar por FLOP ignora diferencias de rendimiento real entre brazos; (3) el esqueleto del producto aplica MobileLLM a código sin medida previa en código (objeción I1-P1/1).
- Falsación: E5 y E6.
- Supuestos eje:
  - S1: el orden entre variantes se conserva de 20M a 100M — indicador de cambio: concordancia de orden del par de formas en E6 entre las dos escalas — evidencia discriminante: concordante frente a discordante.
  - S2: el completado de Python se beneficia más de profundidad que de anchura cerca de 100M — indicador de cambio: orden de bpb y NKS entre 24×576 y 12×768 (regla propuesta) a igual FLOP — evidencia discriminante: brazos de 100M de E6.
- Estado: PROPUESTA.

### Propuesta B (fusión I2-P1 + I1-P2 a) — Tabla viva de nombres con filas calculadas desde caracteres

- Casilla: H4, H6, H1.
- Mecanismo: en cada posición la salida es una softmax sobre la tabla libre más un conjunto de candidatos que cambia por paso. Los candidatos son, como máximo, 2,000 nombres filtrados por ámbito: definiciones del archivo y del repositorio, extraídas con `ast`, y nombres públicos de los paquetes instalados que el archivo importa (`__all__` o definiciones de nivel superior). La fila de cada candidato es e(nombre) = MLP(suma de las cubetas de sus tripletas de caracteres), al estilo de [R09], así que un nombre nunca visto en entrenamiento tiene fila. Pérdida (corrección de la crítica, objeción I2-P1/2): cada tramo copiable suma con logsumexp sus dos derivaciones, la fila entera y la secuencia de piezas, y tras emitir una fila entera se prohíbe seguir con una pieza de identificador, de modo que la marginal por tramo es exacta, como en RFC-CHISPA-1 P2. En entrenamiento, los candidatos salen de los archivos hermanos del repositorio, con un 30% descartado al azar (umbral propuesto). En inferencia, las filas se recalculan solo cuando cambia el índice.
- Novedad frente al mapa: la copia y los punteros existen ([R18] [R11]) y la representación desde caracteres también ([R09] [R03]). Lo que se reclama es combinarlas en un modelo ≤100M con candidatos de fuera de la ventana y de los paquetes instalados, cuya fila se calcula y no se guarda (corrección de la crítica, objeción I2-P1/3). Fuera del mapa hay sistemas de contexto con tablas de símbolos para modelos grandes (SaraCoder, IDECoder; V0-conjetura, no abiertos en esta corrida).
- Coste (estimación): cubetas y MLP `{der: 4,096 × 384 + 2 × (384² + 384) = 1,868,544}` a 20M y `{der: 8,192 × 576 + 2 × (576² + 576) = 5,383,296}` a 100M; candidatos por paso `{der: 2,000 × 576 ≈ 1.15e6 MAC}`.
- Riesgos: (1) nombres casi iguales (`get_user`, `get_users`) comparten cubetas; (2) los paquetes instalados meten ruido; (3) los archivos hermanos pueden no simular bien el repositorio real; (4) el indexado puede añadir latencia al abrir un archivo.
- Falsación: E2 y E4.
- Supuestos eje:
  - S1: las tripletas separan nombres casi iguales — indicador de cambio: confusión en pares con distancia de edición ≤2 frente al brazo de filas libres — evidencia discriminante: matriz de confusión de E4 sobre esos pares.
  - S2: la ventaja viene de calcular la fila desde caracteres y no solo del conjunto de candidatos — indicador de cambio: el brazo de filas libres iguala al de caracteres — evidencia discriminante: control de E4.
- Estado: PROPUESTA.

### Propuesta C (fusión I2-P2 + I1-P2 c) — Puerta de resolución de nombres antes de mostrar

- Casilla: H5.
- Mecanismo: el decodificador produce 4 candidatos por haz para el resto de la línea. Un resolutor determinista sin parámetros (`ast.parse` tolerante más `symtable`) comprueba que cada nombre del candidato esté definido, importado o sea builtin, sin ejecutar código. Regla única (corrección de la crítica, objeción I2-P2/1): se muestra el candidato mejor puntuado que resuelve todos sus nombres; si ninguno resuelve, no se muestra nada. La cabeza de abstención de RFC-CHISPA-1 P3 decide después sobre el candidato que pasa. El resolutor sustituye a jedi (corrección de la crítica, objeción I1-P2/2) para no pasar de la latencia del editor. Opcional, como experimento secundario: usar el resolutor como señal de ajuste fino, con probabilidad inversa sobre los candidatos que no resuelven.
- Novedad frente al mapa: un modelo de aceptación entrenado con registros de uso ya filtra lo que se muestra [R19], y los monitores y los tipos restringen la decodificación con modelos ≥1B [R12] [R15]. Lo nuevo es un filtro estático, sin entrenamiento ni datos de uso, en un modelo ≤100M y medido con NKS (corrección de la crítica, objeción I2-P2/2).
- Coste (estimación): el resolutor corre en CPU por candidato; el haz de 4 multiplica ×4 la decodificación.
- Riesgos: (1) alias dinámicos, `setattr` e `import *` producen falsos descartes; (2) el haz de 4 sube la latencia en CPU; (3) callar de más baja el ahorro.
- Falsación: E1 y E3.
- Supuestos eje:
  - S1: los errores mostrados por un modelo pequeño son en buena parte nombres no resueltos — indicador de cambio: fracción de sugerencias erróneas que el resolutor rechaza — evidencia discriminante: ≥30% frente a <30% (umbral propuesto).
  - S2: los falsos descartes son raros — indicador de cambio: descartes sobre completados correctos — evidencia discriminante: ≤5% frente a >5% (umbral propuesto) en E1.
- Estado: PROPUESTA.

## 4. Alternativas

- I1-P1 «Esqueleto y receta CHISPA-100M»: cayó por refrito de [R01] y por una falsación de ~8.6 h sin umbral; residuo: el esqueleto 24×576 del producto en la Propuesta A.
- I1-P2 «Contexto simbólico ampliado»: cayó por la latencia de jedi y por una tabla de copia sin representación para nombres no vistos; residuo: nombres de paquetes instalados (Propuesta B) y descarte antes de mostrar (Propuesta C); la cabecera de firmas en el contexto pasa a preguntas abiertas.
- I2-P1 e I2-P2 no se adoptan tal como se escribieron (INESTABLE por pérdida sin definir y por puerta ambigua); sus mecanismos corregidos son las Propuestas B y C.
- RFC-CHISPA-1 P1 «Tabla generada»: retirada como apuesta por arte previo directo; residuo en la Propuesta B.

## 5. Riesgos, costes y preguntas abiertas

Experimentos asesinos, en orden ascendente de coste dentro de cada unidad. Ninguno se ejecutó en esta corrida.

- E1 — Falsos descartes del resolutor, en CPU: pasar `ast`+`symtable` sobre 10,000 líneas reales de 300 repositorios retenidos y contar cuántas líneas correctas rechaza. Mata la Propuesta C si rechaza >5% (umbral propuesto). Coste: 0.2 h CPU.
- E2 — Oportunidad de la tabla viva, en CPU sobre 500 repositorios retenidos: fracción de identificadores fuera de la ventana de 1,024 tokens que están entre los candidatos, separando repositorio y paquetes instalados. Mata la Propuesta B si es <5% o si los candidatos cubren <60% de esos nombres (umbrales propuestos). Coste: 1 h CPU.
- E3 — Puerta sin reentrenar: modelo base de 20M a 0.5B tokens, 4 candidatos por hueco del banco NKS, elegir el primero que resuelve frente a elegir el primero sin más. Mata la Propuesta C si la tasa de nombres resueltos sube <5 puntos absolutos o si la NKS no sube ≥0.5 (umbrales propuestos). Coste: 1 h GPU.
- E4 — Tabla viva a 20M, a igual FLOP: base, tabla viva desde tripletas y control con filas libres; acierto top-1 en identificadores ausentes de la ventana pero presentes en archivos hermanos, más NKS. Mata la Propuesta B si la ganancia sobre la base es <3 puntos o si la NLL fuera de tramos empeora más de 0.02 nats/token (umbrales propuestos). Coste: 1.5 h GPU.
- E5 — FIM de línea y Muon a 20M, diseño 2×2 a igual FLOP (AdamW o Muon; FIM estándar o FIM de línea), bpb y NKS con 2 semillas en el mejor brazo. Fija la receta del producto; mata el FIM de línea o Muon si no ganan por ≥3σ de semillas (umbral propuesto). Coste: 2 h GPU.
- E6 — Transferencia de forma: 24×576 frente a 12×768 (≈96M, estimación) a 0.5B tokens con la receta de E5, y el mismo par escalado a 20M. Mata la regla «decidir a 20M» de la Propuesta A si el orden del par se invierte entre escalas. Coste: 3.6 h GPU.

Costes y riesgos globales:
- Laboratorio: `{der: 1 + 1.5 + 2 + 3.6 = 8.1 h}` de RTX 4090 y 1.2 h de CPU, ~3 USD (estimación). Producto: ~20 h facturables, ~7 USD (estimación). En la RTX 4060 de portátil el producto no es viable: `{der: 16.6 h × 5 × 2.4 ≈ 200 h}` con un factor 5 de velocidad frente a la 4090 y un factor térmico de 2.4 (V0-conjetura).
- La validación del banco NKS (E5 de RFC-CHISPA-1) sigue vigente y se corre antes de usar la NKS para decidir.
- Toda la evidencia del mapa viene de resúmenes extraídos por WebFetch; ninguna tabla se leyó y ninguna cita se cotejó contra texto crudo.
- El recall del sondeo sellado fue 3/11 en la ola 1: el barrido barato no encontró solo la mayor parte del arte previo esperado.

Preguntas abiertas: la cabecera con firmas del repositorio en el contexto, que el mapa solo apoya con modelos grandes; la CNN Softmax de 2016, vista en el buscador pero no leída en texto completo; destilación o sintéticos hacia ≤100M (H8, G1); cuantización INT8/INT4 con el motor fijado y la latencia real en CPU; el uso del resolutor como recompensa de entrenamiento.

## 6. Referencias

Consultadas el 2026-11-08.

R01 Liu et al., MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases, 2024-02 — https://arxiv.org/abs/2402.14905
R03 Kim et al., Character-Aware Neural Language Models, 2015-08 — https://arxiv.org/abs/1508.06615
R06 Dagan et al., Getting the most out of your tokenizer for pre-training and domain adaptation, 2024-02 — https://arxiv.org/abs/2402.01035
R09 Deiseroth et al., T-FREE: Tokenizer-Free Generative LLMs via Sparse Representations for Memory-Efficient Embeddings, 2024-06 — https://arxiv.org/abs/2406.19223
R11 Li et al., Code Completion with Neural Attention and Pointer Networks, 2017-11 — https://arxiv.org/abs/1711.09573
R12 Agrawal et al., Guiding Language Models of Code with Global Context using Monitors, 2023-06 — https://arxiv.org/abs/2306.10763
R13 Zhang et al., RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation, 2023-03 — https://arxiv.org/abs/2303.12570
R15 Mündler et al., Type-Constrained Code Generation with Language Models, 2025-04 — https://arxiv.org/abs/2504.09246
R16 ContextModule: contexto del repositorio para completado de código, 2024-12 — https://arxiv.org/abs/2412.08063
R17 Zan et al., When Language Model Meets Private Library, 2022-10 — https://arxiv.org/abs/2210.17236
R18 Bhoopchand et al., Learning Python Code Suggestion with a Sparse Pointer Network, 2016-11 — https://arxiv.org/abs/1611.08307
R19 Toward Less Hidden Cost of Code Completion with Acceptance and Ranking Models, 2021-06 — https://arxiv.org/abs/2106.13928
R20 Dunay et al., Multi-line AI-assisted Code Authoring, 2024-02 — https://arxiv.org/abs/2402.04141
R21 Wen et al., Fantastic Pretraining Optimizers and Where to Find Them, 2025-09 — https://arxiv.org/abs/2509.02046
R23 Currículo y contexto para FIM en completado de código, 2024-12 — https://arxiv.org/abs/2412.16589
R24 Ajuste supervisado para FIM sin posprocesado, 2025-05 — https://arxiv.org/abs/2505.18789
R26 Liu et al., Muon is Scalable for LLM Training, 2025-02 — https://arxiv.org/abs/2502.16982
