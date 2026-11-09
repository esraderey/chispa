# RFC-CHISPA-1 — CHISPA-20M: un modelo de código de 19.2M parámetros entrenable en una RTX 4060

- Estado: PROPUESTO
- Fecha: 2026-11-08
- Firmado: ESRADEREY
- Expedición: ¿cómo diseñar un modelo de lenguaje para código de ~20M parámetros, entrenable desde cero en una GPU de consumo (RTX 4060, 8 GB), que maximice la ayuda útil en programación para su tamaño?
- Método: expedición mediana; 4 exploradores Sonnet y 1 lector de ola 2; 115 anclas en 46 obras, todas leídas por extracto; verificación ciega por fuente de las 46 obras (fuentes defectuosas: C 0/6, F 0/32, R 1/7; cotas al 90 %: 31.9 %, 6.9 % y 45.3 %); canarios de verificación 4/4; sondeo sellado con recall 6/13; 2 innovadores Opus, 6 propuestas, 21 objeciones ancladas (0 inválidas; incoherencias sembradas detectadas 2/2); 3 supervivientes; 0 experimentos ejecutados por restricción de presupuesto.

## 1. Resumen

Se propone CHISPA-20M, un decodificador de 19.2M parámetros para completar Python dentro del editor, entrenable en una RTX 4060 en una noche (estimación). Tres decisiones lo separan de un GPT diminuto. Primera: la tabla de embeddings no se almacena, se genera a partir de los bytes de cada token, de modo que un vocabulario de 32,768 cuesta ~1.6M parámetros y el resto se invierte en profundidad. Segunda: una cabeza de copia de 0.3M consulta una tabla simbólica con los nombres del repositorio y copia identificadores completos, porque 20M no alcanza para memorizar un proyecto. Tercera: el modelo se juzga con un banco offline de pulsaciones netas ahorradas y abstención aprendida, porque a esta escala HumanEval no mide nada útil. Nada está ejecutado: seis experimentos de ≤4 h cada uno pueden matar el diseño, empezando por medir el rendimiento real de la GPU.

## 2. Estado del arte

Los modelos de código diminutos publicados rinden poco en generación de funciones completas: tiny_starcoder_py, de 164M, declara 7.84% pass@1 en HumanEval [R18], y TinyCodeLM de 150M obtiene 12.8 pass@1 [R44]. Wisp Coder 110M reconoce que alucina con confianza en APIs desconocidas y no midió su latencia [R40]. El precedente útil es otro: el completado local de JetBrains usa un modelo tipo LLaMA de 100M con vocabulario de 16,384 y contexto de 1,536 [R45], y en producción alcanza una aceptación de cerca de 38% en Python con inferencia de 50 ms [R45].

En la forma del modelo, entre 70M y 125M las variantes con 30 o más capas superan a las de 12 [R01] [R02], y un barrido byte-level elige profundidad 24 a 20M [R09]. La profundidad tiene un precio: en modelos diminutos mejora el rendimiento a costa de velocidad de inferencia [R13], y los modelos más delgados gastan más de 200% de las horas GPU de la mejor configuración [R14]. A esta escala el embedding y la cabeza pueden suponer del 45.20% al 62.25% de los parámetros [R07]; los vocabularios grandes degradan el rendimiento más allá de cierto umbral [R06], y quitar los embeddings factorizados perjudica de forma sustancial [R04]. A 1.5B, en cambio, el tamaño de vocabulario entre 32k y 256k cambia poco el resultado en HumanEval [R23]. Reutilizar capas compra profundidad sin parámetros: un modelo recurrente de 12.18M se acerca al objetivo de entrenamiento de uno no recurrente de 29.9M, aunque en BabyLM y sin igualar FLOPs [R03], y sus variantes con bucles quedan por debajo en seguimiento de entidades [R03].

En datos, filtrar el corpus y añadir libros sintéticos sube un modelo de 350M de 12.19% a 20.12% en HumanEval [R15]; el filtrado educativo Stack-Edu lleva un modelo de 1.7B de 20.7 a 25.6 en Python [R19]. Hasta 4 épocas de datos repetidos cambian la pérdida de forma insignificante [R16], y cualquier tasa FIM entre 50% y 90% es razonable [R17]. En entrenamiento, con baselines afinados la ventaja de Muon sobre AdamW es de 1.4× a 0.1B y baja a 1.1× a 1.2B [R27]; el speedrun de referencia se mide con 8xH100 [R26]. La destilación tiene una brecha de capacidad, porque un maestro más fuerte puede producir un estudiante peor [R29], y en código con pocos datos la corrección de las salidas no cambió el resultado [R32].

En inferencia, RepoCoder mejora en más de 10% el completado que solo ve el archivo [R38], y la recuperación de código del dominio desacoplada del modelo mejora el completado sin ajuste fino [R42] [R41]. Memory Decoder añade 1.28× de sobrecarga frente a 2.17× de kNN-LM [R39]. La restricción por tipos reduce los errores de compilación en más de la mitad, medida con modelos grandes [R36]. Un modelo de aceptación sobre un GPT-2 de 108M decide si mostrar cada completado y baja las listas inválidas de 55.09% a 17.44% [R46]. La capacidad de conocimiento ronda los 2 bits por parámetro con y sin bucles [R11].

{obs} Por debajo de 100M el barrido no localizó ningún resultado de código con métrica funcional ni con aceptación en editor. {obs} Los huecos que este RFC ataca tienen grado G2 en el mapa y se redactan como cota del barrido, no como inexistencia:

H2 — Ley de forma a 5-30M con BPE y corpus de código.

H4 — Reparto entre embeddings y bloques a 20M con vocabularios BPE de distinto tamaño.

H6 — Tamaño de vocabulario en modelos de código ≤100M.

H13 — Memoria externa en modelos de código <100M.

H15 — Latencia, tamaño y aceptación de completado local <100M.

H16 — Métrica validada para modelos de código de 20M.

## 3. Propuesta(s)

### Diseño integrado CHISPA-20M (regla propuesta por el director)

- Estado: PROPUESTA.
- Supuestos eje:
  - S1: componer P1 y P2 no anula sus ganancias — indicador de cambio: la ventaja de la cabeza de copia medida sobre la tabla generada frente a la medida sobre la tabla libre — evidencia discriminante: E4 repetido con la variante C de E6.

Las tres propuestas supervivientes se componen sobre un mismo esqueleto. Todas las cifras de esta tabla son nuestras:

| Pieza | Parámetros |
|---|---|
| 11 bloques B384 (d=384, 6 cabezas de 64, 2 cabezas kv, SwiGLU 1,024, RMSNorm, RoPE) | `{der: 11 × 1,573,632 = 17,309,952}` |
| Compositor de la tabla generada (P1) | `{der: 413,184}` |
| Residuo libre para los 3,072 tokens más frecuentes (P1) | `{der: 3,072 × 384 = 1,179,648}` |
| Norma final | `{der: 384}` |
| Cabeza de copia (P2) | `{der: 2 × 384² + 8 × 384 + 385 = 298,369}` |
| Cabeza de abstención (P3) | `{der: 388 × 64 + 64 + 64 + 1 = 24,961}` |
| **Total** | `{der: 19,226,498}` |

- Datos (regla propuesta): Python monolingüe filtrado con un clasificador educativo de umbral 3 al estilo de [R19], deduplicado y separado por repositorio para validar; ~400M tokens únicos × 3 épocas = 1.2B tokens (estimación, dentro de lo que [R16] considera casi inocuo); FIM 50%, dentro del rango de [R17].
- Tokenizador (regla propuesta): BPE byte-level de 32,768 con una regex de pretokenización alineada a los lexemas de Python: ninguna pieza cruza la frontera entre identificador, número, operador, delimitador o comilla, y las corridas de salto de línea más sangría son piezas propias. Es condición de P2 y de P3.
- Receta (estimación): contexto 1,024; lote de 131,072 tokens con microlote 16×1,024; AdamW con calendario WSD; bf16, atención fusionada y `torch.compile`. Muon queda como ablación, no como apuesta, porque [R27] mide su ventaja solo desde 0.1B.
- Cómputo (estimación): `{der: 6 × 17.31e6 + 6 × 32,768 × 384 + 11 × 12 × 512 × 384 ≈ 2.05e8 FLOP/token}`; con 1e13 FLOP/s efectivos (V0-conjetura que E1 mide) salen `{der: 1e13 / 2.05e8 ≈ 48,700 tokens/s}` y la corrida de 1.2B tokens dura `{der: 1.2e9 / 48,700 / 3,600 ≈ 6.8 h}`. VRAM estimada entre 5 y 6 GB.
- Inferencia (estimación): la tabla se materializa una vez al cargar (`{der: 32,768 × 384 × 2 B ≈ 25.2 MB}` en fp16), las claves de la cabeza de copia se recalculan solo cuando cambia la tabla de nombres y la caché KV se reutiliza entre pulsaciones.

### Propuesta P1 (I1-P1) — Tabla generada: embeddings compuestos desde los bytes del token

- Casilla: H4, H6, H2.
- Mecanismo: la matriz E (V×d) no se guarda; un compositor la genera en cada paso desde los bytes de cada token (byte-level BPE, máximo 16 bytes): embedding de byte de dimensión 128 más posición, una convolución residual de núcleo 3, agregación por media y máximo, un MLP 256→512→384 y RMSNorm. A los 3,072 tokens más frecuentes se les suma un residuo libre inicializado en cero. E se ata a la cabeza de salida (logits = h·Eᵀ), así que todo gradiente de salida entrena el compositor compartido y los tokens raros, sobre todo los fragmentos de identificadores, aprenden de su ortografía. Con V=32,768 el vocabulario cuesta `{der: 413,184 + 1,179,648 ≈ 1.59M}` frente a `{der: 32,768 × 384 ≈ 12.58M}` de una tabla libre, y la diferencia se invierte en profundidad, en la dirección que favorecen [R01] [R02]. El código de referencia del compositor está en `propuestas/I1.md`.
- Novedad frente al mapa: el mapa registra la fracción de parámetros [R07], la degradación de vocabularios grandes [R06] y el daño de quitar embeddings factorizados [R04], siempre con tablas libres o factorizadas por rango. Aquí el coste en parámetros del vocabulario se vuelve casi constante para cualquier V, y la pregunta de H4 y H6 deja de ser de parámetros para ser de cómputo y calidad. Fuera del mapa existen embeddings compuestos desde caracteres (V0-conjetura, no verificado en esta corrida): la novedad reclamada es su uso como palanca de reparto a 20M en código con BPE, con residuo libre y control de ortografía barajada.
- Coste (estimación): sobrecoste del compositor `{der: ≈ 2.2e11 / 3.5e12 ≈ 6%}` del cómputo de un microlote; la cabeza de 32k se queda con `{der: 75.5 / 216 ≈ 35%}` del cómputo por token en la variante de 12 bloques.
- Riesgos: (1) a igual cómputo, un vocabulario menor puede ganar aunque la tabla sea gratis en parámetros, por el coste de la cabeza; (2) tokens con ortografía parecida y significado distinto (`in`/`int`) se acoplan; (3) los tokens de más de 16 bytes se truncan; (4) si la ventaja nace de que la cola está subentrenada, puede encoger con más tokens (objeción I1-P1/1 de la crítica).
- Falsación: E6.
- Supuestos eje:
  - S1: con 20M y ~4B tokens, los embeddings libres de la cola quedan subentrenados — indicador de cambio: la pérdida de B (tabla libre de 32k) en la cubeta de rango >16k deja de ser peor que la de C — evidencia discriminante: pérdida por cubeta de frecuencia en B frente a C a igual número de bytes vistos.
  - S2: la ortografía de un token de código predice su rol — indicador de cambio: el brazo con bytes barajados entre tokens iguala al brazo real — evidencia discriminante: brazo E de E6.
  - S3: la profundidad comprada vale más que la capacidad léxica perdida — indicador de cambio: el orden de los brazos en bpb a igual tiempo — evidencia discriminante: brazos A, B y C de E6.
- Estado: PROPUESTA.

### Propuesta P2 (I2-P2) — Tabla de nombres del repositorio con copia por puntero de identificador completo

- Casilla: H13, H15.
- Mecanismo: un indexador basado en `ast` mantiene por repositorio la tabla de nombres definidos fuera de la ventana: funciones, clases y métodos de nivel superior, atributos `self.x`, constantes de módulo y nombres de los módulos que el archivo importa; máximo 512 entradas priorizadas, cada una en dos variantes («nombre» y «␣nombre»). La cabeza de copia calcula claves desde la media de los embeddings de las piezas del nombre más un embedding de tipo y de espacio, una consulta desde el estado final y una compuerta; el modelo elige entre emitir una pieza o copiar un nombre completo. La pérdida suma exactamente las dos derivaciones de cada tramo copiable con logsumexp. Corrección de la crítica (objeción I2-P2/2): tras una copia se prohíbe continuar con una pieza de identificador, condición que hace exacta la marginal por tramo. En el editor, la tabla se recalcula al guardar y se copia el nombre completo cuando la probabilidad de copia supera 0.5 (umbral propuesto).
- Novedad frente al mapa: el vecino de H13 es la recuperación desacoplada sin ajuste fino [R42] [R41]; [R39] usa una memoria paramétrica de 124M, mayor que todo este modelo, y [R38] mete fragmentos en el contexto. Aquí la memoria es simbólica, la produce un indexador determinista, no gasta ventana, se aprende desde el preentrenamiento y cuesta `{der: 298,369 / 19,226,498 ≈ 1.6%}` de los parámetros. Con unos 2 bits por parámetro [R11], un modelo de 19.2M guarda del orden de `{der: 2 × 19.2e6 / 8 ≈ 4.8 MB}` (estimación): no puede memorizar un proyecto. Alcance corregido (objeción I2-P2/1): cubre los nombres del repositorio, no las APIs de terceros que [R40] alucina. Fuera del mapa existen redes de copia y de punteros para completado de código (V0-conjetura, no verificado).
- Coste (estimación): la atención a la tabla añade `{der: 3 × 2 × 1,024 × 384 ≈ 2.4e6 FLOP/token}`, ~1% del cómputo; preproceso en CPU de ~2 h con 8 núcleos.
- Riesgos: (1) el filtrado puede romper la estructura por repositorio y dejar pocas oportunidades de copia; (2) la compuerta puede colapsar a cero al principio (se mitiga con sesgo inicial −2); (3) nombres dinámicos (`getattr`, `__all__` generado) quedan fuera.
- Falsación: E2 y E4.
- Supuestos eje:
  - S1: el corpus conserva estructura de repositorio — indicador de cambio: fracción de archivos de entrenamiento con al menos un hermano — evidencia discriminante: ≥30% frente a <30% (umbral propuesto).
  - S2: la ganancia viene de consultar la tabla por contexto y no de copiar nombres frecuentes — indicador de cambio: una cabeza con consulta constante iguala a la cabeza contextual — evidencia discriminante: brazo de consulta constante de E4 (añadido por la objeción I2-P2/3).
  - S3: un bono sin entrenamiento no basta — indicador de cambio: brecha entre la copia entrenada y el bono gratuito — evidencia discriminante: ≥5 puntos de EM frente a <5 (umbral propuesto); si el bono basta, se adopta el bono y se retira la cabeza.
- Estado: PROPUESTA.

### Propuesta P3 (I2-P3, reducida) — Banco NKS: pulsaciones netas ahorradas con usuario simulado y abstención aprendida

- Casilla: H16, H15.
- Mecanismo: se muestrean 10,000 líneas de 300 repositorios excluidos del entrenamiento. En cada frontera de pretoken el sistema propone el resto de la línea o se abstiene; el usuario simulado acepta con Tab si la sugerencia es prefijo exacto del resto verdadero y ahorra su longitud menos una pulsación; una sugerencia errónea mostrada cuesta c pulsaciones. NKS = (ahorro − c·errores) / caracteres, con c = 1 como cifra principal (umbral propuesto) e intervalo de confianza del 90% por bootstrap de repositorios; la definición exacta está en `nks.py` (`propuestas/I2.md`). La cabeza de abstención (24,961 parámetros, modelo base congelado) recibe el estado del cursor y cuatro rasgos de la sugerencia voraz, y se entrena con etiquetas on-policy: 1 si el propio modelo habría acertado. Residuo adoptado de I1-P3: el banco suma el completado de asserts en repositorios no vistos como segunda métrica funcional, con el control de valores permutados.
- Validez (corregida por la crítica): resolución con 3 semillas y checkpoints decaídos, no intermedios de WSD; «orden por construcción» solo para la cadena de checkpoints del mismo modelo y para el par de tamaño a igual número de tokens; el 4-grama se reporta como expectativa, no como prueba.
- Novedad frente al mapa: un filtro de aceptación aprendido ya existe a 108M con registros reales [R46], y [R45] y [R21] miden aceptación con usuarios reales. Lo nuevo es el instrumento offline, determinista y barato, con su validez demostrada a 20M y etiquetas on-policy; el EM y el pass@k [R43] [R44] no castigan mostrar sugerencias erróneas, y la NKS sí.
- Retirado de esta versión: la rama INT4/QAT y la medida de latencia, porque no fijaba el motor de CPU (objeción I2-P3/4); pasa a §5.
- Coste (estimación): cabeza de abstención ≈ 0.3 h de GPU; banco memorizado por prefijo ≈ 0.15 h de GPU por sistema.
- Riesgos: (1) el usuario simulado valida el orden interno, no la cifra absoluta frente a [R45]; (2) los casi-duplicados entre entrenamiento y prueba inflan la NKS si no se deduplican; (3) la cabeza puede aprender a abstenerse siempre, lo que da NKS = 0.
- Falsación: E5.
- Supuestos eje:
  - S1: un usuario simulado determinista ordena sistemas como su calidad real — indicador de cambio: pares concordantes en la cadena de checkpoints y en el par de tamaño — evidencia discriminante: todos concordantes frente a alguno discordante.
  - S2: la NKS resuelve diferencias del tamaño que producen P1 y P2 — indicador de cambio: razón entre esas diferencias y σ de 3 semillas — evidencia discriminante: ≥3 frente a <3 (umbral propuesto).
  - S3: a 20M callar vale tanto como acertar — indicador de cambio: NKS sin abstención a c = 1 — evidencia discriminante: si es negativa y la abstención la vuelve positiva, la cabeza es pieza central del uso.
- Estado: PROPUESTA.

## 4. Alternativas

- I1-P2 «Bucle que crece» (bloques compartidos cuyas pasadas crecen durante el entrenamiento sin cambiar la función; cifras de la propuesta, no del mapa): cayó por atacar un hueco de grado G1 y por pagar ~3× de prellenado en un completador local; residuo: E1 mide tokens/s con 1 y con 4 pasadas, y la perilla de salidas tempranas queda abierta.
- I1-P3 «El intérprete como maestro» (asserts con valores obtenidos al ejecutar funciones del corpus): cayó porque su mecanismo semántico choca con [R32] y su prueba principal se gana con la marginal de valores; residuo: la métrica de completado de asserts entra en P3.
- I2-P1 «Molde léxico» (softmax renormalizada por un autómata léxico de 76 estados): cayó por prometer garantías que un cierre empírico no da (sangría inexpresable, palabras clave blandas como `match`); residuo: la regex de pretokenización alineada a lexemas y E3.
- Rama INT4/QAT/latencia de I2-P3: retirada por no fijar el motor de inferencia; pasa a preguntas abiertas.

## 5. Riesgos, costes y preguntas abiertas

Experimentos asesinos, en orden ascendente de coste (horas de RTX 4060 salvo donde se indica CPU). Ninguno se ejecutó en esta corrida.

- E1 — Rendimiento real de la GPU. Medir tokens/s y VRAM pico de CHISPA-20M, del control de 11 bloques con tabla libre de 8,192 y de una variante con 4 pasadas compartidas, 10 minutos cada uno. Mata todas las estimaciones de coste si el rendimiento queda por debajo de la mitad de 48,700 tokens/s (umbral propuesto); en ese caso se reescalan los presupuestos de tokens. Coste: 0.5 h.
- E2 — Oportunidad de copia, en CPU sobre 500 repositorios excluidos: fracción de identificadores ausentes de la ventana pero presentes en la tabla priorizada. Mata P2 si es <5% o si la tabla cubre <60% de los nombres fuera de ventana (umbrales propuestos). Coste: 1 h.
- E3 — Bits retirados por el autómata léxico, en CPU: violación fuera de muestra y bits medios retirados por token. Decide si una máscara léxica blanda merece entrar en la decodificación; mata la idea si retira <0.5 bits/token (umbral propuesto). Coste: 1 h.
- E4 — Copia contra bono gratuito: brazos de 200M tokens con el modelo base, con cabeza de copia contextual, con cabeza de consulta constante y con bono sin entrenar; EM del nombre completo en posiciones cuyo nombre no está en la ventana; control con tablas de otro repositorio. Mata P2 si la ventaja sobre el mejor rival es <5 puntos de EM o si la NLL fuera de tramos empeora más de 0.02 nats/token (umbrales propuestos). Coste: 2.5 h.
- E5 — Validez del banco NKS: 3 semillas del modelo base a 200M tokens con checkpoints decaídos a 50M, 100M y 200M, más el modelo de 6 capas; abstención aprendida frente a abstención por log-prob medio. Mata la NKS si algún par de orden por construcción sale discordante, si la diferencia 100M→200M es <3σ o si su orden coincide siempre con el de la NLL; mata la cabeza si gana <0.5 puntos de NKS (umbrales propuestos). Coste: 3.9 h.
- E6 — Tabla generada contra tablas libres, 45 min por brazo a igual tiempo de reloj: A (8,192 libre, 11 bloques), B (32,768 libre, 5 bloques), C (tabla generada, 12 bloques), E (como C con bytes barajados entre tokens) y una segunda semilla de A. Mata P1 si bpb(C) no queda por debajo de mín(bpb(A), bpb(B)) por al menos máx(1%, 2× el ruido entre semillas), o si E iguala a C (umbrales propuestos); si sobrevive, un brazo largo B/C a doble número de tokens mide si la ventaja encoge. Coste: 3.9 h.

Costes y riesgos globales:
- Total de matanza estimado: `{der: 0.5 + 2.5 + 3.9 + 3.9 = 10.8 h}` de GPU y 2 h de CPU, menos si E4, E5 y E6 comparten los brazos base. La corrida completa de CHISPA-20M se estima en ~6.8 h para 1.2B tokens.
- Toda la evidencia del mapa pasó por extractos de WebFetch mediados por modelo; ninguna cita se cotejó contra texto crudo.
- La novedad de P1 y P2 no se contrastó fuera del mapa: no hubo explorador de novedad en esta corrida mediana.
- El rendimiento efectivo de la RTX 4060 es una V0-conjetura compartida por todas las horas de este documento.

Preguntas abiertas: Muon frente a AdamW por debajo de la escala mínima medida en [R27]; bucles y recurrencia en profundidad medidos con código; QAT y cuantización INT4 con un motor de inferencia fijado (por ejemplo, el que el usuario vaya a usar en el editor); alternativas SSM o híbridas a 20M con tokens igualados; destilación desde un maestro grande hacia 20M, que el mapa deja sin cifras.

## 6. Referencias

Consultadas el 2026-11-08.

R01 Liu et al., MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases, 2024-02 — https://arxiv.org/pdf/2402.14905
R02 codelion, Optimal model architecture (blog), 2025-12 — https://huggingface.co/blog/codelion/optimal-model-architecture
R03 Looped GPT-BERT: Trading Parameters for Computation in Small Language Modeling, 2026-09 — https://arxiv.org/pdf/2609.09691
R04 RecursiveGPT, 2026-08 — https://arxiv.org/pdf/2608.26973
R06 SCONE, 2025-02 — https://arxiv.org/pdf/2502.01637v1
R07 Super Tiny Language Models, 2024-05 — https://arxiv.org/pdf/2405.14159v1
R09 Barrido byte-level de profundidad hasta 50M, 2026-05 — https://arxiv.org/pdf/2605.27989
R11 Ouro: Scaling latent reasoning via looped language models, 2025-10 — https://arxiv.org/html/2510.25741v2
R13 PanGu-π Pro: Rethinking Optimization and Architecture for Tiny Language Models, 2024-02 — https://arxiv.org/abs/2402.02791
R14 Gemstones: A Model Suite for Multi-Faceted Scaling Laws, 2025-02 — https://arxiv.org/pdf/2502.06857
R15 Gunasekar et al., Textbooks Are All You Need, 2023-06 — https://arxiv.org/pdf/2306.11644
R16 Muennighoff et al., Scaling Data-Constrained Language Models, 2023-05 — https://arxiv.org/abs/2305.16264
R17 Bavarian et al., Efficient Training of Language Models to Fill in the Middle, 2022-07 — https://arxiv.org/pdf/2207.14255
R18 BigCode, tiny_starcoder_py (tarjeta del modelo), 2023 — https://huggingface.co/bigcode/tiny_starcoder_py
R19 SmolLM2, 2025-02 — https://arxiv.org/pdf/2502.02737
R21 Currículo y contexto para completado de código (CMFT), 2024-12 — https://arxiv.org/html/2412.16589v1
R23 Getting the most out of your tokenizer for pre-training and domain adaptation, 2024-02 — https://arxiv.org/pdf/2402.01035
R26 Jordan et al., modded-nanogpt (repositorio), 2026-08 — https://github.com/KellerJordan/modded-nanogpt
R27 Fantastic Pretraining Optimizers and Where to Find Them, 2025-09 — https://arxiv.org/abs/2509.02046
R29 Distillation Scaling Laws, 2025-02 — https://arxiv.org/pdf/2502.08606
R32 The Valley of Code Reasoning: Scaling Knowledge Distillation, 2025-10 — https://arxiv.org/abs/2510.06101
R36 Type-Constrained Code Generation, 2025-04 — https://arxiv.org/abs/2504.09246
R38 RepoCoder, 2023-03 — https://arxiv.org/abs/2303.12570
R39 Memory Decoder, 2025-08 — https://arxiv.org/html/2508.09874v1
R40 Wisp Coder 110M (tarjeta del modelo), s.f. — https://huggingface.co/philipjohnbasile/wisp-coder-110m
R41 FT2Ra, 2024-04 — https://arxiv.org/abs/2404.01554
R42 kNM-LM, 2023-08 — https://arxiv.org/abs/2308.09313
R43 Large Language Monkeys: Scaling Inference Compute with Repeated Sampling, 2024-07 — https://arxiv.org/abs/2407.21787
R44 Training Language Models on Synthetic Edit Sequences (LintSeq, TinyCodeLM), 2024-10 — https://arxiv.org/html/2410.02749v1
R45 Full Line Code Completion: Bringing AI to Desktop, 2024-05 — https://arxiv.org/abs/2405.08704
R46 Toward Less Hidden Cost of Code Completion with Acceptance and Ranking Models, 2021-06 — https://arxiv.org/abs/2106.13928
