# Resultados E1–E5 de RFC-CHISPA-3

Generado por `python -m chispa.informe` desde los resúmenes crudos de `res/`. Hipótesis y umbrales: `PREREGISTRO.md`; cambios y defectos corregidos: `DESVIACIONES.md`; predicciones previas: `PREDICCIONES.md`.

## Veredictos

| Exp. | Propuesta | Resultado principal | Umbral de muerte | Veredicto |
|---|---|---|---|---|
| E1 | P1 memoria (oportunidad) | 5.3% (IC90 4.9%–5.7%); E p95 2495 | < 5 % o p95 > 20,000 | SOBREVIVE |
| E2 | P2 prueba del sufijo | AUROC 0.727 → 0.791 (+0.0636, IC90 +0.0544 a +0.0718); latencia extra p95: 1 hilo(s) 54.0 ms, 4 hilo(s) 44.5 ms | mejora < 0.02 o > 50 ms | SOBREVIVE |
| E3 | P3 monitor (origen) | invenciones 53.5% (IC90 49.9%–57.2%) | < 30 % | SOBREVIVE |
| E4 | P3 monitor | reducción de nombres indefinidos 5.0% (IC90 2.8%–7.0%); sesgo sin entrenar 11.8%; caída de EM +0.67 pt | < 25 %, IC toca 0, EM > 1 pt o sesgo ≥ 80 % | MUERE |
| E5 | P1 memoria | EM A 24.4%, B 25.6%, C 17.1%; B − A +1.2 pt (IC90 -0.3 a +9.1); latencia p50 (4 hilos) A 197 ms, B 106 ms, C 136 ms | B − A < 3 pt o B más lento | MUERE (semántica: IMPORTA, B − C +8.5 pt) |

## Detalle y sensibilidad

- **E1.** 472 repositorios retenidos, 1,117,037 apariciones de identificadores. Solo con el repositorio (sin paquetes): 4.2%. Origen de la oportunidad: {'archivo': 7551, 'repo': 4116, 'paquete': 2553, 'modulo': 11533, 'miembro': 33295}. El umbral se supera con la estimación puntual, pero el intervalo lo toca.
- **E3.** Modelo base de 20M: el 11.0% de los completados introduce un nombre indefinido; primer identificador correcto 17.1%; línea exacta 1.2%. Clases: {'invencion': 1945, 'local_hermano': 1397, 'otro_archivo': 291}.
- **E3, sensibilidad (exploratoria, no cambia el veredicto).** Sin los 1446 cursores dentro de docstrings: invenciones 44.2%. Solo pyflakes (11633 completados que parsean): invenciones 27.9%, clases {'invencion': 536, 'local_hermano': 1265, 'otro_archivo': 121}. El veredicto no es robusto al detector; en todos los análisis los nombres de otro archivo son minoría.
- **E4.** Tasa con MLE 11.9%, con UL 11.3%, con sesgo β sin entrenar 10.5%. El sesgo se aplicó al brazo MLE de la semilla 1337 y se compara con la media de las dos semillas MLE.
- **E5.** 26 repositorios retenidos aportan los objetivos: el intervalo pareado de B − A es ancho y el experimento no tiene potencia para descartar un efecto de 3 puntos. El brazo B elige la memoria en casi todas las posiciones objetivo; no se midió cuándo dispara fuera de ellas.

## Predicciones previas frente a resultados

| Exp. | Predicción | P(sobrevive) predicha | Resultado | Veredicto |
|---|---|---|---|---|
| E1 | oportunidad ≈ 10 % (6–18 %) | 0.75 | 5.3% | SOBREVIVE |
| E2 | AUROC +0.02 (−0.01 a +0.06) | 0.45 | +0.064 | SOBREVIVE |
| E3 | invenciones ≈ 55 % (35–70 %) | 0.70 | 53.5% | SOBREVIVE |
| E4 | reducción ≈ 30 % (10–50 %) | 0.40 | 5.0% | MUERE |
| E5 | B − A ≈ +3 pt (−2 a +10) | 0.50 | +1.2 pt | MUERE |
