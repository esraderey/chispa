# Predicciones del director antes de ver resultados

**Escritas el 2026-11-09, con el B20 entrenándose y sin ningún resultado de E1–E5 disponible.** Son creencias previas,
no datos; sirven para medir después cuánto se movieron. Probabilidad = de que la propuesta sobreviva su experimento.

| Exp. | Predicción central | Rango plausible | P(sobrevive) |
|---|---|---|---|
| E1 | oportunidad ≈ 10% de las apariciones de identificadores; E p95 ≈ 3,000 | 6–18%; 1,000–10,000 | 0.75 |
| E3 | invenciones ≈ 55% de los nombres indefinidos nuevos (resto: otro archivo ~30%, local hermano ~15%) | 35–70% | 0.70 |
| E2 | mejora de AUROC ≈ +0.02 con Δ; latencia extra p95 ≈ 30 ms con 4 hilos | −0.01 a +0.06; 15–60 ms | 0.45 |
| E4 | reducción de Undefined Name ≈ 30% con UL; el sesgo sin entrenar logra ≈ 60–90% de eso | 10–50% | 0.40 |
| E5 | B − A ≈ +3 puntos de EM en identificadores de otro archivo; B − C ≈ +1 punto; B más rápido que A en CPU | −2 a +10; −2 a +4 | 0.50 |

Descriptivos del B20 (20M, 0.5B tokens): completados con ≥1 nombre indefinido nuevo ≈ 8–15% (frente al ~4.5% de
CodeGen-350M); coincidencia exacta del primer identificador ≈ 35–50%; coincidencia exacta de la línea ≈ 10–20%.

Razones principales: E1 y E3 dependen de la estructura del código y no del modelo, y la literatura apunta a que los
identificadores del proyecto son el punto débil de los modelos pequeños. E2 y E4 tienen controles fuertes (el log-prob
medio y el sesgo sin entrenar) que pueden igualarlos. E5 se juega en la calibración: B entrena contra ≤128 negativos del
lote e infiere contra todo E.
