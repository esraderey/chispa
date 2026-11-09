"""Compone RESULTADOS.md de E1–E5 desde los resúmenes crudos (ninguna cifra se escribe a mano).

Uso: python -m chispa.informe --res res --out RESULTADOS.md
"""
from __future__ import annotations

import argparse, json
from pathlib import Path


def _j(p):
    p = Path(p)
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def pct(x):
    return f"{x:.1%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); R = Path(a.res)
    e1, e3, s3, e2, e4, e5 = (_j(R / "e1/resumen.json"), _j(R / "base/resumen_e3.json"), _j(R / "base/sensibilidad.json"),
                              _j(R / "e2/resumen.json"), _j(R / "e4/resumen.json"), _j(R / "e5/resumen.json"))
    L = ["# Resultados E1–E5 de RFC-CHISPA-3", "",
         "Generado por `python -m chispa.informe` desde los resúmenes crudos de `res/`. Hipótesis y umbrales: `PREREGISTRO.md`; "
         "cambios y defectos corregidos: `DESVIACIONES.md`; predicciones previas: `PREDICCIONES.md`.", "",
         "## Veredictos", "",
         "| Exp. | Propuesta | Resultado principal | Umbral de muerte | Veredicto |", "|---|---|---|---|---|"]
    if e1:
        L.append(f"| E1 | P1 memoria (oportunidad) | {pct(e1['oportunidad'])} (IC90 {pct(e1['ic90'][0])}–{pct(e1['ic90'][1])}); "
                 f"E p95 {e1['E_p95']:.0f} | < 5 % o p95 > 20,000 | {e1['veredicto']} |")
    if e2:
        lat = ", ".join(f"{k} hilo(s) {v:.1f} ms" for k, v in e2["latencia_extra_p95_ms"].items()) or "pendiente"
        L.append(f"| E2 | P2 prueba del sufijo | AUROC {e2['auroc_base']:.3f} → {e2['auroc_delta']:.3f} ({e2['mejora']:+.4f}, IC90 "
                 f"{e2['mejora_ic90'][0]:+.4f} a {e2['mejora_ic90'][1]:+.4f}); latencia extra p95: {lat} | mejora < 0.02 o > 50 ms | {e2['veredicto']} |")
    if e3:
        L.append(f"| E3 | P3 monitor (origen) | invenciones {pct(e3['frac_invencion'])} (IC90 {pct(e3['frac_invencion_ic90'][0])}–"
                 f"{pct(e3['frac_invencion_ic90'][1])}) | < 30 % | {e3['veredicto_E3']} |")
    if e4:
        L.append(f"| E4 | P3 monitor | reducción de nombres indefinidos {pct(e4['reduccion_ul'])} (IC90 {pct(e4['reduccion_ul_ic90'][0])}–"
                 f"{pct(e4['reduccion_ul_ic90'][1])}); sesgo sin entrenar {pct(e4['reduccion_sesgo'])}; caída de EM {e4['caida_em']*100:+.2f} pt "
                 f"| < 25 %, IC toca 0, EM > 1 pt o sesgo ≥ 80 % | {e4['veredicto']} |")
    if e5:
        pa = e5.get("pareado_B_menos_A", {}); pc = e5.get("pareado_B_menos_C", {})
        lat = ", ".join(f"{k} {v:.0f} ms" for k, v in e5["latencia_p50_ms_4hilos"].items()) or "pendiente"
        L.append(f"| E5 | P1 memoria | EM A {pct(e5['em_A'])}, B {pct(e5['em_B'])}, C {pct(e5['em_C'])}; B − A {pa.get('dif', 0)*100:+.1f} pt "
                 f"(IC90 {pa.get('ic90', [0, 0])[0]*100:+.1f} a {pa.get('ic90', [0, 0])[1]*100:+.1f}); latencia p50 (4 hilos) {lat} "
                 f"| B − A < 3 pt o B más lento | {e5['veredicto_P1']} (semántica: {e5['semantica']}, B − C {pc.get('dif', 0)*100:+.1f} pt) |")
    L += ["", "## Detalle y sensibilidad", ""]
    if e1:
        L.append(f"- **E1.** {e1['repos']} repositorios retenidos, {e1['ocurrencias']:,} apariciones de identificadores. Solo con el "
                 f"repositorio (sin paquetes): {pct(e1['oportunidad_solo_repo'])}. Origen de la oportunidad: {e1['origen']}. "
                 "El umbral se supera con la estimación puntual, pero el intervalo lo toca.")
    if e3:
        L.append(f"- **E3.** Modelo base de 20M: el {pct(e3['tasa_undefined'])} de los completados introduce un nombre indefinido; "
                 f"primer identificador correcto {pct(e3['em_primer_id'])}; línea exacta {pct(e3['em_linea'])}. Clases: {e3['clases']}.")
    if s3:
        sa, sb = s3["a_sin_cadenas"], s3["b_pyflakes"]
        L.append(f"- **E3, sensibilidad (exploratoria, no cambia el veredicto).** Sin los {s3['cursores_en_cadenas']} cursores dentro de "
                 f"docstrings: invenciones {pct(sa['frac_invencion'])}. Solo pyflakes ({sb['completados_que_parsean']} completados que "
                 f"parsean): invenciones {pct(sb['frac_invencion'])}, clases {sb['clases']}. El veredicto no es robusto al detector; "
                 "en todos los análisis los nombres de otro archivo son minoría.")
    if e4:
        L.append(f"- **E4.** Tasa con MLE {pct(e4['tasa_mle'])}, con UL {pct(e4['tasa_ul'])}, con sesgo β sin entrenar {pct(e4['tasa_sesgo'])}. "
                 "El sesgo se aplicó al brazo MLE de la semilla 1337 y se compara con la media de las dos semillas MLE.")
    if e5:
        L.append(f"- **E5.** {e5.get('pareado_B_menos_A', {}).get('repos', '?')} repositorios retenidos aportan los objetivos: el intervalo "
                 "pareado de B − A es ancho y el experimento no tiene potencia para descartar un efecto de 3 puntos. El brazo B "
                 "elige la memoria en casi todas las posiciones objetivo; no se midió cuándo dispara fuera de ellas.")
    # Predicciones selladas antes de los resultados (PREDICCIONES.md, commit 6138e84)
    PRED = {"E1": ("oportunidad ≈ 10 % (6–18 %)", 0.75), "E2": ("AUROC +0.02 (−0.01 a +0.06)", 0.45),
            "E3": ("invenciones ≈ 55 % (35–70 %)", 0.70), "E4": ("reducción ≈ 30 % (10–50 %)", 0.40),
            "E5": ("B − A ≈ +3 pt (−2 a +10)", 0.50)}
    real = {"E1": (e1 and f"{pct(e1['oportunidad'])}", e1 and e1["veredicto"]),
            "E2": (e2 and f"{e2['mejora']:+.3f}", e2 and e2["veredicto"]),
            "E3": (e3 and f"{pct(e3['frac_invencion'])}", e3 and e3["veredicto_E3"]),
            "E4": (e4 and f"{pct(e4['reduccion_ul'])}", e4 and e4["veredicto"]),
            "E5": (e5 and f"{e5['B_menos_A']*100:+.1f} pt", e5 and e5["veredicto_P1"])}
    L += ["", "## Predicciones previas frente a resultados", "",
          "| Exp. | Predicción | P(sobrevive) predicha | Resultado | Veredicto |", "|---|---|---|---|---|"]
    for k, (p, prob) in PRED.items():
        r, v = real[k]
        L.append(f"| {k} | {p} | {prob:.2f} | {r} | {v} |")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
