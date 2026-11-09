"""Compone el informe de E1–E5 desde los resúmenes crudos (nunca a mano).

Uso: python -m chispa.informe --res RES --out RESULTADOS.md
"""
from __future__ import annotations

import argparse, json
from pathlib import Path


def _j(p):
    p = Path(p)
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); R = Path(a.res)
    e1, e3, e2, e4, e5 = _j(R / "e1/resumen.json"), _j(R / "base/resumen_e3.json"), _j(R / "e2/resumen.json"), \
        _j(R / "e4/resumen.json"), _j(R / "e5/resumen.json")
    L = ["# Resultados E1–E5 de RFC-CHISPA-3", "",
         "Generado por `python -m chispa.informe` desde los resúmenes crudos de `res/`. Umbrales: PREREGISTRO.md; cambios: DESVIACIONES.md.", "",
         "| Exp. | Propuesta | Métrica principal | Umbral de muerte | Resultado | Veredicto | Artefacto |", "|---|---|---|---|---|---|---|"]
    if e1:
        L.append(f"| E1 | P1 memoria | oportunidad (IC90) · E p95 | <5% o p95>20,000 | {e1['oportunidad']:.2%} ({e1['ic90'][0]:.2%}–{e1['ic90'][1]:.2%}) · "
                 f"{e1['E_p95']:.0f} | {e1['veredicto']} | res/e1/resumen.json |")
    if e2:
        L.append(f"| E2 | P2 sufijo | mejora de AUROC (IC90) · latencia p95 | <0.02 o >50 ms | {e2['mejora']:+.4f} "
                 f"({e2['mejora_ic90'][0]:+.4f}–{e2['mejora_ic90'][1]:+.4f}) · {e2['latencia_extra_p95_ms'] or 'pendiente'} | {e2['veredicto']} | res/e2/resumen.json |")
    if e3:
        L.append(f"| E3 | P3 monitor | fracción de invenciones (IC90) | <30% | {e3['frac_invencion']:.2%} "
                 f"({e3['frac_invencion_ic90'][0]:.2%}–{e3['frac_invencion_ic90'][1]:.2%}) | {e3['veredicto_E3']} | res/base/resumen_e3.json |")
    if e4:
        L.append(f"| E4 | P3 monitor | reducción de Undefined Name (IC90) · caída EM · sesgo | <25%, IC toca 0, EM>1 pt, sesgo≥80% | "
                 f"{e4['reduccion_ul']:.2%} ({e4['reduccion_ul_ic90'][0]:.2%}–{e4['reduccion_ul_ic90'][1]:.2%}) · {e4['caida_em']:+.4f} · "
                 f"{e4['reduccion_sesgo']:.2%} | {e4['veredicto']} | res/e4/resumen.json |")
    if e5:
        L.append(f"| E5 | P1 memoria | EM B−A · B−C · latencia | B−A<3 pt o latencia B>A; B−C<1 pt | {e5['B_menos_A']:+.4f} · "
                 f"{e5['B_menos_C']:+.4f} · {e5['latencia_p50_ms_4hilos'] or 'pendiente'} | {e5['veredicto_P1']} / semántica {e5['semantica']} | res/e5/resumen.json |")
    L += ["", "## Descriptivos", ""]
    if e3:
        L.append(f"- Base B20: tasa de completados con ≥1 nombre indefinido nuevo {e3['tasa_undefined']:.2%} "
                 f"(IC90 {e3['tasa_ic90'][0]:.2%}–{e3['tasa_ic90'][1]:.2%}); EM del primer identificador {e3['em_primer_id']}; "
                 f"EM de línea {e3['em_linea']}; clases {e3['clases']}.")
    if e1:
        L.append(f"- E1: {e1['repos']} repos, {e1['ocurrencias']} apariciones; solo repositorio {e1['oportunidad_solo_repo']:.2%}; origen {e1['origen']}.")
    if e4:
        L.append(f"- E4: tasa MLE {e4['tasa_mle']:.2%}, UL {e4['tasa_ul']:.2%}, sesgo {e4['tasa_sesgo']:.2%}; EM MLE {e4['em_mle']}, UL {e4['em_ul']}.")
    if e5:
        L.append(f"- E5: EM A {e5['em_A']}, B {e5['em_B']}, C {e5['em_C']}.")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
