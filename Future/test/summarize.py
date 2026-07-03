"""Carga diagnostico.json y emite:
  - Tabla resumen por profundidad
  - Trayectorias por generación del mejor reinicio (texto)
  - Detección de patrones (convergencia prematura, plateau, STOP redundante)
"""
from __future__ import annotations
import json, statistics
from pathlib import Path
from typing import Dict, List


def fmt_pct(x: float) -> str:
    return f"{x*100:5.1f}%"


def main(path: str = "diagnostico.json") -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = payload["config"]
    summaries = payload["summaries"]
    per_run = payload["per_run"]

    print("\n" + "=" * 88)
    print("CONFIGURACIÓN")
    print("=" * 88)
    for k, v in cfg.items():
        print(f"  {k:>14s} = {v}")

    # ------------------------------------------------------------
    # Tabla 1: resumen agregado
    # ------------------------------------------------------------
    print("\n" + "=" * 120)
    print("TABLA 1 — RESUMEN AGREGADO POR PROFUNDIDAD (medianas, salvo success%)")
    print("=" * 120)
    headers = [
        ("depth",      ">6"),
        ("success",    ">8"),
        ("init_pdb_d", ">10"),
        ("gens_run",   ">8"),
        ("stag_gen",   ">8"),
        ("ent_final",  ">9"),
        ("elite_div",  ">9"),
        ("pop%>init",  ">9"),
        ("elite%>init",">11"),
        ("learn_len",  ">9"),
        ("mxpref_len", ">10"),
        ("decod_len",  ">9"),
        ("best_pdb_d", ">10"),
    ]
    print("  " + "  ".join(f"{h:{fmt}s}" for h, fmt in headers))
    for d in sorted(summaries.keys(), key=int):
        s = summaries[d]
        row = [
            f"{int(d):6d}",
            f"{fmt_pct(s['success_rate']):>8s}",
            f"{(s.get('initial_pdb_dist') or 0):>10.4f}",
            f"{s['gens_run_median']:>8.1f}",
            f"{s['stagnation_gen_median']:>8.1f}",
            f"{s['entropy_final_median']:>9.3f}",
            f"{s['elite_unique_costs_median']:>9.1f}",
            f"{fmt_pct(s['pop_pct_better_than_initial_median']):>9s}",
            f"{fmt_pct(s['elite_pct_better_than_initial_median']):>11s}",
            f"{s['best_learned_len_median']:>9.1f}",
            f"{s['best_max_prefix_len_median']:>10.1f}",
            f"{s['best_decoded_len_median']:>9.1f}",
            f"{s['best_prefix_pdb_dist_median']:>10.4f}",
        ]
        print("  " + "  ".join(row))
    print("=" * 120)
    print(
        "Leyenda:\n"
        "  init_pdb_d   = distancia PDB normalizada (1-sim) del scramble inicial\n"
        "  ent_final    = entropía media del modelo (max ~log2(10)=3.32) al final del run del mejor reinicio\n"
        "  elite_div    = nº de costes distintos en la élite (top-alpha) en la última gen — 1.0 = élite colapsada\n"
        "  pop%>init    = % individuos con coste mejor que la secuencia vacía (encuentran prefijo mejor que el scramble)\n"
        "  elite%>init  = idem restringido a la élite\n"
        "  learn_len    = índice del STOP aprendido por el best individuo\n"
        "  mxpref_len   = paso donde el max-prefix encuentra el mejor estado\n"
        "  decod_len    = nº de movimientos tras simplify() en el best\n"
        "  best_pdb_d   = distancia PDB normalizada del mejor estado encontrado\n"
    )

    # ------------------------------------------------------------
    # Tabla 2: evolución por gen del mejor reinicio (1ª run de cada profundidad)
    # ------------------------------------------------------------
    print("\n" + "=" * 100)
    print("TABLA 2 — TRAYECTORIA POR GENERACIÓN (run #1 de cada profundidad, mejor reinicio)")
    print("=" * 100)
    for d in sorted(per_run.keys(), key=int):
        runs = per_run[d]
        if not runs:
            continue
        r = runs[0]
        if not r["restart_summaries"]:
            continue
        best_r = min(r["restart_summaries"], key=lambda x: x["best_cost"])
        logs = best_r["gen_logs"]
        if not logs:
            continue
        print(f"\n  depth={d}  scramble='{r['scramble']}'  seed={r['seed']}  "
              f"reinicio={best_r['restart_idx']}  gl={best_r['genome_length']}  "
              f"solved={best_r['solved']}  best_cost={best_r['best_cost']:+.4f}")
        print("    gen  entropy  best_cost  best_pdb_d  elite_div  pop%>init  mxpref/learn")
        for log in logs:
            print(f"    {log['gen']:>3d}  "
                  f"{log['entropy']:>7.3f}  "
                  f"{log['best_cost']:>+9.4f}  "
                  f"{log['best_prefix_pdb_dist']:>10.4f}  "
                  f"{log['elite_unique_costs']:>9d}  "
                  f"{log['pop_pct_better_than_initial']*100:>8.1f}%  "
                  f"{log['best_max_prefix_len']:>4d}/{log['best_learned_len']}")
    print("=" * 100)

    # ------------------------------------------------------------
    # Diagnóstico: detección de patrones
    # ------------------------------------------------------------
    print("\n" + "=" * 100)
    print("PATRONES DETECTADOS")
    print("=" * 100)
    for d in sorted(summaries.keys(), key=int):
        s = summaries[d]
        flags: List[str] = []
        if s["entropy_final_median"] < 1.0:
            flags.append("ENTROPIA_COLAPSADA")
        if s["elite_unique_costs_median"] <= 1.5:
            flags.append("ELITE_COLAPSADA")
        if s["stagnation_gen_median"] < 0.3 * s["gens_run_median"] and s["gens_run_median"] > 5:
            flags.append("CONVERGENCIA_PREMATURA")
        if (s["best_max_prefix_len_median"] < 0.7 * s["best_learned_len_median"]
                and s["best_learned_len_median"] > 0):
            flags.append("STOP_REDUNDANTE")
        if s["pop_pct_better_than_initial_median"] < 0.5:
            flags.append("PLATEAU_FITNESS")
        if not flags:
            flags.append("(sin patron severo)")
        print(f"  d={d}: {', '.join(flags)}")
    print("=" * 100)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "diagnostico.json")
