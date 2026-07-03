"""
verify_solutions.py
===================
Verificacion independiente de los resultados del benchmark.

Para cada solucion marcada como "resuelta", reconstruye el estado aplicando
el scramble y luego la solucion, y comprueba igualdad EXACTA con el estado
resuelto (cp, co, ep, eo). Debe reportar 0 mismatches. Esto blinda el capitulo
de resultados frente a un posible bug del tracker del solver.

USO:
  python -X utf8 verify_solutions.py results_baseline_univariado.json
"""

import json
import sys

from fusecube_eda_edaspy import FuseState, apply_algorithm, is_solved


def main(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    runs = data["runs"]

    checked = 0
    mismatches = 0
    for r in runs:
        if not r["solved"]:
            continue
        checked += 1
        st = FuseState.solved()
        st = apply_algorithm(st, r["scramble"])
        st = apply_algorithm(st, r["solution"])
        if not is_solved(st):
            mismatches += 1
            print(f"MISMATCH  d={r['depth']} idx={r['idx']}")
            print(f"   scramble: {r['scramble']}")
            print(f"   solucion: {r['solution']}")

    print(f"\nSoluciones comprobadas (marcadas resueltas): {checked}")
    print(f"Mismatches: {mismatches}")
    if mismatches == 0:
        print("OK: 0 mismatches. Todas las soluciones reportadas resuelven el cubo.")
    else:
        print("REVISAR: hay soluciones reportadas como validas que NO resuelven el cubo.")
        sys.exit(1)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results_baseline_univariado.json"
    main(path)
