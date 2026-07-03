"""Compara dos (o más) ficheros JSON de diagnostico lado a lado.

Uso:
  python compare.py baseline=diagnostico.json no_stop=diagnostico_no_stop.json
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
from typing import Dict, List, Tuple


def load(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt_pct(x: float) -> str:
    return f"{x*100:5.1f}%"


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Intervalo de confianza Wilson para una proporción.

    Devuelve (lo, hi) en [0, 1]. Si n==0, devuelve (0, 1).
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def count_solved(payload: Dict, d: int) -> Tuple[int, int]:
    """Cuenta (solved, total) para una profundidad dada."""
    per_run = payload["per_run"]
    runs = per_run.get(str(d), per_run.get(d, []))
    n = len(runs)
    k = sum(1 for r in runs if r["solved"])
    return k, n


def evals_per_run_mean(payload: Dict, d: int) -> float:
    """Media de evaluaciones de fitness por run (sumando todos los reinicios)."""
    import statistics
    per_run = payload["per_run"]
    runs = per_run.get(str(d), per_run.get(d, []))
    vals = []
    for r in runs:
        total = 0
        for rs in r.get("restart_summaries", []):
            n_evals = rs.get("n_evals_total")
            if n_evals is None:
                # Fallback: gens_run × size_gen aproximado
                cfg = payload["config"]
                n_evals = rs.get("n_gens_run", 0) * cfg.get("size_gen", 0)
            total += n_evals
        vals.append(total)
    if not vals:
        return float("nan")
    return float(statistics.mean(vals))


def scramble_pdb_median(payload: Dict, d: int) -> float:
    """Mediana de la distancia PDB del SCRAMBLE inicial (depende solo del scramble,
    no del modelo). Se toma del primer reinicio de cada run (todos los reinicios
    parten del mismo scramble, así que es indistinto)."""
    import statistics
    per_run = payload["per_run"]
    runs = per_run.get(str(d), per_run.get(d, []))
    vals = []
    for r in runs:
        if r["restart_summaries"]:
            vals.append(r["restart_summaries"][0]["initial_pdb_dist"])
    if not vals:
        return float("nan")
    return float(statistics.median(vals))


def main(args: List[str]) -> None:
    if not args:
        print("Uso: python compare.py LABEL=FILE [LABEL=FILE ...]")
        sys.exit(1)

    payloads: List[Tuple[str, Dict]] = []
    for a in args:
        if "=" not in a:
            print(f"Argumento mal formado (esperado LABEL=FILE): {a}")
            sys.exit(1)
        label, path = a.split("=", 1)
        payloads.append((label, load(path)))

    depths = sorted({int(d) for _, p in payloads for d in p["summaries"].keys()})

    # Config compartida (asumimos que coinciden los parámetros relevantes)
    print("\n" + "=" * 100)
    print("CONFIG (de cada fichero)")
    print("=" * 100)
    for label, p in payloads:
        cfg = p["config"]
        print(f"  [{label}] " + ", ".join(f"{k}={v}" for k, v in cfg.items()
                                          if k != "depths"))

    # Métricas a comparar (columna -> (key, formato))
    # init_pdb_d se calcula aparte desde per_run (el summary tenía un bug que solo
    # guardaba el último valor en lugar de la mediana). Excluida de la lista.
    metrics = [
        ("success%",   "success_rate",                       lambda v: fmt_pct(v)),
        ("gens_run",   "gens_run_median",                    lambda v: f"{v:.1f}"),
        ("stag_gen",   "stagnation_gen_median",              lambda v: f"{v:.1f}"),
        ("ent_final",  "entropy_final_median",               lambda v: f"{v:.3f}"),
        ("elite_div",  "elite_unique_costs_median",          lambda v: f"{v:.1f}"),
        ("pop%>init",  "pop_pct_better_than_initial_median", lambda v: fmt_pct(v)),
        ("learn_len",  "best_learned_len_median",            lambda v: f"{v:.1f}"),
        ("mxpref_len", "best_max_prefix_len_median",         lambda v: f"{v:.1f}"),
        ("decod_len",  "best_decoded_len_median",            lambda v: f"{v:.1f}"),
        ("best_pdb_d", "best_prefix_pdb_dist_median",        lambda v: f"{v:.4f}"),
    ]

    # Tabla 0: success con Wilson CI 95% (resumen ejecutivo)
    print("\n" + "=" * 100)
    print("ÉXITO POR PROFUNDIDAD CON WILSON CI 95%  (k/n  →  p%  [lo, hi])")
    print("=" * 100)
    header = [f"{'depth':<7}"] + [f"{lab:>26s}" for lab, _ in payloads]
    print("    " + " | ".join(header))
    for d in depths:
        row = [f"{d:<7d}"]
        for plabel, payload in payloads:
            k, n = count_solved(payload, d)
            if n == 0:
                row.append(f"{'n/a':>26s}")
                continue
            p = k / n
            lo, hi = wilson_ci(k, n)
            row.append(
                f"{f'{k:>2d}/{n:<2d} → {p*100:5.1f}% [{lo*100:5.1f},{hi*100:5.1f}]':>26s}"
            )
        print("    " + " | ".join(row))
    print("=" * 100)

    # Tabla 0b: scramble_pdb_d (propiedad del scramble, debería coincidir si
    # los scrambles son compartidos)
    print("\n" + "=" * 100)
    print("scramble_pdb_d (mediana por profundidad) — propiedad del SCRAMBLE, no del modelo")
    print("=" * 100)
    header = [f"{'depth':<7}"] + [f"{lab:>14s}" for lab, _ in payloads]
    print("    " + " | ".join(header))
    for d in depths:
        row = [f"{d:<7d}"]
        for plabel, payload in payloads:
            row.append(f"{scramble_pdb_median(payload, d):>14.4f}")
        print("    " + " | ".join(row))
    print("=" * 100)

    # Tabla 0c: evals/run medios — para validar que el presupuesto es comparable
    print("\n" + "=" * 100)
    print("EVALS / RUN (media, suma sobre todos los reinicios) — debe ser comparable")
    print("=" * 100)
    header = [f"{'depth':<7}"] + [f"{lab:>14s}" for lab, _ in payloads]
    print("    " + " | ".join(header))
    for d in depths:
        row = [f"{d:<7d}"]
        for plabel, payload in payloads:
            v = evals_per_run_mean(payload, d)
            row.append(f"{v:>14,.0f}")
        print("    " + " | ".join(row))
    print("=" * 100)

    # Tabla comparativa por profundidad (resto de métricas)
    print("\n" + "=" * 100)
    print("MÉTRICAS DETALLADAS (medianas)")
    print("=" * 100)

    for d in depths:
        print(f"\n  depth = {d}")
        # Cabecera
        header_cols = [f"{'metric':<11}"] + [f"{lab:>14}" for lab, _ in payloads]
        print("    " + " | ".join(header_cols))
        print("    " + "-+-".join(["-" * 11] + ["-" * 14 for _ in payloads]))

        for label, key, formatter in metrics:
            if label == "success%":
                continue  # ya en tabla superior
            vals = []
            for plabel, payload in payloads:
                s = payload["summaries"].get(str(d)) or payload["summaries"].get(d)
                if s is None:
                    vals.append(None)
                else:
                    vals.append(s.get(key))

            row = [f"{label:<11}"]
            for v in vals:
                row.append(f"{formatter(v) if v is not None else 'n/a':>14}")
            print("    " + " | ".join(row))

    print("\n" + "=" * 100)
    print("Nota: 'ent_final' no es directamente comparable entre variantes con vocabularios distintos:")
    print("  - baseline: max log2(10)≈3.32   - no_stop: max log2(9)≈3.17")
    print("  - markov*: max log2(6)≈2.58 para condicionales (post-máscara), log2(9) para pos 0")
    print("=" * 100)


if __name__ == "__main__":
    main(sys.argv[1:])
