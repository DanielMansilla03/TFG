"""
benchmark_final.py
==================
Corre el solver EDA sobre el banco congelado scrambles.json y produce:
  - results_<VARIANT>.json : crudos por scramble + agregados por profundidad (Wilson CI)
  - results_<VARIANT>.csv  : una fila por scramble
  - curva_success.(pdf/svg/png) : tasa de resolucion vs profundidad

USO (Windows / PowerShell):
  python -X utf8 benchmark_final.py

El flag  -X utf8  evita los errores de codificacion cp1252.
Necesita EDAspy instalado (pip install EDAspy) y matplotlib para la curva.
"""

import json
import csv
import time
import math

from pdb_similarity import load_or_build_pdbs
from fusecube_eda_edaspy import solve_fuse


# ------------------------------------------------------------------
#  CONFIGURACION
# ------------------------------------------------------------------
VARIANT   = "baseline_univariado"   
BASE_SEED = 1000                    
SCRAMBLES = "scrambles.json"

SOLVER_KW = dict(
    size_gen       = 500,      
    max_iter       = 200,      
    dead_iter      = 30,       
    alpha          = 0.5,      
    length_penalty = 0.0005,   
    genome_step    = 2,        
    n_restarts     = 10,       
)


def wilson(k, n, z=1.96):
    """Intervalo de Wilson al 95% para una proporcion. Devuelve (p%, lo%, hi%)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom  = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half   = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100 * p, 100 * (center - half), 100 * (center + half))


def main():
    print("Cargando/construyendo PDBs (pdbs_v3.pkl)...")
    corner_pdb, edge_pdb = load_or_build_pdbs()

    with open(SCRAMBLES, encoding="utf-8") as f:
        bank = json.load(f)
    depths = sorted(int(d) for d in bank["depths"])
    print(f"Banco: {SCRAMBLES}  profundidades={depths}  n/prof={bank.get('n_per_depth')}")

    rows = []
    per_depth = {d: [] for d in depths}

    t_start = time.perf_counter()
    for depth in depths:
        scrambles = bank["depths"][str(depth)]
        for idx, scr in enumerate(scrambles):
            seed = BASE_SEED + depth * 1000 + idx   # determinista y unico por scramble
            r = solve_fuse(
                scr, seed=seed,
                corner_pdb=corner_pdb, edge_pdb=edge_pdb,
                verbose=False, **SOLVER_KW,
            )
            sol    = r.get("best_algorithm", "") or ""
            solved = bool(r["solved"])
            secs   = float(r.get("elapsed_seconds", 0.0))
            row = dict(
                depth=depth, idx=idx, scramble=scr, seed=seed,
                solved=solved, solution=sol,
                sol_len=len(sol.split()) if sol else 0,
                sim_pdb=float(r.get("similarity_imp", 0.0)),
                seconds=secs,
            )
            rows.append(row)
            per_depth[depth].append(solved)
            print(f"[d={depth:>2} {idx+1:>2}/{len(scrambles)}] "
                  f"solved={str(solved):>5} len={row['sol_len']:>2} t={secs:5.1f}s")

    # ---- agregados por profundidad ----
    agg = []
    for depth in depths:
        flags = per_depth[depth]
        k, n = sum(flags), len(flags)
        p, lo, hi = wilson(k, n)
        solved_lens = [r["sol_len"] for r in rows if r["depth"] == depth and r["solved"]]
        mean_t = sum(r["seconds"] for r in rows if r["depth"] == depth) / n if n else 0.0
        agg.append(dict(
            depth=depth, n=n, solved=k,
            success_pct=round(p, 1), wilson_lo=round(lo, 1), wilson_hi=round(hi, 1),
            mean_sol_len=round(sum(solved_lens) / len(solved_lens), 2) if solved_lens else None,
            mean_seconds=round(mean_t, 2),
        ))

    out = dict(
        variant=VARIANT, base_seed=BASE_SEED, solver_kw=SOLVER_KW,
        scrambles_schema=bank.get("schema"),
        total_seconds=round(time.perf_counter() - t_start, 1),
        by_depth=agg, runs=rows,
    )
    with open(f"results_{VARIANT}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    with open(f"results_{VARIANT}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["depth", "idx", "solved", "sol_len", "seconds", "sim_pdb", "scramble", "solution"])
        for r in rows:
            w.writerow([r["depth"], r["idx"], r["solved"], r["sol_len"],
                        f"{r['seconds']:.3f}", f"{r['sim_pdb']:.4f}", r["scramble"], r["solution"]])

    # ---- resumen por consola ----
    print("\n=== RESUMEN por profundidad ===")
    print(f"{'d':>3} {'n':>3} {'solv':>4} {'succ%':>6} {'Wilson95%':>15} {'len':>6} {'t(s)':>7}")
    for a in agg:
        ci  = f"[{a['wilson_lo']:.0f},{a['wilson_hi']:.0f}]"
        ln  = f"{a['mean_sol_len']:.1f}" if a['mean_sol_len'] is not None else "-"
        print(f"{a['depth']:>3} {a['n']:>3} {a['solved']:>4} {a['success_pct']:>6.1f} "
              f"{ci:>15} {ln:>6} {a['mean_seconds']:>7.1f}")

    # ---- curva ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [a["depth"] for a in agg]
        ys = [a["success_pct"] for a in agg]
        lo = [a["success_pct"] - a["wilson_lo"] for a in agg]
        hi = [a["wilson_hi"] - a["success_pct"] for a in agg]
        plt.figure(figsize=(6, 4))
        plt.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=4)
        plt.xlabel("Longitud del scramble (profundidad)")
        plt.ylabel("Tasa de resolucion (%)")
        plt.title(f"Tasa de resolucion vs profundidad ({VARIANT})")
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        for ext in ("pdf", "svg", "png"):
            plt.savefig(f"curva_success.{ext}", dpi=150)
        print("\nGuardado: curva_success.pdf / .svg / .png")
    except Exception as e:
        print(f"\n(No se genero la curva: {e}. Instala matplotlib con: pip install matplotlib)")

    print(f"\nHecho. Ficheros: results_{VARIANT}.json / .csv")
    print("Ahora ejecuta:  python -X utf8 verify_solutions.py results_%s.json" % VARIANT)


if __name__ == "__main__":
    main()