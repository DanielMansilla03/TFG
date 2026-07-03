"""benchmark_final.py — Regenera de cero todos los artefactos del cierre.

Ejecuta los 5 experimentos (baseline, no_stop, markov, markov_uniformT,
markov_anchor) sobre los MISMOS scrambles versionados en scrambles.json,
con n=50 en d∈{15,19} y n=20 en d∈{5,10,25}, y produce:

  - diagnostico.json                    (baseline)
  - diagnostico_no_stop.json
  - diagnostico_markov_uniformT.json
  - diagnostico_markov_d25.json         (markov puro, incluye d=25)
  - diagnostico_markov_anchor.json      (incluye d=25)
  - curva_success.{pdf,svg}             (figura para LaTeX)
  - Tablas Wilson CI 95% + presupuestos por stdout

Reproducibilidad: scrambles.json fija los inputs; --base_seed=1000 fija
las semillas internas del EDA (compartidas entre variantes).

Uso:
  python benchmark_final.py
  python benchmark_final.py --skip-existing   (no re-corre si el JSON existe)
  python benchmark_final.py --only baseline,markov_anchor
  python benchmark_final.py --quick           (n=5 por profundidad, smoke test)
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
PY = sys.executable

# Profundidades y tamaño de muestra por variante.
DEPTHS_ALL = (5, 10, 15, 19, 25)
N_RUNS_PER_DEPTH = {5: 20, 10: 20, 15: 50, 19: 50, 25: 20}

# (variant_id, output_json, depths). Se conserva la nomenclatura ya usada
# en el resto de scripts y en el artículo.
RUNS: List[Tuple[str, str, Tuple[int, ...]]] = [
    ("baseline",        "diagnostico.json",                    DEPTHS_ALL),
    ("no_stop",         "diagnostico_no_stop.json",            DEPTHS_ALL),
    ("markov_uniformT", "diagnostico_markov_uniformT.json",    DEPTHS_ALL),
    ("markov",          "diagnostico_markov_d25.json",         DEPTHS_ALL),
    ("markov_anchor",   "diagnostico_markov_anchor.json",      DEPTHS_ALL),
]


def n_runs_per_depth_arg(depths, n_map) -> str:
    return ",".join(f"{d}:{n_map[d]}" for d in depths)


def ensure_scrambles() -> None:
    """Garantiza que scrambles.json existe (lo regenera si no)."""
    sc = HERE / "scrambles.json"
    if sc.exists():
        print(f"[ok] scrambles.json ya existe ({sc})")
        return
    print("[gen] scrambles.json no existe — generando...")
    rc = subprocess.call([PY, str(HERE / "generate_scrambles.py")])
    if rc != 0 or not sc.exists():
        sys.exit("ERROR: no se pudo generar scrambles.json")


def run_one(variant: str, output_json: str, depths: Tuple[int, ...],
            n_map: Dict[int, int], base_seed: int) -> None:
    out = HERE / output_json
    cmd = [
        PY, str(HERE / "instrument.py"),
        "--variant", variant,
        "--depths", *map(str, depths),
        "--n_runs_per_depth", n_runs_per_depth_arg(depths, n_map),
        "--base_seed", str(base_seed),
        "--output", str(out),
    ]
    print("\n" + "=" * 72)
    print(f"  RUN: variant={variant}  depths={depths}  → {out.name}")
    print("=" * 72)
    print("  " + " ".join(cmd))
    t0 = time.time()
    rc = subprocess.call(cmd)
    dt = time.time() - t0
    if rc != 0:
        sys.exit(f"ERROR: variant={variant} salió con código {rc}")
    print(f"[ok] {variant} terminado en {dt/60:.1f} min")


def run_compare() -> None:
    cmd = [
        PY, str(HERE / "compare.py"),
        "baseline=diagnostico.json",
        "no_stop=diagnostico_no_stop.json",
        "mkvU=diagnostico_markov_uniformT.json",
        "markov=diagnostico_markov_d25.json",
        "anchor=diagnostico_markov_anchor.json",
    ]
    print("\n" + "=" * 72)
    print("  TABLAS COMPARATIVAS (Wilson CI 95% + presupuestos)")
    print("=" * 72)
    subprocess.call(cmd)


def run_figure() -> None:
    print("\n" + "=" * 72)
    print("  FIGURA curva_success.{pdf,svg}")
    print("=" * 72)
    rc = subprocess.call([PY, str(HERE / "make_figure.py")])
    if rc != 0:
        sys.exit("ERROR: make_figure.py falló")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_seed", type=int, default=1000)
    ap.add_argument("--skip-existing", action="store_true",
                    help="No re-correr variantes cuyo JSON ya existe.")
    ap.add_argument("--only", default=None,
                    help="Lista de variantes separadas por coma a ejecutar "
                         "(p.ej. 'baseline,markov_anchor').")
    ap.add_argument("--quick", action="store_true",
                    help="Smoke test: n=5 por profundidad, sin d=25.")
    args = ap.parse_args()

    if args.quick:
        depths = (5, 10, 15, 19)
        n_map = {d: 5 for d in depths}
        runs = [(v, o, depths) for v, o, _ in RUNS]
    else:
        depths = DEPTHS_ALL
        n_map = N_RUNS_PER_DEPTH
        runs = RUNS

    if args.only:
        wanted = set(args.only.split(","))
        runs = [r for r in runs if r[0] in wanted]
        if not runs:
            sys.exit(f"ERROR: --only={args.only} no coincide con ninguna variante.")

    ensure_scrambles()

    t_start = time.time()
    for variant, out_name, ds in runs:
        out = HERE / out_name
        if args.skip_existing and out.exists():
            print(f"[skip] {out_name} ya existe — saltando {variant}")
            continue
        run_one(variant, out_name, ds, n_map, args.base_seed)

    run_compare()
    run_figure()

    print(f"\n[done] benchmark completo en {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
