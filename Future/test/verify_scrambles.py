"""Verifica que los scrambles son idénticos entre variantes para el mismo (d, seed)."""
import json, sys
from pathlib import Path

paths = {
    "baseline":     "diagnostico.json",
    "no_stop":      "diagnostico_no_stop.json",
    "mkv_uniformT": "diagnostico_markov_uniformT.json",
    "markov":       "diagnostico_markov.json",
}
here = Path(__file__).resolve().parent
data = {k: json.loads((here / v).read_text(encoding="utf-8")) for k, v in paths.items()}

print("Comparando scrambles por (depth, run_idx). Debe coincidir entre TODAS las variantes en los primeros min(n).")
for d in (5, 10, 15, 19):
    per_runs = {k: data[k]["per_run"].get(str(d), data[k]["per_run"].get(d, [])) for k in paths}
    n_min = min(len(rs) for rs in per_runs.values())
    if n_min == 0:
        print(f"  d={d}: alguna variante sin datos, salto.")
        continue
    diffs = 0
    for i in range(n_min):
        scrs = {k: per_runs[k][i]["scramble"] for k in paths}
        seeds = {k: per_runs[k][i]["seed"] for k in paths}
        if len(set(scrs.values())) > 1:
            diffs += 1
            if diffs <= 3:
                print(f"  d={d} i={i}: DIFIEREN")
                for k, s in scrs.items():
                    print(f"    [{k}] seed={seeds[k]}  scr='{s}'")
    print(f"  d={d}: {diffs} mismatches en los primeros {n_min} runs")
