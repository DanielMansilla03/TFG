"""Genera scrambles.json — fichero canónico, versionado, de scrambles para los
benchmarks. Usa la misma función determinista que el código de instrumentación,
de modo que el archivo coincide con lo que generaría el código en vuelo.

Formato:
{
  "schema": "fuse-eda-scrambles/v1",
  "base_seed": 1000,
  "depths": {
    "5":  ["scramble1", "scramble2", ...],
    "10": [...],
    ...
  }
}
"""
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PARENT))
os.chdir(str(PARENT))

from instrument import random_scramble


DEPTHS = (5, 10, 15, 19, 25)
N_PER_DEPTH = 50
BASE_SEED = 1000


def main(out_path: Path) -> None:
    payload = {
        "schema": "fuse-eda-scrambles/v1",
        "base_seed": BASE_SEED,
        "n_per_depth": N_PER_DEPTH,
        "comment": (
            "Scrambles deterministas usados por todos los benchmarks. "
            "Generados con random_scramble (sin repetición de cara contigua) "
            "y semilla random.Random(base_seed + d*1000) por profundidad. "
            "Los primeros 20 elementos de cada lista son los mismos que con n_runs=20; "
            "los 50 elementos son los usados con n_runs=50."
        ),
        "depths": {},
    }
    for d in DEPTHS:
        rng = random.Random(BASE_SEED + d * 1000)
        payload["depths"][str(d)] = [random_scramble(d, rng) for _ in range(N_PER_DEPTH)]

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Generado {out_path} con {len(DEPTHS)} profundidades × {N_PER_DEPTH} scrambles.")


if __name__ == "__main__":
    out = HERE / "scrambles.json"
    main(out)
