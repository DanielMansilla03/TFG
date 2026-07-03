"""Sustituye las entradas de profundidades concretas en un JSON de diagnostico.

Uso:
  python merge_json.py base.json patch.json out.json
Donde patch.json contiene un subconjunto de profundidades. Las profundidades
presentes en patch se sobrescriben en base; el resto se mantiene.
"""
import json, sys
from pathlib import Path


def main(base_path: str, patch_path: str, out_path: str) -> None:
    base = json.loads(Path(base_path).read_text(encoding="utf-8"))
    patch = json.loads(Path(patch_path).read_text(encoding="utf-8"))

    patched_depths = set(patch["summaries"].keys())
    for d in patched_depths:
        base["summaries"][d] = patch["summaries"][d]
        base["per_run"][d] = patch["per_run"][d]

    Path(out_path).write_text(json.dumps(base, indent=2), encoding="utf-8")
    print(f"Profundidades sobrescritas: {sorted(patched_depths)} → {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
