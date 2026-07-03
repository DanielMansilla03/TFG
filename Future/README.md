# Future — Resultados y mejoras futuras

Material asociado a la sección de **resultados futuros y mejoras futuras** del TFG.
Documenta el historial experimental completo del solver (línea base → ablaciones
→ `markov_anchor`), preservando también los **resultados negativos** como
evidencia, y enumera las líneas de trabajo futuro para atacar profundidades
d ≥ 25 sin romper la pureza EDA.

El documento principal de esta sección es [`MEJORAS.md`](MEJORAS.md).

## Estructura

```
Future/
├── MEJORAS.md              # Narrativa experimental + tablas Wilson + trabajo futuro
├── src/                    # Código de las variantes e instrumentación
│   ├── solver_variants.py  # Variantes del EDA (no_stop, markov, markov_anchor, ...)
│   └── instrument.py       # Motor de instrumentación generación a generación
├── test/                   # Orquestación, reproducción y análisis
│   ├── benchmark_final.py  # Orquestador end-to-end del benchmark
│   ├── generate_scrambles.py
│   ├── verify_scrambles.py
│   ├── compare.py          # Tablas de éxito (Wilson 95%) y presupuestos
│   ├── make_figure.py      # Genera curva_success.{pdf,svg}
│   ├── merge_json.py
│   └── summarize.py
└── results/                # Salidas experimentales (regenerables)
    ├── curva_success.{pdf,svg}
    ├── diagnostico*.json / .log   # Volcados de instrument.py por variante
    ├── sanity*.json
    └── benchmark_log.txt
```

## Dependencias

El código de `src/` y `test/` importa los módulos del solver principal
(`fusecube_eda_edaspy`, `pdb_similarity`, `corner_pdb`, `edge_pdb`), que residen
en el [`src/`](../src) de la raíz del repositorio. Para ejecutar los scripts,
dicha carpeta debe estar en el `PYTHONPATH`:

```bash
PYTHONPATH=../src python test/benchmark_final.py
```

> Los ficheros de `results/` son artefactos regenerables mediante
> `benchmark_final.py`; se incluyen como evidencia de los experimentos.
