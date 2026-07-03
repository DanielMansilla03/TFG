# TFG — Solver EDA para el cubo Fuse 3×3×3

Repositorio de código asociado al Trabajo de Fin de Grado. Implementa un solver
basado en **Algoritmos de Estimación de Distribuciones (EDA)** para el cubo
*Fuse* 3×3×3, utilizando `UMDAcat` de la librería
[EDAspy](https://github.com/VicentePerezSoloviev/EDAspy) y bases de datos de
patrones (PDBs) como heurística de similitud.

El documento completo del TFG se encuentra en la carpeta [`TFG/`](TFG/).

## Estructura del repositorio

```
.
├── src/                          # Código fuente del solver
│   ├── fusecube_eda_edaspy.py    # Solver EDA (UMDAcat) e infraestructura del cubo Fuse
│   ├── pdb_similarity.py         # Heurística de similitud y construcción/carga de PDBs
│   ├── corner_pdb.py             # BFS de la PDB de esquinas
│   └── edge_pdb.py               # BFS de la PDB de aristas (factorizada 5+4)
├── test/                         # Pruebas, benchmark y verificación
│   ├── test_fusecube_basico.py   # Pruebas unitarias básicas
│   ├── benchmark_final.py        # Benchmark sobre el banco congelado de scrambles
│   ├── verify_solutions.py       # Verificación de soluciones
│   └── scrambles.json            # Banco congelado de scrambles
├── results/                      # Resultados y figuras del benchmark
├── TFG/                          # Memoria del TFG (PDF)
├── requirements.txt
└── README.md
```

## Requisitos

- Python 3.9 o superior
- Dependencias en [`requirements.txt`](requirements.txt): `numpy`, `EDAspy`, `matplotlib`

Instalación:

```bash
pip install -r requirements.txt
```

## Uso

Las pruebas y el benchmark importan directamente los módulos de `src/`, por lo
que dicha carpeta debe estar en el `PYTHONPATH` al ejecutarlos.

En Windows / PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -X utf8 test/benchmark_final.py
```

En Linux / macOS:

```bash
PYTHONPATH=src python test/benchmark_final.py
```

> El flag `-X utf8` evita errores de codificación `cp1252` en Windows.

La primera ejecución construye las PDBs mediante BFS y las cachea en un fichero
`*.pkl` (ignorado por Git); las siguientes ejecuciones las cargan directamente.

## Salidas del benchmark

- `results_<VARIANT>.json` — resultados crudos por scramble y agregados por profundidad (Wilson CI)
- `results_<VARIANT>.csv` — una fila por scramble
- `curva_success.(pdf/svg/png)` — tasa de resolución frente a profundidad
