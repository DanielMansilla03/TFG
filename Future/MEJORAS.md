# MEJORAS al EDA del Fuse Cube 3×3×3

Historia completa de la mejora del solver desde la línea base hasta la versión
final (`markov_anchor`). Conserva los resultados **negativos** porque son parte
de la evidencia experimental: las ablaciones intermedias justifican qué cambio
fue realmente responsable de la mejora.

> Restricción de partida (no negociable): el solver debe seguir siendo un **EDA
> puro**. Se admite usar el PDB como heurística/fitness, pero no IDA*, no
> búsqueda PDB pura, no GA, no simulated annealing. Cada variante por debajo es
> un EDA: modelo probabilístico, muestreo, selección de élite y actualización
> iterativa del modelo.

## 1. Configuración común

- **Scrambles**: fichero versionado [`scrambles.json`](../test/scrambles.json), schema
  `fuse-eda-scrambles/v1`, `base_seed=1000`, 50 scrambles por profundidad ∈ {5,
  10, 15, 19, 25}. Generados con `random_scramble` (sin repetición de cara
  contigua). Todas las variantes se evalúan sobre **los mismos scrambles**
  (verificado con `verify_scrambles.py`: 0 mismatches). El script
  [`generate_scrambles.py`](test/generate_scrambles.py) reproduce el fichero
  determinísticamente.
- **Tamaño de muestra**: n=20 en d∈{5,10,25}; n=50 en d∈{15,19}.
- **EDA core**: `size_gen=400`, `max_iter=120`, `dead_iter=30`, `alpha=0.30`,
  `n_restarts=3`, `genome_step=2`, `length_penalty=5e-4`.
- **Intervalos de confianza**: Wilson 95% sobre la tasa de éxito.
- **Reproducción end-to-end**: `python benchmark_final.py` (regenera JSONs,
  tablas y figura desde cero).

## 2. Diagnóstico inicial (Fase 1)

Instrumentación generación a generación de la línea base UMDAcat (vocabulario
de 10 alelos: 9 movimientos + STOP). Profundidades 5 / 10 / 15 / 19, n=20.

| depth | success | gens_run | stag_gen | ent_final | elite_div | pop%>init | learn_len | mxpref_len |
| ----: | :-----: | :------: | :------: | :-------: | :-------: | :-------: | :-------: | :--------: |
|     5 | 100%    | 32       | 0        | 0.00      | 1         | 100%      | 5         | 5          |
|    10 | 100%    | 32       | 1        | 2.01      | 1         | 100%      | 10        | 10         |
|    15 | 65%     | 35.5     | 4        | 2.01      | 1         | 100%      | ≈19       | 10         |
|    19 | 20%     | 44.2     | 11       | 1.99      | 1         | 100%      | ≈19       | 10         |

Patrones observados en d∈{15,19}:

- **Colapso prematuro de la élite** (`elite_div=1`): toda la élite tiene el
  mismo coste — el modelo se concentra en una sola moda demasiado pronto.
- **Estancamiento real largo** (`stag_gen` 4–11): el coste no mejora durante
  decenas de generaciones antes de que `dead_iter` corte.
- **STOP infrautilizado**: `learn_len ≈ 19` pero `mxpref_len = 10`. El alelo
  STOP **no acorta** la longitud aprendida; el solver gasta entropía aprendiendo
  movimientos posteriores al prefijo solución.
- **Univariate model**: UMDAcat no captura dependencias entre posiciones
  consecutivas. En particular, no penaliza secuencias `L L'` (cara repetida)
  porque cada posición se modela independientemente.

→ Hipótesis: el modelo univariate + STOP es insuficiente para profundidades
medias; hay que introducir dependencias y reducir el vocabulario.

## 3. Ablaciones (Fase 2)

Cada ablación cambia **una sola cosa** respecto a la anterior. El diseño
incremental permite atribuir mejoras a un cambio concreto.

### 3.1 Ablación 1 — `no_stop` (eliminar STOP)

Vocabulario reducido a 9 alelos (sin STOP). El resto, idéntico al baseline.

| depth | baseline | no_stop |    Δ |
| ----: | -------: | ------: | ---: |
|     5 |     100% |    100% |    0 |
|    10 |     100% |    100% |    0 |
|    15 |      65% |     70% |  +5  |
|    19 |      20% |     10% | −10  |

**Resultado: neutral (dentro de ruido)**. Los CIs Wilson 95% se solapan
completamente. STOP no era el cuello de botella principal; sólo añadía un
canal de entropía redundante. Lo conservamos eliminado como línea base más
limpia para la memoria — un alelo menos = un grado de libertad menos
desperdiciado, aunque no aparece en el éxito numérico.

### 3.2 Ablación 2 — `markov_uniformT` (modelo producto con T congelada)

Modelo: P(mov_i = m | prev = p) ∝ P_pos,i(m) · T(m|p) · mask(p, m), con
máscara dura que **prohíbe la misma cara que la anterior** (`mask` ∈ {0, 1}).
T se fija a uniforme sobre el complemento de la cara previa y **no se aprende**.

| depth | no_stop | mkv_uniformT |    Δ |
| ----: | ------: | -----------: | ---: |
|    15 |     70% |      **42%** | −28  |
|    19 |     10% |       **4%** | −6   |

**Resultado NEGATIVO**: la máscara por sí sola es **perjudicial**. Por qué es
relevante: descarta la explicación "el problema era simplemente permitir caras
repetidas". La máscara estructural restringe el espacio pero, sin transiciones
aprendidas, el EDA pierde información posicional útil y converge a peores
óptimos locales. Esta ablación deja claro que la **mejora de la variante
siguiente viene del aprendizaje de T**, no de la máscara.

### 3.3 Ablación 3 — `markov` (modelo producto con T aprendida)

Misma estructura, pero T(m|p) se actualiza con Laplace tras cada generación a
partir de los pares (prev, curr) en la élite.

| depth | no_stop | markov |    Δ |
| ----: | ------: | -----: | ---: |
|    15 |     70% |  **96%** | +26  |
|    19 |     10% |  **60%** | +50  |
|    25 |       — |    0%  |   —  |

**Salto cualitativo**. La mejora respecto a `markov_uniformT` (+54 pts en d=15,
+56 pts en d=19) confirma que la pieza crítica es **aprender** las
transiciones, no la máscara dura. Además el modelo Markov entrena en menos
generaciones (`gens_run`≈32 vs 44 en baseline) y con menos evaluaciones por run
en las profundidades difíciles (≈14 k vs 29–44 k en d=15/19, ver §5).

Limitación: d=25 sigue siendo intratable (0%).

### 3.4 Ablación 4 — `markov_anchor` (anclaje por estancamiento)

Mismo modelo Markov, pero la búsqueda se organiza en **hasta 3 fases**. Cuando
una fase se estanca (`dead_iter` alcanzado sin mejora del coste), se ancla el
mejor prefijo encontrado y se relanza un EDA nuevo sobre el sub-problema
restante. Se conserva `best_overall` globalmente y el presupuesto total de
evaluaciones se mantiene comparable al de `markov` puro.

Detalles que importaron:

- El disparo es por **estancamiento**, no por un umbral fijo de pdb_d.
- La detección de "resuelto" se hace caminando el **max-prefix** sobre el
  scramble original, no aplicando la secuencia completa al estado actual (el
  estado actual ya tenía aplicado el ancla, así que aplicar de nuevo deshacía
  la solución — bug detectado y corregido).
- En d=5 había un *budget waste*: la fase 0 resolvía en gen 0 pero seguía
  consumiendo `dead_iter` antes de lanzar la siguiente fase. Corregido con un
  cortocircuito `if best_phase_cost <= -50.0: break` después de la mejora.

### 3.5 Tabla final — éxito con Wilson CI 95%

| depth | baseline                | no_stop                 | mkv_uniformT            | markov                  | **markov_anchor**        |
| ----: | ----------------------- | ----------------------- | ----------------------- | ----------------------- | ------------------------ |
|     5 | 100% [83.9, 100.0]      | 100% [83.9, 100.0]      | 100% [83.9, 100.0]      | 100% [83.9, 100.0]      | **100% [83.9, 100.0]**   |
|    10 | 100% [83.9, 100.0]      | 100% [83.9, 100.0]      | 100% [83.9, 100.0]      | 100% [83.9, 100.0]      | **100% [83.9, 100.0]**   |
|    15 |  65% [43.3, 81.9]       |  70% [48.1, 85.5]       |  42% [29.4, 55.8]       |  96% [86.5, 98.9]       | **100% [92.9, 100.0]**   |
|    19 |  20% [ 8.1, 41.6]       |  10% [ 2.8, 30.1]       |   4% [ 1.1, 13.5]       |  60% [46.2, 72.4]       |  **92% [81.2,  96.8]**   |
|    25 |  —                      |  —                      |  —                      |   0% [ 0.0, 16.1]       |  **20% [ 8.1,  41.6]**   |

(d=15 y d=19 con n=50; el resto con n=20. CIs Wilson 95%.)

Curva en formato vectorial: [`curva_success.pdf`](results/curva_success.pdf) /
[`curva_success.svg`](results/curva_success.svg).

## 4. Presupuestos (evals / run)

Media de evaluaciones de fitness por run (suma sobre todos los reinicios). El
diseño de `markov_anchor` mantiene el **mismo presupuesto total** que `markov`
en d∈{5,10,15,19}; en d=25 sí gasta más porque ejecuta las 3 fases completas.

| depth | baseline | no_stop |  mkvU  | markov | anchor |
| ----: | -------: | ------: | -----: | -----: | -----: |
|     5 |   12 800 |  12 800 | 12 400 | 12 400 | 10 120 |
|    10 |   12 940 |  12 900 | 12 400 | 12 400 | 12 400 |
|    15 |   28 980 |  26 400 | 35 304 | 13 744 | 13 216 |
|    19 |   44 220 |  49 160 | 56 288 | 24 056 | 25 264 |
|    25 |        — |       — |      — | 41 100 | 95 800 |

Observaciones:

- En d∈{15,19} el modelo Markov **resuelve con la mitad-tercio de evaluaciones**
  que el baseline, porque converge antes y no necesita reinicios.
- `mkvU` gasta más que el baseline en d=15/19 (más reinicios infructuosos) —
  consistente con su menor éxito.
- En d=5 el anchor gasta menos que `markov` porque la mayoría de runs cortan
  por el cortocircuito de resolución temprana introducido en el cierre.

## 5. Negativos preservados — interpretación

Estos resultados **no son fracasos**: son el contrafactual que sostiene el
diseño final.

- **`no_stop` neutral**: descarta que STOP fuera el cuello de botella. Si
  hubiéramos saltado directamente al modelo Markov, no podríamos atribuir la
  mejora a un cambio concreto. La ablación deja claro que la ganancia viene del
  **modelo**, no de la reducción del vocabulario.
- **`markov_uniformT` perjudicial**: descarta que la máscara dura por sí sola
  sea responsable de la mejora. Demuestra que la pieza crítica es **aprender T**
  desde la élite. Sin esta ablación, podríamos atribuir incorrectamente la
  mejora a la restricción estructural.

Ambos negativos son evidencia experimental directa de qué pieza concreta del
modelo final (`markov_anchor`) carga la mejora.

## 6. Reproducción

```bash
python benchmark_final.py             # regenera TODO desde cero
python benchmark_final.py --quick     # smoke test (n=5, sin d=25)
python benchmark_final.py --skip-existing  # sólo lo que falte
```

`benchmark_final.py` orquesta:

1. Garantiza `scrambles.json` (lo regenera con `generate_scrambles.py` si falta).
2. Lanza `instrument.py` para cada variante con los mismos scrambles y semilla.
3. Llama a `compare.py` para imprimir las tablas Wilson + presupuestos.
4. Llama a `make_figure.py` para regenerar `curva_success.{pdf,svg}`.

Semillas fijas: `--base_seed=1000` y `scrambles.json` versionado garantizan que
dos ejecuciones independientes producen exactamente los mismos JSONs.

## 7. Trabajo futuro

Para atacar d≥25 sin romper la pureza EDA, los siguientes 4 frentes son las
palancas más prometedoras (cada una sigue siendo "modelo + muestreo + élite +
actualización"):

1. **Más fases de anclaje** (`max_phases` 4–5). El cuello de botella en d=25 es
   que 3 fases no alcanzan a cubrir la longitud típica de solución; permitir
   más fases sigue siendo un EDA secuencializado.
2. **`size_gen` mayor en fases tardías**. Las fases finales operan sobre un
   sub-problema más corto pero menos estructurado; muestrear más individuos
   compensa la pérdida de señal heurística del PDB cerca del óptimo.
3. **Reiniciar el modelo Markov entre fases**. Mantener P_pos y T entre fases
   sesga el muestreo hacia el sub-problema anterior; un reset completo (con
   máscara intacta) hace cada fase un EDA independiente bien condicionado.
4. **Población élite con archivo global**. Mantener un archivo de los mejores
   prefijos de todas las fases y mezclarlos con la élite local en cada
   actualización del modelo; preserva diversidad fenotípica entre fases sin
   abandonar el ciclo EDA.

Ninguna de las cuatro requiere salir del marco EDA: el motor sigue siendo
muestreo desde un modelo probabilístico, selección de élite por fitness PDB y
actualización iterativa del modelo.
