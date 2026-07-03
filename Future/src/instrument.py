"""
instrument.py — Instrumentación diagnóstica del EDA Fuse 3x3x3.

Mide por generación:
  - Entropía media del modelo UMDAcat
  - Mejor coste de la generación
  - Distancia PDB del mejor prefijo del mejor individuo
  - Diversidad fenotípica de la élite (# costes distintos, proxy de # distancias PDB)
  - % individuos cuyo coste mejora el de la secuencia vacía (≈ mejoran la PDB inicial)
  - Longitud aprendida (índice del STOP) vs longitud del best prefix (donde max-prefix corta)

Configuración: profundidades {5, 10, 15, 19}, N=20 runs, semilla fija.
"""

from __future__ import annotations

import os
import sys
import io
import json
import time
import random
import statistics
import contextlib
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np

# ------------------------------------------------------------
# sys.path: aux (este dir) y el padre (donde están las PDBs y los modulos
# pdb_similarity / corner_pdb / edge_pdb / pdbs_v3.pkl)
# ------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PARENT))

# Para que load_or_build_pdbs encuentre el cache pre-construido.
os.chdir(str(PARENT))

from fusecube_eda_edaspy import (
    FuseState, FuseUMDAProblem, apply_algorithm, apply_move, is_solved,
    MOVES, STOP_TOKEN, N_GENES, GENES,
    _ImprovementTracker, sample_initial_generation_from_frequency,
    make_inverse_biased_frequency, _make_frequency_no_repeat_face,
    entropy_of_distribution, umda_frequency_as_array, simplify, _bias_toward_best,
)
from solver_variants import (
    FuseProblemNoStop,
    make_inverse_biased_frequency_no_stop,
    make_no_repeat_face_no_stop,
    bias_toward_best_no_stop,
    ProductMaskedModel,
    make_markov_initial_P_pos,
    N_MOVES as VARIANTS_N_MOVES,
)
from pdb_similarity import load_or_build_pdbs, pdb_similarity, phased_state_reward
from EDAspy.optimization import UMDAcat


# ============================================================
#  Fábricas de variante
# ============================================================

VARIANT_BASELINE = "baseline"
VARIANT_NO_STOP = "no_stop"
VARIANT_MARKOV = "markov"
VARIANT_MARKOV_UNIFORM_T = "markov_uniformT"
VARIANT_MARKOV_ANCHOR = "markov_anchor"
MARKOV_VARIANTS = (VARIANT_MARKOV, VARIANT_MARKOV_UNIFORM_T, VARIANT_MARKOV_ANCHOR)


def make_problem(variant: str, scramble_alg, genome_length, length_penalty,
                 corner_pdb, edge_pdb):
    if variant == VARIANT_NO_STOP:
        return FuseProblemNoStop(
            scramble_alg=scramble_alg, genome_length=genome_length,
            length_penalty=length_penalty,
            corner_pdb=corner_pdb, edge_pdb=edge_pdb,
        )
    return FuseUMDAProblem(
        scramble_alg=scramble_alg, genome_length=genome_length,
        length_penalty=length_penalty,
        corner_pdb=corner_pdb, edge_pdb=edge_pdb,
    )


def make_freq_init(variant: str, scramble_alg, genome_length, r_idx,
                   best_overall_raw):
    """Devuelve la matriz de frecuencias inicial para el reinicio actual.

    Para variantes Markov, devuelve la marginal posicional P_pos inicial
    (mismo formato (gl, 9) que no_stop).
    """
    if variant in MARKOV_VARIANTS:
        return make_markov_initial_P_pos(
            scramble_alg=scramble_alg, genome_length=genome_length,
            r_idx=r_idx, best_overall_raw=best_overall_raw,
        )
    if variant == VARIANT_NO_STOP:
        base_freq = make_no_repeat_face_no_stop(genome_length)
        if best_overall_raw is not None:
            seed_genome = list(best_overall_raw[:genome_length])
            if len(seed_genome) < genome_length:
                # Sin STOP: rellena con muestreo aleatorio uniforme de movs.
                seed_genome.extend([MOVES[0]] * (genome_length - len(seed_genome)))
            return bias_toward_best_no_stop(
                base_freq, np.array(seed_genome, dtype=object), strength=0.35,
            )
        if r_idx == 0:
            return make_inverse_biased_frequency_no_stop(scramble_alg, genome_length)
        return base_freq

    # Baseline
    base_freq = _make_frequency_no_repeat_face(genome_length)
    if best_overall_raw is not None:
        seed_genome = list(best_overall_raw[:genome_length])
        if len(seed_genome) < genome_length:
            seed_genome.extend([STOP_TOKEN] * (genome_length - len(seed_genome)))
        return _bias_toward_best(
            base_freq, np.array(seed_genome, dtype=object), strength=0.35,
        )
    if r_idx == 0:
        return make_inverse_biased_frequency(scramble_alg, genome_length)
    return base_freq


# ============================================================
#  Generación reproducible de scrambles (sin repetir cara contigua)
# ============================================================

def random_scramble(depth: int, rng: random.Random) -> str:
    tokens: List[str] = []
    prev_face: Optional[str] = None
    for _ in range(depth):
        choices = [m for m in MOVES if m[0] != prev_face]
        m = rng.choice(choices)
        tokens.append(m)
        prev_face = m[0]
    return " ".join(tokens)


def inverse_alg(scramble: str) -> str:
    inv = {"L": "L'", "L'": "L", "L2": "L2",
           "D": "D'", "D'": "D", "D2": "D2",
           "B": "B'", "B'": "B", "B2": "B2"}
    return " ".join(inv[t] for t in reversed(scramble.split()))


# ============================================================
#  Tracker instrumentado
# ============================================================

class InstrumentedTracker(_ImprovementTracker):
    """Subclase que captura métricas por generación.

    Acumula los costes/individuos de cada batch de `size_gen` evaluaciones
    y, al cerrarse el batch, registra una fila en `gen_logs`.
    """

    def __init__(self, problem, restart_idx, verbose, size_gen, alpha: float):
        super().__init__(problem, restart_idx, verbose, size_gen)
        self.size_gen = size_gen
        self.alpha = alpha
        self.gen_logs: List[Dict] = []
        self._buf_costs: List[float] = []
        self._buf_inds: List[np.ndarray] = []
        # Hook opcional para entropía no-UMDA (p.ej. Markov):
        # callable() -> float | None. Si None, se cae al cálculo via umda.pm.
        self.entropy_provider = None

        # Estado/coste inicial (secuencia vacía). Cualquier individuo con
        # coste menor habrá encontrado un prefijo mejor que el scramble inicial.
        self.initial_reward = phased_state_reward(
            problem.scramble_state, problem.corner_pdb, problem.edge_pdb
        )
        self.initial_cost = 1.0 - self.initial_reward
        self.initial_pdb_dist = 1.0 - pdb_similarity(
            problem.scramble_state, problem.corner_pdb, problem.edge_pdb
        )

    def __call__(self, solution: np.ndarray) -> float:
        cost = super().__call__(solution)
        self._buf_costs.append(cost)
        self._buf_inds.append(np.array(solution, dtype=object, copy=True))
        if len(self._buf_costs) >= self.size_gen:
            self._snapshot_gen()
            self._buf_costs.clear()
            self._buf_inds.clear()
        return cost

    def _snapshot_gen(self) -> None:
        costs = np.asarray(self._buf_costs, dtype=float)
        n = len(costs)
        k = max(1, int(self.alpha * n))
        sort_idx = np.argsort(costs)
        elite_costs = costs[sort_idx[:k]]
        best_idx = int(sort_idx[0])
        best_cost = float(costs[best_idx])
        best_ind = self._buf_inds[best_idx]

        # Entropía del modelo (distribución con la que se acaba de muestrear/se va a muestrear)
        if self.entropy_provider is not None:
            ent = float(self.entropy_provider())
        else:
            freq = umda_frequency_as_array(self.umda, self.problem.possible_values)
            ent = float(entropy_of_distribution(freq)) if freq is not None else float("nan")

        # Diversidad fenotípica (proxy: nº de costes únicos en la élite, redondeados)
        elite_unique = int(len({round(float(c), 4) for c in elite_costs}))

        # % individuos cuyo coste mejora el inicial (== max-prefix encontró algo mejor que el scramble)
        thr = self.initial_cost - 1e-9
        pop_better = float(np.mean(costs < thr))
        elite_better = float(np.mean(elite_costs < thr))

        # Métricas STOP / max-prefix para el best individuo de la gen.
        # En variantes sin STOP, learned_len == genome_length por convención.
        learned_len = len(best_ind)
        for i, x in enumerate(best_ind):
            if str(x) == STOP_TOKEN:
                learned_len = i
                break

        tokens = self.problem.decode(best_ind)  # ya aplica simplify
        decoded_len = len(tokens)

        # Recorre la decodificación buscando el paso del mejor reward (max-prefix)
        st = self.problem.scramble_state
        best_r = self.initial_reward
        best_step = 0
        best_state = st
        for i, tok in enumerate(tokens, start=1):
            st = apply_move(st, tok)
            r = phased_state_reward(st, self.problem.corner_pdb, self.problem.edge_pdb)
            if r > best_r + 1e-12:
                best_r = r
                best_step = i
                best_state = st
        max_prefix_len = best_step
        best_prefix_pdb_dist = 1.0 - pdb_similarity(
            best_state, self.problem.corner_pdb, self.problem.edge_pdb
        )

        self.gen_logs.append({
            "gen": len(self.gen_logs),
            "entropy": ent,
            "best_cost": best_cost,
            "best_prefix_pdb_dist": best_prefix_pdb_dist,
            "elite_unique_costs": elite_unique,
            "elite_size": k,
            "pop_pct_better_than_initial": pop_better,
            "elite_pct_better_than_initial": elite_better,
            "best_learned_len": int(learned_len),
            "best_decoded_len": int(decoded_len),
            "best_max_prefix_len": int(max_prefix_len),
        })


# ============================================================
#  Driver de un EDA single-restart con instrumentación
# ============================================================

def _run_markov_eda(
    problem,
    tracker: "InstrumentedTracker",
    model: ProductMaskedModel,
    size_gen: int,
    max_iter: int,
    dead_iter: int,
    alpha: float,
    rng,
    prefix_anchored: Optional[List[str]] = None,
    eval_budget_remaining: Optional[List[int]] = None,
):
    """Bucle EDA propio para el modelo Markov producto.

    Si se pasa `prefix_anchored`, cada evaluación se hace sobre la secuencia
    completa (prefix_anchored + tokens_genome). El modelo Markov solo decide
    la parte del genome (lo anclado es inmutable durante la fase).

    `eval_budget_remaining` es un contenedor mutable [N] que se decrementa por
    cada eval; permite repartir un budget total entre fases sin pasar de él.

    Devuelve (best_raw_phase, best_cost_phase, ran_to_completion):
      best_raw_phase: SOLO los tokens del genome de fase (no incluye prefix).
      best_cost_phase: coste min(gen_costs) visto durante esta fase (en términos
                       de la secuencia COMPLETA = prefix + tokens).
      ran_to_completion: True si terminó por max_iter o dead_iter, False si
                         budget agotado.
    """
    if prefix_anchored is None:
        prefix_anchored = []
    best_phase_cost = float("inf")
    best_phase_raw_ints = None
    iters_no_improve = 0
    elite_n = max(1, int(alpha * size_gen))
    ran_to_completion = True

    for _ in range(max_iter):
        ints = model.sample(size_gen, rng)                # (size_gen, gl) int
        gen_costs = np.empty(size_gen, dtype=np.float64)
        budget_exhausted = False
        for k in range(size_gen):
            tokens_obj = np.array(
                prefix_anchored + [MOVES[a] for a in ints[k]],
                dtype=object,
            )
            c = tracker(tokens_obj)
            gen_costs[k] = c
            if eval_budget_remaining is not None:
                eval_budget_remaining[0] -= 1
                if eval_budget_remaining[0] <= 0:
                    budget_exhausted = True
                    # Trunca gen_costs a las evaluadas
                    gen_costs = gen_costs[:k + 1]
                    ints = ints[:k + 1]
                    break

        if gen_costs.size:
            gen_best = float(gen_costs.min())
            if gen_best < best_phase_cost - 1e-9:
                best_phase_cost = gen_best
                iters_no_improve = 0
                best_phase_raw_ints = ints[int(np.argmin(gen_costs))].copy()
            else:
                iters_no_improve += 1

        if budget_exhausted:
            ran_to_completion = False
            break

        # Salida temprana: si ya hemos resuelto (coste centinela <= -50), no
        # malgastar el resto del presupuesto en gens que no pueden mejorar.
        # Evita que anchoring lance una fase nueva tras un solve trivial.
        if best_phase_cost <= -50.0:
            break

        if iters_no_improve >= dead_iter:
            break

        # Selección élite + update
        if gen_costs.size >= elite_n:
            elite_idx = np.argsort(gen_costs)[:elite_n]
            model.update_from_elite(ints[elite_idx])

    if best_phase_raw_ints is None:
        best_phase_raw = None
    else:
        best_phase_raw = np.array([MOVES[a] for a in best_phase_raw_ints], dtype=object)
    return best_phase_raw, best_phase_cost, ran_to_completion


def _compute_max_prefix(
    tokens: List[str],
    state_start,
    corner_pdb,
    edge_pdb,
) -> Tuple[List[str], object, float]:
    """Recorre tokens desde state_start y devuelve (prefix_hasta_mejor, state_mejor, reward_mejor)."""
    st = state_start
    best_reward = phased_state_reward(st, corner_pdb, edge_pdb)
    best_step = 0
    best_state = st
    for i, tok in enumerate(tokens, 1):
        st = apply_move(st, tok)
        if is_solved(st):
            return tokens[:i], st, 1.0
        r = phased_state_reward(st, corner_pdb, edge_pdb)
        if r > best_reward + 1e-12:
            best_reward = r
            best_step = i
            best_state = st
    return tokens[:best_step], best_state, best_reward


def _run_markov_anchored(
    problem,
    tracker: "InstrumentedTracker",
    initial_P_pos: np.ndarray,
    size_gen: int,
    total_eval_budget: int,
    anchor_stagnation_gens: int,
    alpha: float,
    rng_seed: int,
    max_phases: int = 3,
    min_phase_genome: int = 6,
):
    """Markov producto con anchoring por estancamiento.

    Fases:
      - phase 0: P_pos sembrada con `initial_P_pos`; sin prefix anclado.
      - phase >=1: P_pos uniforme con máscara; prefix anclado = mejor max-prefix
        encontrado hasta el momento.

    Disparador del anchor: `anchor_stagnation_gens` generaciones sin mejora en
    la fase actual. Si una fase no encuentra mejora del best global, no se
    actualiza el anchor.

    Budget total = `total_eval_budget` evals, repartidos como
        budget_fase = remaining / phases_restantes.
    """
    rng = np.random.default_rng(rng_seed)
    eval_budget_remaining = [int(total_eval_budget)]

    prefix_anchored: List[str] = []
    phase_log: List[Dict] = []

    for phase_idx in range(max_phases):
        if eval_budget_remaining[0] <= 0:
            break

        # Genome de fase: el largo original menos el prefix ya anclado.
        phase_genome_length = max(
            min_phase_genome,
            problem.genome_length - len(prefix_anchored),
        )

        if phase_idx == 0:
            P_pos_init = initial_P_pos
            # Si initial_P_pos no encaja en phase_genome_length, recortar/rellenar.
            if P_pos_init.shape[0] != phase_genome_length:
                P_pos_init = P_pos_init[:phase_genome_length] if P_pos_init.shape[0] > phase_genome_length \
                    else make_no_repeat_face_no_stop(phase_genome_length)
        else:
            P_pos_init = make_no_repeat_face_no_stop(phase_genome_length)

        model = ProductMaskedModel(
            genome_length=phase_genome_length,
            P_pos_init=P_pos_init,
            laplace=1.0,
            learn_T=True,
        )
        tracker.entropy_provider = model.entropy_avg

        phases_remaining = max_phases - phase_idx
        phase_budget = max(size_gen, eval_budget_remaining[0] // phases_remaining)
        phase_max_iter = max(1, phase_budget // size_gen)

        # Snapshot del best_cost del tracker antes de esta fase
        cost_before_phase = tracker.best_cost

        phase_eval_budget = [phase_budget]
        # Pasamos eval_budget_remaining ligado al global Y al de la fase
        # tomando el mínimo. Lo encadenamos en _run_markov_eda con el global.
        # (Simplificación: usamos solo el global con un cap por fase.)
        cap = min(phase_budget, eval_budget_remaining[0])
        # Truco: contador local para detectar fin de fase por budget;
        # _run_markov_eda decrementa eval_budget_remaining.
        # Para evitar pasarse, usamos un sentinel envoltorio.
        budget_at_phase_start = eval_budget_remaining[0]
        phase_floor = budget_at_phase_start - cap  # evals_remaining cuando la fase termine

        # Envolvemos eval_budget_remaining: el bucle _run_markov_eda parará
        # cuando llegue a 0 o por max_iter/dead_iter. Para limitarlo a cap,
        # decrementamos cap usando un wrapper local.
        class _Counter(list):
            pass

        local_budget = _Counter([cap])

        def _decrement_callback():
            pass  # no se usa; el contador se modifica directamente abajo

        # Como _run_markov_eda admite eval_budget_remaining directamente:
        # le pasamos local_budget y, en paralelo, decrementamos el global
        # tras la fase mirando cuánto consumió.

        phase_best_raw, phase_best_cost, ran_to_completion = _run_markov_eda(
            problem=problem,
            tracker=tracker,
            model=model,
            size_gen=size_gen,
            max_iter=phase_max_iter,
            dead_iter=anchor_stagnation_gens,
            alpha=alpha,
            rng=rng,
            prefix_anchored=prefix_anchored,
            eval_budget_remaining=local_budget,
        )

        evals_consumed = cap - local_budget[0]
        eval_budget_remaining[0] -= evals_consumed

        phase_log.append({
            "phase_idx": phase_idx,
            "phase_genome_length": phase_genome_length,
            "phase_budget": cap,
            "evals_consumed": evals_consumed,
            "prefix_len_in": len(prefix_anchored),
            "cost_before_phase": float(cost_before_phase),
            "phase_best_cost": float(phase_best_cost),
            "tracker_best_after": float(tracker.best_cost),
        })

        # Si ya está resuelto, salir
        if tracker.best_cost <= -50.0:
            break

        # Para anclar, usar el MEJOR GLOBAL conocido por el tracker
        if tracker.best_solution is None:
            break
        best_full_tokens = problem.decode(tracker.best_solution)  # ya simplificado
        # Calcular max-prefix desde el scramble original
        new_anchor, _, _ = _compute_max_prefix(
            best_full_tokens, problem.scramble_state,
            problem.corner_pdb, problem.edge_pdb,
        )

        # Si el nuevo anchor no extiende lo ya anclado, no avanzar
        if len(new_anchor) <= len(prefix_anchored):
            # No hubo progreso útil → terminar
            break
        prefix_anchored = list(new_anchor)

    return phase_log


def run_instrumented_restart(
    scramble_alg: str,
    genome_length: int,
    size_gen: int,
    max_iter: int,
    dead_iter: int,
    alpha: float,
    length_penalty: float,
    freq_init: np.ndarray,
    restart_idx: int,
    corner_pdb,
    edge_pdb,
    variant: str = VARIANT_BASELINE,
    rng_seed: int = 0,
) -> Dict:
    problem = make_problem(
        variant=variant,
        scramble_alg=scramble_alg,
        genome_length=genome_length,
        length_penalty=length_penalty,
        corner_pdb=corner_pdb,
        edge_pdb=edge_pdb,
    )
    problem.frequency = freq_init

    tracker = InstrumentedTracker(
        problem, restart_idx, verbose=False, size_gen=size_gen, alpha=alpha
    )

    phase_log_anchor = None
    if variant in MARKOV_VARIANTS:
        tracker.umda = None  # desactiva warm-restart heredado del baseline

        if variant == VARIANT_MARKOV_ANCHOR:
            # Budget total comparable a markov puro: max_iter*size_gen evals.
            total_budget = max_iter * size_gen
            t0 = time.perf_counter()
            phase_log_anchor = _run_markov_anchored(
                problem=problem,
                tracker=tracker,
                initial_P_pos=np.asarray(freq_init, dtype=np.float64),
                size_gen=size_gen,
                total_eval_budget=total_budget,
                anchor_stagnation_gens=dead_iter,  # reuse dead_iter como stag threshold
                alpha=alpha,
                rng_seed=rng_seed,
                max_phases=3,
                min_phase_genome=6,
            )
            elapsed = time.perf_counter() - t0
        else:
            learn_T = (variant == VARIANT_MARKOV)
            model = ProductMaskedModel(
                genome_length=genome_length,
                P_pos_init=np.asarray(freq_init, dtype=np.float64),
                laplace=1.0,
                learn_T=learn_T,
            )
            tracker.entropy_provider = model.entropy_avg
            rng = np.random.default_rng(rng_seed)
            t0 = time.perf_counter()
            _markov_best_raw, _markov_best_cost, _ = _run_markov_eda(
                problem, tracker, model,
                size_gen=size_gen, max_iter=max_iter, dead_iter=dead_iter,
                alpha=alpha, rng=rng,
            )
            elapsed = time.perf_counter() - t0

        if tracker._buf_costs:
            tracker._snapshot_gen()

        # Mejor del tracker prevalece (incluye -100 si se resolvió)
        best_raw = tracker.best_solution if tracker.best_solution is not None else np.array([], dtype=object)
        best_cost = float(tracker.best_cost) if tracker.best_cost != float("inf") else 1.0
    else:
        umda = UMDAcat(
            size_gen=size_gen,
            max_iter=max_iter,
            dead_iter=dead_iter,
            n_variables=genome_length,
            alpha=alpha,
            frequency=problem.frequency,
            possible_values=problem.possible_values,
            init_data=sample_initial_generation_from_frequency(
                problem.possible_values, problem.frequency, size_gen,
            ),
        )
        tracker.umda = umda

        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            result = umda.minimize(tracker, verbose=False)
        elapsed = time.perf_counter() - t0

        if tracker._buf_costs:
            tracker._snapshot_gen()

        if tracker.best_solution is not None and tracker.best_cost <= float(result.best_cost) + 1e-12:
            best_raw = tracker.best_solution
            best_cost = float(tracker.best_cost)
        else:
            best_raw = result.best_ind
            best_cost = float(result.best_cost)

    best_tokens = problem.decode(np.array(best_raw, dtype=object))
    # Solved se decide vía MAX-PREFIX (acorde con la función de fitness):
    # algún prefijo de best_tokens debe llevar a estado resuelto.
    solved = False
    state_step = problem.scramble_state
    solving_prefix = best_tokens
    for i, tok in enumerate(best_tokens, 1):
        state_step = apply_move(state_step, tok)
        if is_solved(state_step):
            solved = True
            solving_prefix = best_tokens[:i]
            break
    best_state = state_step if solved else apply_algorithm(problem.scramble_state, " ".join(best_tokens))
    if solved:
        best_tokens = solving_prefix

    # Generación en que se estanca el mejor coste (último mejor coste estricto)
    stagnation_gen = None
    best_so_far = float("inf")
    for log in tracker.gen_logs:
        if log["best_cost"] < best_so_far - 1e-9:
            best_so_far = log["best_cost"]
            stagnation_gen = log["gen"]
    if stagnation_gen is None and tracker.gen_logs:
        stagnation_gen = 0

    return {
        "genome_length": genome_length,
        "elapsed_seconds": elapsed,
        "solved": solved,
        "best_cost": best_cost,
        "best_tokens": best_tokens,
        "stagnation_gen": stagnation_gen,
        "n_gens_run": len(tracker.gen_logs),
        "n_evals_total": int(tracker.n_evals),
        "gen_logs": tracker.gen_logs,
        "phase_log_anchor": phase_log_anchor if variant == VARIANT_MARKOV_ANCHOR else None,
        "initial_pdb_dist": tracker.initial_pdb_dist,
        "initial_cost": tracker.initial_cost,
        "_best_raw": list(best_tokens),
    }


def solve_instrumented(
    scramble_alg: str,
    size_gen: int,
    max_iter: int,
    dead_iter: int,
    alpha: float,
    length_penalty: float,
    min_genome: int,
    max_genome: int,
    genome_step: int,
    n_restarts: int,
    seed: int,
    corner_pdb,
    edge_pdb,
    variant: str = VARIANT_BASELINE,
) -> Dict:
    np.random.seed(seed)
    random.seed(seed)

    genome_seq: List[int] = []
    gl = min_genome
    for _ in range(n_restarts):
        genome_seq.append(gl)
        gl = min(gl + genome_step, max_genome)

    best_overall: Optional[Dict] = None
    all_restart_logs: List[Dict] = []
    t0_total = time.perf_counter()

    for r_idx, gl in enumerate(genome_seq):
        best_raw_for_seed = None
        if best_overall is not None and best_overall["best_cost"] < 1.0:
            best_raw_for_seed = np.array(best_overall["_best_raw"], dtype=object)
        freq_init = make_freq_init(
            variant=variant, scramble_alg=scramble_alg, genome_length=gl,
            r_idx=r_idx, best_overall_raw=best_raw_for_seed,
        )

        r = run_instrumented_restart(
            scramble_alg=scramble_alg,
            genome_length=gl,
            size_gen=size_gen,
            max_iter=max_iter,
            dead_iter=dead_iter,
            alpha=alpha,
            length_penalty=length_penalty,
            freq_init=freq_init,
            restart_idx=r_idx + 1,
            corner_pdb=corner_pdb,
            edge_pdb=edge_pdb,
            variant=variant,
            rng_seed=seed + r_idx * 97,
        )
        r["restart_idx"] = r_idx + 1
        all_restart_logs.append(r)

        if best_overall is None or r["best_cost"] < best_overall["best_cost"]:
            best_overall = r

        if r["solved"]:
            break

    elapsed = time.perf_counter() - t0_total
    if best_overall is None:
        best_overall = {"solved": False, "best_cost": float("inf"), "_best_raw": []}

    return {
        "scramble": scramble_alg,
        "seed": seed,
        "solved": best_overall["solved"],
        "best_cost": best_overall["best_cost"],
        "elapsed_seconds": elapsed,
        "n_restarts_used": len(all_restart_logs),
        "restarts": all_restart_logs,
    }


# ============================================================
#  Agregación por profundidad
# ============================================================

def aggregate_runs(runs: List[Dict]) -> Dict:
    n = len(runs)
    n_solved = sum(1 for r in runs if r["solved"])

    # Métricas tomadas de la ÚLTIMA gen del MEJOR reinicio de cada run (el que minimiza best_cost)
    entropies, divs, pop_betters, elite_betters = [], [], [], []
    learned_lens, max_prefix_lens, decoded_lens = [], [], []
    stagnation_gens, n_gens = [], []
    best_pdb_dists = []
    init_pdb = None

    for r in runs:
        if not r["restarts"]:
            continue
        # Reinicio que produjo el best
        best_r = min(r["restarts"], key=lambda x: x["best_cost"])
        logs = best_r["gen_logs"]
        if not logs:
            continue
        last = logs[-1]
        entropies.append(last["entropy"])
        divs.append(last["elite_unique_costs"])
        pop_betters.append(last["pop_pct_better_than_initial"])
        elite_betters.append(last["elite_pct_better_than_initial"])
        learned_lens.append(last["best_learned_len"])
        max_prefix_lens.append(last["best_max_prefix_len"])
        decoded_lens.append(last["best_decoded_len"])
        best_pdb_dists.append(last["best_prefix_pdb_dist"])
        stagnation_gens.append(best_r["stagnation_gen"] or 0)
        n_gens.append(best_r["n_gens_run"])
        init_pdb = best_r["initial_pdb_dist"]

    def _stat(xs: List[float]) -> Tuple[float, float]:
        if not xs:
            return float("nan"), float("nan")
        return float(statistics.mean(xs)), float(statistics.median(xs))

    return {
        "n_runs": n,
        "success_rate": n_solved / n if n else 0.0,
        "initial_pdb_dist": init_pdb,
        "entropy_final_mean": _stat(entropies)[0],
        "entropy_final_median": _stat(entropies)[1],
        "elite_unique_costs_median": _stat(divs)[1],
        "pop_pct_better_than_initial_median": _stat(pop_betters)[1],
        "elite_pct_better_than_initial_median": _stat(elite_betters)[1],
        "best_learned_len_median": _stat(learned_lens)[1],
        "best_max_prefix_len_median": _stat(max_prefix_lens)[1],
        "best_decoded_len_median": _stat(decoded_lens)[1],
        "best_prefix_pdb_dist_median": _stat(best_pdb_dists)[1],
        "stagnation_gen_median": _stat(stagnation_gens)[1],
        "gens_run_median": _stat(n_gens)[1],
    }


# ============================================================
#  Main benchmark
# ============================================================

def run_benchmark(
    depths: Tuple[int, ...] = (5, 10, 15, 19),
    n_runs: int = 20,
    base_seed: int = 1000,
    size_gen: int = 400,
    max_iter: int = 120,
    dead_iter: int = 30,
    alpha: float = 0.30,
    length_penalty: float = 0.0005,
    genome_step: int = 2,
    n_restarts: int = 3,
    output_json: str = "diagnostico.json",
    variant: str = VARIANT_BASELINE,
    n_runs_map: Optional[Dict[int, int]] = None,
) -> Dict:
    print("Cargando PDBs (puede tardar la primera vez)...", flush=True)
    t_pdb = time.perf_counter()
    corner_pdb, edge_pdb = load_or_build_pdbs()
    print(f"PDBs cargadas en {time.perf_counter()-t_pdb:.1f}s", flush=True)

    all_results: Dict[int, List[Dict]] = {}
    for d in depths:
        n_runs_d = (n_runs_map.get(d, n_runs) if n_runs_map else n_runs)
        print(f"\n{'='*64}\n  PROFUNDIDAD {d}  (n_runs={n_runs_d})\n{'='*64}", flush=True)

        # Scrambles deterministas: si existe scrambles.json en el directorio del
        # script, se carga de ahí (fuente canónica versionada en el repo).
        # Si no, fallback al generador determinista in-place con la misma semilla.
        scrambles_file = HERE / "scrambles.json"
        if scrambles_file.exists():
            payload_scr = json.loads(scrambles_file.read_text(encoding="utf-8"))
            scrambles_all = payload_scr["depths"].get(str(d), [])
            if len(scrambles_all) < n_runs_d:
                raise RuntimeError(
                    f"scrambles.json contiene {len(scrambles_all)} scrambles para d={d}, "
                    f"se piden {n_runs_d}. Regenera scrambles.json con más."
                )
            scrambles = scrambles_all[:n_runs_d]
        else:
            scr_rng = random.Random(base_seed + d * 1000)
            max_runs = max(n_runs_d, 50)
            scrambles_all = [random_scramble(d, scr_rng) for _ in range(max_runs)]
            scrambles = scrambles_all[:n_runs_d]

        runs: List[Dict] = []
        for i, scr in enumerate(scrambles):
            run_seed = base_seed + d * 1000 + i
            # Para no truncar el sesgo inverso, min_genome = d, max_genome = d+6
            r = solve_instrumented(
                scramble_alg=scr,
                size_gen=size_gen,
                max_iter=max_iter,
                dead_iter=dead_iter,
                alpha=alpha,
                length_penalty=length_penalty,
                min_genome=d,
                max_genome=d + 6,
                genome_step=genome_step,
                n_restarts=n_restarts,
                seed=run_seed,
                corner_pdb=corner_pdb,
                edge_pdb=edge_pdb,
                variant=variant,
            )
            runs.append(r)
            tag = "OK" if r["solved"] else "--"
            print(
                f"  [d={d} {i+1:02d}/{n_runs_d}] {tag}  "
                f"best_cost={r['best_cost']:+.4f}  "
                f"restarts={r['n_restarts_used']}  "
                f"t={r['elapsed_seconds']:.1f}s",
                flush=True,
            )

        all_results[d] = runs
        summary = aggregate_runs(runs)
        print(f"\n  Resumen d={d}: success={summary['success_rate']*100:.0f}%  "
              f"entropy_final={summary['entropy_final_median']:.3f}  "
              f"elite_div={summary['elite_unique_costs_median']:.1f}  "
              f"pop%_mejor_init={summary['pop_pct_better_than_initial_median']*100:.1f}%",
              flush=True)

    # Persistir
    out_path = HERE / output_json
    serializable = {
        "config": {
            "variant": variant,
            "depths": list(depths), "n_runs": n_runs, "base_seed": base_seed,
            "size_gen": size_gen, "max_iter": max_iter, "dead_iter": dead_iter,
            "alpha": alpha, "length_penalty": length_penalty,
            "genome_step": genome_step, "n_restarts": n_restarts,
        },
        "summaries": {int(d): aggregate_runs(rs) for d, rs in all_results.items()},
        "per_run": {
            int(d): [
                {
                    "seed": r["seed"], "scramble": r["scramble"],
                    "solved": r["solved"], "best_cost": r["best_cost"],
                    "n_restarts_used": r["n_restarts_used"],
                    "elapsed_seconds": r["elapsed_seconds"],
                    "restart_summaries": [
                        {
                            "restart_idx": rs["restart_idx"],
                            "genome_length": rs["genome_length"],
                            "solved": rs["solved"],
                            "best_cost": rs["best_cost"],
                            "stagnation_gen": rs["stagnation_gen"],
                            "n_gens_run": rs["n_gens_run"],
                            "n_evals_total": rs.get("n_evals_total"),
                            "initial_pdb_dist": rs["initial_pdb_dist"],
                            "elapsed_seconds": rs["elapsed_seconds"],
                            "phase_log_anchor": rs.get("phase_log_anchor"),
                            "gen_logs": rs["gen_logs"],
                        } for rs in r["restarts"]
                    ],
                } for r in rs_list
            ] for d, rs_list in all_results.items()
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nGuardado: {out_path}", flush=True)
    return serializable


def print_summary_table(payload: Dict) -> None:
    summaries = payload["summaries"]
    headers = [
        "depth", "success%", "init_pdb_d", "gens_run", "stag_gen",
        "entropy", "elite_div", "pop%>init", "elite%>init",
        "learn_len", "maxpref_len", "decod_len", "best_pdb_d",
    ]
    print("\n" + "=" * 120)
    print("RESUMEN POR PROFUNDIDAD")
    print("=" * 120)
    print("  " + "  ".join(f"{h:>11s}" for h in headers))
    for d in sorted(summaries.keys(), key=int):
        s = summaries[d]
        row = [
            f"{int(d):>11d}",
            f"{s['success_rate']*100:>10.1f}%",
            f"{(s.get('initial_pdb_dist') or 0):>11.4f}",
            f"{s['gens_run_median']:>11.1f}",
            f"{s['stagnation_gen_median']:>11.1f}",
            f"{s['entropy_final_median']:>11.3f}",
            f"{s['elite_unique_costs_median']:>11.1f}",
            f"{s['pop_pct_better_than_initial_median']*100:>10.1f}%",
            f"{s['elite_pct_better_than_initial_median']*100:>10.1f}%",
            f"{s['best_learned_len_median']:>11.1f}",
            f"{s['best_max_prefix_len_median']:>11.1f}",
            f"{s['best_decoded_len_median']:>11.1f}",
            f"{s['best_prefix_pdb_dist_median']:>11.4f}",
        ]
        print("  " + "  ".join(row))
    print("=" * 120)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[5, 10, 15, 19])
    ap.add_argument("--n_runs", type=int, default=20)
    ap.add_argument("--size_gen", type=int, default=400)
    ap.add_argument("--max_iter", type=int, default=120)
    ap.add_argument("--dead_iter", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.30)
    ap.add_argument("--n_restarts", type=int, default=3)
    ap.add_argument("--genome_step", type=int, default=2)
    ap.add_argument("--base_seed", type=int, default=1000)
    ap.add_argument(
        "--n_runs_per_depth",
        default=None,
        help='Override por profundidad, p.ej. "5:20,10:20,15:50,19:50". Si se da, --n_runs queda como fallback.',
    )
    ap.add_argument("--output", default="diagnostico.json")
    ap.add_argument(
        "--variant",
        choices=[VARIANT_BASELINE, VARIANT_NO_STOP, VARIANT_MARKOV,
                 VARIANT_MARKOV_UNIFORM_T, VARIANT_MARKOV_ANCHOR],
        default=VARIANT_BASELINE,
    )
    args = ap.parse_args()

    n_runs_map = None
    if args.n_runs_per_depth:
        n_runs_map = {}
        for spec in args.n_runs_per_depth.split(","):
            d_str, n_str = spec.split(":")
            n_runs_map[int(d_str)] = int(n_str)

    payload = run_benchmark(
        depths=tuple(args.depths),
        n_runs=args.n_runs,
        base_seed=args.base_seed,
        size_gen=args.size_gen,
        max_iter=args.max_iter,
        dead_iter=args.dead_iter,
        alpha=args.alpha,
        genome_step=args.genome_step,
        n_restarts=args.n_restarts,
        output_json=args.output,
        variant=args.variant,
        n_runs_map=n_runs_map,
    )
    print_summary_table(payload)
