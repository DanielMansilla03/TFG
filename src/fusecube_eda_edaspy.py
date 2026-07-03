"""
fusecube_eda_edaspy.py
======================
Solver EDA para el cubo Fuse 3×3×3 usando UMDAcat de EDAspy.

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import csv, random, statistics, time

import numpy as np
from pdb_similarity import (
    pdb_similarity,
    load_or_build_pdbs,
    evaluate_phased_max_prefix,
    phased_state_reward,
)
from EDAspy.optimization import UMDAcat


# ============================================================
#  INFRAESTRUCTURA DEL CUBO FUSE
# ============================================================

CORNER_SLOTS: Tuple[str, ...] = ("UFL","ULB","UBR","DFR","DLF","DBL","DRB")
EDGE_SLOTS:   Tuple[str, ...] = ("UL","UB","DR","DF","DL","DB","FL","BL","BR")

FIXED_BLOCK = {
    "corner":  "URF",
    "edges":   ("UR","UF","FR"),
    "centers": ("U","R","F"),
}


@dataclass(frozen=True)
class FuseState:
    cp: Tuple[int, ...]
    co: Tuple[int, ...]
    ep: Tuple[int, ...]
    eo: Tuple[int, ...]

    @staticmethod
    def solved() -> "FuseState":
        return FuseState(tuple(range(7)), (0,)*7, tuple(range(9)), (0,)*9)


BASE_MOVES: Dict = {
    "L": {"corners": ((1,5,2,3,0,4,6),(1,2,0,0,2,1,0)),
          "edges":   ((7,1,2,3,6,5,0,4,8),(0,0,0,0,0,0,0,0,0))},
    "D": {"corners": ((0,1,2,4,5,6,3),(0,0,0,0,0,0,0)),
          "edges":   ((0,1,3,4,5,2,6,7,8),(0,0,0,0,0,0,0,0,0))},
    "B": {"corners": ((0,2,6,3,4,1,5),(0,1,2,0,0,2,1)),
          "edges":   ((0,8,2,3,4,7,6,1,5),(0,1,0,0,0,1,0,1,1))},
}


def apply_base_move(state: FuseState, move: str) -> FuseState:
    cp_p, co_d = BASE_MOVES[move]["corners"]
    ep_p, eo_d = BASE_MOVES[move]["edges"]
    return FuseState(
        tuple(state.cp[cp_p[d]] for d in range(7)),
        tuple((state.co[cp_p[d]] + co_d[d]) % 3 for d in range(7)),
        tuple(state.ep[ep_p[d]] for d in range(9)),
        tuple((state.eo[ep_p[d]] + eo_d[d]) % 2 for d in range(9)),
    )


def apply_move(state: FuseState, move: str) -> FuseState:
    move = move.strip()
    if move in {"L","D","B"}:
        return apply_base_move(state, move)
    if len(move) == 2 and move[1] == "2":
        return apply_base_move(apply_base_move(state, move[0]), move[0])
    if len(move) == 2 and move[1] == "'":
        s = state
        for _ in range(3):
            s = apply_base_move(s, move[0])
        return s
    raise ValueError(f"Movimiento no permitido: {move}")


def apply_algorithm(state: FuseState, algorithm: str) -> FuseState:
    s = state
    for tok in algorithm.split():
        if tok:
            s = apply_move(s, tok)
    return s


def is_solved(state: FuseState) -> bool:
    return state == FuseState.solved()


def validate_state(state: FuseState) -> None:
    if sorted(state.cp) != list(range(7)):
        raise ValueError("Permutación de esquinas no válida.")
    if sorted(state.ep) != list(range(9)):
        raise ValueError("Permutación de aristas no válida.")


def pretty_state(state: FuseState) -> str:
    lines = ["=== ESQUINAS ==="]
    for i, slot in enumerate(CORNER_SLOTS):
        lines.append(f"  {slot}: pieza={CORNER_SLOTS[state.cp[i]]}  ori={state.co[i]}")
    lines.append("=== ARISTAS ===")
    for i, slot in enumerate(EDGE_SLOTS):
        lines.append(f"  {slot}: pieza={EDGE_SLOTS[state.ep[i]]}  ori={state.eo[i]}")
    return "\n".join(lines)


# ============================================================
#  VOCABULARIO DE MOVIMIENTOS Y GENOTIPO
# ============================================================

MOVES:      List[str]      = ["L","L2","L'","D","D2","D'","B","B2","B'"]
STOP_TOKEN: str            = "STOP"
GENES:      List[str]      = MOVES + [STOP_TOKEN]
N_MOVES:    int            = len(MOVES)
N_GENES:    int            = len(GENES)
MOVE_FACES: List[str]      = [m[0] for m in MOVES]


def _face_power(move: str) -> Tuple[str, int]:
    f = move[0]
    return f, (1 if len(move) == 1 else (2 if move[1] == "2" else 3))


def _power_to_move(face: str, power: int) -> Optional[str]:
    return {0: None, 1: face, 2: face+"2", 3: face+"'"}[power % 4]


def simplify(tokens: List[str]) -> List[str]:
    """Cancela movimientos adyacentes en la misma cara: L L → L2, L L' → ∅."""
    out: List[str] = []
    for m in tokens:
        if not out:
            out.append(m); continue
        f1, p1 = _face_power(out[-1])
        f2, p2 = _face_power(m)
        if f1 == f2:
            mg = _power_to_move(f1, p1 + p2)
            out.pop()
            if mg:
                out.append(mg)
        else:
            out.append(m)
    return out


# ============================================================
#  FUNCIÓN DE SIMILITUD (v3)
# ============================================================

def _cycle_scores(perm: Tuple[int, ...]) -> List[float]:
    """
    Devuelve para cada pieza i su score de posición = 1 / longitud_del_ciclo.

    Fixed point (posición correcta): score = 1.0
    2-ciclo (un intercambio): score = 0.5
    k-ciclo: score = 1/k

    Suma total = número de ciclos = mismo valor que _count_cycles, pero
    la versión per-pieza permite combinarla con la orientación individual.
    """
    n = len(perm)
    visited = [False] * n
    scores  = [0.0] * n
    for i in range(n):
        if not visited[i]:
            cycle: List[int] = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle.append(j)
                j = perm[j]
            s = 1.0 / len(cycle)
            for k in cycle:
                scores[k] = s
    return scores


def similarity_to_solved(state: FuseState) -> float:
    """
    Similitud mejorada (v3) ∈ [0, 1]. Devuelve 1.0 solo si resuelto.

    Score de posición por pieza: 1/longitud_ciclo.
    Score combinado: score_pos × 1.5 si orientación también correcta,
                     score_pos × 1.0 si orientación incorrecta.
    Normalizado para que pieza completamente correcta contribuya 1.
    Bonus cuadrático para el tramo final (similitud > 0.7).
    """
    cs = _cycle_scores(state.cp)   # scores de posición por esquina
    es = _cycle_scores(state.ep)   # scores de posición por arista

    # Combinado posición + orientación individual
    pc = sum(cs[i] * (1.5 if state.co[i] == 0 else 1.0) for i in range(7)) / (7 * 1.5)
    pe = sum(es[i] * (1.5 if state.eo[i] == 0 else 1.0) for i in range(9)) / (9 * 1.5)

    # Orientación global independiente de posición (señal adicional)
    ori_c = sum(1 for i in range(7) if state.co[i] == 0) / 7
    ori_e = sum(1 for i in range(9) if state.eo[i] == 0) / 9

    # Bonus cuadrático: crece rápido cuando el cubo está casi resuelto
    sol = FuseState.solved()
    fc = sum(1 for i in range(7)
             if state.cp[i] == sol.cp[i] and state.co[i] == sol.co[i]) / 7
    fe = sum(1 for i in range(9)
             if state.ep[i] == sol.ep[i] and state.eo[i] == sol.eo[i]) / 9
    bonus = (0.4 * fc + 0.6 * fe) ** 2

    return 0.25*pc + 0.10*ori_c + 0.35*pe + 0.10*ori_e + 0.20*bonus


def similarity_original(state: FuseState) -> float:
    """Función original del TFG (para reportar en los experimentos)."""
    sol = FuseState.solved()
    pc = oc = pe = oe = 0
    for i in range(7):
        if state.cp[i] == sol.cp[i]:
            pc += 1
            if state.co[i] == sol.co[i]: oc += 1
    for i in range(9):
        if state.ep[i] == sol.ep[i]:
            pe += 1
            if state.eo[i] == sol.eo[i]: oe += 1
    return 0.35*(pc/7) + 0.15*(oc/7) + 0.40*(pe/9) + 0.10*(oe/9)


# ============================================================
#  ENVOLTORIO DEL PROBLEMA PARA EDAspy
# ============================================================

class FuseUMDAProblem:
    """
    Adapta el cubo Fuse a la interfaz de UMDAcat de EDAspy.

    Evaluación por max-prefix
    --------------------------
    En lugar de evaluar el estado FINAL del genoma, evalúa el MEJOR
    estado intermedio alcanzado durante la ejecución de la secuencia.

    Ejemplo: genoma = [D' B L2 D B' L' L D B L D]
      → la solución real es D' B L2 D B' L' (paso 6)
      → los genes 7-11 son basura que destruye el cubo
      → max-prefix detecta el pico de similitud en el paso 6 y
        puntúa ese individuo como excelente, no como malo.
    """

    def __init__(
        self,
        scramble_alg:   str,
        genome_length:  int   = 12,
        length_penalty: float = 0.0005,
        corner_pdb:     dict = None,
        edge_pdb:       dict = None,
    ):
        self.scramble_alg   = scramble_alg
        self.scramble_state = apply_algorithm(FuseState.solved(), scramble_alg)
        self.genome_length  = genome_length
        self.length_penalty = length_penalty
        self.corner_pdb     = corner_pdb
        self.edge_pdb       = edge_pdb

        self.possible_values = np.array(
            [np.array(GENES, dtype=object) for _ in range(genome_length)],
            dtype=object,
        )
        self.frequency = np.array(
            [[1.0 / N_GENES] * N_GENES for _ in range(genome_length)],
            dtype=object,
        )

    def decode(self, solution: np.ndarray) -> List[str]:
        tokens: List[str] = []
        for x in solution:
            tok = str(x)
            if tok == STOP_TOKEN:
                break
            if tok in MOVES:
                tokens.append(tok)
        return simplify(tokens)

    def evaluate(self, solution: np.ndarray) -> float:
        tokens = self.decode(solution)
        return evaluate_phased_max_prefix(
            self.scramble_state,
            tokens,
            apply_move,
            is_solved,
            self.corner_pdb,
            self.edge_pdb,
            self.length_penalty,
        )


# ============================================================
#  CALLABLE CON SEGUIMIENTO DE MEJORA (impresión selectiva)
# ============================================================

class _ImprovementTracker:
    """
    Envuelve evaluate() e imprime solo cuando el coste mejora globalmente.

    UMDAcat llama a evaluate para cada individuo de cada generacion.
    Este wrapper mantiene el mejor coste visto y solo emite una linea
    cuando ese valor disminuye, evitando el ruido de imprimir en cada
    evaluacion o en cada generacion sin progreso.
    """

    def __init__(
        self,
        problem: FuseUMDAProblem,
        restart_idx: int,
        verbose: bool,
        size_gen: int,
    ):
        self.problem          = problem
        self.restart_idx      = restart_idx
        self.verbose          = verbose
        self.best_cost        = float("inf")
        self.best_solution     = None
        self.n_evals          = 0
        self.freq_init        = np.array(problem.frequency, dtype=float, copy=True)
        self.iters_no_improve = 0
        self.warm_restart_log = []
        self.patience         = size_gen * 15
        self.umda             = None
        self.elites           = []
        self.max_elites       = 24

    def _record_elite(self, cost: float, solution: np.ndarray) -> None:
        worst_cost = self.elites[-1][0] if len(self.elites) >= self.max_elites else float("inf")
        if cost >= worst_cost - 1e-12:
            return

        tokens = self.problem.decode(solution)
        key = tuple(tokens)
        if any(tuple(saved_tokens) == key for _, saved_tokens in self.elites):
            return

        self.elites.append((float(cost), tokens))
        self.elites.sort(key=lambda item: item[0])
        del self.elites[self.max_elites:]

    def __call__(self, solution: np.ndarray) -> float:
        cost = self.problem.evaluate(solution)
        self.n_evals += 1
        self._record_elite(cost, solution)

        improved = cost < self.best_cost - 1e-6
        if improved:
            self.best_cost = cost
            self.best_solution = np.array(solution, dtype=object, copy=True)
            self.iters_no_improve = 0
            if cost < 0:
                sim_str = "RESUELTO"
            else:
                tokens = self.problem.decode(solution)
                _, state, _ = best_prefix_snapshot(
                    tokens,
                    self.problem.scramble_state,
                    self.problem.corner_pdb,
                    self.problem.edge_pdb,
                    self.problem.length_penalty,
                )
                fitness = max(0.0, 1.0 - cost)
                sim_pdb = pdb_similarity(state, self.problem.corner_pdb, self.problem.edge_pdb)
                sim_orig = similarity_original(state)
                sim_str = (
                    f"fitness={fitness:.4f}  sim_pdb={sim_pdb:.4f}  "
                    f"sim_original={sim_orig:.4f}"
                )
            gl = self.problem.genome_length
            if self.verbose:
                print(
                    f"  [R{self.restart_idx} gl={gl} eval#{self.n_evals}] "
                    f"nueva mejora -> {sim_str}",
                    flush=True,
                )
        else:
            self.iters_no_improve += 1

        if self.iters_no_improve >= self.patience:
            mixed_freq = None
            if self.umda is not None and hasattr(self.umda, "pm"):
                mixed_freq = mix_umda_frequency(
                    self.umda,
                    self.problem.possible_values,
                    self.freq_init,
                    keep_weight=0.3,
                )
            if mixed_freq is not None:
                self.iters_no_improve = 0
                self.warm_restart_log.append(self.n_evals)
                if self.verbose:
                    entropy = entropy_of_distribution(mixed_freq)
                    gl = self.problem.genome_length
                    print(
                        f"  [R{self.restart_idx} gl={gl} eval#{self.n_evals}] "
                        f"reinicio calido -> entropy={entropy:.4f}",
                        flush=True,
                    )

        return cost


# ============================================================
#  UTILIDADES DE FRECUENCIA
# ============================================================

def entropy_of_distribution(freq: np.ndarray) -> float:
    """Entropia media de Shannon por posicion."""
    probs = np.asarray(freq, dtype=float)
    return float(np.mean(-np.sum(probs * np.log2(probs + 1e-10), axis=1)))


def umda_frequency_as_array(umda, possible_values: np.ndarray) -> Optional[np.ndarray]:
    """Extrae la tabla de probabilidades de UMDAcat como matriz numerica."""
    if umda is None or not hasattr(umda, "pm"):
        return None

    pm = umda.pm
    if hasattr(pm, "prob_table"):
        freq = np.zeros((len(possible_values), len(possible_values[0])), dtype=float)
        for i, values in enumerate(possible_values):
            label = str(i)
            probs = pm.prob_table.get(label)
            if probs is None:
                return None
            freq[i] = [float(probs.get(value, 0.0)) for value in values]
        return freq

    raw_pm = getattr(pm, "pm", None)
    if isinstance(raw_pm, np.ndarray):
        return np.array(raw_pm, dtype=float, copy=True)

    return None


def set_umda_frequency_from_array(
    umda,
    possible_values: np.ndarray,
    freq: np.ndarray,
) -> bool:
    """Escribe una matriz de probabilidades dentro del modelo de UMDAcat."""
    if umda is None or not hasattr(umda, "pm"):
        return False

    pm = umda.pm
    if hasattr(pm, "prob_table"):
        pm.prob_table = {
            str(i): {
                value: float(freq[i, j])
                for j, value in enumerate(possible_values[i])
            }
            for i in range(len(possible_values))
        }
        return True

    if hasattr(pm, "pm") and isinstance(pm.pm, np.ndarray):
        pm.pm = np.array(freq, dtype=float, copy=True)
        return True

    return False


def mix_umda_frequency(
    umda,
    possible_values: np.ndarray,
    freq_init: np.ndarray,
    keep_weight: float = 0.3,
) -> Optional[np.ndarray]:
    """Mezcla el modelo actual con la frecuencia inicial y lo reinyecta en UMDAcat."""
    current_freq = umda_frequency_as_array(umda, possible_values)
    if current_freq is None:
        return None

    mixed_freq = keep_weight * current_freq + (1.0 - keep_weight) * freq_init
    mixed_freq /= mixed_freq.sum(axis=1, keepdims=True)

    if not set_umda_frequency_from_array(umda, possible_values, mixed_freq):
        return None

    return mixed_freq

def _make_frequency_no_repeat_face(genome_length: int) -> np.ndarray:
    """
    Tabla de frecuencias inicial que penaliza repetir cara en posiciones
    contiguas. Reduce movimientos trivialmente redundantes en la
    generación 0 y acelera la convergencia inicial.
    """
    small = 0.02
    stop_prob = 0.03
    base  = (1.0 - stop_prob - 3 * small) / 6
    freq  = np.zeros((genome_length, N_GENES))
    for pos in range(genome_length):
        if pos == 0:
            freq[pos, :N_MOVES] = (1.0 - stop_prob) / N_MOVES
            freq[pos, N_MOVES] = stop_prob
        else:
            pf_idx = pos % 3
            for j, m in enumerate(MOVES):
                fi = ["L","D","B"].index(m[0])
                freq[pos, j] = small if fi == pf_idx else base
            freq[pos, N_MOVES] = stop_prob
        freq[pos] /= freq[pos].sum()
    return freq


def _bias_toward_best(
    base_freq:    np.ndarray,
    best_genome:  np.ndarray,
    strength:     float = 0.35,
) -> np.ndarray:
    """Sesga la distribución hacia el mejor individuo conocido."""
    biased = (1.0 - strength) * base_freq.copy()
    gl = base_freq.shape[0]
    for pos in range(min(gl, len(best_genome))):
        gene = str(best_genome[pos])
        if gene not in GENES:
            continue
        idx = GENES.index(gene)
        biased[pos, idx] += strength
    biased /= biased.sum(axis=1, keepdims=True)
    return biased


def make_inverse_biased_frequency(
    scramble_alg: str,
    genome_length: int,
    strength: float = 0.5,
) -> np.ndarray:
    """
    Inicializa la frecuencia sesgando cada posicion hacia el inverso
    del scramble. Si la cara coincide con la posicion anterior del
    inverso, se penaliza esa repeticion y se redistribuye la masa hacia
    las otras dos caras.
    """
    small = 0.02
    inverse_map = {
        "L": "L'", "L'": "L", "L2": "L2",
        "D": "D'", "D'": "D", "D2": "D2",
        "B": "B'", "B'": "B", "B2": "B2",
    }

    scramble_tokens = [tok for tok in scramble_alg.split() if tok]
    inverse_tokens = [inverse_map[tok] for tok in reversed(scramble_tokens)]

    freq = np.zeros((genome_length, N_GENES), dtype=float)
    fallback = _make_frequency_no_repeat_face(genome_length)

    for pos in range(genome_length):
        if pos == len(inverse_tokens):
            row = np.full(N_GENES, (1.0 - strength) / N_MOVES, dtype=float)
            row[N_MOVES] = strength
            freq[pos] = row / row.sum()
            continue

        if pos > len(inverse_tokens):
            freq[pos] = fallback[pos]
            continue

        target = inverse_tokens[pos]
        target_idx = MOVES.index(target)
        stop_prob = 0.01
        row = np.zeros(N_GENES, dtype=float)
        row[:N_MOVES] = (1.0 - strength - stop_prob) / (N_MOVES - 1)
        row[target_idx] = strength
        row[N_MOVES] = stop_prob

        if pos > 0 and inverse_tokens[pos - 1][0] == target[0]:
            row[:] = 0.0
            other_face_idxs = [j for j, move in enumerate(MOVES) if move[0] != target[0]]
            row[other_face_idxs] = (1.0 - small - stop_prob) / len(other_face_idxs)
            row[target_idx] = small
            row[N_MOVES] = stop_prob

        freq[pos] = row / row.sum()

    return freq


def sample_initial_generation_from_frequency(
    possible_values: np.ndarray,
    frequency: np.ndarray,
    size_gen: int,
) -> np.ndarray:
    """
    EDAspy CategoricalSampling ignora frequency en la generacion inicial.
    Generamos init_data nosotros para que el sesgo inicial sea real.
    """
    n_variables = len(possible_values)
    generation = np.empty((size_gen, n_variables), dtype=object)
    for pos in range(n_variables):
        probs = np.asarray(frequency[pos], dtype=float)
        probs = probs / probs.sum()
        generation[:, pos] = np.random.choice(possible_values[pos], size=size_gen, p=probs)
    return generation


# ============================================================
#  SOLVER PRINCIPAL CON EDASPY
# ============================================================

def best_prefix_snapshot(
    tokens: List[str],
    scramble_state: FuseState,
    corner_pdb: dict,
    edge_pdb: dict,
    length_penalty: float = 0.0005,
) -> Tuple[List[str], FuseState, float]:
    state = scramble_state
    best_tokens: List[str] = []
    best_state = state
    best_reward = phased_state_reward(state, corner_pdb, edge_pdb)
    best_cost = 1.0 - best_reward

    for i, tok in enumerate(tokens, start=1):
        state = apply_move(state, tok)
        prefix = tokens[:i]

        if is_solved(state):
            return prefix, state, -100.0

        reward = phased_state_reward(state, corner_pdb, edge_pdb)
        cost = 1.0 - reward + length_penalty * i
        if cost < best_cost - 1e-12:
            best_tokens = prefix
            best_state = state
            best_cost = cost

    return best_tokens, best_state, best_cost


def solve_fuse_with_edaspy(
    scramble_alg:   str,
    genome_length:  int   = 12,
    size_gen:       int   = 300,
    max_iter:       int   = 200,
    dead_iter:      int   = 40,
    alpha:          float = 0.4,
    length_penalty: float = 0.0005,
    freq_init:      Optional[np.ndarray] = None,
    restart_idx:    int   = 1,
    verbose:        bool  = True,
    disp:           bool  = False,
    corner_pdb:     dict  = None,
    edge_pdb:       dict  = None,
) -> dict:
    """
    Una ejecución de UMDAcat con max-prefix fitness.

    Parámetros
    ----------
    freq_init   : tabla de frecuencias inicial (None = uniforme con penalización
                  de cara repetida)
    restart_idx : número de reinicio actual (solo para el mensaje de consola)
    disp        : si True, EDAspy imprime su propio log interno (muy verboso)
    """
    problem = FuseUMDAProblem(
        scramble_alg   = scramble_alg,
        genome_length  = genome_length,
        length_penalty = length_penalty,
        corner_pdb     = corner_pdb,
        edge_pdb       = edge_pdb,
    )

    if freq_init is not None:
        problem.frequency = freq_init
    elif restart_idx == 1:
        problem.frequency = make_inverse_biased_frequency(scramble_alg, genome_length)
    else:
        problem.frequency = _make_frequency_no_repeat_face(genome_length)

    tracker = _ImprovementTracker(problem, restart_idx, verbose, size_gen)

    umda = UMDAcat(
        size_gen        = size_gen,
        max_iter        = max_iter,
        dead_iter       = dead_iter,
        n_variables     = genome_length,
        alpha           = alpha,
        frequency       = problem.frequency,
        possible_values = problem.possible_values,
        init_data       = sample_initial_generation_from_frequency(
            problem.possible_values,
            problem.frequency,
            size_gen,
        ),
    )
    tracker.umda = umda

    t0     = time.perf_counter()
    result = umda.minimize(tracker, verbose=disp)
    elapsed = time.perf_counter() - t0

    result_best_cost = float(result.best_cost)
    if tracker.best_solution is not None and tracker.best_cost <= result_best_cost + 1e-12:
        best_raw = tracker.best_solution
        best_cost = float(tracker.best_cost)
    else:
        best_raw = result.best_ind
        best_cost = result_best_cost
    history    = list(result.history) if hasattr(result, "history") else []

    decoded_tokens = problem.decode(np.array(best_raw, dtype=object))

    best_tokens, best_state, prefix_cost = best_prefix_snapshot(
        decoded_tokens,
        problem.scramble_state,
        problem.corner_pdb,
        problem.edge_pdb,
        problem.length_penalty,
    )

    best_cost = min(best_cost, prefix_cost)

    return {
        "scramble": scramble_alg,
        "genome_length": genome_length,
        "best_algorithm": " ".join(best_tokens),
        "best_cost": best_cost,
        "best_state": best_state,
        "solved": is_solved(best_state),
        "similarity_imp": pdb_similarity(best_state, problem.corner_pdb, problem.edge_pdb),
        "similarity_v3": similarity_to_solved(best_state),
        "similarity_orig": similarity_original(best_state),
        "history": history,
        "elapsed_seconds": elapsed,
        "warm_restart_log": list(tracker.warm_restart_log),
        "_best_raw": list(best_tokens),
        "solution_length": len(best_tokens),
        "scramble_length": len(scramble_alg.split()),
        "solution_algorithm": " ".join(best_tokens),
    }



def solve_fuse(
    scramble_alg:   str,
    size_gen:       int   = 300,
    max_iter:       int   = 200,
    dead_iter:      int   = 40,
    alpha:          float = 0.4,
    length_penalty: float = 0.0005,
    min_genome:     Optional[int] = None,
    max_genome:     Optional[int] = None,
    genome_step:    int   = 2,
    n_restarts:     int   = 8,
    seed:           Optional[int] = None,
    verbose:        bool  = True,
    corner_pdb:     dict  = None,
    edge_pdb:       dict  = None,
) -> dict:
    """
    Solver EDA puro con reinicios y longitud de genoma creciente.

    Estrategia de reinicios
    -----------------------
    Reinicio 1: genome_length = min_genome  (= n_scr por defecto)
    Reinicio 2: genome_length = min_genome + genome_step
    ...
    Hasta max_genome o hasta resolver.

    En cada reinicio la frecuencia inicial se sesga hacia el mejor
    individuo conocido (reinicio cálido), acelerando la convergencia.

    Parámetros clave
    ----------------
    min_genome  : genome_length inicial (None → n_movimientos_scramble)
    max_genome  : genome_length máximo  (None → 3 × n_movimientos_scramble)
    genome_step : incremento por reinicio
    n_restarts  : número máximo de reinicios
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    n_scr = len(scramble_alg.split())
    lo    = min_genome if min_genome is not None else n_scr
    hi    = max_genome if max_genome is not None else max(lo + 2, 3 * n_scr)

    if verbose:
        sc0 = apply_algorithm(FuseState.solved(), scramble_alg)
        print("=" * 60)
        print(f"  Scramble: '{scramble_alg}' ({n_scr} movimientos)")
        print(f"  Sim inicial (orig):     {similarity_original(sc0):.4f}")
        if corner_pdb is not None and edge_pdb is not None:
            print(f"  Sim inicial (PDB):      {pdb_similarity(sc0, corner_pdb, edge_pdb):.4f}")
        print(f"  Sim inicial (v3 vieja): {similarity_to_solved(sc0):.4f}")
        print(f"  Genomas: {lo}..{hi}  paso={genome_step}  reinicios={n_restarts}")
        print("=" * 60)

    best_overall: Optional[dict] = None
    t_total = time.perf_counter()

    genome_seq = []
    gl = lo
    for _ in range(n_restarts):
        genome_seq.append(gl)
        gl = min(gl + genome_step, hi)

    for r_idx, gl in enumerate(genome_seq):
        if verbose:
            print(f"\n[Reinicio {r_idx+1}/{n_restarts}  genome_length={gl}]")

        # Frecuencia inicial
        base_freq = _make_frequency_no_repeat_face(gl)
        if best_overall is not None and best_overall["best_cost"] < 1.0:
            print("Aplicando 1");
            prev_raw = np.array(best_overall["_best_raw"], dtype=object)
            seed_genome = list(prev_raw[:gl])
            if len(seed_genome) < gl:
                seed_genome.extend([STOP_TOKEN] * (gl - len(seed_genome)))
            freq_init = _bias_toward_best(base_freq, np.array(seed_genome, dtype=object), strength=0.35)
        elif r_idx == 0:
            print("Aplicando 2");
            freq_init = make_inverse_biased_frequency(scramble_alg, gl)
        else:
            print("Aplicando 3");
            freq_init = base_freq

        run_seed = None if seed is None else seed + r_idx * 97

        r = solve_fuse_with_edaspy(
            scramble_alg   = scramble_alg,
            genome_length  = gl,
            size_gen       = size_gen,
            max_iter       = max_iter,
            dead_iter      = dead_iter,
            alpha          = alpha,
            length_penalty = length_penalty,
            freq_init      = freq_init,
            restart_idx    = r_idx + 1,
            verbose        = verbose,
            disp           = False,
            corner_pdb     = corner_pdb,
            edge_pdb       = edge_pdb,
        )

        if best_overall is None or r["best_cost"] < best_overall["best_cost"]:
            best_overall = r

        if r["solved"]:
            break

    elapsed = time.perf_counter() - t_total
    best_overall["elapsed_seconds"] = elapsed
    best_overall.pop("_best_raw", None)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  RESULTADO FINAL")
        print(f"  ¿Resuelto?:      {best_overall['solved']}")
        print(f"  Sim original:    {best_overall['similarity_orig']:.4f}")
        print(f"  Sim PDB:         {best_overall['similarity_imp']:.4f}")
        print(f"  Sim v3 vieja:    {best_overall.get('similarity_v3', 0.0):.4f}")
        print(f"  Tiempo total:    {elapsed:.1f}s")
        print(f"  Algoritmo:       {best_overall['best_algorithm']}")
        print(f"{'='*60}")

    return best_overall


# ============================================================
#  EXPERIMENTACIÓN
# ============================================================

def run_experiments(
    scramble_alg: str,
    n_runs:       int  = 30,
    base_seed:    int  = 1000,
    verbose_runs: bool = False,
    **solver_kwargs,
) -> Tuple[list, dict]:
    runs: List[dict] = []

    for i in range(n_runs):
        seed = base_seed + i
        r = solve_fuse(
            scramble_alg = scramble_alg,
            seed         = seed,
            verbose      = verbose_runs,
            **solver_kwargs,
        )
        runs.append(r)
        tag = "✓" if r["solved"] else "✗"
        print(
            f"[{i+1:02d}/{n_runs}] {tag} "
            f"seed={seed}  "
            f"sim_orig={r['similarity_orig']:.4f}  "
            f"sim_imp={r['similarity_imp']:.4f}  "
            f"t={r['elapsed_seconds']:.1f}s  "
            f"alg='{r['best_algorithm'][:55]}'"
        )

    ok    = [r for r in runs if r["solved"]]
    sims  = [r["similarity_imp"] for r in runs]
    times = [r["elapsed_seconds"] for r in runs]

    summary = {
        "scramble":       scramble_alg,
        "n_runs":         n_runs,
        "success_count":  len(ok),
        "success_rate":   len(ok) / n_runs,
        "mean_sim_imp":   statistics.mean(sims),
        "best_sim_imp":   max(sims),
        "std_sim_imp":    statistics.pstdev(sims) if len(sims) > 1 else 0.0,
        "mean_time":      statistics.mean(times),
        "best_run":       max(runs, key=lambda r: (r["solved"], r["similarity_imp"])),
    }
    return runs, summary


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("  RESUMEN EXPERIMENTAL")
    print("=" * 60)
    print(f"  Scramble:          {summary['scramble']}")
    print(f"  Corridas:          {summary['n_runs']}")
    print(f"  Éxitos:            {summary['success_count']} ({100*summary['success_rate']:.1f}%)")
    print(f"  Sim mejorada media:{summary['mean_sim_imp']:.4f}")
    print(f"  Sim mejorada máx:  {summary['best_sim_imp']:.4f}")
    print(f"  Desviación típica: {summary['std_sim_imp']:.4f}")
    print(f"  Tiempo medio:      {summary['mean_time']:.1f}s")
    b = summary["best_run"]
    print(f"\n  --- Mejor corrida ---")
    print(f"  ¿Resuelto?: {b['solved']}")
    print(f"  Sim orig:   {b['similarity_orig']:.4f}")
    print(f"  Sim imp:    {b['similarity_imp']:.4f}")
    print(f"  Algoritmo:  {b['best_algorithm']}")


def save_csv(runs: list, filename: str = "experimentos_fuse.csv") -> None:
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scramble","solved","genome_length","similarity_original",
                    "similarity_improved","best_cost","elapsed_seconds","best_algorithm"])
        for r in runs:
            w.writerow([r["scramble"], r["solved"], r["genome_length"],
                        r["similarity_orig"], r["similarity_imp"],
                        r["best_cost"], r["elapsed_seconds"], r["best_algorithm"]])


# ============================================================
#  EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":
    corner_pdb, edge_pdb = load_or_build_pdbs()
    # 1) Verificación básica
    s0 = FuseState.solved()
    assert is_solved(s0)
    #assert abs(similarity_to_solved(s0) - 1.0) < 1e-9
    #assert abs(similarity_original(s0) - 1.0) < 1e-9
    print("Verificación básica: OK\n")

    # 2) Scrambles de distintas profundidades
    ejemplos = [
        # (scramble,  min_genome, max_genome, size_gen, n_restarts)
        ("L D B",                                              3,  9, 200, 5),
        ("L B2 D L' B",                                        5, 13, 250, 6),
        ("L B D' L2 B' D L B2 D' L'",                          10, 12, 350, 8),
        ("L B D' L2 B' D L B2 D' L' B D2 L2 B' D L' B2 D'",    18, 22, 450, 10),
    ]

    r = solve_fuse(
        scramble_alg = "L D L' D' L D L' D'",
        size_gen     = 1000,
        max_iter     = 500,
        dead_iter    = 120,
        alpha        = 0.30,
        min_genome   = 18,
        max_genome   = 30,
        genome_step  = 1,
        n_restarts   = 20,
        seed         = 42,
        verbose      = True,
        corner_pdb   = corner_pdb,
        edge_pdb     = edge_pdb,
    )
    print("¿Resuelto?:", r["solved"])
