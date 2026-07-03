"""
solver_variants.py — Variantes del problema EDA Fuse 3x3x3.

Mantiene el baseline `FuseUMDAProblem` (de fusecube_eda_edaspy) intacto
y añade variantes incrementales para ablation:

  - no_stop : elimina el alelo STOP del vocabulario (9 alelos en lugar de 10).
              El corte de secuencia lo decide max-prefix, no el modelo.
"""
from __future__ import annotations
from typing import List
import numpy as np

from fusecube_eda_edaspy import (
    FuseState, apply_algorithm, apply_move, is_solved, MOVES, simplify,
)
from pdb_similarity import evaluate_phased_max_prefix


N_MOVES = len(MOVES)
FACES = ("L", "D", "B")
_INVERSE_MAP = {
    "L": "L'", "L'": "L", "L2": "L2",
    "D": "D'", "D'": "D", "D2": "D2",
    "B": "B'", "B'": "B", "B2": "B2",
}


# ============================================================
#  Variante NO_STOP
# ============================================================

class FuseProblemNoStop:
    """Versión de FuseUMDAProblem sin el alelo STOP.

    Justificación: el max-prefix ya selecciona el punto óptimo de corte
    durante la evaluación, por lo que aprender STOP es redundante. Eliminarlo
    libera entropía del modelo para los movimientos reales (9 alelos
    en vez de 10).
    """

    def __init__(
        self,
        scramble_alg: str,
        genome_length: int,
        length_penalty: float = 0.0005,
        corner_pdb: dict = None,
        edge_pdb: dict = None,
    ):
        self.scramble_alg = scramble_alg
        self.scramble_state = apply_algorithm(FuseState.solved(), scramble_alg)
        self.genome_length = genome_length
        self.length_penalty = length_penalty
        self.corner_pdb = corner_pdb
        self.edge_pdb = edge_pdb

        self.possible_values = np.array(
            [np.array(MOVES, dtype=object) for _ in range(genome_length)],
            dtype=object,
        )
        self.frequency = np.array(
            [[1.0 / N_MOVES] * N_MOVES for _ in range(genome_length)],
            dtype=object,
        )

    def decode(self, solution: np.ndarray) -> List[str]:
        # Sin STOP: todos los tokens válidos, simplificados.
        tokens = [str(x) for x in solution if str(x) in MOVES]
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


def make_inverse_biased_frequency_no_stop(
    scramble_alg: str,
    genome_length: int,
    strength: float = 0.5,
) -> np.ndarray:
    """Igual que `make_inverse_biased_frequency` del baseline pero sin masa en STOP.

    En posiciones más allá del inverso: distribución uniforme sobre los 9 movs
    (en el baseline se sesgaba a STOP con prob `strength`, lo cual no aplica).
    """
    small = 0.02
    scramble_tokens = [tok for tok in scramble_alg.split() if tok]
    inverse_tokens = [_INVERSE_MAP[tok] for tok in reversed(scramble_tokens)]

    freq = np.zeros((genome_length, N_MOVES), dtype=float)
    fallback = make_no_repeat_face_no_stop(genome_length)

    for pos in range(genome_length):
        if pos >= len(inverse_tokens):
            # Más allá del inverso: distribución base (con anti-repetición de cara)
            freq[pos] = fallback[pos]
            continue

        target = inverse_tokens[pos]
        target_idx = MOVES.index(target)
        row = np.full(N_MOVES, (1.0 - strength) / (N_MOVES - 1), dtype=float)
        row[target_idx] = strength

        if pos > 0 and inverse_tokens[pos - 1][0] == target[0]:
            # Misma cara que el anterior del inverso: redistribuir hacia otras caras.
            row[:] = 0.0
            other_face_idxs = [j for j, move in enumerate(MOVES) if move[0] != target[0]]
            row[other_face_idxs] = (1.0 - small) / len(other_face_idxs)
            row[target_idx] = small

        freq[pos] = row / row.sum()

    return freq


def make_no_repeat_face_no_stop(genome_length: int) -> np.ndarray:
    """Tabla inicial uniforme que penaliza cara repetida cíclicamente. Sin STOP."""
    small = 0.02
    base = (1.0 - 3 * small) / 6
    freq = np.zeros((genome_length, N_MOVES), dtype=float)
    for pos in range(genome_length):
        if pos == 0:
            freq[pos, :] = 1.0 / N_MOVES
        else:
            pf_idx = pos % 3  # mismo esquema que el baseline
            for j, m in enumerate(MOVES):
                fi = FACES.index(m[0])
                freq[pos, j] = small if fi == pf_idx else base
        freq[pos] /= freq[pos].sum()
    return freq


def bias_toward_best_no_stop(
    base_freq: np.ndarray,
    best_genome: np.ndarray,
    strength: float = 0.35,
) -> np.ndarray:
    """Sesga la distribución hacia el mejor individuo conocido (9 alelos)."""
    biased = (1.0 - strength) * base_freq.copy()
    gl = base_freq.shape[0]
    for pos in range(min(gl, len(best_genome))):
        gene = str(best_genome[pos])
        if gene not in MOVES:
            continue
        idx = MOVES.index(gene)
        biased[pos, idx] += strength
    biased /= biased.sum(axis=1, keepdims=True)
    return biased


# ============================================================
#  Variante MARKOV (modelo producto P_pos · T con máscara dura de cara)
# ============================================================

# Pre-cómputo de máscara de cara: MASK_FACE[p] es un vector de 9 ints {0,1}
# que vale 0 en los índices de movimientos cuya cara coincide con MOVES[p].
_MOVE_FACE_IDX = np.array([FACES.index(m[0]) for m in MOVES], dtype=np.int64)
MASK_FACE = np.ones((N_MOVES, N_MOVES), dtype=np.float64)
for _p in range(N_MOVES):
    same_face = _MOVE_FACE_IDX == _MOVE_FACE_IDX[_p]
    MASK_FACE[_p, same_face] = 0.0  # prohíbe los 3 movs de la misma cara

_EPS = 1e-12


class ProductMaskedModel:
    """EDA con modelo factorizado P_pos[i] (marginal posicional) y T (matriz
    homogénea 9x9 de transición), combinados como producto y enmascarados:

        P(mov_i = m | prev = p) ∝ P_pos[i, m] · T[p, m] · mask(p, m)

    donde mask(p, m) = 0 si face(m) == face(p) (prohíbe la misma cara consecutiva).

    Para la posición 0 no hay prev → se muestrea directamente de P_pos[0].

    Suavizado Laplace (α=1) sobre celdas legales, aplicado siempre en
    update_from_elite, de modo que ninguna celda permitida llegue jamás a 0.
    """

    def __init__(
        self,
        genome_length: int,
        P_pos_init: np.ndarray,
        laplace: float = 1.0,
        learn_T: bool = True,
    ):
        assert P_pos_init.shape == (genome_length, N_MOVES), \
            f"P_pos_init shape {P_pos_init.shape} ≠ ({genome_length},{N_MOVES})"
        self.gl = genome_length
        self.laplace = float(laplace)
        self.learn_T = bool(learn_T)

        # Marginal posicional inicial (no se renormaliza aquí; se asume válida)
        self.P_pos = np.asarray(P_pos_init, dtype=np.float64).copy()
        self.P_pos /= self.P_pos.sum(axis=1, keepdims=True)

        # T inicial: uniforme sobre transiciones legales (1/6 por celda permitida)
        T0 = MASK_FACE.copy()
        T0 /= T0.sum(axis=1, keepdims=True)
        self.T = T0

    # --------------------------------------------------------------
    #  Muestreo
    # --------------------------------------------------------------
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        out = np.empty((n, self.gl), dtype=np.int64)
        # Pos 0: muestreo directo de P_pos[0]
        out[:, 0] = rng.choice(N_MOVES, size=n, p=self.P_pos[0])
        # Pos i>=1: combinación posición × transición × máscara
        for i in range(1, self.gl):
            p_pos_i = self.P_pos[i]                           # (9,)
            # Combinación condicional para cada posible prev
            cond = self.T * p_pos_i[np.newaxis, :] * MASK_FACE  # (9, 9)
            row_sum = cond.sum(axis=1, keepdims=True)
            # Safety: si alguna fila quedó toda en 0 (no debería con Laplace),
            # caer a uniforme legal.
            zero_rows = (row_sum.squeeze() <= _EPS)
            if zero_rows.any():
                cond[zero_rows] = MASK_FACE[zero_rows]
                row_sum = cond.sum(axis=1, keepdims=True)
            cond = cond / row_sum                              # (9, 9) renormalizada
            prev = out[:, i - 1]                               # (n,)
            probs_for_each = cond[prev]                        # (n, 9)
            # rng.choice no vectoriza con prob distinto por fila → muestreo manual
            u = rng.random(n)[:, None]
            cdf = np.cumsum(probs_for_each, axis=1)
            out[:, i] = (u < cdf).argmax(axis=1)
        return out

    # --------------------------------------------------------------
    #  Re-estimación desde la élite
    # --------------------------------------------------------------
    def update_from_elite(self, elite_ints: np.ndarray) -> None:
        """Actualiza P_pos y T contando frecuencias en la élite + Laplace."""
        k, gl = elite_ints.shape
        assert gl == self.gl

        # P_pos: conteo por posición (sin máscara: en pos 0 todo es legal,
        # en pos i>=1 la máscara aplica pero el ind ya respetó la máscara
        # en el muestreo, así que el conteo natural ya es legal).
        new_P = np.full((gl, N_MOVES), self.laplace, dtype=np.float64)
        for i in range(gl):
            counts = np.bincount(elite_ints[:, i], minlength=N_MOVES)
            new_P[i] += counts
        new_P /= new_P.sum(axis=1, keepdims=True)
        self.P_pos = new_P

        if self.learn_T:
            # T: conteo de pares (prev, curr) sólo sobre celdas legales.
            new_T = self.laplace * MASK_FACE.copy()
            # Vectorización por pares consecutivos.
            prevs = elite_ints[:, :-1].reshape(-1)
            currs = elite_ints[:, 1:].reshape(-1)
            np.add.at(new_T, (prevs, currs), 1.0)
            row_sum = new_T.sum(axis=1, keepdims=True)
            # Garantía: ninguna fila a 0 (Laplace sobre máscara asegura ≥6·laplace)
            new_T /= row_sum
            self.T = new_T
        # Si learn_T es False, T se mantiene como uniforme legal del init.

    # --------------------------------------------------------------
    #  Entropía media (condicional) para reporting
    # --------------------------------------------------------------
    def entropy_avg(self) -> float:
        """Entropía promedio del modelo:
          - pos 0: H(P_pos[0])  (rango [0, log2(9)])
          - pos i>=1: E_{prev ~ pi_{i-1}}[ H(X_i | prev) ]  con  pi_{i-1}
            = distribución empírica del modelo en pos i-1 (P_pos[i-1]).
        Se devuelve la media simple por posición.
        """
        ents = []

        p0 = self.P_pos[0]
        ents.append(_entropy_bits(p0))

        for i in range(1, self.gl):
            p_pos_i = self.P_pos[i]
            cond = self.T * p_pos_i[np.newaxis, :] * MASK_FACE
            row_sum = cond.sum(axis=1, keepdims=True)
            # Filas degeneradas: las omitimos del promedio (pesarán 0 si pi[p]=0).
            with np.errstate(invalid="ignore", divide="ignore"):
                cond_norm = np.where(row_sum > _EPS, cond / row_sum, 0.0)
            # H(X_i | prev = p)
            h_per_prev = -np.sum(
                np.where(cond_norm > 0, cond_norm * np.log2(cond_norm + _EPS), 0.0),
                axis=1,
            )
            # Promedio ponderado por pi_{i-1} = P_pos[i-1]
            pi_prev = self.P_pos[i - 1]
            ents.append(float(np.sum(pi_prev * h_per_prev)))

        return float(np.mean(ents))


def _entropy_bits(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    return float(-np.sum(np.where(p > 0, p * np.log2(p + _EPS), 0.0)))


def make_markov_initial_P_pos(
    scramble_alg: str,
    genome_length: int,
    r_idx: int,
    best_overall_raw=None,
    strength: float = 0.5,
) -> np.ndarray:
    """Sembrado de la marginal posicional para el modelo Markov.

    Reutiliza los inicializadores de la variante no_stop para mantener la
    misma semántica de información a priori (sesgo inverso en reinicio 1,
    sesgo hacia el mejor en reinicios subsiguientes).
    """
    base = make_no_repeat_face_no_stop(genome_length)
    if best_overall_raw is not None:
        seed_genome = list(best_overall_raw[:genome_length])
        if len(seed_genome) < genome_length:
            seed_genome.extend([MOVES[0]] * (genome_length - len(seed_genome)))
        return bias_toward_best_no_stop(
            base, np.array(seed_genome, dtype=object), strength=0.35,
        )
    if r_idx == 0:
        return make_inverse_biased_frequency_no_stop(
            scramble_alg, genome_length, strength=strength
        )
    return base
