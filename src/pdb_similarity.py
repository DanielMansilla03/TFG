from __future__ import annotations

import os
import pickle
from typing import Any, Callable, Dict, Iterable, Tuple, Union

from corner_pdb import build_corner_pdb
from edge_pdb import build_edge_pdb


CornerKey = Tuple[Tuple[int, ...], Tuple[int, ...]]  # (cp, co)
Edge6Key = Tuple[Tuple[int, ...], Tuple[int, ...]]  # (legacy: ep[:6], eo[:6])
Edge5Key = Tuple[Tuple[int, ...], Tuple[int, ...]]  # positions/orientations of pieces 0..4
Edge4Key = Tuple[Tuple[int, ...], Tuple[int, ...]]  # positions/orientations of pieces 5..8

CornerPDB = Dict[CornerKey, int]
LegacyEdgePDB = Dict[Edge6Key, int]
EdgePDB5 = Dict[Edge5Key, int]
EdgePDB4 = Dict[Edge4Key, int]
EdgePDB = Union[LegacyEdgePDB, Tuple[EdgePDB5, int, EdgePDB4, int]]

CORNER_MAX_DIST = 11
EDGE6_MAX_DIST = 17  # legacy


def edge_pattern_key(state: Any, pieces: Tuple[int, ...]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    positions = []
    orientations = []
    for piece in pieces:
        slot = state.ep.index(piece)
        positions.append(slot)
        orientations.append(state.eo[slot])
    return tuple(positions), tuple(orientations)


def pdb_similarity(state, corner_pdb: CornerPDB, edge_pdb: EdgePDB) -> float:
    """
    Similaridad basada en PDBs:
      - corners: (cp, co) -> distancia (max 11)
      - edges:   PDB factorizada 5+4 (piezas 0..4 y 5..8)
    """
    d_corner = int(corner_pdb.get((state.cp, state.co), CORNER_MAX_DIST))
    if isinstance(edge_pdb, tuple) and len(edge_pdb) == 4:
        pdb5, max5, pdb4, max4 = edge_pdb
        d5 = int(pdb5.get(edge_pattern_key(state, (0, 1, 2, 3, 4)), max5))
        d4 = int(pdb4.get(edge_pattern_key(state, (5, 6, 7, 8)), max4))
        n5 = d5 / max5
        n4 = d4 / max4
        edge_distance = 0.35 * ((n5 + n4) / 2.0) + 0.65 * max(n5, n4)
        sim_edge = 1.0 - edge_distance
        if d_corner == 0 and d5 == 0 and d4 == 0:
            return 1.0
    else:
        d_edge = int(edge_pdb.get((state.ep[:6], state.eo[:6]), EDGE6_MAX_DIST))
        sim_edge = 1.0 - (d_edge / EDGE6_MAX_DIST)
        if d_corner == 0 and d_edge == 0:
            return 1.0

    sim_corner = 1.0 - (d_corner / CORNER_MAX_DIST)

    if sim_corner < 0.0:
        sim_corner = 0.0
    elif sim_corner > 1.0:
        sim_corner = 1.0
    if sim_edge < 0.0:
        sim_edge = 0.0
    elif sim_edge > 1.0:
        sim_edge = 1.0

    return 0.45 * sim_corner + 0.55 * sim_edge


def load_or_build_pdbs(cache_file: str = "pdbs_v3.pkl") -> Tuple[CornerPDB, EdgePDB]:
    """
    Carga corner_pdb y edge_pdb desde pickle si existe; si no, los construye
    con BFS y los guarda en cache_file.
    """
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, tuple) and len(payload) == 5:
            corner_pdb, edge_pdb5, max5, edge_pdb4, max4 = payload
            return corner_pdb, (edge_pdb5, max5, edge_pdb4, max4)
        if isinstance(payload, tuple) and len(payload) == 2:
            return payload  
        raise ValueError("Cache PDB invalido o version desconocida.")

    corner_pdb = build_corner_pdb()
    edge_pdb5, max5, edge_pdb4, max4 = build_edge_pdb()

    with open(cache_file, "wb") as f:
        pickle.dump(
            (corner_pdb, edge_pdb5, max5, edge_pdb4, max4),
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return corner_pdb, (edge_pdb5, max5, edge_pdb4, max4)


def count_correct_edges(state: Any) -> int:
    return sum(1 for i in range(9) if state.ep[i] == i and state.eo[i] == 0)


def count_correct_corners(state: Any) -> int:
    return sum(1 for i in range(7) if state.cp[i] == i and state.co[i] == 0)


def phased_state_reward(state: Any, corner_pdb: CornerPDB, edge_pdb: EdgePDB) -> float:
    """
    Reward en [0, 1] para una fase concreta del cubo.

    Reward densa: evita plateaus por umbrales discretos y mantiene segnal
    para las ultimas piezas.
    """
    base = pdb_similarity(state, corner_pdb, edge_pdb)
    edge_fraction = count_correct_edges(state) / 9
    corner_fraction = count_correct_corners(state) / 7

    if edge_fraction == 1.0 and corner_fraction == 1.0:
        return 1.0

    fine_progress = 0.65 * edge_fraction + 0.35 * corner_fraction
    reward = 0.90 * base + 0.10 * fine_progress
    return min(0.999, reward)


def evaluate_phased_max_prefix(
    scramble_state: Any,
    tokens: Iterable[str],
    apply_move: Callable[[Any, str], Any],
    is_solved: Callable[[Any], bool],
    corner_pdb: CornerPDB,
    edge_pdb: EdgePDB,
    length_penalty: float = 0.0005,
) -> float:
    """
    Max-prefix fitness por fases.

    Devuelve un coste en [-100, 1] compatible con UMDAcat.minimize().
    """
    state = scramble_state
    best_state = state
    best_reward = phased_state_reward(state, corner_pdb, edge_pdb)
    best_index = 0

    for i, tok in enumerate(tokens, start=1):
        state = apply_move(state, tok)
        if is_solved(state):
            return -100.0 + length_penalty * i

        reward = phased_state_reward(state, corner_pdb, edge_pdb)
        if reward > best_reward + 1e-12:
            best_reward = reward
            best_state = state
            best_index = i

    if is_solved(best_state):
        return -100.0 + length_penalty * best_index

    cost = 1.0 - best_reward + length_penalty * best_index
    if cost > 1.0:
        cost = 1.0
    if cost < -100.0:
        cost = -100.0
    return cost


if __name__ == "__main__":
    corner_pdb, edge_pdb = load_or_build_pdbs()

    class _SolvedState:
        cp = tuple(range(7))
        co = (0,) * 7
        ep = tuple(range(9))
        eo = (0,) * 9

    s = pdb_similarity(_SolvedState(), corner_pdb, edge_pdb)
    print(f"pdb_similarity(solved) = {s!r}")
    assert s == 1.0

    phase_cost = evaluate_phased_max_prefix(
        _SolvedState(),
        tokens=[],
        apply_move=lambda state, move: state,
        is_solved=lambda state: True,
        corner_pdb=corner_pdb,
        edge_pdb=edge_pdb,
    )
    print(f"evaluate_phased_max_prefix(solved) = {phase_cost!r}")
    assert phase_cost == -100.0
