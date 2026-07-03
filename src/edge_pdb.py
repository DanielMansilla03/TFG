from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable, Tuple


EdgePatternKey = Tuple[Tuple[int, ...], Tuple[int, ...]]
EdgePDB = Dict[EdgePatternKey, int]


BASE_MOVES = {
    "L": {"ep_perm": (7, 1, 2, 3, 6, 5, 0, 4, 8), "eo_delta": (0, 0, 0, 0, 0, 0, 0, 0, 0)},
    "D": {"ep_perm": (0, 1, 3, 4, 5, 2, 6, 7, 8), "eo_delta": (0, 0, 0, 0, 0, 0, 0, 0, 0)},
    "B": {"ep_perm": (0, 8, 2, 3, 4, 7, 6, 1, 5), "eo_delta": (0, 1, 0, 0, 0, 1, 0, 1, 1)},
}

MOVES = ("L", "L2", "L'", "D", "D2", "D'", "B", "B2", "B'")


def _apply_base_to_pattern(
    key: EdgePatternKey,
    pieces: Tuple[int, ...],
    face: str,
) -> EdgePatternKey:
    positions, orientations = key
    perm = BASE_MOVES[face]["ep_perm"]
    delta = BASE_MOVES[face]["eo_delta"]

    new_positions = []
    new_orientations = []
    for old_slot, old_ori in zip(positions, orientations):
        new_slot = next(dst for dst in range(9) if perm[dst] == old_slot)
        new_positions.append(new_slot)
        new_orientations.append((old_ori + delta[new_slot]) % 2)

    return tuple(new_positions), tuple(new_orientations)


def _apply_move_to_pattern(
    key: EdgePatternKey,
    pieces: Tuple[int, ...],
    move: str,
) -> EdgePatternKey:
    if move in ("L", "D", "B"):
        return _apply_base_to_pattern(key, pieces, move)
    if len(move) == 2 and move[1] == "2":
        key = _apply_base_to_pattern(key, pieces, move[0])
        return _apply_base_to_pattern(key, pieces, move[0])
    if len(move) == 2 and move[1] == "'":
        for _ in range(3):
            key = _apply_base_to_pattern(key, pieces, move[0])
        return key
    raise ValueError(f"Movimiento no permitido: {move}")


def build_edge_pattern_pdb(pieces: Iterable[int], label: str) -> Tuple[EdgePDB, int]:
    """
    PDB abstracta correcta: trackea posiciones y orientaciones de un subconjunto
    de piezas de arista, ignorando las demas piezas.
    """
    tracked = tuple(pieces)
    solved_key: EdgePatternKey = (tracked, (0,) * len(tracked))
    pdb: EdgePDB = {solved_key: 0}
    q: Deque[EdgePatternKey] = deque([solved_key])
    max_dist = 0

    while q:
        key = q.popleft()
        dist = pdb[key]
        if dist > max_dist:
            max_dist = dist
        next_dist = dist + 1

        for move in MOVES:
            next_key = _apply_move_to_pattern(key, tracked, move)
            if next_key not in pdb:
                pdb[next_key] = next_dist
                q.append(next_key)

    print(f"Edge PDB ({label}): {len(pdb)} estados")
    print(f"Distancia maxima ({label}): {max_dist}")
    return pdb, max_dist


def build_edge_pdb_5() -> Tuple[EdgePDB, int]:
    return build_edge_pattern_pdb((0, 1, 2, 3, 4), "pieces 0..4")


def build_edge_pdb_4() -> Tuple[EdgePDB, int]:
    return build_edge_pattern_pdb((5, 6, 7, 8), "pieces 5..8")


def build_edge_pdb() -> Tuple[EdgePDB, int, EdgePDB, int]:
    """
    Construye PDBs factorizadas por piezas (5+4):
      - piezas 0..4
      - piezas 5..8
    """
    pdb5, max5 = build_edge_pdb_5()
    pdb4, max4 = build_edge_pdb_4()
    return pdb5, max5, pdb4, max4


if __name__ == "__main__":
    build_edge_pdb()
