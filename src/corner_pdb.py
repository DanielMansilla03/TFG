from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Tuple


def build_corner_pdb() -> Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int]:
    """
    BFS desde el estado resuelto sobre (cp, co) únicamente.

    Devuelve:
      {(cp_tuple, co_tuple): distancia_minima}

    Además imprime:
      - numero de estados encontrados
      - distancia maxima (diametro respecto al resuelto, en el grafo generado)
    """

    base_moves = {
        "L": {"cp_perm": (1, 5, 2, 3, 0, 4, 6), "co_delta": (1, 2, 0, 0, 2, 1, 0)},
        "D": {"cp_perm": (0, 1, 2, 4, 5, 6, 3), "co_delta": (0, 0, 0, 0, 0, 0, 0)},
        "B": {"cp_perm": (0, 2, 6, 3, 4, 1, 5), "co_delta": (0, 1, 2, 0, 0, 2, 1)},
    }

    moves = ("L", "L2", "L'", "D", "D2", "D'", "B", "B2", "B'")

    def apply_base(cp: Tuple[int, ...], co: Tuple[int, ...], face: str) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        perm = base_moves[face]["cp_perm"]
        delta = base_moves[face]["co_delta"]
        new_cp = tuple(cp[perm[d]] for d in range(7))
        new_co = tuple((co[perm[d]] + delta[d]) % 3 for d in range(7))
        return new_cp, new_co

    def apply_move(cp: Tuple[int, ...], co: Tuple[int, ...], mv: str) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        mv = mv.strip()
        if mv in ("L", "D", "B"):
            return apply_base(cp, co, mv)
        if len(mv) == 2 and mv[1] == "2":
            cp2, co2 = apply_base(cp, co, mv[0])
            return apply_base(cp2, co2, mv[0])
        if len(mv) == 2 and mv[1] == "'":
            cpt, cot = cp, co
            for _ in range(3):
                cpt, cot = apply_base(cpt, cot, mv[0])
            return cpt, cot
        raise ValueError(f"Movimiento no permitido: {mv}")

    solved_cp = (0, 1, 2, 3, 4, 5, 6)
    solved_co = (0, 0, 0, 0, 0, 0, 0)
    start = (solved_cp, solved_co)

    pdb: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int] = {start: 0}
    q: Deque[Tuple[Tuple[int, ...], Tuple[int, ...]]] = deque([start])
    max_dist = 0

    while q:
        cp, co = q.popleft()
        dist = pdb[(cp, co)]
        if dist > max_dist:
            max_dist = dist
        nd = dist + 1
        for mv in moves:
            ncp, nco = apply_move(cp, co, mv)
            key = (ncp, nco)
            if key not in pdb:
                pdb[key] = nd
                q.append(key)

    print(f"Corner PDB: {len(pdb)} estados")
    print(f"Distancia maxima: {max_dist}")
    return pdb


if __name__ == "__main__":
    build_corner_pdb()
