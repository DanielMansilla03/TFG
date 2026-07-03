from fusecube_eda_edaspy import (
    FuseState,
    MOVES,
    apply_move,
    apply_algorithm,
    is_solved,
    validate_state,
    similarity_original,
    similarity_to_solved,
)
from pdb_similarity import load_or_build_pdbs, pdb_similarity


INVERSE = {
    "L": "L'", "L'": "L", "L2": "L2",
    "D": "D'", "D'": "D", "D2": "D2",
    "B": "B'", "B'": "B", "B2": "B2",
}


def inverse_algorithm(alg: str) -> str:
    tokens = [t for t in alg.split() if t]
    return " ".join(INVERSE[t] for t in reversed(tokens))


def test_estado_resuelto():
    s = FuseState.solved()
    assert is_solved(s)
    assert similarity_original(s) == 1.0
    assert similarity_to_solved(s) == 1.0


def test_movimiento_mas_inverso():
    s0 = FuseState.solved()

    for move in MOVES:
        s1 = apply_move(s0, move)
        s2 = apply_move(s1, INVERSE[move])
        assert s2 == s0


def test_cuatro_giros_misma_cara():
    s0 = FuseState.solved()

    for face in ["L", "D", "B"]:
        s = s0
        for _ in range(4):
            s = apply_move(s, face)
        assert s == s0


def test_movimiento_doble():
    s0 = FuseState.solved()

    for face in ["L", "D", "B"]:
        s1 = apply_move(s0, face + "2")
        s2 = apply_move(apply_move(s0, face), face)
        assert s1 == s2


def test_scramble_mas_inverso():
    scrambles = [
        "L D B",
        "L B2 D L' B",
        "D B' L2 D2 B",
        "L B D' L2 B' D L",
    ]

    for scramble in scrambles:
        inverse = inverse_algorithm(scramble)
        final_state = apply_algorithm(FuseState.solved(), scramble + " " + inverse)
        assert is_solved(final_state)


def test_estado_valido_tras_scramble():
    scrambles = [
        "L D B L2 D' B'",
        "B2 L D' B L2 D",
        "D L' B2 D' L2 B",
    ]

    for scramble in scrambles:
        state = apply_algorithm(FuseState.solved(), scramble)
        validate_state(state)


def test_pdb_estado_resuelto():
    corner_pdb, edge_pdb = load_or_build_pdbs()
    state = FuseState.solved()
    assert pdb_similarity(state, corner_pdb, edge_pdb) == 1.0
