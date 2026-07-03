"""Genera figura success% vs profundidad con barras de error Wilson 95%.

4 variantes principales (baseline, no_stop, markov, markov_anchor).
Exporta PDF y SVG listos para LaTeX.
"""
from __future__ import annotations
import json, math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")  # backend sin display
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def collect_curve(payload_path: Path):
    """Devuelve dict {depth: (k, n)} a partir de un JSON de diagnostico."""
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    per_run = payload["per_run"]
    out = {}
    for d_str, runs in per_run.items():
        d = int(d_str)
        n = len(runs)
        k = sum(1 for r in runs if r["solved"])
        out[d] = (k, n)
    return out


VARIANTS = [
    ("baseline",     "diagnostico.json",               "#1f77b4", "o", "-"),
    ("no_stop",      "diagnostico_no_stop.json",       "#ff7f0e", "s", "-"),
    ("markov",       "diagnostico_markov_d25.json",    "#2ca02c", "^", "-"),
    ("markov_anchor","diagnostico_markov_anchor.json", "#d62728", "D", "-"),
]


def main(out_basename: str = "curva_success") -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for label, fname, color, marker, ls in VARIANTS:
        curve = collect_curve(HERE / fname)
        depths = sorted(curve.keys())
        ps, los, his, ns = [], [], [], []
        for d in depths:
            k, n = curve[d]
            p = k / n
            lo, hi = wilson_ci(k, n)
            ps.append(p * 100)
            los.append((p - lo) * 100)
            his.append((hi - p) * 100)
            ns.append(n)

        ax.errorbar(
            depths, ps, yerr=[los, his],
            color=color, marker=marker, markersize=7, linewidth=1.6,
            linestyle=ls, capsize=4, capthick=1.2, elinewidth=1.0,
            label=label,
        )
        # Anotación de n por punto (debajo del marcador)
        for d, p_pct, n in zip(depths, ps, ns):
            ax.annotate(
                f"n={n}", (d, p_pct), textcoords="offset points",
                xytext=(0, -14), fontsize=7, color=color, ha="center",
                alpha=0.7,
            )

    ax.set_xlabel("Profundidad del scramble (movimientos)", fontsize=11)
    ax.set_ylabel("Tasa de resolución (%)", fontsize=11)
    ax.set_title(
        "EDA Fuse 3x3x3: tasa de resolución vs profundidad\n"
        "(barras = IC 95% Wilson)",
        fontsize=12,
    )
    ax.set_xticks([5, 10, 15, 19, 25])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(-5, 108)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    fig.tight_layout()

    pdf_path = HERE / f"{out_basename}.pdf"
    svg_path = HERE / f"{out_basename}.svg"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {pdf_path}")
    print(f"Guardado: {svg_path}")


if __name__ == "__main__":
    main()
