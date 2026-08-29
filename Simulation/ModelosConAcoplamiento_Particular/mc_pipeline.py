"""Punto de entrada reutilizable por el notebook, pruebas y exportador."""

from __future__ import annotations

from pathlib import Path

from mc_case0 import build_case0, evaluate_btz_ansatz
from mc_case1 import build_case1, evaluate_case1_ansatz
from mc_case2 import build_case2, evaluate_case2_ansatz
from mc_core import CouplingContext
from mc_general import build_general_theory


def run_pipeline(output_dir: str | Path | None = None) -> CouplingContext:
    base = Path(__file__).resolve().parent
    ctx = CouplingContext(Path(output_dir) if output_dir else base / "salidas")
    build_general_theory(ctx)
    build_case0(ctx)
    build_case1(ctx)
    build_case2(ctx)
    evaluate_btz_ansatz(ctx)
    evaluate_case1_ansatz(ctx)
    evaluate_case2_ansatz(ctx)
    return ctx


if __name__ == "__main__":
    context = run_pipeline()
    print(f"Pasos construidos: {len(context.steps)}")
    print(f"Verificaciones exactas: {len(context.checks)}")
    print(f"Todas nulas: {all(value == 0 for value in context.checks.values())}")
