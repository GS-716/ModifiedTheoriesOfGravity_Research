"""Punto de entrada reutilizable por el notebook, pruebas y exportador."""

from __future__ import annotations

from pathlib import Path

from mc_case0 import build_case0, evaluate_btz_ansatz
from mc_case1 import build_case1, evaluate_case1_ansatz
from mc_case2 import build_case2, evaluate_case2_ansatz
from mc_core import CouplingContext
from mc_eqt import build_eqt_general, evaluate_eqt_general_ansatz
from mc_general import build_general_theory
from mc_invariants import EQTModelSpec, symbolic_eqt_spec


def run_pipeline(
    output_dir: str | Path | None = None,
    eqt_spec: EQTModelSpec | None = None,
) -> CouplingContext:
    """Ejecuta casos de regresion y un modelo EQT configurable.

    La configuracion por defecto activa alpha_1, alpha_2 y beta_1. Con ello se
    ejercitan ordenes nuevos de ambas torres, mientras los Casos 1 y 2 ya cubren
    alpha_1 y beta_0 por separado.
    """
    base = Path(__file__).resolve().parent
    ctx = CouplingContext(Path(output_dir) if output_dir else base / "salidas")
    spec = eqt_spec or symbolic_eqt_spec(alpha_orders=(1, 2), beta_orders=(1,))
    build_general_theory(ctx)
    build_case0(ctx)
    build_case1(ctx)
    build_case2(ctx)
    build_eqt_general(ctx, spec)
    evaluate_btz_ansatz(ctx)
    evaluate_case1_ansatz(ctx)
    evaluate_case2_ansatz(ctx)
    evaluate_eqt_general_ansatz(ctx, spec)
    return ctx


if __name__ == "__main__":
    context = run_pipeline()
    print(f"Pasos construidos: {len(context.steps)}")
    print(f"Verificaciones exactas: {len(context.checks)}")
    print(f"Todas nulas: {all(value == 0 for value in context.checks.values())}")
