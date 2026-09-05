"""Reproducible formal-solver examples; no initial/boundary data.

From FieldEquationsSolver: python examples/solve_field_equations.py
Wolfram and a LaTeX compiler are optional; diagnostics remain in the bundles.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "TensorEngine" / "src"))
sys.path.insert(0, str(ROOT / "src"))

from tensor_engine import (  # noqa: E402
    AnsatzSpecialization, DimensionSpec, EngineOptions, LagrangianSourceSpec,
    ParameterSpec, Scalar, TensorEngine, WolframXActBridge,
    draft4_circular_ansatz, spatially_flat_flrw_ansatz,
)
from field_equations_solver import FieldEquationWolframBridge, solveFieldEquations  # noqa: E402


def main():
    cases = (
        ("eh_draft4", "R + 2/ell**2", draft4_circular_ansatz()),
        ("case2_draft4", "R + 2/ell**2 + ell**2*beta0*(3*RicciUU - X*R)", draft4_circular_ansatz()),
        ("eh_flrw", "R", spatially_flat_flrw_ansatz()),
    )
    xact_bridge = WolframXActBridge(timeout_seconds=180)
    solver_bridge = FieldEquationWolframBridge(timeout_seconds=180)
    for name, expression, ansatz in cases:
        print("Calculando", name, flush=True)
        model = LagrangianSourceSpec(
            name=name, expression=expression, dimension=DimensionSpec(ansatz.dimension),
            parameters=(ParameterSpec("ell"), ParameterSpec("beta0")),
        ).compile()
        run = TensorEngine(options=EngineOptions(include_noether=False, include_export=False)).run(
            model, ansatz=ansatz, wolfram_bridge=xact_bridge if xact_bridge.available else None,
        )
        solution = solveFieldEquations(
            run,
            specialization=(AnsatzSpecialization(scalar_field=Scalar("q")*Scalar("varphi"))
                            if ansatz.name == "draft4_circular" else None),
            wolfram_bridge=solver_bridge,
            output_root=ROOT/"output"/"pdf"/"field-equation-solving"/name,
        )
        print(name, solution.status, [s.status for s in solution.solutions], flush=True)
        print("xAct/source:", run.package.verification.summary, flush=True)
        print(solution.output_directory, flush=True)


if __name__ == "__main__":
    main()
