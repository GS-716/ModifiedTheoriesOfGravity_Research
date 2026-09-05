"""Public API for the optional formal field-equation solver."""

from .bridge import FieldEquationWolframBridge
from .solving import (
    FieldEquationSolution,
    FormalSolution,
    ReducedEquation,
    analyze_redundancy,
    classify_system,
    raise_metric_equation,
    solve_field_equations,
    solveFieldEquations,
)

__all__ = [
    "FieldEquationSolution",
    "FieldEquationWolframBridge",
    "FormalSolution",
    "ReducedEquation",
    "analyze_redundancy",
    "classify_system",
    "raise_metric_equation",
    "solve_field_equations",
    "solveFieldEquations",
]

