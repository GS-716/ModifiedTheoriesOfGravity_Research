"""Infraestructura compartida para la derivacion simbolica.

La matematica tensorial abstracta se conserva como LaTeX legible, mientras que las
comprobaciones algebraicas y el calculo coordenado se guardan como objetos SymPy.
Esto evita fingir que SymPy conoce una derivada funcional tensorial que realmente
requiere convenciones de simetrizacion, y permite auditar cada paso por separado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import sympy as sp


@dataclass
class Step:
    """Una igualdad, identidad o verificacion mostrable y exportable."""

    key: str
    title: str
    lhs: str
    rhs: str
    group: str
    note: str = ""
    check: Any | None = None


@dataclass
class CouplingContext:
    """Estado completo de una corrida, sin variables globales ocultas."""

    output_dir: Path
    steps: list[Step] = field(default_factory=list)
    objects: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, sp.Expr] = field(default_factory=dict)

    def add(
        self,
        key: str,
        title: str,
        lhs: str,
        rhs: str,
        group: str,
        note: str = "",
        check: Any | None = None,
    ) -> Step:
        if any(step.key == key for step in self.steps):
            raise KeyError(f"La clave {key!r} ya existe")
        step = Step(key, title, lhs, rhs, group, note, check)
        self.steps.append(step)
        if check is not None:
            value = sp.simplify(check)
            self.checks[key] = value
            if value != 0:
                raise AssertionError(f"La verificacion {key} no dio cero: {value}")
        return step

    def put(self, key: str, value: Any) -> Any:
        self.objects[key] = value
        return value

    def show(self, keys: Iterable[str] | None = None) -> None:
        """Muestra pasos en Jupyter; fuera de Jupyter imprime texto compacto."""
        selected = self.steps if keys is None else [
            next(step for step in self.steps if step.key == key) for key in keys
        ]
        try:
            from IPython.display import Markdown, display

            for step in selected:
                note = f"\n\n{step.note}" if step.note else ""
                display(Markdown(
                    f"#### {step.title}\n\n"
                    f"Objeto: `ctx.steps[{self.steps.index(step)}]` / clave `{step.key}`\n\n"
                    f"$$\n{step.lhs} = {step.rhs}\n$$"
                    f"{note}"
                ))
        except ImportError:
            for step in selected:
                print(f"[{step.key}] {step.title}: {step.lhs} = {step.rhs}")


def latex_expr(expr: Any) -> str:
    """LaTeX estable para escalares, matrices y arreglos pequenos."""
    if isinstance(expr, str):
        return expr
    return sp.latex(sp.simplify(expr))


def matrix_is_zero(matrix: sp.MatrixBase) -> bool:
    return all(sp.simplify(entry) == 0 for entry in matrix)

