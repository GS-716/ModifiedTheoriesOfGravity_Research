"""Evaluación coordenada de expresiones tensoriales sobre un ansatz concreto.

La IR de TensorEngine sigue siendo el formato canónico. SymPy se usa aquí como
motor escalar para invertir la métrica, derivar respecto a coordenadas y
simplificar cada componente.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import product
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import sympy as sp
from sympy.core.function import AppliedUndef

from .errors import BackendExecutionError, ModelValidationError, TensorAlgebraError
from .ir import (
    Add,
    CovariantDerivative,
    Expr,
    Function,
    FunctionDerivative,
    Index,
    Mul,
    Number,
    Power,
    Scalar,
    Tensor,
    Variance,
    Variation,
    VolumeElement,
    add,
    as_expr,
    expr_from_data,
    infer_free_indices,
    mul,
    power,
    walk,
)
from .model import GeometrySymbols, ModelSpec


COMPONENT_SCHEMA_VERSION = "1.0"
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_ELEMENTARY_FUNCTIONS: dict[str, Any] = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "Abs": sp.Abs,
}


class ScalarFieldMode(str, Enum):
    """Nivel de especificación del campo escalar dentro de un ansatz."""

    ABSENT = "absent"
    GENERIC = "generic"
    SPECIALIZED = "specialized"


def _validate_name(name: str, label: str) -> None:
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ModelValidationError(f"{label} inválido: {name!r}.")


def _require_scalar(expr: Expr, label: str) -> None:
    if infer_free_indices(expr):
        raise ModelValidationError(f"{label} debe ser una expresión escalar.")


@dataclass(frozen=True, slots=True)
class CoordinateChart:
    """Carta ordenada; la posición de cada coordenada fija su componente."""

    name: str
    coordinates: tuple[Scalar, ...]

    def __post_init__(self) -> None:
        _validate_name(self.name, "nombre de carta")
        object.__setattr__(self, "coordinates", tuple(self.coordinates))
        if len(self.coordinates) < 2:
            raise ModelValidationError("Una carta debe contener al menos dos coordenadas.")
        names = [coordinate.name for coordinate in self.coordinates]
        if len(names) != len(set(names)):
            raise ModelValidationError("Las coordenadas de una carta deben ser únicas.")

    @property
    def dimension(self) -> int:
        return len(self.coordinates)

    def to_data(self) -> dict[str, Any]:
        return {"name": self.name, "coordinates": [item.name for item in self.coordinates]}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "CoordinateChart":
        return cls(
            str(data["name"]),
            tuple(Scalar(str(item)) for item in data["coordinates"]),
        )


@dataclass(frozen=True, slots=True)
class GeometryAnsatz:
    """Métrica covariante y campo escalar definidos sobre una carta."""

    name: str
    chart: CoordinateChart
    metric_covariant: tuple[tuple[Expr, ...], ...]
    scalar_field: Expr | None = None
    assumptions: tuple[str, ...] = ()
    scalar_field_mode: ScalarFieldMode | None = None
    schema_version: str = COMPONENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_name(self.name, "nombre de ansatz")
        rows = tuple(tuple(row) for row in self.metric_covariant)
        object.__setattr__(self, "metric_covariant", rows)
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        if self.schema_version != COMPONENT_SCHEMA_VERSION:
            raise ModelValidationError(
                f"Versión de GeometryAnsatz no soportada: {self.schema_version!r}."
            )
        dimension = self.chart.dimension
        if len(rows) != dimension or any(len(row) != dimension for row in rows):
            raise ModelValidationError(
                "La matriz métrica debe ser cuadrada y coincidir con la dimensión de la carta."
            )
        for a, row in enumerate(rows):
            for b, entry in enumerate(row):
                _require_scalar(entry, f"g[{a},{b}]")
                if entry != rows[b][a]:
                    raise ModelValidationError("La métrica covariante del ansatz debe ser simétrica.")
        if self.scalar_field is not None:
            _require_scalar(self.scalar_field, "El campo escalar del ansatz")
        mode = self.scalar_field_mode
        if mode is None:
            mode = (
                ScalarFieldMode.ABSENT
                if self.scalar_field is None
                else ScalarFieldMode.GENERIC
                if isinstance(self.scalar_field, Function)
                else ScalarFieldMode.SPECIALIZED
            )
        else:
            mode = ScalarFieldMode(mode)
        if self.scalar_field is None and mode is not ScalarFieldMode.ABSENT:
            raise ModelValidationError(
                "Un ansatz sin campo escalar debe usar scalar_field_mode='absent'."
            )
        if self.scalar_field is not None and mode is ScalarFieldMode.ABSENT:
            raise ModelValidationError(
                "Un ansatz con campo escalar no puede usar scalar_field_mode='absent'."
            )
        object.__setattr__(self, "scalar_field_mode", mode)
        if len(set(self.assumptions)) != len(self.assumptions):
            raise ModelValidationError("El ansatz contiene hipótesis repetidas.")

    @property
    def dimension(self) -> int:
        return self.chart.dimension

    def validate_for_model(self, model: ModelSpec) -> "GeometryAnsatz":
        if model.dimension.is_symbolic:
            raise ModelValidationError(
                "La evaluación por componentes requiere fijar una dimensión entera en ModelSpec."
            )
        if model.dimension.value != self.dimension:
            raise ModelValidationError(
                f"El modelo tiene dimensión {model.dimension.value}, pero el ansatz tiene "
                f"dimensión {self.dimension}."
            )
        return self

    def validate_scalar_specialization(self, scalar_field: Expr) -> Expr:
        """Impide que una especialización introduzca dependencias no declaradas.

        La dependencia coordenada permitida se obtiene de los argumentos del
        campo genérico del ansatz. Esto mantiene, por ejemplo, ``Phi(r,varphi)``
        estacionario en Draft 4 sin imponer la misma restricción a FLRW u otros
        ansatz que declaren coordenadas diferentes.
        """

        _require_scalar(scalar_field, "El perfil escalar especializado")
        if not isinstance(self.scalar_field, Function):
            return scalar_field
        chart_names = {coordinate.name for coordinate in self.chart.coordinates}
        allowed = {
            node.name
            for argument in self.scalar_field.arguments
            for node in walk(argument)
            if isinstance(node, Scalar) and node.name in chart_names
        }
        used = {
            node.name
            for node in walk(scalar_field)
            if isinstance(node, Scalar) and node.name in chart_names
        }
        forbidden = sorted(used.difference(allowed))
        if forbidden:
            declared = ", ".join(sorted(allowed)) or "ninguna coordenada"
            raise ModelValidationError(
                "El perfil escalar especializado introduce dependencias coordenadas "
                f"no declaradas por {self.scalar_field.name}: {', '.join(forbidden)}. "
                f"Dependencias permitidas: {declared}."
            )
        return scalar_field

    def specialize_scalar(
        self,
        scalar_field: Expr,
        *,
        name: str | None = None,
        assumptions: tuple[str, ...] = (),
    ) -> "GeometryAnsatz":
        """Impone explícitamente un perfil escalar sin cambiar la métrica."""

        self.validate_scalar_specialization(scalar_field)
        return GeometryAnsatz(
            name=self.name if name is None else name,
            chart=self.chart,
            metric_covariant=self.metric_covariant,
            scalar_field=scalar_field,
            assumptions=tuple(dict.fromkeys((*self.assumptions, *assumptions))),
            scalar_field_mode=ScalarFieldMode.SPECIALIZED,
        )

    def specialize(
        self,
        replacements: Mapping[Expr, Expr],
        *,
        name: str | None = None,
        assumptions: tuple[str, ...] = (),
    ) -> "GeometryAnsatz":
        """Sustituye funciones del ansatz para evaluar soluciones concretas.

        La transformación se aplica únicamente a la métrica y al campo escalar
        coordenados; nunca modifica la IR covariante del modelo.
        """

        from .transform import substitute

        mapping = dict(replacements)
        metric = tuple(
            tuple(substitute(entry, mapping) for entry in row)
            for row in self.metric_covariant
        )
        scalar = (
            None
            if self.scalar_field is None
            else substitute(self.scalar_field, mapping)
        )
        scalar_mode = self.scalar_field_mode
        if scalar != self.scalar_field and scalar is not None:
            self.validate_scalar_specialization(scalar)
            scalar_mode = ScalarFieldMode.SPECIALIZED
        return GeometryAnsatz(
            name=self.name if name is None else name,
            chart=self.chart,
            metric_covariant=metric,
            scalar_field=scalar,
            assumptions=tuple(dict.fromkeys((*self.assumptions, *assumptions))),
            scalar_field_mode=scalar_mode,
        )

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "chart": self.chart.to_data(),
            "metric_covariant": [
                [entry.to_data() for entry in row] for row in self.metric_covariant
            ],
            "scalar_field": None if self.scalar_field is None else self.scalar_field.to_data(),
            "scalar_field_mode": self.scalar_field_mode.value,
            "assumptions": list(self.assumptions),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "GeometryAnsatz":
        scalar_data = data.get("scalar_field")
        return cls(
            name=str(data["name"]),
            chart=CoordinateChart.from_data(data["chart"]),
            metric_covariant=tuple(
                tuple(expr_from_data(entry) for entry in row)
                for row in data["metric_covariant"]
            ),
            scalar_field=None if scalar_data is None else expr_from_data(scalar_data),
            assumptions=tuple(str(item) for item in data.get("assumptions", ())),
            scalar_field_mode=(
                None
                if data.get("scalar_field_mode") is None
                else ScalarFieldMode(str(data["scalar_field_mode"]))
            ),
            schema_version=str(data.get("schema_version", COMPONENT_SCHEMA_VERSION)),
        )


@dataclass(frozen=True, slots=True)
class AnsatzSpecialization:
    """Sustituciones coordenadas aplicadas después de la proyección genérica.

    Los valores son expresiones escalares de la IR. La especialización crea un
    nuevo ``GeometryAnsatz`` y nunca modifica el modelo ni su derivación
    covariante.
    """

    metric_functions: Mapping[str, Expr] = field(default_factory=dict)
    scalar_field: Expr | None = None
    name: str | None = None
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized: dict[str, Expr] = {}
        for function_name, value in self.metric_functions.items():
            _validate_name(function_name, "nombre de función métrica")
            expression = as_expr(value)
            _require_scalar(expression, f"La especialización de {function_name}")
            normalized[function_name] = expression
        object.__setattr__(
            self,
            "metric_functions",
            MappingProxyType(dict(sorted(normalized.items()))),
        )
        if self.scalar_field is not None:
            scalar = as_expr(self.scalar_field)
            _require_scalar(scalar, "El perfil escalar especializado")
            object.__setattr__(self, "scalar_field", scalar)
        if self.name is not None:
            _validate_name(self.name, "nombre de especialización")
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        if len(set(self.assumptions)) != len(self.assumptions):
            raise ModelValidationError("La especialización contiene hipótesis repetidas.")
        if not normalized and self.scalar_field is None:
            raise ModelValidationError(
                "AnsatzSpecialization requiere al menos una función métrica o un campo escalar."
            )

    def apply(self, ansatz: GeometryAnsatz) -> GeometryAnsatz:
        """Aplica la sustitución a una copia del ansatz base."""

        from .transform import substitute

        available: dict[str, set[Function]] = {}
        for row in ansatz.metric_covariant:
            for entry in row:
                for node in walk(entry):
                    if isinstance(node, Function):
                        available.setdefault(node.name, set()).add(node)
        missing = set(self.metric_functions).difference(available)
        if missing:
            raise ModelValidationError(
                "El ansatz no contiene las funciones métricas solicitadas: "
                + ", ".join(sorted(missing))
            )
        replacements = {
            function: self.metric_functions[name]
            for name, functions in available.items()
            if name in self.metric_functions
            for function in functions
        }
        metric = tuple(
            tuple(substitute(entry, replacements) for entry in row)
            for row in ansatz.metric_covariant
        )
        scalar = ansatz.scalar_field if self.scalar_field is None else self.scalar_field
        mode = ansatz.scalar_field_mode
        if self.scalar_field is not None:
            ansatz.validate_scalar_specialization(self.scalar_field)
            mode = ScalarFieldMode.SPECIALIZED
        return GeometryAnsatz(
            name=self.name or f"{ansatz.name}_specialized",
            chart=ansatz.chart,
            metric_covariant=metric,
            scalar_field=scalar,
            assumptions=tuple(dict.fromkeys((*ansatz.assumptions, *self.assumptions))),
            scalar_field_mode=mode,
        )

    def to_data(self) -> dict[str, Any]:
        return {
            "metric_functions": {
                name: expression.to_data()
                for name, expression in self.metric_functions.items()
            },
            "scalar_field": (
                None if self.scalar_field is None else self.scalar_field.to_data()
            ),
            "name": self.name,
            "assumptions": list(self.assumptions),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "AnsatzSpecialization":
        scalar = data.get("scalar_field")
        return cls(
            metric_functions={
                str(name): expr_from_data(expression)
                for name, expression in data.get("metric_functions", {}).items()
            },
            scalar_field=None if scalar is None else expr_from_data(scalar),
            name=None if data.get("name") is None else str(data["name"]),
            assumptions=tuple(str(item) for item in data.get("assumptions", ())),
        )


def ir_scalar_to_sympy(
    expr: Expr,
    scalar_values: Mapping[str, sp.Expr] | None = None,
) -> sp.Expr:
    """Traduce el subconjunto escalar de la IR a SymPy sin analizar texto."""

    values = scalar_values or {}
    if isinstance(expr, Number):
        return sp.Rational(expr.numerator, expr.denominator)
    if isinstance(expr, Scalar):
        return values.get(expr.name, sp.Symbol(expr.name))
    if isinstance(expr, Add):
        return sp.Add(*(ir_scalar_to_sympy(term, values) for term in expr.terms))
    if isinstance(expr, Mul):
        return sp.Mul(*(ir_scalar_to_sympy(factor, values) for factor in expr.factors))
    if isinstance(expr, Power):
        return ir_scalar_to_sympy(expr.base, values) ** ir_scalar_to_sympy(expr.exponent, values)
    if isinstance(expr, Function):
        arguments = tuple(ir_scalar_to_sympy(item, values) for item in expr.arguments)
        head = _ELEMENTARY_FUNCTIONS.get(expr.name, sp.Function(expr.name))
        return head(*arguments)
    if isinstance(expr, FunctionDerivative):
        arguments = tuple(ir_scalar_to_sympy(item, values) for item in expr.arguments)
        dummy = tuple(sp.Dummy(f"arg{position}") for position in range(len(arguments)))
        formal = sp.Function(expr.name)(*dummy)
        variables: list[sp.Symbol] = []
        for variable, order in zip(dummy, expr.derivative_orders, strict=True):
            variables.extend([variable] * order)
        derivative = sp.Derivative(formal, *variables, evaluate=False)
        return sp.Subs(derivative, dummy, arguments).doit()
    raise BackendExecutionError(
        f"El nodo {type(expr).__name__} no pertenece al subconjunto escalar coordenado."
    )


def _derivative_to_ir(expr: sp.Derivative) -> Expr:
    base = expr.expr
    if not isinstance(base, AppliedUndef):
        raise BackendExecutionError(
            f"No se puede representar en la IR la derivada SymPy {expr!s}."
        )
    arguments = tuple(sympy_scalar_to_ir(item) for item in base.args)
    orders = [0] * len(base.args)
    for variable, count in expr.variable_count:
        try:
            position = base.args.index(variable)
        except ValueError as error:
            raise BackendExecutionError(
                f"La derivada {expr!s} no es respecto a argumentos de {base.func.__name__}."
            ) from error
        orders[position] += int(count)
    return FunctionDerivative(base.func.__name__, tuple(orders), arguments)


def sympy_scalar_to_ir(expr: sp.Expr) -> Expr:
    """Convierte un resultado escalar exacto de SymPy de vuelta a la IR."""

    expr = sp.factor_terms(expr)
    if expr.is_Integer:
        return Number(int(expr))
    if expr.is_Rational:
        return Number(int(expr.p), int(expr.q))
    if isinstance(expr, sp.Symbol):
        return Scalar(str(expr))
    if isinstance(expr, sp.Add):
        return add(*(sympy_scalar_to_ir(item) for item in expr.args))
    if isinstance(expr, sp.Mul):
        return mul(*(sympy_scalar_to_ir(item) for item in expr.args))
    if isinstance(expr, sp.Pow):
        return power(sympy_scalar_to_ir(expr.base), sympy_scalar_to_ir(expr.exp))
    if isinstance(expr, AppliedUndef):
        return Function(
            expr.func.__name__,
            tuple(sympy_scalar_to_ir(item) for item in expr.args),
        )
    if isinstance(expr, sp.Derivative):
        return _derivative_to_ir(expr)
    if isinstance(expr, sp.Subs) and isinstance(expr.expr, sp.Derivative):
        derivative = expr.expr
        base = derivative.expr
        if not isinstance(base, AppliedUndef):
            raise BackendExecutionError(f"Subs no representable en la IR: {expr!s}.")
        substitutions = dict(zip(expr.variables, expr.point, strict=True))
        arguments = tuple(
            sympy_scalar_to_ir(item.xreplace(substitutions)) for item in base.args
        )
        orders = [0] * len(base.args)
        for variable, count in derivative.variable_count:
            position = base.args.index(variable)
            orders[position] += int(count)
        return FunctionDerivative(base.func.__name__, tuple(orders), arguments)
    if expr.is_Function:
        return Function(
            expr.func.__name__,
            tuple(sympy_scalar_to_ir(item) for item in expr.args),
        )
    raise BackendExecutionError(f"Resultado SymPy no representable en la IR: {expr!r}.")


@dataclass(frozen=True, slots=True)
class ComponentTensor:
    """Tensor denso con orden de ejes explícito y componentes SymPy exactas."""

    indices: tuple[Index, ...]
    dimension: int
    values: tuple[tuple[tuple[int, ...], sp.Expr], ...]
    _value_mapping: Mapping[tuple[int, ...], sp.Expr] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "indices", tuple(self.indices))
        object.__setattr__(self, "values", tuple(self.values))
        object.__setattr__(self, "_value_mapping", dict(self.values))
        if self.dimension < 2:
            raise TensorAlgebraError("Un tensor coordenado requiere dimensión D>=2.")
        positions = [position for position, _ in self.values]
        if len(positions) != len(set(positions)):
            raise TensorAlgebraError("Un tensor coordenado repite una componente.")

    @classmethod
    def from_mapping(
        cls,
        indices: Sequence[Index],
        dimension: int,
        values: Mapping[tuple[int, ...], sp.Expr],
    ) -> "ComponentTensor":
        axes = tuple(indices)
        normalized = tuple(
            sorted(
                (
                    (tuple(key), sp.simplify(value))
                    for key, value in values.items()
                    if sp.simplify(value) != 0
                ),
                key=lambda item: item[0],
            )
        )
        if any(len(key) != len(axes) for key, _ in normalized):
            raise TensorAlgebraError("Una componente no coincide con el rango del tensor.")
        if any(any(item < 0 or item >= dimension for item in key) for key, _ in normalized):
            raise TensorAlgebraError("Una componente está fuera del rango de la carta.")
        return cls(axes, dimension, normalized)

    @property
    def mapping(self) -> dict[tuple[int, ...], sp.Expr]:
        return dict(self._value_mapping)

    def component(self, *positions: int) -> sp.Expr:
        if len(positions) != len(self.indices):
            raise TensorAlgebraError(
                f"Se esperaban {len(self.indices)} posiciones y se recibieron {len(positions)}."
            )
        return self._value_mapping.get(tuple(positions), sp.S.Zero)

    @property
    def is_scalar(self) -> bool:
        return not self.indices

    @property
    def scalar(self) -> sp.Expr:
        if not self.is_scalar:
            raise TensorAlgebraError("El resultado conserva índices libres.")
        return self.component()

    def to_ir(self) -> "ComponentEvaluation":
        return ComponentEvaluation(
            self.indices,
            self.dimension,
            tuple((key, sympy_scalar_to_ir(value)) for key, value in self.values),
        )


@dataclass(frozen=True, slots=True)
class ComponentEvaluation:
    """Resultado serializable de proyectar una expresión abstracta."""

    free_indices: tuple[Index, ...]
    dimension: int
    values: tuple[tuple[tuple[int, ...], Expr], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "free_indices", tuple(self.free_indices))
        object.__setattr__(
            self,
            "values",
            tuple((tuple(position), expression) for position, expression in self.values),
        )
        if self.dimension < 2:
            raise TensorAlgebraError("La evaluación requiere dimensión mayor o igual que dos.")
        positions = [position for position, _ in self.values]
        if len(positions) != len(set(positions)):
            raise TensorAlgebraError("Una evaluación repite posiciones de componentes.")
        for position, expression in self.values:
            if len(position) != len(self.free_indices):
                raise TensorAlgebraError("Una posición no coincide con el rango del resultado.")
            if any(item < 0 or item >= self.dimension for item in position):
                raise TensorAlgebraError("Una posición está fuera del rango de la carta.")
            _require_scalar(expression, "Cada componente")

    def component(self, *positions: int) -> Expr:
        return dict(self.values).get(tuple(positions), Number(0))

    @property
    def scalar(self) -> Expr:
        if self.free_indices:
            raise TensorAlgebraError("El resultado conserva índices libres.")
        return self.component()

    def to_data(self) -> dict[str, Any]:
        return {
            "free_indices": [item.to_data() for item in self.free_indices],
            "dimension": self.dimension,
            "components": [
                {"position": list(position), "expression": expression.to_data()}
                for position, expression in self.values
            ],
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ComponentEvaluation":
        return cls(
            free_indices=tuple(Index.from_data(item) for item in data["free_indices"]),
            dimension=int(data["dimension"]),
            values=tuple(
                (
                    tuple(int(item) for item in component["position"]),
                    expr_from_data(component["expression"]),
                )
                for component in data.get("components", ())
            ),
        )


def specialize_scalar_components(
    components: ComponentEvaluation,
    original_field: Function,
    scalar_field: Expr,
) -> ComponentEvaluation:
    """Especializa componentes existentes, incluidas derivadas del campo.

    No vuelve a proyectar ni deriva el lagrangiano. Evalúa la sustitución
    escalar con los mismos traductores IR/SymPy del backend de componentes.
    """

    _require_scalar(scalar_field, "El perfil escalar especializado")
    original = ir_scalar_to_sympy(original_field)
    replacement = ir_scalar_to_sympy(scalar_field)
    values = {
        position: ir_scalar_to_sympy(expression).subs(original, replacement).doit()
        for position, expression in components.values
    }
    return ComponentTensor.from_mapping(
        components.free_indices, components.dimension, values,
    ).to_ir()


@dataclass(frozen=True, slots=True)
class CoordinateGeometry:
    """Geometría de Levi-Civita calculada desde un ``GeometryAnsatz``."""

    ansatz: GeometryAnsatz
    coordinates: tuple[sp.Symbol, ...]
    metric_covariant: sp.ImmutableDenseMatrix
    metric_contravariant: sp.ImmutableDenseMatrix
    determinant: sp.Expr
    christoffel: tuple
    riemann_up: tuple
    riemann_down: tuple
    ricci_covariant: sp.ImmutableDenseMatrix
    ricci_scalar: sp.Expr
    einstein_covariant: sp.ImmutableDenseMatrix
    scalar_field: sp.Expr | None

    @classmethod
    def build(cls, ansatz: GeometryAnsatz) -> "CoordinateGeometry":
        coordinates = tuple(sp.Symbol(item.name) for item in ansatz.chart.coordinates)
        scalar_values: dict[str, sp.Expr] = {
            item.name: coordinate
            for item, coordinate in zip(ansatz.chart.coordinates, coordinates, strict=True)
        }
        metric = sp.ImmutableDenseMatrix(
            [
                [ir_scalar_to_sympy(entry, scalar_values) for entry in row]
                for row in ansatz.metric_covariant
            ]
        )
        determinant = sp.simplify(metric.det())
        if determinant == 0:
            raise ModelValidationError("La métrica del ansatz es degenerada.")
        try:
            inverse = sp.ImmutableDenseMatrix(sp.simplify(metric.inv()))
        except (ValueError, ZeroDivisionError) as error:
            raise ModelValidationError("No se pudo invertir la métrica del ansatz.") from error
        n = ansatz.dimension
        gamma = tuple(
            tuple(
                tuple(
                    sp.simplify(
                        sp.Rational(1, 2)
                        * sum(
                            inverse[rho, sigma]
                            * (
                                sp.diff(metric[sigma, nu], coordinates[mu])
                                + sp.diff(metric[sigma, mu], coordinates[nu])
                                - sp.diff(metric[mu, nu], coordinates[sigma])
                            )
                            for sigma in range(n)
                        )
                    )
                    for nu in range(n)
                )
                for mu in range(n)
            )
            for rho in range(n)
        )
        riemann_up = tuple(
            tuple(
                tuple(
                    tuple(
                        sp.simplify(
                            sp.diff(gamma[rho][nu][sigma], coordinates[mu])
                            - sp.diff(gamma[rho][mu][sigma], coordinates[nu])
                            + sum(
                                gamma[rho][mu][lam] * gamma[lam][nu][sigma]
                                - gamma[rho][nu][lam] * gamma[lam][mu][sigma]
                                for lam in range(n)
                            )
                        )
                        for nu in range(n)
                    )
                    for mu in range(n)
                )
                for sigma in range(n)
            )
            for rho in range(n)
        )
        riemann_down = tuple(
            tuple(
                tuple(
                    tuple(
                        sp.simplify(
                            sum(metric[a, rho] * riemann_up[rho][b][c][d] for rho in range(n))
                        )
                        for d in range(n)
                    )
                    for c in range(n)
                )
                for b in range(n)
            )
            for a in range(n)
        )
        ricci = sp.ImmutableDenseMatrix(
            n,
            n,
            lambda a, b: sp.simplify(
                sum(riemann_up[rho][a][rho][b] for rho in range(n))
            ),
        )
        ricci_scalar = sp.simplify(
            sum(inverse[a, b] * ricci[a, b] for a, b in product(range(n), repeat=2))
        )
        einstein = sp.ImmutableDenseMatrix(
            n,
            n,
            lambda a, b: sp.simplify(
                ricci[a, b] - sp.Rational(1, 2) * metric[a, b] * ricci_scalar
            ),
        )
        scalar_field = (
            None
            if ansatz.scalar_field is None
            else ir_scalar_to_sympy(ansatz.scalar_field, scalar_values)
        )
        return cls(
            ansatz,
            coordinates,
            metric,
            inverse,
            determinant,
            gamma,
            riemann_up,
            riemann_down,
            ricci,
            ricci_scalar,
            einstein,
            scalar_field,
        )

    @property
    def dimension(self) -> int:
        return self.ansatz.dimension

    def scalar_gradient_covariant(self, scalar: sp.Expr | None = None) -> tuple[sp.Expr, ...]:
        value = self.scalar_field if scalar is None else scalar
        if value is None:
            raise ModelValidationError("El ansatz no define un campo escalar.")
        return tuple(sp.simplify(sp.diff(value, coordinate)) for coordinate in self.coordinates)

    def scalar_hessian_covariant(self, scalar: sp.Expr | None = None) -> sp.ImmutableDenseMatrix:
        value = self.scalar_field if scalar is None else scalar
        if value is None:
            raise ModelValidationError("El ansatz no define un campo escalar.")
        gradient = self.scalar_gradient_covariant(value)
        return sp.ImmutableDenseMatrix(
            self.dimension,
            self.dimension,
            lambda a, b: sp.simplify(
                sp.diff(gradient[b], self.coordinates[a])
                - sum(
                    self.christoffel[c][a][b] * gradient[c]
                    for c in range(self.dimension)
                )
            ),
        )

    def scalar_laplacian(self, scalar: sp.Expr | None = None) -> sp.Expr:
        hessian = self.scalar_hessian_covariant(scalar)
        return sp.simplify(
            sum(
                self.metric_contravariant[a, b] * hessian[a, b]
                for a, b in product(range(self.dimension), repeat=2)
            )
        )

    def nonzero_christoffel(self) -> dict[tuple[int, int, int], sp.Expr]:
        return {
            (rho, mu, nu): value
            for rho, mu, nu in product(range(self.dimension), repeat=3)
            if (value := sp.simplify(self.christoffel[rho][mu][nu])) != 0
        }

    def covariant_derivative(self, tensor: ComponentTensor) -> ComponentTensor:
        """Calcula todas las componentes de nabla_k T sin contraer k."""

        new_index = Index("coordinate_derivative", Variance.DOWN)
        result_indices = (*tensor.indices, new_index)
        values: dict[tuple[int, ...], sp.Expr] = {}
        for position in product(range(self.dimension), repeat=len(tensor.indices)):
            for derivative in range(self.dimension):
                value = sp.diff(tensor.component(*position), self.coordinates[derivative])
                for axis, index in enumerate(tensor.indices):
                    slot = position[axis]
                    for replacement in range(self.dimension):
                        changed = (*position[:axis], replacement, *position[axis + 1 :])
                        if index.variance is Variance.UP:
                            value += (
                                self.christoffel[slot][derivative][replacement]
                                * tensor.component(*changed)
                            )
                        else:
                            value -= (
                                self.christoffel[replacement][derivative][slot]
                                * tensor.component(*changed)
                            )
                values[(*position, derivative)] = sp.simplify(value)
        return ComponentTensor.from_mapping(result_indices, self.dimension, values)


def spatially_flat_flrw_ansatz() -> GeometryAnsatz:
    """Referencia 4D: ds²=-dt²+a(t)²(dx²+dy²+dz²), phi=phi(t)."""

    t, x, y, z = (Scalar(name) for name in ("t", "x", "y", "z"))
    scale = Function("a", (t,))
    zero = Number(0)
    scale_squared = power(scale, 2)
    return GeometryAnsatz(
        name="flat_flrw",
        chart=CoordinateChart("cosmological", (t, x, y, z)),
        metric_covariant=(
            (Number(-1), zero, zero, zero),
            (zero, scale_squared, zero, zero),
            (zero, zero, scale_squared, zero),
            (zero, zero, zero, scale_squared),
        ),
        scalar_field=Function("phi", (t,)),
        assumptions=("a(t)>0",),
        scalar_field_mode=ScalarFieldMode.GENERIC,
    )


def draft4_circular_ansatz() -> GeometryAnsatz:
    """Draft 4 con f(r) y el campo estacionario Phi(r,varphi) genéricos."""

    tau, radial, angle = (Scalar(name) for name in ("tau", "r", "varphi"))
    metric_function = Function("f", (radial,))
    zero = Number(0)
    return GeometryAnsatz(
        name="draft4_circular",
        chart=CoordinateChart("draft4_axial", (tau, radial, angle)),
        metric_covariant=(
            (mul(-1, metric_function), zero, zero),
            (zero, power(metric_function, -1), zero),
            (zero, zero, power(radial, 2)),
        ),
        scalar_field=Function("Phi", (radial, angle)),
        assumptions=("r>0", "f(r)!=0"),
        scalar_field_mode=ScalarFieldMode.GENERIC,
    )


def draft4_angular_scalar_profile(parameter_name: str = "p") -> Expr:
    """Perfil opcional Phi=p*varphi para especializar Draft 4 explícitamente."""

    _validate_name(parameter_name, "nombre del parámetro del perfil escalar")
    return mul(Scalar(parameter_name), Scalar("varphi"))


class SympyComponentBackend:
    """Proyecta la IR tensorial a componentes coordenadas exactas."""

    name = "sympy-components"
    version = "0.7.0"
    generic_scalar_derivative_budget = 24

    def __init__(
        self,
        geometry: CoordinateGeometry | GeometryAnsatz,
        symbols: GeometrySymbols | None = None,
        tensors: Mapping[str, ComponentTensor] | None = None,
    ) -> None:
        self.geometry = (
            geometry if isinstance(geometry, CoordinateGeometry) else CoordinateGeometry.build(geometry)
        )
        self.symbols = symbols or GeometrySymbols()
        self.tensors = dict(tensors or {})
        self.scalar_values: dict[str, sp.Expr] = {
            coordinate.name: symbol
            for coordinate, symbol in zip(
                self.geometry.ansatz.chart.coordinates,
                self.geometry.coordinates,
                strict=True,
            )
        }
        if self.geometry.scalar_field is not None:
            self.scalar_values[self.symbols.scalar] = self.geometry.scalar_field
        self._evaluation_cache: dict[Expr, ComponentTensor] = {}

    def projection_limit_reason(self, expression: Expr) -> str | None:
        """Evita expansiones incontroladas sin alterar la expresión abstracta."""

        ansatz = self.geometry.ansatz
        scalar = ansatz.scalar_field
        if (
            ansatz.scalar_field_mode is not ScalarFieldMode.GENERIC
            or not isinstance(scalar, Function)
            or len(scalar.arguments) <= 1
        ):
            return None
        derivative_nodes = sum(
            isinstance(node, CovariantDerivative) for node in walk(expression)
        )
        if derivative_nodes <= self.generic_scalar_derivative_budget:
            return None
        return (
            "La expresión se conserva simbólica: su proyección con el campo "
            f"genérico {scalar.name}({', '.join(item.name for item in scalar.arguments)}) "
            f"contiene {derivative_nodes} nodos de derivada covariante, por encima "
            f"del límite seguro {self.generic_scalar_derivative_budget} del backend. "
            "Especialice el perfil escalar o eleve el límite del backend de componentes."
        )

    @classmethod
    def from_model(
        cls,
        model: ModelSpec,
        ansatz: GeometryAnsatz,
        tensors: Mapping[str, ComponentTensor] | None = None,
    ) -> "SympyComponentBackend":
        ansatz.validate_for_model(model)
        return cls(ansatz, model.symbols, tensors)

    def _scalar(self, expr: Expr) -> ComponentTensor:
        value = ir_scalar_to_sympy(expr, self.scalar_values)
        return ComponentTensor.from_mapping((), self.geometry.dimension, {(): value})

    def _metric_component(self, indices: tuple[Index, ...], positions: tuple[int, ...]) -> sp.Expr:
        if len(indices) != 2:
            raise TensorAlgebraError("La métrica debe tener exactamente dos índices.")
        first, second = indices
        a, b = positions
        if first.variance is Variance.DOWN and second.variance is Variance.DOWN:
            return self.geometry.metric_covariant[a, b]
        if first.variance is Variance.UP and second.variance is Variance.UP:
            return self.geometry.metric_contravariant[a, b]
        return sp.S.One if a == b else sp.S.Zero

    def _riemann_component(self, indices: tuple[Index, ...], positions: tuple[int, ...]) -> sp.Expr:
        if len(indices) != 4:
            raise TensorAlgebraError("Riemann debe tener exactamente cuatro índices.")
        choices: list[range | tuple[int, ...]] = []
        for index, position in zip(indices, positions, strict=True):
            choices.append(
                range(self.geometry.dimension)
                if index.variance is Variance.UP
                else (position,)
            )
        value = sp.S.Zero
        for lowered in product(*choices):
            coefficient = sp.S.One
            for index, position, lower_position in zip(indices, positions, lowered, strict=True):
                if index.variance is Variance.UP:
                    coefficient *= self.geometry.metric_contravariant[position, lower_position]
            value += coefficient * self.geometry.riemann_down[lowered[0]][lowered[1]][lowered[2]][lowered[3]]
        return sp.simplify(value)

    def _gradient_component(self, indices: tuple[Index, ...], positions: tuple[int, ...]) -> sp.Expr:
        if len(indices) != 1:
            raise TensorAlgebraError("El gradiente escalar debe tener rango uno.")
        gradient = self.geometry.scalar_gradient_covariant()
        position = positions[0]
        if indices[0].variance is Variance.DOWN:
            return gradient[position]
        return sp.simplify(
            sum(
                self.geometry.metric_contravariant[position, other] * gradient[other]
                for other in range(self.geometry.dimension)
            )
        )

    def _tensor(self, expr: Tensor) -> ComponentTensor:
        n = self.geometry.dimension
        if expr.name == "delta":
            if (len(expr.indices) != 2
                    or any(i.space != self.symbols.index_space for i in expr.indices)
                    or expr.indices[0].variance is expr.indices[1].variance):
                raise BackendExecutionError(
                    "El delta requiere dos índices de varianza opuesta en el espacio de la geometría activa."
                )
            values = {(a, b): sp.S.One if a == b else sp.S.Zero
                      for a, b in product(range(n), repeat=2)}
        elif expr.name in self.tensors:
            source = self.tensors[expr.name]
            if source.dimension != n:
                raise TensorAlgebraError(
                    f"El tensor registrado {expr.name!r} tiene dimensión incompatible."
                )
            if tuple(item.variance for item in source.indices) != tuple(
                item.variance for item in expr.indices
            ):
                raise TensorAlgebraError(
                    f"El tensor registrado {expr.name!r} no coincide en varianzas."
                )
            values = {
                position: source.component(*position)
                for position in product(range(n), repeat=len(expr.indices))
            }
        elif expr.name == self.symbols.metric:
            values = {
                position: self._metric_component(expr.indices, position)
                for position in product(range(n), repeat=2)
            }
        elif expr.name == self.symbols.curvature:
            values = {
                position: self._riemann_component(expr.indices, position)
                for position in product(range(n), repeat=4)
            }
        elif expr.name == self.symbols.scalar_gradient:
            values = {
                position: self._gradient_component(expr.indices, position)
                for position in product(range(n), repeat=1)
            }
        else:
            raise BackendExecutionError(
                f"No hay componentes registradas para el tensor {expr.name!r}."
            )
        tensor = ComponentTensor.from_mapping(expr.indices, n, values)
        # Identity elimination can expose an intrinsic trace T^a_a. Use the
        # same summation path as products, rather than leaving duplicate axes.
        free = infer_free_indices(expr)
        return tensor if len(free) == len(expr.indices) else self._contract_components([tensor], free)

    @staticmethod
    def _align(tensor: ComponentTensor, indices: tuple[Index, ...]) -> ComponentTensor:
        if tensor.indices == indices:
            return tensor
        source_keys = [(item.space, item.name, item.variance) for item in tensor.indices]
        target_keys = [(item.space, item.name, item.variance) for item in indices]
        if sorted(source_keys, key=str) != sorted(target_keys, key=str):
            raise TensorAlgebraError("No se pueden alinear tensores con índices libres distintos.")
        order = tuple(source_keys.index(key) for key in target_keys)
        values = {
            tuple(position[source_axis] for source_axis in order): value
            for position, value in tensor.values
        }
        return ComponentTensor.from_mapping(indices, tensor.dimension, values)

    def _add(self, expr: Add) -> ComponentTensor:
        indices = infer_free_indices(expr)
        terms = [self._align(self._evaluate(term), indices) for term in expr.terms]
        values = {
            position: sp.simplify(sum(term.component(*position) for term in terms))
            for position in product(range(self.geometry.dimension), repeat=len(indices))
        }
        return ComponentTensor.from_mapping(indices, self.geometry.dimension, values)

    def _multiply(self, expr: Mul) -> ComponentTensor:
        factors = [self._evaluate(factor) for factor in expr.factors]
        output_indices = infer_free_indices(expr)
        return self._contract_components(factors, output_indices)

    def _contract_components(
        self, factors: list[ComponentTensor], output_indices: tuple[Index, ...],
    ) -> ComponentTensor:
        """Shared Einstein summation for products and intrinsic tensor traces."""
        occurrences: dict[tuple[str, str], list[Index]] = {}
        for factor in factors:
            for index in factor.indices:
                occurrences.setdefault((index.space, index.name), []).append(index)
        dummy_names = tuple(
            key for key, items in occurrences.items() if len(items) == 2
        )
        output_names = tuple((item.space, item.name) for item in output_indices)
        n = self.geometry.dimension
        values: dict[tuple[int, ...], sp.Expr] = {}
        for output_position in product(range(n), repeat=len(output_indices)):
            assignment = dict(zip(output_names, output_position, strict=True))
            total = sp.S.Zero
            for dummy_position in product(range(n), repeat=len(dummy_names)):
                local = assignment | dict(zip(dummy_names, dummy_position, strict=True))
                value = sp.S.One
                for factor in factors:
                    positions = tuple(local[(item.space, item.name)] for item in factor.indices)
                    value *= factor.component(*positions)
                total += value
            values[output_position] = sp.simplify(total)
        return ComponentTensor.from_mapping(output_indices, n, values)

    def _covariant_derivative(self, expr: CovariantDerivative) -> ComponentTensor:
        operand = self._evaluate(expr.operand)
        output_indices = infer_free_indices(expr)
        derivative_key = (expr.index.space, expr.index.name)
        contracted_axis = next(
            (
                position
                for position, index in enumerate(operand.indices)
                if (index.space, index.name) == derivative_key
            ),
            None,
        )
        n = self.geometry.dimension
        values: dict[tuple[int, ...], sp.Expr] = {}
        for output_position in product(range(n), repeat=len(output_indices)):
            output_assignment = {
                (index.space, index.name): value
                for index, value in zip(output_indices, output_position, strict=True)
            }
            derivative_values = range(n) if contracted_axis is not None else (
                output_assignment[derivative_key],
            )
            total = sp.S.Zero
            for derivative in derivative_values:
                operand_position = tuple(
                    derivative
                    if (index.space, index.name) == derivative_key
                    else output_assignment[(index.space, index.name)]
                    for index in operand.indices
                )
                value = sp.diff(operand.component(*operand_position), self.geometry.coordinates[derivative])
                for axis, index in enumerate(operand.indices):
                    slot = operand_position[axis]
                    for replacement in range(n):
                        changed = (
                            *operand_position[:axis],
                            replacement,
                            *operand_position[axis + 1 :],
                        )
                        if index.variance is Variance.UP:
                            value += (
                                self.geometry.christoffel[slot][derivative][replacement]
                                * operand.component(*changed)
                            )
                        else:
                            value -= (
                                self.geometry.christoffel[replacement][derivative][slot]
                                * operand.component(*changed)
                            )
                total += value
            values[output_position] = sp.simplify(total)
        return ComponentTensor.from_mapping(output_indices, n, values)

    def _evaluate_uncached(self, expr: Expr) -> ComponentTensor:
        if isinstance(expr, (Number, Scalar, Function, FunctionDerivative)):
            return self._scalar(expr)
        if isinstance(expr, Tensor):
            return self._tensor(expr)
        if isinstance(expr, Add):
            return self._add(expr)
        if isinstance(expr, Mul):
            return self._multiply(expr)
        if isinstance(expr, Power):
            base = self._evaluate(expr.base)
            exponent = self._evaluate(expr.exponent)
            if not base.is_scalar or not exponent.is_scalar:
                raise TensorAlgebraError("Solo se evalúan potencias escalares.")
            return ComponentTensor.from_mapping(
                (), self.geometry.dimension, {(): sp.simplify(base.scalar ** exponent.scalar)}
            )
        if isinstance(expr, CovariantDerivative):
            return self._covariant_derivative(expr)
        if isinstance(expr, VolumeElement):
            value = sp.sqrt(sp.simplify(-self.geometry.determinant))
            return ComponentTensor.from_mapping((), self.geometry.dimension, {(): value})
        if isinstance(expr, Variation):
            raise BackendExecutionError("Una variación formal no puede evaluarse como componente.")
        raise BackendExecutionError(f"Nodo IR no soportado: {type(expr).__name__}.")

    def _evaluate(self, expr: Expr) -> ComponentTensor:
        cached = self._evaluation_cache.get(expr)
        if cached is not None:
            return cached
        result = self._evaluate_uncached(expr)
        self._evaluation_cache[expr] = result
        return result

    def evaluate_sympy(self, expr: Expr) -> ComponentTensor:
        """Evalúa una expresión y conserva componentes SymPy para trabajo numérico."""

        return self._evaluate(expr)

    def evaluate(self, expr: Expr) -> ComponentEvaluation:
        """Evalúa y devuelve un resultado serializable en la IR común."""

        return self._evaluate(expr).to_ir()


@dataclass(frozen=True, slots=True)
class ComponentFieldEquations:
    """Componentes independientes de E_ab y ecuación escalar."""

    ansatz_name: str
    metric: ComponentEvaluation
    independent_metric: tuple[tuple[tuple[int, int], Expr], ...]
    scalar: ComponentEvaluation

    def to_data(self) -> dict[str, Any]:
        return {
            "ansatz": self.ansatz_name,
            "metric": self.metric.to_data(),
            "independent_metric": [
                {"position": list(position), "expression": expression.to_data()}
                for position, expression in self.independent_metric
            ],
            "scalar": self.scalar.to_data(),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ComponentFieldEquations":
        return cls(
            ansatz_name=str(data["ansatz"]),
            metric=ComponentEvaluation.from_data(data["metric"]),
            independent_metric=tuple(
                (
                    tuple(int(item) for item in component["position"]),
                    expr_from_data(component["expression"]),
                )
                for component in data.get("independent_metric", ())
            ),
            scalar=ComponentEvaluation.from_data(data["scalar"]),
        )


def evaluate_field_equations(
    metric_euler: Expr,
    scalar_euler: Expr,
    backend: SympyComponentBackend,
) -> ComponentFieldEquations:
    """Proyecta E_ab y E_phi y selecciona a<=b sin descartar ceros."""

    metric = backend.evaluate(metric_euler)
    scalar = backend.evaluate(scalar_euler)
    if len(metric.free_indices) != 2 or any(
        index.variance is not Variance.DOWN for index in metric.free_indices
    ):
        raise TensorAlgebraError("La ecuación métrica debe tener dos índices covariantes libres.")
    if scalar.free_indices:
        raise TensorAlgebraError("La ecuación escalar debe ser de rango cero.")
    independent = tuple(
        ((a, b), metric.component(a, b))
        for a in range(metric.dimension)
        for b in range(a, metric.dimension)
    )
    return ComponentFieldEquations(
        backend.geometry.ansatz.name,
        metric,
        independent,
        scalar,
    )
