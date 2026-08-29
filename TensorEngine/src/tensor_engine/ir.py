"""Representación intermedia tensorial independiente del backend.

La IR conserva estructura matemática; no contiene sintaxis de SymPy, xAct ni
LaTeX. Todos sus nodos son inmutables y serializables a datos básicos de Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
import re
from typing import Any, Iterable, Mapping

from .errors import IRValidationError


_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _validate_name(name: str, label: str = "nombre") -> None:
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise IRValidationError(
            f"{label} inválido {name!r}; use letras, números y guion bajo."
        )


class Variance(str, Enum):
    UP = "up"
    DOWN = "down"


@dataclass(frozen=True, slots=True)
class Index:
    name: str
    variance: Variance
    space: str = "M"

    def __post_init__(self) -> None:
        _validate_name(self.name, "índice")
        _validate_name(self.space, "espacio de índices")
        if not isinstance(self.variance, Variance):
            object.__setattr__(self, "variance", Variance(self.variance))

    def flipped(self) -> "Index":
        variance = Variance.DOWN if self.variance is Variance.UP else Variance.UP
        return Index(self.name, variance, self.space)

    def to_data(self) -> dict[str, str]:
        return {"name": self.name, "variance": self.variance.value, "space": self.space}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "Index":
        return cls(str(data["name"]), Variance(data["variance"]), str(data.get("space", "M")))


class Expr:
    """Clase base de una expresión de la IR."""

    def to_data(self) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def free_indices(self) -> tuple[Index, ...]:
        return infer_free_indices(self)

    @property
    def is_scalar(self) -> bool:
        return not self.free_indices

    def __add__(self, other: ExprLike) -> "Expr":
        return add(self, other)

    def __radd__(self, other: ExprLike) -> "Expr":
        return add(other, self)

    def __sub__(self, other: ExprLike) -> "Expr":
        return add(self, mul(-1, other))

    def __rsub__(self, other: ExprLike) -> "Expr":
        return add(other, mul(-1, self))

    def __mul__(self, other: ExprLike) -> "Expr":
        return mul(self, other)

    def __rmul__(self, other: ExprLike) -> "Expr":
        return mul(other, self)

    def __truediv__(self, other: ExprLike) -> "Expr":
        return mul(self, power(other, -1))

    def __neg__(self) -> "Expr":
        return mul(-1, self)

    def __pow__(self, exponent: ExprLike) -> "Expr":
        return power(self, exponent)


@dataclass(frozen=True, slots=True)
class Number(Expr):
    numerator: int
    denominator: int = 1

    def __post_init__(self) -> None:
        value = Fraction(self.numerator, self.denominator)
        object.__setattr__(self, "numerator", value.numerator)
        object.__setattr__(self, "denominator", value.denominator)

    @property
    def value(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def to_data(self) -> dict[str, Any]:
        return {"type": "number", "numerator": self.numerator, "denominator": self.denominator}


@dataclass(frozen=True, slots=True)
class Scalar(Expr):
    name: str

    def __post_init__(self) -> None:
        _validate_name(self.name, "escalar")

    def to_data(self) -> dict[str, Any]:
        return {"type": "scalar", "name": self.name}


@dataclass(frozen=True, slots=True)
class Tensor(Expr):
    name: str
    indices: tuple[Index, ...]

    def __post_init__(self) -> None:
        _validate_name(self.name, "tensor")
        object.__setattr__(self, "indices", tuple(self.indices))
        if not self.indices:
            raise IRValidationError("Un Tensor debe tener al menos un índice; use Scalar para rango cero.")

    def to_data(self) -> dict[str, Any]:
        return {
            "type": "tensor",
            "name": self.name,
            "indices": [index.to_data() for index in self.indices],
        }


@dataclass(frozen=True, slots=True)
class Add(Expr):
    terms: tuple[Expr, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        if len(self.terms) < 2:
            raise IRValidationError("Add requiere al menos dos términos.")

    def to_data(self) -> dict[str, Any]:
        return {"type": "add", "terms": [term.to_data() for term in self.terms]}


@dataclass(frozen=True, slots=True)
class Mul(Expr):
    factors: tuple[Expr, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "factors", tuple(self.factors))
        if len(self.factors) < 2:
            raise IRValidationError("Mul requiere al menos dos factores.")

    def to_data(self) -> dict[str, Any]:
        return {"type": "mul", "factors": [factor.to_data() for factor in self.factors]}


@dataclass(frozen=True, slots=True)
class Power(Expr):
    base: Expr
    exponent: Expr

    def to_data(self) -> dict[str, Any]:
        return {"type": "power", "base": self.base.to_data(), "exponent": self.exponent.to_data()}


@dataclass(frozen=True, slots=True)
class Function(Expr):
    name: str
    arguments: tuple[Expr, ...]

    def __post_init__(self) -> None:
        _validate_name(self.name, "función")
        object.__setattr__(self, "arguments", tuple(self.arguments))
        if not self.arguments:
            raise IRValidationError("Una función simbólica debe recibir al menos un argumento.")

    def to_data(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "arguments": [argument.to_data() for argument in self.arguments],
        }


@dataclass(frozen=True, slots=True)
class FunctionDerivative(Expr):
    """Derivada parcial formal de una función respecto a sus argumentos."""

    name: str
    derivative_orders: tuple[int, ...]
    arguments: tuple[Expr, ...]

    def __post_init__(self) -> None:
        _validate_name(self.name, "función")
        object.__setattr__(self, "derivative_orders", tuple(self.derivative_orders))
        object.__setattr__(self, "arguments", tuple(self.arguments))
        if not self.arguments or len(self.derivative_orders) != len(self.arguments):
            raise IRValidationError(
                "FunctionDerivative requiere una orden no negativa por argumento."
            )
        if any(isinstance(order, bool) or not isinstance(order, int) or order < 0 for order in self.derivative_orders):
            raise IRValidationError("Los órdenes de derivación deben ser enteros no negativos.")
        if not any(self.derivative_orders):
            raise IRValidationError("FunctionDerivative debe contener al menos una derivada.")

    def to_data(self) -> dict[str, Any]:
        return {
            "type": "function_derivative",
            "name": self.name,
            "derivative_orders": list(self.derivative_orders),
            "arguments": [argument.to_data() for argument in self.arguments],
        }


@dataclass(frozen=True, slots=True)
class CovariantDerivative(Expr):
    index: Index
    operand: Expr

    def __post_init__(self) -> None:
        if self.index.variance is not Variance.DOWN:
            raise IRValidationError("La derivada covariante elemental debe portar un índice inferior.")

    def to_data(self) -> dict[str, Any]:
        return {
            "type": "covariant_derivative",
            "index": self.index.to_data(),
            "operand": self.operand.to_data(),
        }


@dataclass(frozen=True, slots=True)
class Variation(Expr):
    """Variación formal de una expresión, sin imponer aún su origen geométrico."""

    operand: Expr

    def __post_init__(self) -> None:
        if isinstance(self.operand, Number):
            raise IRValidationError("La variación formal de un número debe simplificarse a cero.")

    def to_data(self) -> dict[str, Any]:
        return {"type": "variation", "operand": self.operand.to_data()}


@dataclass(frozen=True, slots=True)
class VolumeElement(Expr):
    """Densidad de volumen sqrt(-g), distinguida de un escalar ordinario."""

    metric_name: str = "g"

    def __post_init__(self) -> None:
        _validate_name(self.metric_name, "métrica de la densidad de volumen")

    def to_data(self) -> dict[str, Any]:
        return {"type": "volume_element", "metric_name": self.metric_name}


ExprLike = Expr | int | Fraction


def as_expr(value: ExprLike) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, Fraction):
        return Number(value.numerator, value.denominator)
    if isinstance(value, int):
        return Number(value)
    raise TypeError(f"No se puede convertir {type(value).__name__} a una expresión IR.")


def add(*terms: ExprLike) -> Expr:
    flat: list[Expr] = []
    constant = Fraction(0)
    for item in terms:
        expr = as_expr(item)
        children = expr.terms if isinstance(expr, Add) else (expr,)
        for child in children:
            if isinstance(child, Number):
                constant += child.value
            else:
                flat.append(child)
    if constant:
        flat.append(Number(constant.numerator, constant.denominator))
    if not flat:
        return Number(0)
    if len(flat) == 1:
        return flat[0]
    result = Add(tuple(flat))
    infer_free_indices(result)
    return result


def mul(*factors: ExprLike) -> Expr:
    flat: list[Expr] = []
    constant = Fraction(1)
    for item in factors:
        expr = as_expr(item)
        # No se aplanan productos anidados: cada subexpresión escalar delimita
        # el alcance de sus índices mudos. La canonización futura podrá
        # aplanarlos después de renombrarlos de forma segura.
        if isinstance(expr, Number):
            constant *= expr.value
        else:
            flat.append(expr)
    if constant == 0:
        return Number(0)
    if constant != 1:
        flat.insert(0, Number(constant.numerator, constant.denominator))
    if not flat:
        return Number(1)
    if len(flat) == 1:
        return flat[0]
    result = Mul(tuple(flat))
    infer_free_indices(result)
    return result


def power(base: ExprLike, exponent: ExprLike) -> Expr:
    base_expr = as_expr(base)
    exponent_expr = as_expr(exponent)
    if not base_expr.is_scalar or not exponent_expr.is_scalar:
        raise IRValidationError("Las potencias de la IR inicial deben tener base y exponente escalares.")
    if isinstance(exponent_expr, Number):
        if exponent_expr.value == 0:
            return Number(1)
        if exponent_expr.value == 1:
            return base_expr
    return Power(base_expr, exponent_expr)


def function(name: str, *arguments: ExprLike) -> Function:
    result = Function(name, tuple(as_expr(argument) for argument in arguments))
    infer_free_indices(result)
    return result


def _contract_occurrences(indices: Iterable[Index], context: str) -> tuple[Index, ...]:
    grouped: dict[tuple[str, str], list[Index]] = {}
    order: list[tuple[str, str]] = []
    for index in indices:
        key = (index.space, index.name)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(index)

    free: list[Index] = []
    for key in order:
        occurrences = grouped[key]
        if len(occurrences) == 1:
            free.append(occurrences[0])
            continue
        if len(occurrences) == 2 and {item.variance for item in occurrences} == {
            Variance.UP,
            Variance.DOWN,
        }:
            continue
        label = f"{key[0]}:{key[1]}"
        raise IRValidationError(
            f"Contracción inválida del índice {label} en {context}: "
            "debe aparecer una vez libre o exactamente dos veces con varianza opuesta."
        )
    return tuple(free)


def _signature(indices: Iterable[Index]) -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted((item.space, item.name, item.variance.value) for item in indices))


def infer_free_indices(expr: Expr) -> tuple[Index, ...]:
    """Valida recursivamente una expresión y devuelve sus índices libres."""

    if isinstance(expr, (Number, Scalar, VolumeElement)):
        return ()
    if isinstance(expr, Tensor):
        return _contract_occurrences(expr.indices, f"tensor {expr.name}")
    if isinstance(expr, Add):
        signatures: list[tuple[tuple[str, str, str], ...]] = []
        free_terms: list[tuple[Index, ...]] = []
        for term in expr.terms:
            free = infer_free_indices(term)
            free_terms.append(free)
            signatures.append(_signature(free))
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise IRValidationError("Todos los términos de una suma deben tener los mismos índices libres.")
        return free_terms[0]
    if isinstance(expr, Mul):
        indices: list[Index] = []
        for factor in expr.factors:
            indices.extend(infer_free_indices(factor))
        return _contract_occurrences(indices, "producto")
    if isinstance(expr, Power):
        if infer_free_indices(expr.base) or infer_free_indices(expr.exponent):
            raise IRValidationError("Las potencias solo se admiten para expresiones escalares.")
        return ()
    if isinstance(expr, (Function, FunctionDerivative)):
        for argument in expr.arguments:
            if infer_free_indices(argument):
                raise IRValidationError(f"El argumento de {expr.name} debe ser escalar.")
        return ()
    if isinstance(expr, CovariantDerivative):
        indices = [*infer_free_indices(expr.operand), expr.index]
        return _contract_occurrences(indices, "derivada covariante")
    if isinstance(expr, Variation):
        return infer_free_indices(expr.operand)
    raise TypeError(f"Nodo IR no reconocido: {type(expr).__name__}")


def walk(expr: Expr) -> Iterable[Expr]:
    """Recorre un árbol IR en preorden."""

    yield expr
    if isinstance(expr, Add):
        for term in expr.terms:
            yield from walk(term)
    elif isinstance(expr, Mul):
        for factor in expr.factors:
            yield from walk(factor)
    elif isinstance(expr, Power):
        yield from walk(expr.base)
        yield from walk(expr.exponent)
    elif isinstance(expr, (Function, FunctionDerivative)):
        for argument in expr.arguments:
            yield from walk(argument)
    elif isinstance(expr, CovariantDerivative):
        yield from walk(expr.operand)
    elif isinstance(expr, Variation):
        yield from walk(expr.operand)


def expr_from_data(data: Mapping[str, Any]) -> Expr:
    """Reconstruye una expresión IR desde datos JSON-compatibles."""

    node_type = data.get("type")
    if node_type == "number":
        return Number(int(data["numerator"]), int(data.get("denominator", 1)))
    if node_type == "scalar":
        return Scalar(str(data["name"]))
    if node_type == "tensor":
        return Tensor(
            str(data["name"]),
            tuple(Index.from_data(item) for item in data["indices"]),
        )
    if node_type == "add":
        result = Add(tuple(expr_from_data(item) for item in data["terms"]))
    elif node_type == "mul":
        result = Mul(tuple(expr_from_data(item) for item in data["factors"]))
    elif node_type == "power":
        result = Power(expr_from_data(data["base"]), expr_from_data(data["exponent"]))
    elif node_type == "function":
        result = Function(
            str(data["name"]),
            tuple(expr_from_data(item) for item in data["arguments"]),
        )
    elif node_type == "function_derivative":
        result = FunctionDerivative(
            str(data["name"]),
            tuple(int(item) for item in data["derivative_orders"]),
            tuple(expr_from_data(item) for item in data["arguments"]),
        )
    elif node_type == "covariant_derivative":
        result = CovariantDerivative(Index.from_data(data["index"]), expr_from_data(data["operand"]))
    elif node_type == "variation":
        result = Variation(expr_from_data(data["operand"]))
    elif node_type == "volume_element":
        result = VolumeElement(str(data.get("metric_name", "g")))
    else:
        raise IRValidationError(f"Tipo de nodo IR desconocido: {node_type!r}")
    infer_free_indices(result)
    return result
