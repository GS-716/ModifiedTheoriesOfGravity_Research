"""Cálculo diferencial covariante formal sobre la IR tensorial."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .errors import TensorAlgebraError
from .indices import all_indices, canonicalize_dummy_indices, index_key, rename_free_indices
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
    infer_free_indices,
    mul,
    power,
)
from .model import ModelSpec


@dataclass(frozen=True, slots=True)
class DifferentialContext:
    metric_name: str = "g"
    delta_name: str = "delta"
    curvature_name: str = "Riemann"
    scalar_name: str = "phi"
    scalar_gradient_name: str = "u"
    lie_vector_name: str = "xi"
    constant_scalars: frozenset[str] = frozenset({"D"})
    dimension: int | str | Expr = "D"

    @classmethod
    def from_model(cls, model: ModelSpec) -> "DifferentialContext":
        constants = {item.name for item in model.parameters}
        if model.dimension.is_symbolic:
            constants.add(str(model.dimension.value))
        return cls(
            metric_name=model.symbols.metric,
            curvature_name=model.symbols.curvature,
            scalar_name=model.symbols.scalar,
            scalar_gradient_name=model.symbols.scalar_gradient,
            constant_scalars=frozenset(constants),
            dimension=model.dimension.value,
        )


def _index_order(index: Index) -> tuple[str, str, str]:
    return index.space, index.name, index.variance.value


def _fresh_names(
    expr: Expr,
    count: int,
    prefix: str,
    extra: tuple[Index, ...] = (),
) -> tuple[str, ...]:
    occupied = {(item.space, item.name) for item in all_indices(expr)}
    occupied.update((item.space, item.name) for item in extra)
    space = extra[0].space if extra else "M"
    names: list[str] = []
    counter = 0
    while len(names) < count:
        candidate = f"{prefix}{counter}"
        counter += 1
        if (space, candidate) not in occupied:
            names.append(candidate)
            occupied.add((space, candidate))
    return tuple(names)


def _ordered_scalar_hessian(outer: Index, inner: Index, scalar: Expr) -> Expr:
    first, second = sorted((outer, inner), key=_index_order)
    return CovariantDerivative(first, CovariantDerivative(second, scalar))


def _ordered_gradient_hessian(
    derivative_index: Index,
    gradient: Tensor,
    context: DifferentialContext,
) -> Expr:
    gradient_index = gradient.indices[0]
    if gradient_index.variance is Variance.UP:
        return CovariantDerivative(derivative_index, gradient)
    first, second = sorted((derivative_index, gradient_index), key=_index_order)
    return CovariantDerivative(first, Tensor(context.scalar_gradient_name, (second,)))


def _function_chain_rule(
    name: str,
    orders: tuple[int, ...],
    arguments: tuple[Expr, ...],
    index: Index,
    context: DifferentialContext,
) -> Expr:
    terms: list[Expr] = []
    for position, argument in enumerate(arguments):
        derivative = _covariant_derivative(argument, index, context)
        if derivative == Number(0):
            continue
        next_orders = list(orders)
        next_orders[position] += 1
        coefficient = FunctionDerivative(name, tuple(next_orders), arguments)
        terms.append(mul(coefficient, derivative))
    return add(*terms)


def _covariant_derivative(
    expr: Expr,
    index: Index,
    context: DifferentialContext,
) -> Expr:
    if isinstance(expr, Number):
        return Number(0)
    if isinstance(expr, Scalar):
        if expr.name in context.constant_scalars:
            return Number(0)
        if expr.name == context.scalar_name:
            return Tensor(context.scalar_gradient_name, (index,))
        return CovariantDerivative(index, expr)
    if isinstance(expr, Tensor):
        if expr.name in {context.metric_name, context.delta_name}:
            return Number(0)
        if expr.name == context.scalar_gradient_name and len(expr.indices) == 1:
            return _ordered_gradient_hessian(index, expr, context)
        return CovariantDerivative(index, expr)
    if isinstance(expr, Add):
        return add(*(_covariant_derivative(term, index, context) for term in expr.terms))
    if isinstance(expr, Mul):
        terms: list[Expr] = []
        for position, factor in enumerate(expr.factors):
            derivative = _covariant_derivative(factor, index, context)
            if derivative == Number(0):
                continue
            factors = list(expr.factors)
            factors[position] = derivative
            terms.append(mul(*factors))
        return add(*terms)
    if isinstance(expr, Power):
        if isinstance(expr.exponent, Number):
            exponent = expr.exponent.value
            derivative = _covariant_derivative(expr.base, index, context)
            if derivative == Number(0):
                return Number(0)
            reduced = Fraction(exponent.numerator - exponent.denominator, exponent.denominator)
            return mul(
                Number(exponent.numerator, exponent.denominator),
                power(expr.base, Number(reduced.numerator, reduced.denominator)),
                derivative,
            )
        return CovariantDerivative(index, expr)
    if isinstance(expr, Function):
        return _function_chain_rule(
            expr.name,
            (0,) * len(expr.arguments),
            expr.arguments,
            index,
            context,
        )
    if isinstance(expr, FunctionDerivative):
        return _function_chain_rule(
            expr.name,
            expr.derivative_orders,
            expr.arguments,
            index,
            context,
        )
    if isinstance(expr, CovariantDerivative):
        if expr.operand.is_scalar:
            return _ordered_scalar_hessian(index, expr.index, expr.operand)
        return CovariantDerivative(index, expr)
    if isinstance(expr, (Variation, VolumeElement)):
        return CovariantDerivative(index, expr)
    raise TypeError(f"Nodo IR no reconocido: {type(expr).__name__}")


def covariant_derivative(
    expr: Expr,
    index: Index,
    context: DifferentialContext | None = None,
) -> Expr:
    """Aplica linealidad, Leibniz, cadena y compatibilidad métrica."""

    if index.variance is not Variance.DOWN:
        raise TensorAlgebraError("La derivada covariante elemental debe tener índice inferior.")
    context = context or DifferentialContext()
    hygienic = canonicalize_dummy_indices(expr)
    result = _covariant_derivative(hygienic, index, context)
    return canonicalize_dummy_indices(result)


def gradient(
    scalar: Expr,
    index: Index,
    context: DifferentialContext | None = None,
) -> Expr:
    if not scalar.is_scalar:
        raise TensorAlgebraError("gradient requiere una expresión escalar.")
    return covariant_derivative(scalar, index, context)


def hessian(
    scalar: Expr,
    first: Index,
    second: Index,
    context: DifferentialContext | None = None,
) -> Expr:
    if not scalar.is_scalar:
        raise TensorAlgebraError("hessian requiere una expresión escalar.")
    context = context or DifferentialContext()
    inner = covariant_derivative(scalar, second, context)
    return covariant_derivative(inner, first, context)


def divergence(
    expr: Expr,
    index: Index,
    context: DifferentialContext | None = None,
) -> Expr:
    """Contrae una derivada covariante con el índice libre indicado."""

    context = context or DifferentialContext()
    free = {index_key(item): item for item in infer_free_indices(expr)}
    key = index_key(index)
    if key not in free or free[key].variance is not index.variance:
        raise TensorAlgebraError("El índice de divergencia no es libre en la expresión.")
    hygienic = canonicalize_dummy_indices(expr)
    if index.variance is Variance.UP:
        derivative_index = Index(index.name, Variance.DOWN, index.space)
        return covariant_derivative(hygienic, derivative_index, context)

    derivative_name, tensor_name = _fresh_names(hygienic, 2, "v", (index,))
    derivative_index = Index(derivative_name, Variance.DOWN, index.space)
    metric_tensor_index = Index(tensor_name, Variance.UP, index.space)
    renamed = rename_free_indices(hygienic, {key: tensor_name})
    metric = Tensor(
        context.metric_name,
        (Index(derivative_name, Variance.UP, index.space), metric_tensor_index),
    )
    return mul(metric, covariant_derivative(renamed, derivative_index, context))


def laplacian(
    scalar: Expr,
    context: DifferentialContext | None = None,
    index_space: str = "M",
) -> Expr:
    if not scalar.is_scalar:
        raise TensorAlgebraError("laplacian requiere una expresión escalar.")
    context = context or DifferentialContext()
    first_name, second_name = _fresh_names(scalar, 2, "l")
    first = Index(first_name, Variance.DOWN, index_space)
    second = Index(second_name, Variance.DOWN, index_space)
    metric = Tensor(
        context.metric_name,
        (
            Index(first_name, Variance.UP, index_space),
            Index(second_name, Variance.UP, index_space),
        ),
    )
    return mul(metric, hessian(scalar, first, second, context))


def derivative_commutator(
    expr: Expr,
    first: Index,
    second: Index,
    context: DifferentialContext | None = None,
) -> Expr:
    """Construye [nabla_first,nabla_second] aplicado a una expresión."""

    context = context or DifferentialContext()
    forward = covariant_derivative(covariant_derivative(expr, second, context), first, context)
    backward = covariant_derivative(covariant_derivative(expr, first, context), second, context)
    return add(forward, mul(-1, backward))


def curvature_commutator_action(
    expr: Expr,
    first: Index,
    second: Index,
    context: DifferentialContext | None = None,
) -> Expr:
    """Construye la acción algebraica de Riemann sobre cada índice libre."""

    context = context or DifferentialContext()
    if first.variance is not Variance.DOWN or second.variance is not Variance.DOWN:
        raise TensorAlgebraError("Los índices del conmutador deben ser inferiores.")
    if index_key(first) == index_key(second):
        raise TensorAlgebraError("Los índices del conmutador deben ser distintos.")
    free_indices = infer_free_indices(expr)
    occupied_free = {index_key(item) for item in free_indices}
    if index_key(first) in occupied_free or index_key(second) in occupied_free:
        raise TensorAlgebraError(
            "Los índices del conmutador deben ser nuevos respecto a los índices libres del operando."
        )
    if not free_indices:
        return Number(0)

    terms: list[Expr] = []
    for position, free_index in enumerate(free_indices):
        replacement_name, metric_dummy_name = _fresh_names(
            expr,
            2,
            f"c{position}",
            (first, second, free_index),
        )
        replacement = rename_free_indices(
            expr,
            {index_key(free_index): replacement_name},
        )
        if free_index.variance is Variance.UP:
            metric = Tensor(
                context.metric_name,
                (
                    Index(free_index.name, Variance.UP, free_index.space),
                    Index(metric_dummy_name, Variance.UP, free_index.space),
                ),
            )
            curvature = Tensor(
                context.curvature_name,
                (
                    Index(metric_dummy_name, Variance.DOWN, free_index.space),
                    Index(replacement_name, Variance.DOWN, free_index.space),
                    first,
                    second,
                ),
            )
            terms.append(mul(metric, curvature, replacement))
        else:
            metric = Tensor(
                context.metric_name,
                (
                    Index(replacement_name, Variance.UP, free_index.space),
                    Index(metric_dummy_name, Variance.UP, free_index.space),
                ),
            )
            curvature = Tensor(
                context.curvature_name,
                (
                    Index(metric_dummy_name, Variance.DOWN, free_index.space),
                    Index(free_index.name, Variance.DOWN, free_index.space),
                    first,
                    second,
                ),
            )
            terms.append(mul(-1, metric, curvature, replacement))
    return add(*terms)


def commutator_residual(
    expr: Expr,
    first: Index,
    second: Index,
    context: DifferentialContext | None = None,
) -> Expr:
    context = context or DifferentialContext()
    return add(
        derivative_commutator(expr, first, second, context),
        mul(-1, curvature_commutator_action(expr, first, second, context)),
    )


def lie_derivative(
    expr: Expr,
    context: DifferentialContext | None = None,
) -> Expr:
    """Derivada de Lie respecto a xi^a para un tensor de rango arbitrario."""

    context = context or DifferentialContext()
    free_indices = infer_free_indices(expr)
    space = free_indices[0].space if free_indices else "M"
    space_hint = (Index("space_hint", Variance.DOWN, space),)
    transport_name = _fresh_names(expr, 1, "x", space_hint)[0]
    transport_up = Index(transport_name, Variance.UP, space)
    transport_down = Index(transport_name, Variance.DOWN, space)
    vector = Tensor(context.lie_vector_name, (transport_up,))
    terms: list[Expr] = [mul(vector, covariant_derivative(expr, transport_down, context))]

    for position, free_index in enumerate(free_indices):
        contracted_name = _fresh_names(expr, 1, f"x{position}", (free_index,))[0]
        replacement = rename_free_indices(
            expr,
            {index_key(free_index): contracted_name},
        )
        if free_index.variance is Variance.UP:
            xi = Tensor(context.lie_vector_name, (free_index,))
            dxi = covariant_derivative(
                xi,
                Index(contracted_name, Variance.DOWN, free_index.space),
                context,
            )
            terms.append(mul(-1, replacement, dxi))
        else:
            xi = Tensor(
                context.lie_vector_name,
                (Index(contracted_name, Variance.UP, free_index.space),),
            )
            dxi = covariant_derivative(
                xi,
                Index(free_index.name, Variance.DOWN, free_index.space),
                context,
            )
            terms.append(mul(replacement, dxi))
    return add(*terms)


def differential_bianchi_residual(
    tensor_name: str,
    e: Index,
    a: Index,
    b: Index,
    c: Index,
    d: Index,
    context: DifferentialContext | None = None,
) -> Expr:
    """Construye nabla_e R_abcd + nabla_a R_becd + nabla_b R_eacd."""

    context = context or DifferentialContext(curvature_name=tensor_name)
    indices = (e, a, b, c, d)
    if any(item.variance is not Variance.DOWN for item in indices):
        raise TensorAlgebraError("Bianchi diferencial requiere índices inferiores.")
    if len({item.space for item in indices}) != 1:
        raise TensorAlgebraError("Los índices deben pertenecer al mismo espacio.")
    return add(
        CovariantDerivative(e, Tensor(tensor_name, (a, b, c, d))),
        CovariantDerivative(a, Tensor(tensor_name, (b, e, c, d))),
        CovariantDerivative(b, Tensor(tensor_name, (e, a, c, d))),
    )
