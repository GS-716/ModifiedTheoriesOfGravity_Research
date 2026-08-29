"""Cálculo variacional elemental para L(g, R, phi, u)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .errors import TensorAlgebraError
from .indices import canonicalize_dummy_indices, index_key, rename_free_indices, used_index_names
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
    expr_from_data,
    function,
    infer_free_indices,
    mul,
    power,
)
from .model import ModelSpec, TensorSymmetry
from .transform import antisymmetrize, symmetrize


@dataclass(frozen=True, slots=True)
class VariationalContext:
    metric_name: str = "g"
    curvature_name: str = "Riemann"
    scalar_name: str = "phi"
    scalar_gradient_name: str = "u"
    delta_name: str = "delta"
    connection_variation_name: str = "delta_Gamma"
    index_space: str = "M"
    constant_scalars: frozenset[str] = frozenset({"D"})

    @classmethod
    def from_model(cls, model: ModelSpec) -> "VariationalContext":
        constants = {item.name for item in model.parameters}
        if model.dimension.is_symbolic:
            constants.add(str(model.dimension.value))
        return cls(
            metric_name=model.symbols.metric,
            curvature_name=model.symbols.curvature,
            scalar_name=model.symbols.scalar,
            scalar_gradient_name=model.symbols.scalar_gradient,
            index_space=model.symbols.index_space,
            constant_scalars=frozenset(constants),
        )


@dataclass(frozen=True, slots=True)
class LagrangianMomenta:
    """Los cuatro coeficientes de la regla de cadena variacional."""

    metric: Expr
    curvature: Expr
    scalar_gradient: Expr
    scalar: Expr

    def to_data(self) -> dict[str, Any]:
        return {
            "metric_momentum": self.metric.to_data(),
            "curvature_momentum": self.curvature.to_data(),
            "scalar_gradient_momentum": self.scalar_gradient.to_data(),
            "scalar_derivative": self.scalar.to_data(),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "LagrangianMomenta":
        return cls(
            metric=expr_from_data(data["metric_momentum"]),
            curvature=expr_from_data(data["curvature_momentum"]),
            scalar_gradient=expr_from_data(data["scalar_gradient_momentum"]),
            scalar=expr_from_data(data["scalar_derivative"]),
        )


def _is_zero(expr: Expr) -> bool:
    return isinstance(expr, Number) and expr.value == 0


def _dual(variance: Variance) -> Variance:
    return Variance.DOWN if variance is Variance.UP else Variance.UP


def _delta_for_slot(actual: Index, output: Index, delta_name: str) -> Tensor:
    if actual.variance is Variance.UP and output.variance is Variance.DOWN:
        return Tensor(delta_name, (actual, output))
    if actual.variance is Variance.DOWN and output.variance is Variance.UP:
        return Tensor(delta_name, (output, actual))
    raise TensorAlgebraError("El índice del momento debe tener varianza dual a la variable.")


def _function_partial(
    name: str,
    orders: tuple[int, ...],
    arguments: tuple[Expr, ...],
    derivative,
) -> Expr:
    terms: list[Expr] = []
    for position, argument in enumerate(arguments):
        argument_derivative = derivative(argument)
        if _is_zero(argument_derivative):
            continue
        next_orders = list(orders)
        next_orders[position] += 1
        terms.append(
            mul(
                FunctionDerivative(name, tuple(next_orders), arguments),
                argument_derivative,
            )
        )
    return add(*terms)


def _power_partial(base: Expr, exponent: Expr, derivative) -> Expr:
    base_derivative = derivative(base)
    exponent_derivative = derivative(exponent)
    terms: list[Expr] = []
    if not _is_zero(base_derivative):
        terms.append(mul(exponent, power(base, add(exponent, -1)), base_derivative))
    if not _is_zero(exponent_derivative):
        terms.append(mul(power(base, exponent), function("Log", base), exponent_derivative))
    return add(*terms)


def scalar_partial_derivative(expr: Expr, scalar_name: str) -> Expr:
    """Derivada parcial algebraica respecto a un escalar independiente."""

    if isinstance(expr, Number):
        return Number(0)
    if isinstance(expr, Scalar):
        return Number(1) if expr.name == scalar_name else Number(0)
    if isinstance(expr, (Tensor, VolumeElement)):
        return Number(0)
    if isinstance(expr, Add):
        return add(*(scalar_partial_derivative(term, scalar_name) for term in expr.terms))
    if isinstance(expr, Mul):
        terms: list[Expr] = []
        for position, factor in enumerate(expr.factors):
            derivative = scalar_partial_derivative(factor, scalar_name)
            if _is_zero(derivative):
                continue
            factors = list(expr.factors)
            factors[position] = derivative
            terms.append(mul(*factors))
        return add(*terms)
    if isinstance(expr, Power):
        return _power_partial(
            expr.base,
            expr.exponent,
            lambda item: scalar_partial_derivative(item, scalar_name),
        )
    if isinstance(expr, Function):
        return _function_partial(
            expr.name,
            (0,) * len(expr.arguments),
            expr.arguments,
            lambda item: scalar_partial_derivative(item, scalar_name),
        )
    if isinstance(expr, FunctionDerivative):
        return _function_partial(
            expr.name,
            expr.derivative_orders,
            expr.arguments,
            lambda item: scalar_partial_derivative(item, scalar_name),
        )
    if isinstance(expr, (CovariantDerivative, Variation)):
        raise TensorAlgebraError("La derivada parcial variacional solo admite expresiones algebraicas.")
    raise TypeError(f"Nodo IR no reconocido: {type(expr).__name__}")


def _tensor_partial_derivative(
    expr: Expr,
    target_name: str,
    target_slots: tuple[Variance, ...],
    output_indices: tuple[Index, ...],
    delta_name: str,
) -> Expr:
    if isinstance(expr, (Number, Scalar)):
        return Number(0)
    if isinstance(expr, VolumeElement):
        if target_name == expr.metric_name and target_slots == (Variance.UP, Variance.UP):
            return mul(-1, Number(1, 2), expr, Tensor(expr.metric_name, output_indices))
        return Number(0)
    if isinstance(expr, Tensor):
        if expr.name != target_name:
            return Number(0)
        actual_slots = tuple(index.variance for index in expr.indices)
        if actual_slots != target_slots:
            raise TensorAlgebraError(
                f"{target_name} apareció con varianzas incompatibles con su variable independiente."
            )
        return mul(
            *(
                _delta_for_slot(actual, output, delta_name)
                for actual, output in zip(expr.indices, output_indices, strict=True)
            )
        )
    if isinstance(expr, Add):
        return add(
            *(
                _tensor_partial_derivative(
                    term, target_name, target_slots, output_indices, delta_name
                )
                for term in expr.terms
            )
        )
    if isinstance(expr, Mul):
        terms: list[Expr] = []
        for position, factor in enumerate(expr.factors):
            derivative = _tensor_partial_derivative(
                factor, target_name, target_slots, output_indices, delta_name
            )
            if _is_zero(derivative):
                continue
            factors = list(expr.factors)
            factors[position] = derivative
            terms.append(mul(*factors))
        return add(*terms)
    if isinstance(expr, Power):
        return _power_partial(
            expr.base,
            expr.exponent,
            lambda item: _tensor_partial_derivative(
                item, target_name, target_slots, output_indices, delta_name
            ),
        )
    if isinstance(expr, Function):
        return _function_partial(
            expr.name,
            (0,) * len(expr.arguments),
            expr.arguments,
            lambda item: _tensor_partial_derivative(
                item, target_name, target_slots, output_indices, delta_name
            ),
        )
    if isinstance(expr, FunctionDerivative):
        return _function_partial(
            expr.name,
            expr.derivative_orders,
            expr.arguments,
            lambda item: _tensor_partial_derivative(
                item, target_name, target_slots, output_indices, delta_name
            ),
        )
    if isinstance(expr, (CovariantDerivative, Variation)):
        raise TensorAlgebraError("La derivada tensorial parcial solo admite expresiones algebraicas.")
    raise TypeError(f"Nodo IR no reconocido: {type(expr).__name__}")


def project_riemann_symmetry(expr: Expr, indices: tuple[Index, Index, Index, Index]) -> Expr:
    """Proyecta sobre simetrías de Riemann, incluida la identidad cíclica."""

    a, b, c, d = indices
    free = {index_key(item): item for item in infer_free_indices(expr)}
    if any(index_key(item) not in free for item in indices):
        raise TensorAlgebraError("La proyección de Riemann requiere cuatro índices libres.")
    if len({item.space for item in indices}) != 1 or len({item.variance for item in indices}) != 1:
        raise TensorAlgebraError("Los índices proyectados deben compartir espacio y varianza.")

    pair_antisymmetric = antisymmetrize(antisymmetrize(expr, (a, b)), (c, d))
    pair_swapped = rename_free_indices(
        pair_antisymmetric,
        {
            index_key(a): c.name,
            index_key(b): d.name,
            index_key(c): a.name,
            index_key(d): b.name,
        },
    )
    pair_symmetric = mul(Number(1, 2), add(pair_antisymmetric, pair_swapped))
    four_form = antisymmetrize(pair_symmetric, indices)
    return add(pair_symmetric, mul(-1, four_form))


def tensor_partial_derivative(
    expr: Expr,
    target_name: str,
    target_slots: tuple[Variance, ...],
    output_indices: tuple[Index, ...],
    symmetry: TensorSymmetry = TensorSymmetry.NONE,
    delta_name: str = "delta",
) -> Expr:
    """Derivada parcial tensorial con proyección sobre la simetría declarada."""

    target_slots = tuple(target_slots)
    output_indices = tuple(output_indices)
    if len(target_slots) != len(output_indices):
        raise TensorAlgebraError("La variable y su momento deben tener el mismo rango.")
    if len({index_key(item) for item in output_indices}) != len(output_indices):
        raise TensorAlgebraError("Los índices libres del momento deben ser distintos.")
    if any(item.variance is not _dual(slot) for item, slot in zip(output_indices, target_slots)):
        raise TensorAlgebraError("La varianza de cada índice del momento debe ser dual al argumento.")

    hygienic = canonicalize_dummy_indices(expr, prefix="q")
    occupied = used_index_names(hygienic, output_indices[0].space if output_indices else None)
    collisions = occupied.intersection(item.name for item in output_indices)
    if collisions:
        raise TensorAlgebraError(
            "Los índices de salida deben ser nuevos respecto a la expresión: "
            + ", ".join(sorted(collisions))
        )
    result = _tensor_partial_derivative(
        hygienic, target_name, target_slots, output_indices, delta_name
    )
    if _is_zero(result):
        return result
    if symmetry is TensorSymmetry.SYMMETRIC:
        if len(output_indices) != 2:
            raise TensorAlgebraError("La proyección simétrica requiere rango dos.")
        result = symmetrize(result, output_indices)
    elif symmetry is TensorSymmetry.RIEMANN:
        if len(output_indices) != 4:
            raise TensorAlgebraError("La proyección de Riemann requiere rango cuatro.")
        result = project_riemann_symmetry(result, output_indices)  # type: ignore[arg-type]
    return result


def derive_momenta(
    lagrangian: Expr,
    context: VariationalContext | None = None,
) -> LagrangianMomenta:
    """Calcula M_ab, P^abcd, J^a y F_phi manteniendo argumentos independientes."""

    if not lagrangian.is_scalar:
        raise TensorAlgebraError("Los momentos solo se definen para un lagrangiano escalar.")
    context = context or VariationalContext()
    space = context.index_space
    a_down, b_down = (Index(name, Variance.DOWN, space) for name in ("a", "b"))
    abcd_up = tuple(Index(name, Variance.UP, space) for name in ("a", "b", "c", "d"))
    a_up = Index("a", Variance.UP, space)
    return LagrangianMomenta(
        metric=tensor_partial_derivative(
            lagrangian,
            context.metric_name,
            (Variance.UP, Variance.UP),
            (a_down, b_down),
            TensorSymmetry.SYMMETRIC,
            context.delta_name,
        ),
        curvature=tensor_partial_derivative(
            lagrangian,
            context.curvature_name,
            (Variance.DOWN,) * 4,
            abcd_up,
            TensorSymmetry.RIEMANN,
            context.delta_name,
        ),
        scalar_gradient=tensor_partial_derivative(
            lagrangian,
            context.scalar_gradient_name,
            (Variance.DOWN,),
            (a_up,),
            TensorSymmetry.NONE,
            context.delta_name,
        ),
        scalar=scalar_partial_derivative(lagrangian, context.scalar_name),
    )


def direct_variation(expr: Expr, context: VariationalContext | None = None) -> Expr:
    """Aplica linealidad, Leibniz y cadena a la variación de una expresión algebraica."""

    context = context or VariationalContext()
    if isinstance(expr, Number):
        return Number(0)
    if isinstance(expr, Scalar):
        if expr.name in context.constant_scalars:
            return Number(0)
        return Variation(expr)
    if isinstance(expr, Tensor):
        return Variation(expr)
    if isinstance(expr, VolumeElement):
        return volume_element_variation(context)
    if isinstance(expr, Add):
        return add(*(direct_variation(term, context) for term in expr.terms))
    if isinstance(expr, Mul):
        terms: list[Expr] = []
        for position, factor in enumerate(expr.factors):
            derivative = direct_variation(factor, context)
            if _is_zero(derivative):
                continue
            factors = list(expr.factors)
            factors[position] = derivative
            terms.append(mul(*factors))
        return add(*terms)
    if isinstance(expr, Power):
        return _power_partial(
            expr.base,
            expr.exponent,
            lambda item: direct_variation(item, context),
        )
    if isinstance(expr, Function):
        return _function_partial(
            expr.name,
            (0,) * len(expr.arguments),
            expr.arguments,
            lambda item: direct_variation(item, context),
        )
    if isinstance(expr, FunctionDerivative):
        return _function_partial(
            expr.name,
            expr.derivative_orders,
            expr.arguments,
            lambda item: direct_variation(item, context),
        )
    if isinstance(expr, CovariantDerivative):
        return Variation(expr)
    if isinstance(expr, Variation):
        raise TensorAlgebraError("La segunda variación no pertenece al alcance de la fase 4.")
    raise TypeError(f"Nodo IR no reconocido: {type(expr).__name__}")


def _variation_term(momentum: Expr, variable_name: str, scalar: bool = False) -> Expr:
    if _is_zero(momentum):
        return Number(0)
    free = infer_free_indices(momentum)
    if scalar:
        if free:
            raise TensorAlgebraError("El momento escalar no puede portar índices libres.")
        return mul(momentum, Variation(Scalar(variable_name)))
    if not free:
        raise TensorAlgebraError("Un momento tensorial no nulo debe portar índices libres.")
    variable = Tensor(variable_name, tuple(index.flipped() for index in free))
    return mul(momentum, Variation(variable))


def raw_lagrangian_variation(
    momenta: LagrangianMomenta,
    context: VariationalContext | None = None,
) -> Expr:
    """Reconstruye delta L = M delta g + P delta R + F delta phi + J delta u."""

    context = context or VariationalContext()
    return add(
        _variation_term(momenta.metric, context.metric_name),
        _variation_term(momenta.curvature, context.curvature_name),
        _variation_term(momenta.scalar, context.scalar_name, scalar=True),
        _variation_term(momenta.scalar_gradient, context.scalar_gradient_name),
    )


def covariant_metric_variation(
    first: Index,
    second: Index,
    context: VariationalContext | None = None,
) -> Expr:
    """Construye delta g_ab = -g_ac g_bd delta g^cd."""

    context = context or VariationalContext()
    if first.variance is not Variance.DOWN or second.variance is not Variance.DOWN:
        raise TensorAlgebraError("La variación solicitada debe corresponder a g_ab.")
    if first.space != second.space:
        raise TensorAlgebraError("Los índices métricos deben pertenecer al mismo espacio.")
    occupied = {first.name, second.name}
    names: list[str] = []
    counter = 0
    while len(names) < 2:
        candidate = f"m{counter}"
        counter += 1
        if candidate not in occupied:
            names.append(candidate)
    c, d = names
    expression = mul(
        -1,
        Tensor(context.metric_name, (first, Index(c, Variance.DOWN, first.space))),
        Tensor(context.metric_name, (second, Index(d, Variance.DOWN, first.space))),
        Variation(
            Tensor(
                context.metric_name,
                (Index(c, Variance.UP, first.space), Index(d, Variance.UP, first.space)),
            )
        ),
    )
    return symmetrize(expression, (first, second))


def volume_element_variation(context: VariationalContext | None = None) -> Expr:
    """Construye delta sqrt(-g) para variación respecto a g^ab."""

    context = context or VariationalContext()
    a = Index("v0", Variance.DOWN, context.index_space)
    b = Index("v1", Variance.DOWN, context.index_space)
    return mul(
        -1,
        Number(1, 2),
        VolumeElement(context.metric_name),
        Tensor(context.metric_name, (a, b)),
        Variation(Tensor(context.metric_name, (a.flipped(), b.flipped()))),
    )


def scalar_gradient_geometric_variation(
    index: Index,
    context: VariationalContext | None = None,
) -> Expr:
    """Restituye u_a=nabla_a phi: delta u_a=nabla_a delta phi."""

    context = context or VariationalContext()
    if index.variance is not Variance.DOWN:
        raise TensorAlgebraError("u_a y su variación geométrica llevan índice inferior.")
    return CovariantDerivative(index, Variation(Scalar(context.scalar_name)))


def riemann_independent_variation(
    indices: tuple[Index, Index, Index, Index],
    context: VariationalContext | None = None,
) -> Variation:
    """Variación independiente de R_abcd antes de aplicar Palatini."""

    context = context or VariationalContext()
    if any(index.variance is not Variance.DOWN for index in indices):
        raise TensorAlgebraError("R_abcd debe proporcionarse completamente covariante.")
    if len({index.space for index in indices}) != 1:
        raise TensorAlgebraError("Los índices de Riemann deben compartir espacio.")
    return Variation(Tensor(context.curvature_name, indices))
