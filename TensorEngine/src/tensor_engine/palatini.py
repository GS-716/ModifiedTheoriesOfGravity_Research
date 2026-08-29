"""Variaciones geométricas de conexión y curvatura."""

from __future__ import annotations

from .errors import TensorAlgebraError
from .ir import CovariantDerivative, Expr, Index, Number, Tensor, Variance, Variation, add, mul
from .variational import VariationalContext


def _validate_connection_indices(upper: Index, first: Index, second: Index) -> None:
    if upper.variance is not Variance.UP:
        raise TensorAlgebraError("El primer índice de delta Gamma debe ser superior.")
    if first.variance is not Variance.DOWN or second.variance is not Variance.DOWN:
        raise TensorAlgebraError("Los índices inferiores de delta Gamma deben ser covariantes.")
    if len({upper.space, first.space, second.space}) != 1:
        raise TensorAlgebraError("Los índices de delta Gamma deben compartir espacio.")


def connection_variation_symbol(
    upper: Index,
    first: Index,
    second: Index,
    context: VariationalContext | None = None,
) -> Tensor:
    """Tensor formal delta Gamma^a_bc, simétrico conceptualmente en b,c."""

    context = context or VariationalContext()
    _validate_connection_indices(upper, first, second)
    lower = sorted((first, second), key=lambda item: (item.space, item.name))
    return Tensor(context.connection_variation_name, (upper, *lower))


def connection_variation(
    upper: Index,
    first: Index,
    second: Index,
    context: VariationalContext | None = None,
) -> Expr:
    """Expande delta Gamma con la variación de la métrica covariante."""

    context = context or VariationalContext()
    _validate_connection_indices(upper, first, second)
    occupied = {upper.name, first.name, second.name}
    counter = 0
    while f"p{counter}" in occupied:
        counter += 1
    dummy_name = f"p{counter}"
    dummy_up = Index(dummy_name, Variance.UP, upper.space)
    dummy_down = dummy_up.flipped()
    metric = Tensor(context.metric_name, (upper, dummy_up))
    return mul(
        Number(1, 2),
        metric,
        add(
            CovariantDerivative(
                first,
                Variation(Tensor(context.metric_name, (dummy_down, second))),
            ),
            CovariantDerivative(
                second,
                Variation(Tensor(context.metric_name, (dummy_down, first))),
            ),
            mul(
                -1,
                CovariantDerivative(
                    dummy_down,
                    Variation(Tensor(context.metric_name, (first, second))),
                ),
            ),
        ),
    )


def mixed_curvature_variation(
    upper: Index,
    lower: Index,
    first: Index,
    second: Index,
    context: VariationalContext | None = None,
    *,
    expand_connection: bool = False,
) -> Expr:
    """Construye delta R^a_bcd mediante la identidad de Palatini."""

    context = context or VariationalContext()
    if upper.variance is not Variance.UP:
        raise TensorAlgebraError("delta R^a_bcd requiere un primer índice superior.")
    if any(item.variance is not Variance.DOWN for item in (lower, first, second)):
        raise TensorAlgebraError("Los tres índices restantes de delta R^a_bcd son inferiores.")
    if len({item.space for item in (upper, lower, first, second)}) != 1:
        raise TensorAlgebraError("Los índices de curvatura deben compartir espacio.")

    if expand_connection:
        forward_connection = connection_variation(upper, second, lower, context)
        backward_connection = connection_variation(upper, first, lower, context)
    else:
        forward_connection = connection_variation_symbol(upper, second, lower, context)
        backward_connection = connection_variation_symbol(upper, first, lower, context)
    return add(
        CovariantDerivative(first, forward_connection),
        mul(-1, CovariantDerivative(second, backward_connection)),
    )


def all_down_curvature_variation(
    indices: tuple[Index, Index, Index, Index],
    context: VariationalContext | None = None,
    *,
    expand_connection: bool = False,
) -> Expr:
    """Restituye R_abcd=g_ae R^e_bcd y construye su variación geométrica."""

    context = context or VariationalContext()
    a, b, c, d = indices
    if any(item.variance is not Variance.DOWN for item in indices):
        raise TensorAlgebraError("delta R_abcd requiere cuatro índices inferiores.")
    if len({item.space for item in indices}) != 1:
        raise TensorAlgebraError("Los índices de curvatura deben compartir espacio.")

    occupied = {item.name for item in indices}
    names: list[str] = []
    counter = 0
    while len(names) < 3:
        candidate = f"p{counter}"
        counter += 1
        if candidate not in occupied:
            names.append(candidate)
    m_name, n_name, e_name = names
    m_down = Index(m_name, Variance.DOWN, a.space)
    n_down = Index(n_name, Variance.DOWN, a.space)
    e_down = Index(e_name, Variance.DOWN, a.space)
    e_up = e_down.flipped()

    lowering_variation = mul(
        -1,
        Tensor(context.metric_name, (a, m_down)),
        Tensor(context.curvature_name, (n_down, b, c, d)),
        Variation(Tensor(context.metric_name, (m_down.flipped(), n_down.flipped()))),
    )
    palatini = mul(
        Tensor(context.metric_name, (a, e_down)),
        mixed_curvature_variation(
            e_up,
            b,
            c,
            d,
            context,
            expand_connection=expand_connection,
        ),
    )
    return add(lowering_variation, palatini)
