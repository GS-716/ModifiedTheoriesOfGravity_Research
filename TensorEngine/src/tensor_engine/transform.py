"""Sustitución estructural, expansión y permutación de índices libres."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import permutations, product
from math import factorial

from .errors import TensorAlgebraError
from .indices import IndexKey, index_key, rename_free_indices
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
    Variation,
    VolumeElement,
    add,
    infer_free_indices,
    mul,
)


def _free_signature(expr: Expr) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        sorted((item.space, item.name, item.variance.value) for item in infer_free_indices(expr))
    )


def substitute(expr: Expr, replacements: Mapping[Expr, Expr]) -> Expr:
    """Sustitución exacta que preserva la firma de índices libres."""

    if expr in replacements:
        replacement = replacements[expr]
        if _free_signature(expr) != _free_signature(replacement):
            raise TensorAlgebraError(
                "Una sustitución tensorial debe conservar sus índices libres y varianzas."
            )
        return replacement
    if isinstance(expr, (Number, Scalar, Tensor, VolumeElement)):
        return expr
    if isinstance(expr, Add):
        return add(*(substitute(term, replacements) for term in expr.terms))
    if isinstance(expr, Mul):
        return mul(*(substitute(factor, replacements) for factor in expr.factors))
    if isinstance(expr, Power):
        return Power(substitute(expr.base, replacements), substitute(expr.exponent, replacements))
    if isinstance(expr, Function):
        return Function(
            expr.name,
            tuple(substitute(argument, replacements) for argument in expr.arguments),
        )
    if isinstance(expr, FunctionDerivative):
        return FunctionDerivative(
            expr.name,
            expr.derivative_orders,
            tuple(substitute(argument, replacements) for argument in expr.arguments),
        )
    if isinstance(expr, CovariantDerivative):
        return CovariantDerivative(expr.index, substitute(expr.operand, replacements))
    if isinstance(expr, Variation):
        return Variation(substitute(expr.operand, replacements))
    raise TypeError(f"Nodo IR no reconocido: {type(expr).__name__}")


def expand(expr: Expr) -> Expr:
    """Distribuye productos sobre sumas sin expandir potencias simbólicas."""

    if isinstance(expr, (Number, Scalar, Tensor, VolumeElement)):
        return expr
    if isinstance(expr, Add):
        return add(*(expand(term) for term in expr.terms))
    if isinstance(expr, Mul):
        expanded_factors = [expand(factor) for factor in expr.factors]
        choices = [factor.terms if isinstance(factor, Add) else (factor,) for factor in expanded_factors]
        return add(*(mul(*combination) for combination in product(*choices)))
    if isinstance(expr, Power):
        return Power(expand(expr.base), expand(expr.exponent))
    if isinstance(expr, Function):
        return Function(expr.name, tuple(expand(argument) for argument in expr.arguments))
    if isinstance(expr, FunctionDerivative):
        return FunctionDerivative(
            expr.name,
            expr.derivative_orders,
            tuple(expand(argument) for argument in expr.arguments),
        )
    if isinstance(expr, CovariantDerivative):
        operand = expand(expr.operand)
        if isinstance(operand, Add):
            return add(*(CovariantDerivative(expr.index, term) for term in operand.terms))
        return CovariantDerivative(expr.index, operand)
    if isinstance(expr, Variation):
        operand = expand(expr.operand)
        if isinstance(operand, Add):
            return add(*(Variation(term) for term in operand.terms))
        return Variation(operand)
    raise TypeError(f"Nodo IR no reconocido: {type(expr).__name__}")


def permute_free_indices(expr: Expr, mapping: Mapping[IndexKey, str]) -> Expr:
    """Permuta simultáneamente índices libres declarados."""

    return rename_free_indices(expr, mapping)


def _validate_symmetry_indices(expr: Expr, indices: tuple[Index, ...]) -> None:
    if len(indices) < 2:
        raise TensorAlgebraError("La simetrización requiere al menos dos índices.")
    free = {index_key(index): index for index in infer_free_indices(expr)}
    requested = [index_key(index) for index in indices]
    if len(requested) != len(set(requested)):
        raise TensorAlgebraError("No se puede repetir un índice en la lista de simetrización.")
    if any(key not in free for key in requested):
        raise TensorAlgebraError("Solo se pueden simetrizar índices libres.")
    spaces = {index.space for index in indices}
    variances = {index.variance for index in indices}
    if len(spaces) != 1 or len(variances) != 1:
        raise TensorAlgebraError("Los índices simetrizados deben compartir espacio y varianza.")


def _permutation_parity(order: tuple[int, ...]) -> int:
    inversions = sum(
        1 for left in range(len(order)) for right in range(left + 1, len(order))
        if order[left] > order[right]
    )
    return -1 if inversions % 2 else 1


def symmetrize(expr: Expr, indices: tuple[Index, ...]) -> Expr:
    """Aplica simetrización normalizada con peso 1/n!."""

    _validate_symmetry_indices(expr, indices)
    terms: list[Expr] = []
    keys = [index_key(index) for index in indices]
    for order in permutations(range(len(indices))):
        mapping = {keys[position]: indices[target].name for position, target in enumerate(order)}
        terms.append(permute_free_indices(expr, mapping))
    return mul(Number(1, factorial(len(indices))), add(*terms))


def antisymmetrize(expr: Expr, indices: tuple[Index, ...]) -> Expr:
    """Aplica antisimetrización normalizada con peso 1/n!."""

    _validate_symmetry_indices(expr, indices)
    terms: list[Expr] = []
    keys = [index_key(index) for index in indices]
    for order in permutations(range(len(indices))):
        mapping = {keys[position]: indices[target].name for position, target in enumerate(order)}
        terms.append(mul(_permutation_parity(order), permute_free_indices(expr, mapping)))
    return mul(Number(1, factorial(len(indices))), add(*terms))
