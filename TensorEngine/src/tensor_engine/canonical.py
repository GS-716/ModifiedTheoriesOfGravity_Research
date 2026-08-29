"""Canonización monótérmino y contracciones métricas estructurales."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import json

from .errors import TensorAlgebraError
from .indices import (
    all_indices,
    canonicalize_dummy_indices,
    index_key,
    rename_free_indices,
)
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
)
from .model import TensorDeclaration, TensorSymmetry
from .transform import expand


def _sort_key(expr: Expr) -> str:
    return json.dumps(expr.to_data(), sort_keys=True, separators=(",", ":"))


def _index_order(index: Index) -> tuple[str, str, str]:
    return index.space, index.name, index.variance.value


def canonicalize_tensor(
    tensor: Tensor,
    declarations: dict[str, TensorDeclaration],
) -> Expr:
    """Aplica simetrías que relacionan un monomio con otro por signo."""

    declaration = declarations.get(tensor.name)
    if declaration is None or declaration.symmetry is TensorSymmetry.NONE:
        return tensor

    indices = list(tensor.indices)
    if declaration.symmetry is TensorSymmetry.SYMMETRIC:
        if len(indices) == 2 and indices[0].variance is indices[1].variance:
            if _index_order(indices[1]) < _index_order(indices[0]):
                indices[0], indices[1] = indices[1], indices[0]
        return Tensor(tensor.name, tuple(indices))

    if declaration.symmetry is TensorSymmetry.RIEMANN:
        if len(indices) != 4:
            raise TensorAlgebraError(f"{tensor.name} fue declarado Riemann pero no tiene rango cuatro.")
        # Las simetrías por pares se aplican a representaciones completamente
        # covariantes o completamente contravariantes.
        if len({index.variance for index in indices}) != 1:
            return tensor
        if index_key(indices[0]) == index_key(indices[1]):
            return Number(0)
        if index_key(indices[2]) == index_key(indices[3]):
            return Number(0)

        sign = 1
        if _index_order(indices[1]) < _index_order(indices[0]):
            indices[0], indices[1] = indices[1], indices[0]
            sign *= -1
        if _index_order(indices[3]) < _index_order(indices[2]):
            indices[2], indices[3] = indices[3], indices[2]
            sign *= -1
        first_pair = (_index_order(indices[0]), _index_order(indices[1]))
        second_pair = (_index_order(indices[2]), _index_order(indices[3]))
        if second_pair < first_pair:
            indices = indices[2:4] + indices[0:2]
        result = Tensor(tensor.name, tuple(indices))
        return result if sign == 1 else mul(-1, result)

    return tensor


def _canonicalize_nodes(expr: Expr, declarations: dict[str, TensorDeclaration]) -> Expr:
    if isinstance(expr, (Number, Scalar, VolumeElement)):
        return expr
    if isinstance(expr, Tensor):
        return canonicalize_tensor(expr, declarations)
    if isinstance(expr, Add):
        return add(*(_canonicalize_nodes(term, declarations) for term in expr.terms))
    if isinstance(expr, Mul):
        return mul(*(_canonicalize_nodes(factor, declarations) for factor in expr.factors))
    if isinstance(expr, Power):
        return Power(
            _canonicalize_nodes(expr.base, declarations),
            _canonicalize_nodes(expr.exponent, declarations),
        )
    if isinstance(expr, Function):
        return Function(
            expr.name,
            tuple(_canonicalize_nodes(argument, declarations) for argument in expr.arguments),
        )
    if isinstance(expr, FunctionDerivative):
        return FunctionDerivative(
            expr.name,
            expr.derivative_orders,
            tuple(_canonicalize_nodes(argument, declarations) for argument in expr.arguments),
        )
    if isinstance(expr, CovariantDerivative):
        return CovariantDerivative(expr.index, _canonicalize_nodes(expr.operand, declarations))
    if isinstance(expr, Variation):
        operand = _canonicalize_nodes(expr.operand, declarations)
        if isinstance(operand, Add):
            return add(*(Variation(term) for term in operand.terms))
        if isinstance(operand, Mul):
            coefficient = Fraction(1)
            factors: list[Expr] = []
            for factor in _flatten_product(operand):
                if isinstance(factor, Number):
                    coefficient *= factor.value
                else:
                    factors.append(factor)
            if coefficient != 1 and factors:
                inner = factors[0] if len(factors) == 1 else mul(*factors)
                return mul(Number(coefficient.numerator, coefficient.denominator), Variation(inner))
        return Variation(operand)
    raise TypeError(f"Nodo IR no reconocido: {type(expr).__name__}")


def _flatten_product(expr: Expr) -> list[Expr]:
    if isinstance(expr, Mul):
        result: list[Expr] = []
        for factor in expr.factors:
            result.extend(_flatten_product(factor))
        return result
    return [expr]


def _normalize_product(expr: Expr) -> Expr:
    if not isinstance(expr, Mul):
        return expr
    coefficient = Fraction(1)
    factors: list[Expr] = []
    for factor in _flatten_product(expr):
        if isinstance(factor, Number):
            coefficient *= factor.value
        else:
            factors.append(factor)
    if coefficient == 0:
        return Number(0)
    factors.sort(key=_sort_key)
    coefficient_expr = Number(coefficient.numerator, coefficient.denominator)
    return mul(coefficient_expr, *factors)


def _split_coefficient(expr: Expr) -> tuple[Fraction, Expr | None]:
    if isinstance(expr, Number):
        return expr.value, None
    if isinstance(expr, Mul):
        coefficient = Fraction(1)
        factors: list[Expr] = []
        for factor in _flatten_product(expr):
            if isinstance(factor, Number):
                coefficient *= factor.value
            else:
                factors.append(factor)
        if not factors:
            return coefficient, None
        base = factors[0] if len(factors) == 1 else Mul(tuple(factors))
        return coefficient, base
    return Fraction(1), expr


def _combine_like_terms(expr: Expr) -> Expr:
    if not isinstance(expr, Add):
        return expr
    coefficients: dict[Expr | None, Fraction] = defaultdict(Fraction)
    for term in expr.terms:
        coefficient, base = _split_coefficient(term)
        coefficients[base] += coefficient
    terms: list[Expr] = []
    for base, coefficient in coefficients.items():
        if coefficient == 0:
            continue
        number = Number(coefficient.numerator, coefficient.denominator)
        terms.append(number if base is None else mul(number, base))
    if not terms:
        return Number(0)
    terms.sort(key=_sort_key)
    return terms[0] if len(terms) == 1 else Add(tuple(terms))


def canonicalize_monoterm(
    expr: Expr,
    declarations: tuple[TensorDeclaration, ...] = (),
) -> Expr:
    """Normaliza índices mudos, productos y simetrías monótérmino."""

    declaration_map = {item.name: item for item in declarations}
    current = expand(expr)
    for _ in range(6):
        previous = current
        current = canonicalize_dummy_indices(current)
        current = _canonicalize_nodes(current, declaration_map)
        if isinstance(current, Add):
            terms = [_normalize_product(term) for term in current.terms]
            terms = [canonicalize_dummy_indices(term) for term in terms]
            current = Add(tuple(terms)) if len(terms) > 1 else terms[0]
            current = _combine_like_terms(current)
            if isinstance(current, Add):
                current = Add(tuple(_normalize_product(term) for term in current.terms))
            else:
                current = _normalize_product(current)
        else:
            current = _normalize_product(current)
            current = canonicalize_dummy_indices(current)
        if current == previous:
            break
    infer_free_indices(current)
    return current


def _replace_tensor_index(tensor: Tensor, position: int, replacement: Index) -> Tensor:
    indices = list(tensor.indices)
    indices[position] = replacement
    return Tensor(tensor.name, tuple(indices))


def _dimension_expr(dimension: int | str | Expr) -> Expr:
    if isinstance(dimension, Expr):
        return dimension
    if isinstance(dimension, int):
        return Number(dimension)
    return Scalar(dimension)


def _simplify_metric_product(
    expr: Expr,
    metric_name: str,
    delta_name: str,
    dimension: int | str | Expr,
) -> Expr:
    factors = _flatten_product(canonicalize_dummy_indices(expr))
    factors = [factor for factor in factors if not (isinstance(factor, Number) and factor.value == 1)]
    changed = True
    while changed:
        changed = False

        # Una métrica mixta es el delta de Kronecker; su traza es D.
        for position, factor in enumerate(factors):
            if not isinstance(factor, Tensor) or factor.name != metric_name or len(factor.indices) != 2:
                continue
            first, second = factor.indices
            if first.variance is second.variance:
                continue
            upper, lower = (first, second) if first.variance is Variance.UP else (second, first)
            if index_key(upper) == index_key(lower):
                factors[position] = _dimension_expr(dimension)
            else:
                factors[position] = Tensor(delta_name, (upper, lower))
            changed = True
            break
        if changed:
            continue

        # Contracción de un delta con un tensor directo.
        contracted = False
        for delta_position, factor in enumerate(factors):
            if not isinstance(factor, Tensor) or factor.name != delta_name or len(factor.indices) != 2:
                continue
            upper, lower = factor.indices
            if index_key(upper) == index_key(lower):
                factors[delta_position] = _dimension_expr(dimension)
                changed = contracted = True
                break
            for delta_slot, delta_index in enumerate((upper, lower)):
                other = lower if delta_slot == 0 else upper
                for target_position, target in enumerate(factors):
                    if target_position == delta_position or not isinstance(target, Tensor):
                        continue
                    for target_slot, target_index in enumerate(target.indices):
                        if (
                            index_key(target_index) == index_key(delta_index)
                            and target_index.variance is not delta_index.variance
                        ):
                            replacement = Index(other.name, other.variance, other.space)
                            factors[target_position] = _replace_tensor_index(target, target_slot, replacement)
                            factors.pop(delta_position)
                            changed = contracted = True
                            break
                    if contracted:
                        break
                if contracted:
                    break
            if contracted:
                break
        if changed:
            continue

        # Una métrica de varianza uniforme sube o baja un índice contraído.
        for metric_position, factor in enumerate(factors):
            if not isinstance(factor, Tensor) or factor.name != metric_name or len(factor.indices) != 2:
                continue
            first, second = factor.indices
            if first.variance is not second.variance:
                continue
            for metric_slot, metric_index in enumerate((first, second)):
                other = second if metric_slot == 0 else first
                found = False
                for target_position, target in enumerate(factors):
                    if target_position == metric_position or not isinstance(target, Tensor):
                        continue
                    for target_slot, target_index in enumerate(target.indices):
                        if (
                            index_key(target_index) == index_key(metric_index)
                            and target_index.variance is not metric_index.variance
                        ):
                            replacement = Index(other.name, other.variance, other.space)
                            factors[target_position] = _replace_tensor_index(target, target_slot, replacement)
                            factors.pop(metric_position)
                            changed = found = True
                            break
                    if found:
                        break
                if found:
                    break
            if changed:
                break

    if not factors:
        return Number(1)
    result = mul(*factors)
    infer_free_indices(result)
    return result


def simplify_metrics(
    expr: Expr,
    metric_name: str = "g",
    delta_name: str = "delta",
    dimension: int | str | Expr = "D",
) -> Expr:
    """Contrae métricas y deltas en productos tensoriales directos."""

    if isinstance(expr, (Number, Scalar, Tensor, VolumeElement)):
        if isinstance(expr, Tensor) and expr.name == delta_name and len(expr.indices) == 2:
            upper, lower = expr.indices
            if upper.variance is Variance.DOWN:
                upper, lower = lower, upper
            if upper.variance is Variance.UP and lower.variance is Variance.DOWN:
                if index_key(upper) == index_key(lower):
                    return _dimension_expr(dimension)
        if isinstance(expr, Tensor) and expr.name == metric_name and len(expr.indices) == 2:
            first, second = expr.indices
            if first.variance is not second.variance:
                upper, lower = (first, second) if first.variance is Variance.UP else (second, first)
                if index_key(upper) == index_key(lower):
                    return _dimension_expr(dimension)
                return Tensor(delta_name, (upper, lower))
        return expr
    if isinstance(expr, Add):
        return add(*(simplify_metrics(term, metric_name, delta_name, dimension) for term in expr.terms))
    if isinstance(expr, Mul):
        simplified = mul(
            *(simplify_metrics(factor, metric_name, delta_name, dimension) for factor in expr.factors)
        )
        return _simplify_metric_product(simplified, metric_name, delta_name, dimension)
    if isinstance(expr, Power):
        return Power(
            simplify_metrics(expr.base, metric_name, delta_name, dimension),
            simplify_metrics(expr.exponent, metric_name, delta_name, dimension),
        )
    if isinstance(expr, Function):
        return Function(
            expr.name,
            tuple(simplify_metrics(arg, metric_name, delta_name, dimension) for arg in expr.arguments),
        )
    if isinstance(expr, FunctionDerivative):
        return FunctionDerivative(
            expr.name,
            expr.derivative_orders,
            tuple(simplify_metrics(arg, metric_name, delta_name, dimension) for arg in expr.arguments),
        )
    if isinstance(expr, CovariantDerivative):
        return CovariantDerivative(
            expr.index,
            simplify_metrics(expr.operand, metric_name, delta_name, dimension),
        )
    if isinstance(expr, Variation):
        return Variation(simplify_metrics(expr.operand, metric_name, delta_name, dimension))
    raise TypeError(f"Nodo IR no reconocido: {type(expr).__name__}")


def change_free_index_variance(
    expr: Expr,
    index: Index,
    target: Variance,
    metric_name: str = "g",
) -> Expr:
    """Sube o baja un índice libre introduciendo explícitamente la métrica."""

    free = {index_key(item): item for item in infer_free_indices(expr)}
    key = index_key(index)
    if key not in free or free[key].variance is not index.variance:
        raise TensorAlgebraError("El índice solicitado no es libre con la varianza indicada.")
    if index.variance is target:
        return expr
    occupied = {(item.space, item.name) for item in all_indices(expr)}
    counter = 0
    while (index.space, f"r{counter}") in occupied:
        counter += 1
    dummy_name = f"r{counter}"
    renamed = rename_free_indices(expr, {key: dummy_name})
    free_index = Index(index.name, target, index.space)
    dummy_index = Index(dummy_name, target, index.space)
    metric = Tensor(metric_name, (free_index, dummy_index))
    return mul(metric, renamed)


def raise_index(expr: Expr, index: Index, metric_name: str = "g") -> Expr:
    if index.variance is not Variance.DOWN:
        raise TensorAlgebraError("raise_index requiere un índice libre inferior.")
    return change_free_index_variance(expr, index, Variance.UP, metric_name)


def lower_index(expr: Expr, index: Index, metric_name: str = "g") -> Expr:
    if index.variance is not Variance.UP:
        raise TensorAlgebraError("lower_index requiere un índice libre superior.")
    return change_free_index_variance(expr, index, Variance.DOWN, metric_name)


def first_bianchi_residual(
    tensor_name: str,
    a: Index,
    b: Index,
    c: Index,
    d: Index,
) -> Expr:
    """Construye R_abcd + R_acdb + R_adbc, cuyo valor esperado es cero."""

    if len({item.space for item in (a, b, c, d)}) != 1:
        raise TensorAlgebraError("Los índices de Bianchi deben pertenecer al mismo espacio.")
    if len({item.variance for item in (a, b, c, d)}) != 1:
        raise TensorAlgebraError("Los índices de Bianchi deben tener la misma varianza.")
    return add(
        Tensor(tensor_name, (a, b, c, d)),
        Tensor(tensor_name, (a, c, d, b)),
        Tensor(tensor_name, (a, d, b, c)),
    )
