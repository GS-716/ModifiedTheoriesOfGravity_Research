"""Operaciones higiénicas sobre índices de la representación intermedia."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from .errors import TensorAlgebraError
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
    infer_free_indices,
    mul,
    walk,
)


IndexKey = tuple[str, str]


def index_key(index: Index) -> IndexKey:
    return index.space, index.name


def all_indices(expr: Expr) -> tuple[Index, ...]:
    """Devuelve todas las apariciones de índices, incluidas las derivativas."""

    result: list[Index] = []
    for node in walk(expr):
        if isinstance(node, Tensor):
            result.extend(node.indices)
        elif isinstance(node, CovariantDerivative):
            result.append(node.index)
    return tuple(result)


def used_index_names(expr: Expr, space: str | None = None) -> frozenset[str]:
    return frozenset(
        index.name for index in all_indices(expr) if space is None or index.space == space
    )


def map_indices(expr: Expr, mapper: Callable[[Index], Index]) -> Expr:
    """Aplica una transformación simultánea a todas las apariciones."""

    if isinstance(expr, (Number, Scalar, VolumeElement)):
        return expr
    if isinstance(expr, Tensor):
        result: Expr = Tensor(expr.name, tuple(mapper(index) for index in expr.indices))
    elif isinstance(expr, Add):
        result = Add(tuple(map_indices(term, mapper) for term in expr.terms))
    elif isinstance(expr, Mul):
        result = Mul(tuple(map_indices(factor, mapper) for factor in expr.factors))
    elif isinstance(expr, Power):
        result = Power(map_indices(expr.base, mapper), map_indices(expr.exponent, mapper))
    elif isinstance(expr, Function):
        result = Function(expr.name, tuple(map_indices(arg, mapper) for arg in expr.arguments))
    elif isinstance(expr, FunctionDerivative):
        result = FunctionDerivative(
            expr.name,
            expr.derivative_orders,
            tuple(map_indices(arg, mapper) for arg in expr.arguments),
        )
    elif isinstance(expr, CovariantDerivative):
        result = CovariantDerivative(mapper(expr.index), map_indices(expr.operand, mapper))
    elif isinstance(expr, Variation):
        result = Variation(map_indices(expr.operand, mapper))
    else:
        raise TypeError(f"Nodo IR no reconocido: {type(expr).__name__}")
    infer_free_indices(result)
    return result


def rename_indices(expr: Expr, mapping: Mapping[IndexKey, str]) -> Expr:
    """Renombra índices simultáneamente, conservando espacio y varianza."""

    def mapper(index: Index) -> Index:
        name = mapping.get(index_key(index), index.name)
        return Index(name, index.variance, index.space)

    return map_indices(expr, mapper)


def rename_free_indices(expr: Expr, mapping: Mapping[IndexKey, str]) -> Expr:
    """Renombra solo índices libres; rechaza claves que no sean libres."""

    free_keys = {index_key(index) for index in infer_free_indices(expr)}
    unknown = set(mapping).difference(free_keys)
    if unknown:
        raise TensorAlgebraError(
            f"Se intentó renombrar como libres índices no libres: {sorted(unknown)}"
        )

    # Los índices mudos se hacen higiénicos primero para impedir que compartan
    # accidentalmente el nombre de un índice libre durante el renombrado.
    hygienic = canonicalize_dummy_indices(expr)
    return rename_indices(hygienic, mapping)


@dataclass(slots=True)
class _FreshIndexAllocator:
    reserved: set[tuple[str, str]]
    prefix: str = "d"
    counter: int = 0

    def next(self, space: str) -> str:
        while (space, f"{self.prefix}{self.counter}") in self.reserved:
            self.counter += 1
        name = f"{self.prefix}{self.counter}"
        self.counter += 1
        self.reserved.add((space, name))
        return name


def _rename_local(expr: Expr, key: IndexKey, new_name: str) -> Expr:
    return rename_indices(expr, {key: new_name})


def _canonicalize_dummies(expr: Expr, allocator: _FreshIndexAllocator) -> Expr:
    if isinstance(expr, (Number, Scalar, VolumeElement)):
        return expr
    if isinstance(expr, Tensor):
        result: Expr = expr
        counts = Counter(index_key(index) for index in expr.indices)
        for key, count in counts.items():
            if count == 2:
                occurrences = [index for index in expr.indices if index_key(index) == key]
                if {item.variance for item in occurrences} == {Variance.UP, Variance.DOWN}:
                    result = _rename_local(result, key, allocator.next(key[0]))
        return result
    if isinstance(expr, Add):
        # Cada término de una suma tiene su propio alcance de índices mudos.
        terms: list[Expr] = []
        for term in expr.terms:
            local = _FreshIndexAllocator(set(allocator.reserved), allocator.prefix)
            terms.append(_canonicalize_dummies(term, local))
        result = Add(tuple(terms))
        infer_free_indices(result)
        return result
    if isinstance(expr, Mul):
        factors = [_canonicalize_dummies(factor, allocator) for factor in expr.factors]
        # Una suma-factor puede contener mudos con alcance local que no hicieron
        # avanzar el allocator compartido. Se reservan antes de nombrar las
        # contracciones externas para que ambas capas nunca usen el mismo nombre.
        allocator.reserved.update(
            index_key(index)
            for factor in factors
            for index in all_indices(factor)
        )
        grouped: dict[IndexKey, list[int]] = {}
        for position, factor in enumerate(factors):
            for index in infer_free_indices(factor):
                grouped.setdefault(index_key(index), []).append(position)
        for key, positions in grouped.items():
            if len(positions) == 2:
                occurrences = [
                    next(
                        index
                        for index in infer_free_indices(factors[position])
                        if index_key(index) == key
                    )
                    for position in positions
                ]
                if {item.variance for item in occurrences} == {Variance.UP, Variance.DOWN}:
                    new_name = allocator.next(key[0])
                    for position in set(positions):
                        factors[position] = _rename_local(factors[position], key, new_name)
        result = Mul(tuple(factors))
        infer_free_indices(result)
        return result
    if isinstance(expr, Power):
        return Power(
            _canonicalize_dummies(expr.base, allocator),
            _canonicalize_dummies(expr.exponent, allocator),
        )
    if isinstance(expr, Function):
        return Function(
            expr.name,
            tuple(_canonicalize_dummies(arg, allocator) for arg in expr.arguments),
        )
    if isinstance(expr, FunctionDerivative):
        return FunctionDerivative(
            expr.name,
            expr.derivative_orders,
            tuple(_canonicalize_dummies(arg, allocator) for arg in expr.arguments),
        )
    if isinstance(expr, CovariantDerivative):
        operand = _canonicalize_dummies(expr.operand, allocator)
        derivative_index = expr.index
        matches = [
            index
            for index in infer_free_indices(operand)
            if index_key(index) == index_key(derivative_index)
        ]
        if matches and matches[0].variance is not derivative_index.variance:
            new_name = allocator.next(derivative_index.space)
            operand = _rename_local(operand, index_key(derivative_index), new_name)
            derivative_index = Index(new_name, derivative_index.variance, derivative_index.space)
        result = CovariantDerivative(derivative_index, operand)
        infer_free_indices(result)
        return result
    if isinstance(expr, Variation):
        return Variation(_canonicalize_dummies(expr.operand, allocator))
    raise TypeError(f"Nodo IR no reconocido: {type(expr).__name__}")


def canonicalize_dummy_indices(expr: Expr, prefix: str = "d") -> Expr:
    """Renombra índices mudos determinísticamente sin tocar índices libres."""

    free = {index_key(index) for index in infer_free_indices(expr)}
    # Primera pasada: aparta todos los nombres mudos a un espacio temporal que
    # no colisiona con ningún nombre existente. Segunda pasada: asigna d0, d1,
    # ... reservando únicamente los nombres libres. Así la operación es
    # idempotente incluso si la entrada ya utiliza nombres canónicos.
    occupied = {index_key(index) for index in all_indices(expr)}
    temporary_allocator = _FreshIndexAllocator(set(occupied), f"tmp{prefix}")
    temporary = _canonicalize_dummies(expr, temporary_allocator)
    canonical_allocator = _FreshIndexAllocator(set(free), prefix)
    return _canonicalize_dummies(temporary, canonical_allocator)


def tensor_product(*factors: Expr) -> Expr:
    """Producto tensorial sin contracciones accidentales entre factores."""

    if not factors:
        return Number(1)
    result_factors: list[Expr] = []
    occupied: set[IndexKey] = set()
    for position, original in enumerate(factors):
        factor = canonicalize_dummy_indices(original, prefix=f"t{position}d")
        mapping: dict[IndexKey, str] = {}
        reserved_names = set(occupied).union(index_key(index) for index in all_indices(factor))
        allocator = _FreshIndexAllocator(reserved_names, prefix=f"t{position}f")
        for index in infer_free_indices(factor):
            key = index_key(index)
            if key in occupied:
                mapping[key] = allocator.next(index.space)
        if mapping:
            factor = rename_free_indices(factor, mapping)
        occupied.update(index_key(index) for index in infer_free_indices(factor))
        result_factors.append(factor)
    return mul(*result_factors)
