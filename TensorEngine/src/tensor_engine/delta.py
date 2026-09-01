"""Safe Kronecker identity elimination on existing IR and index scopes.

Only indices free at a product-factor boundary are substituted. Derivative
operators are never commuted, expanded or moved across factors.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

from .errors import TensorEngineError
from .indices import canonicalize_dummy_indices, index_key, rename_free_indices
from .ir import (
    Add, CovariantDerivative, Expr, Function, FunctionDerivative, Index, Mul,
    Number, Power, Scalar, Tensor, Variance, Variation, add, mul, walk,
)


def expression_hash(expr: Expr) -> str:
    data = json.dumps(expr.to_data(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def delta_count(expr: Expr, delta_name: str = "delta") -> int:
    return sum(isinstance(n, Tensor) and n.name == delta_name for n in walk(expr))


def _signature(expr: Expr) -> tuple:
    return tuple(sorted((i.space, i.name, i.variance.value) for i in expr.free_indices))


@dataclass(frozen=True, slots=True)
class DeltaContractionEvent:
    path: str
    action: str
    delta: Tensor
    source: Index | None = None
    replacement: Index | None = None
    reason: str = ""

    def to_data(self) -> dict:
        return {
            "path": self.path, "action": self.action, "delta": self.delta.to_data(),
            "source": None if self.source is None else self.source.to_data(),
            "replacement": None if self.replacement is None else self.replacement.to_data(),
            "reason": self.reason,
        }

    @classmethod
    def from_data(cls, data: Mapping) -> DeltaContractionEvent:
        from .ir import expr_from_data
        delta = expr_from_data(data["delta"])
        if not isinstance(delta, Tensor):
            raise ValueError("Un evento delta requiere un tensor.")
        return cls(str(data["path"]), str(data["action"]), delta,
                   None if data["source"] is None else Index.from_data(data["source"]),
                   None if data["replacement"] is None else Index.from_data(data["replacement"]),
                   str(data["reason"]))


@dataclass(frozen=True, slots=True)
class DeltaContractionAudit:
    input_sha256: str
    output_sha256: str
    status: str
    deltas_before: int
    deltas_after: int
    events: tuple[DeltaContractionEvent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))
        if self.status not in {"canonical", "partial", "symbolic"}:
            raise ValueError("Estado de contracción delta desconocido.")

    def to_data(self) -> dict:
        return {
            "input_sha256": self.input_sha256, "output_sha256": self.output_sha256,
            "status": self.status, "deltas_before": self.deltas_before,
            "deltas_after": self.deltas_after,
            "events": [event.to_data() for event in self.events],
        }

    @classmethod
    def from_data(cls, data: Mapping) -> DeltaContractionAudit:
        return cls(str(data["input_sha256"]), str(data["output_sha256"]), str(data["status"]),
                   int(data["deltas_before"]), int(data["deltas_after"]),
                   tuple(DeltaContractionEvent.from_data(e) for e in data["events"]))


@dataclass(frozen=True, slots=True)
class DeltaContractionResult:
    expression: Expr
    audit: DeltaContractionAudit


def _identity_problem(delta: Tensor) -> str | None:
    if len(delta.indices) != 2:
        return "El delta no tiene rango dos."
    first, second = delta.indices
    if first.space != second.space:
        return "No se identifica un isomorfismo entre espacios de índices distintos."
    if first.variance is second.variance:
        return "La identidad de Kronecker requiere un índice superior y otro inferior."
    return None


def contract_deltas(
    expression: Expr, *, delta_name: str = "delta", dimension: int | str | Expr = "D",
    index_space: str = "M", dimensions: Mapping[str, int | str | Expr] | None = None,
) -> DeltaContractionResult:
    """Eliminate identity factors, retaining unsupported cases with reasons.

    A standalone free identity is not the number one. A trace uses only the
    dimension declared for its own space. Invalid input is not repaired or
    admitted into the pipeline: this operation returns it unchanged for diagnosis.
    """
    dimensions = {index_space: dimension, **dict(dimensions or {})}
    events: list[DeltaContractionEvent] = []
    blocked: dict[Tensor, str] = {}

    def trace(delta: Tensor, path: str) -> Expr:
        problem = _identity_problem(delta)
        if problem:
            blocked[delta] = problem
            return delta
        first, second = delta.indices
        if index_key(first) != index_key(second):
            return delta
        value = dimensions.get(first.space)
        if value is None:
            blocked[delta] = f"No se declaró la dimensión del espacio {first.space}."
            return delta
        result = value if isinstance(value, Expr) else (Number(value) if isinstance(value, int) else Scalar(value))
        events.append(DeltaContractionEvent(path, "trace", delta,
                      reason=f"Traza de la identidad en el espacio {first.space}."))
        return result

    def flatten(expr: Expr) -> list[Expr]:
        if isinstance(expr, Mul):
            return [f for child in expr.factors for f in flatten(child)]
        return [expr]

    def reduce(expr: Expr, path: str) -> Expr:
        if not delta_count(expr, delta_name):
            return expr
        if isinstance(expr, Tensor):
            return trace(expr, path) if expr.name == delta_name else expr
        if isinstance(expr, Add):
            return add(*(reduce(t, f"{path}.terms[{i}]") for i, t in enumerate(expr.terms)))
        if isinstance(expr, Mul):
            # Flatten only after making local scalar scopes disjoint.
            hygienic = canonicalize_dummy_indices(expr)
            factors = [reduce(f, f"{path}.factors[{i}]") for i, f in enumerate(flatten(hygienic))]
            current = mul(*factors)
            if current == Number(0):
                return current
            while True:
                changed = False
                for pos, delta in enumerate(factors):
                    if not isinstance(delta, Tensor) or delta.name != delta_name:
                        continue
                    problem = _identity_problem(delta)
                    if problem:
                        blocked[delta] = problem
                        continue
                    traced = trace(delta, f"{path}.factors[{pos}]")
                    if traced != delta:
                        factors[pos] = traced
                        changed = True
                        break
                    # Prefer replacing the lower slot's partner, without relying
                    # on any fixed ordering of the two delta slots.
                    slots = sorted(delta.indices, key=lambda i: i.variance is Variance.UP)
                    for slot in slots:
                        other = next(i for i in delta.indices if i is not slot)
                        for target_pos, target in enumerate(factors):
                            if target_pos == pos:
                                continue
                            matches = [i for i in target.free_indices if index_key(i) == index_key(slot)
                                       and i.variance is not slot.variance]
                            if len(matches) != 1:
                                continue
                            source = matches[0]
                            replacement = Index(other.name, source.variance, source.space)
                            try:
                                renamed = rename_free_indices(target, {index_key(source): replacement.name})
                                candidate_factors = list(factors)
                                candidate_factors[target_pos] = renamed
                                candidate_factors.pop(pos)
                                candidate = mul(*candidate_factors)
                                if candidate != Number(0) and _signature(candidate) != _signature(current):
                                    raise ValueError("La sustitución cambiaría la firma libre del producto.")
                            except (TensorEngineError, ValueError) as error:
                                blocked[delta] = f"Sustitución no segura: {error}"
                                continue
                            events.append(DeltaContractionEvent(
                                f"{path}.factors[{target_pos}]", "substitute", delta, source, replacement,
                                "Sustitución higiénica en los índices libres del bloque; orden de derivadas preservado.",
                            ))
                            factors, current = candidate_factors, candidate
                            changed = True
                            break
                        if changed:
                            break
                    if changed:
                        break
                if not changed:
                    return mul(*factors)
                current = mul(*factors)
                if current == Number(0):
                    return current
        if isinstance(expr, CovariantDerivative):
            operand = reduce(expr.operand, path + ".operand")
            if operand == Number(0):
                return Number(0)
            return CovariantDerivative(expr.index, operand)
        if isinstance(expr, Variation):
            operand = reduce(expr.operand, path + ".operand")
            return Number(0) if isinstance(operand, Number) else Variation(operand)
        if isinstance(expr, Power):
            return Power(reduce(expr.base, path + ".base"), reduce(expr.exponent, path + ".exponent"))
        if isinstance(expr, (Function, FunctionDerivative)):
            args = tuple(reduce(a, f"{path}.arguments[{i}]") for i, a in enumerate(expr.arguments))
            return Function(expr.name, args) if isinstance(expr, Function) else FunctionDerivative(expr.name, expr.derivative_orders, args)
        return expr

    original_count = delta_count(expression, delta_name)
    invalid_reason = None
    try:
        signature = _signature(expression)
        result = expression
        for _ in range(original_count + 1):
            before = delta_count(result, delta_name)
            result = reduce(result, "root")
            if delta_count(result, delta_name) >= before:
                break
        if result != Number(0) and _signature(result) != signature:
            raise ValueError("La reducción no conserva los índices libres de la entrada.")
    except (TensorEngineError, ValueError, RecursionError) as error:
        result, events = expression, []
        invalid_reason = f"Contracción indeterminada; entrada conservada: {error}"
    for pos, node in enumerate(walk(result)):
        if isinstance(node, Tensor) and node.name == delta_name:
            events.append(DeltaContractionEvent(
                f"result.nodes[{pos}]", "retained", node,
                reason=invalid_reason or blocked.get(node) or _identity_problem(node)
                       or "Identidad explícita: no hay un índice contraíble en este ámbito.",
            ))
    after = delta_count(result, delta_name)
    status = "canonical" if after == 0 else ("partial" if after < original_count else "symbolic")
    return DeltaContractionResult(result, DeltaContractionAudit(
        expression_hash(expression), expression_hash(result), status, original_count, after, tuple(events),
    ))
