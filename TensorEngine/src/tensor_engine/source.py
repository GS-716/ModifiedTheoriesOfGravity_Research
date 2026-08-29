"""Frontend textual seguro para lagrangianos del dominio soportado."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping

from .builders import ModelBuilder
from .errors import SourceCompilationError
from .ir import Expr, Number, Scalar
from .model import DimensionSpec, FunctionSpec, GeometrySymbols, ModelSpec, ParameterSpec


SOURCE_SCHEMA_VERSION = "1.0"
_MAX_SOURCE_LENGTH = 10_000
_MAX_AST_NODES = 1_000
_RESERVED_METADATA = {"source_expression", "source_fingerprint", "source_schema_version"}


def _fingerprint(data: Mapping[str, Any]) -> str:
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


class _Compiler:
    def __init__(
        self,
        builder: ModelBuilder,
        parameters: tuple[ParameterSpec, ...],
        functions: tuple[FunctionSpec, ...],
        dimension: DimensionSpec,
        *,
        normalization: bool = False,
    ) -> None:
        self.builder = builder
        self.parameter_names = {item.name for item in parameters}
        self.function_arities = {item.name: item.arity for item in functions}
        self.dimension_name = str(dimension.value) if dimension.is_symbolic else None
        self.normalization = normalization

    def compile(self, source: str) -> Expr:
        if not source.strip():
            raise SourceCompilationError("La expresión declarativa está vacía.")
        if len(source) > _MAX_SOURCE_LENGTH:
            raise SourceCompilationError(
                f"La expresión supera el límite de {_MAX_SOURCE_LENGTH} caracteres."
            )
        try:
            tree = ast.parse(source, mode="eval")
        except SyntaxError as error:
            location = f"línea {error.lineno}, columna {error.offset}"
            raise SourceCompilationError(f"Sintaxis inválida en {location}: {error.msg}.") from error
        nodes = list(ast.walk(tree))
        if len(nodes) > _MAX_AST_NODES:
            raise SourceCompilationError(
                f"La expresión supera el límite de {_MAX_AST_NODES} nodos sintácticos."
            )
        return self._node(tree.body)

    @staticmethod
    def _location(node: ast.AST) -> str:
        return f"línea {getattr(node, 'lineno', 1)}, columna {getattr(node, 'col_offset', 0) + 1}"

    def _reject(self, node: ast.AST, message: str) -> SourceCompilationError:
        return SourceCompilationError(f"{message} ({self._location(node)}).")

    def _node(self, node: ast.AST) -> Expr:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int):
                if isinstance(node.value, float):
                    raise self._reject(node, "No se admiten decimales; use una fracción exacta como 1/2")
                raise self._reject(node, "Solo se admiten constantes enteras")
            return Number(node.value)

        if isinstance(node, ast.Name):
            name = node.id
            if name in self.parameter_names or name == self.dimension_name:
                return Scalar(name)
            if self.normalization:
                raise self._reject(
                    node,
                    f"El símbolo {name!r} no puede aparecer en la normalización",
                )
            if name == "R":
                return self.builder.ricci_scalar()
            if name == "X":
                return self.builder.kinetic_scalar()
            if name == "phi":
                return self.builder.phi
            if name in self.function_arities:
                raise self._reject(node, f"La función {name!r} debe escribirse con argumentos")
            raise self._reject(node, f"Símbolo no declarado {name!r}")

        if isinstance(node, ast.UnaryOp):
            operand = self._node(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return operand
            raise self._reject(node, "Operador unario no permitido")

        if isinstance(node, ast.BinOp):
            left = self._node(node.left)
            right = self._node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if isinstance(right, Number):
                    if right.numerator == 0:
                        raise self._reject(node.right, "División por cero")
                    return left * Number(right.denominator, right.numerator)
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
            if isinstance(node.op, ast.BitXor):
                raise self._reject(node, "Use ** para potencias; ^ no representa exponenciación")
            raise self._reject(node, "Operador binario no permitido")

        if isinstance(node, ast.Call):
            if self.normalization:
                raise self._reject(node, "La normalización no admite funciones")
            if not isinstance(node.func, ast.Name):
                raise self._reject(node, "Solo se permiten llamadas directas a funciones declaradas")
            if node.keywords:
                raise self._reject(node, "Las funciones no admiten argumentos con nombre")
            name = node.func.id
            arity = self.function_arities.get(name)
            if arity is None:
                raise self._reject(node, f"Función no declarada {name!r}")
            if len(node.args) != arity:
                raise self._reject(
                    node,
                    f"La función {name!r} espera {arity} argumentos y recibió {len(node.args)}",
                )
            return self.builder.function(name, *(self._node(item) for item in node.args))

        raise self._reject(node, f"Construcción {type(node).__name__} no permitida")


@dataclass(frozen=True, slots=True)
class LagrangianSourceSpec:
    name: str
    expression: str
    dimension: DimensionSpec = field(default_factory=lambda: DimensionSpec(4))
    normalization: str = "1"
    parameters: tuple[ParameterSpec, ...] = ()
    functions: tuple[FunctionSpec, ...] = ()
    assumptions: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()
    symbols: GeometrySymbols = field(default_factory=GeometrySymbols)
    schema_version: str = SOURCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "functions", tuple(self.functions))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "metadata", tuple(tuple(item) for item in self.metadata))
        if self.schema_version != SOURCE_SCHEMA_VERSION:
            raise SourceCompilationError(
                f"Versión de fuente no soportada: {self.schema_version!r}."
            )
        if any(len(item) != 2 for item in self.metadata):
            raise SourceCompilationError("La metadata de fuente debe contener pares clave/valor.")
        declared_names = {item.name for item in self.parameters}.union(
            item.name for item in self.functions
        )
        alias_collisions = {"R", "X", "phi"}.intersection(declared_names)
        if alias_collisions:
            raise SourceCompilationError(
                "Nombres reservados por la gramática: " + ", ".join(sorted(alias_collisions))
            )
        collisions = _RESERVED_METADATA.intersection(key for key, _ in self.metadata)
        if collisions:
            raise SourceCompilationError(
                "Metadata reservada por el compilador: " + ", ".join(sorted(collisions))
            )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_data())

    def compile(self) -> ModelSpec:
        builder = ModelBuilder(self.symbols)
        lagrangian = _Compiler(
            builder,
            self.parameters,
            self.functions,
            self.dimension,
        ).compile(self.expression)
        normalization = _Compiler(
            builder,
            self.parameters,
            (),
            self.dimension,
            normalization=True,
        ).compile(self.normalization)
        metadata = self.metadata + (
            ("source_expression", self.expression),
            ("source_fingerprint", self.fingerprint),
            ("source_schema_version", self.schema_version),
        )
        return ModelSpec(
            self.name,
            lagrangian,
            dimension=self.dimension,
            normalization=normalization,
            parameters=self.parameters,
            functions=self.functions,
            assumptions=self.assumptions,
            metadata=metadata,
            symbols=self.symbols,
        )

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "expression": self.expression,
            "normalization": self.normalization,
            "dimension": self.dimension.to_data(),
            "parameters": [item.to_data() for item in self.parameters],
            "functions": [item.to_data() for item in self.functions],
            "assumptions": list(self.assumptions),
            "metadata": {key: value for key, value in self.metadata},
            "symbols": self.symbols.to_data(),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "LagrangianSourceSpec":
        dimension = data.get("dimension", {"value": 4})
        return cls(
            name=str(data["name"]),
            expression=str(data["expression"]),
            dimension=DimensionSpec(dimension.get("value", 4)),
            normalization=str(data.get("normalization", "1")),
            parameters=tuple(ParameterSpec.from_data(item) for item in data.get("parameters", ())),
            functions=tuple(FunctionSpec.from_data(item) for item in data.get("functions", ())),
            assumptions=tuple(str(item) for item in data.get("assumptions", ())),
            metadata=tuple((str(key), str(value)) for key, value in data.get("metadata", {}).items()),
            symbols=GeometrySymbols.from_data(data.get("symbols", GeometrySymbols().to_data())),
            schema_version=str(data.get("schema_version", SOURCE_SCHEMA_VERSION)),
        )
