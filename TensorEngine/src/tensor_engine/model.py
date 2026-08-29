"""Especificación declarativa de un modelo L(g,R,phi,nabla phi)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Mapping

from .errors import ModelValidationError
from .ir import (
    CovariantDerivative,
    Expr,
    Function,
    FunctionDerivative,
    Number,
    Scalar,
    Tensor,
    Variance,
    Variation,
    VolumeElement,
    expr_from_data,
    infer_free_indices,
    walk,
)


MODEL_SCHEMA_VERSION = "1.0"
CONVENTION_ID = "tensor-engine.phase0.v1"
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _name(value: str, label: str) -> None:
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        raise ModelValidationError(f"{label} inválido: {value!r}")


@dataclass(frozen=True, slots=True)
class DimensionSpec:
    """Dimensión abstracta (`D`) o concreta (entero mayor o igual que dos)."""

    value: int | str = "D"

    def __post_init__(self) -> None:
        if isinstance(self.value, bool):
            raise ModelValidationError("La dimensión no puede ser booleana.")
        if isinstance(self.value, int):
            if self.value < 2:
                raise ModelValidationError("La dimensión debe satisfacer D >= 2.")
        elif isinstance(self.value, str):
            _name(self.value, "símbolo de dimensión")
        else:
            raise ModelValidationError("La dimensión debe ser un entero o un símbolo.")

    @property
    def is_symbolic(self) -> bool:
        return isinstance(self.value, str)

    def to_data(self) -> dict[str, Any]:
        return {"value": self.value, "kind": "symbolic" if self.is_symbolic else "concrete"}


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    name: str
    assumptions: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        _name(self.name, "parámetro")
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        if len(set(self.assumptions)) != len(self.assumptions):
            raise ModelValidationError(f"El parámetro {self.name!r} repite hipótesis.")

    def to_data(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "assumptions": list(self.assumptions),
            "description": self.description,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ParameterSpec":
        return cls(
            str(data["name"]),
            tuple(str(item) for item in data.get("assumptions", ())),
            str(data.get("description", "")),
        )


@dataclass(frozen=True, slots=True)
class FunctionSpec:
    name: str
    arity: int = 1
    description: str = ""

    def __post_init__(self) -> None:
        _name(self.name, "función")
        if isinstance(self.arity, bool) or not isinstance(self.arity, int) or self.arity < 1:
            raise ModelValidationError("La aridad debe ser un entero positivo.")

    def to_data(self) -> dict[str, Any]:
        return {"name": self.name, "arity": self.arity, "description": self.description}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "FunctionSpec":
        return cls(str(data["name"]), int(data.get("arity", 1)), str(data.get("description", "")))


@dataclass(frozen=True, slots=True)
class GeometrySymbols:
    """Nombres semánticos reservados por la geometría inicial."""

    index_space: str = "M"
    metric: str = "g"
    curvature: str = "Riemann"
    scalar: str = "phi"
    scalar_gradient: str = "u"

    def __post_init__(self) -> None:
        values = (
            self.index_space,
            self.metric,
            self.curvature,
            self.scalar,
            self.scalar_gradient,
        )
        for value in values:
            _name(value, "símbolo geométrico")
        if len(set(values[1:])) != len(values[1:]):
            raise ModelValidationError("Los símbolos geométricos deben tener nombres distintos.")

    def to_data(self) -> dict[str, str]:
        return {
            "index_space": self.index_space,
            "metric": self.metric,
            "curvature": self.curvature,
            "scalar": self.scalar,
            "scalar_gradient": self.scalar_gradient,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "GeometrySymbols":
        return cls(**{key: str(value) for key, value in data.items()})


class TensorSymmetry(str, Enum):
    NONE = "none"
    SYMMETRIC = "symmetric"
    RIEMANN = "riemann"


@dataclass(frozen=True, slots=True)
class TensorDeclaration:
    """Firma y simetría semántica de una cabeza tensorial."""

    name: str
    slots: tuple[Variance, ...]
    symmetry: TensorSymmetry = TensorSymmetry.NONE

    def __post_init__(self) -> None:
        _name(self.name, "tensor declarado")
        object.__setattr__(self, "slots", tuple(self.slots))
        if not self.slots:
            raise ModelValidationError("Un tensor declarado debe tener rango positivo.")


@dataclass(frozen=True, slots=True)
class ConventionSpec:
    """Referencia inmutable a las convenciones normativas de fase 0."""

    convention_id: str = CONVENTION_ID
    signature: str = "mostly_plus"
    metric_variation: str = "inverse_metric"
    curvature_argument: str = "riemann_all_down"
    connection: str = "levi_civita"

    def validate(self) -> None:
        expected = ConventionSpec()
        if self != expected:
            raise ModelValidationError(
                "La primera versión solo admite las convenciones "
                f"{CONVENTION_ID}; se recibió {self!r}."
            )

    def to_data(self) -> dict[str, str]:
        return {
            "convention_id": self.convention_id,
            "signature": self.signature,
            "metric_variation": self.metric_variation,
            "curvature_argument": self.curvature_argument,
            "connection": self.connection,
        }


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Entrada canónica y validada de una corrida tensorial."""

    name: str
    lagrangian: Expr
    dimension: DimensionSpec = field(default_factory=DimensionSpec)
    normalization: Expr = field(default_factory=lambda: Number(1))
    parameters: tuple[ParameterSpec, ...] = ()
    functions: tuple[FunctionSpec, ...] = ()
    assumptions: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()
    symbols: GeometrySymbols = field(default_factory=GeometrySymbols)
    conventions: ConventionSpec = field(default_factory=ConventionSpec)
    schema_version: str = MODEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _name(self.name, "nombre del modelo")
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "functions", tuple(self.functions))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "metadata", tuple(tuple(item) for item in self.metadata))
        self.validate()

    def validate(self) -> "ModelSpec":
        if self.schema_version != MODEL_SCHEMA_VERSION:
            raise ModelValidationError(
                f"Versión de ModelSpec no soportada: {self.schema_version!r}."
            )
        self.conventions.validate()
        if len(set(self.assumptions)) != len(self.assumptions):
            raise ModelValidationError("El modelo contiene hipótesis repetidas.")
        if any(len(item) != 2 for item in self.metadata):
            raise ModelValidationError("Cada entrada de metadata debe ser un par clave/valor.")
        metadata_keys = [item[0] for item in self.metadata]
        if len(metadata_keys) != len(set(metadata_keys)):
            raise ModelValidationError("Las claves de metadata deben ser únicas.")

        parameter_names = [item.name for item in self.parameters]
        function_names = [item.name for item in self.functions]
        if len(parameter_names) != len(set(parameter_names)):
            raise ModelValidationError("Los parámetros deben tener nombres únicos.")
        if len(function_names) != len(set(function_names)):
            raise ModelValidationError("Las funciones deben tener nombres únicos.")

        reserved = {
            self.symbols.index_space,
            self.symbols.metric,
            self.symbols.curvature,
            self.symbols.scalar,
            self.symbols.scalar_gradient,
        }
        if self.dimension.is_symbolic:
            reserved.add(str(self.dimension.value))
        collisions = reserved.intersection(parameter_names + function_names)
        if collisions:
            raise ModelValidationError(
                "Nombres reservados usados por el modelo: " + ", ".join(sorted(collisions))
            )
        overlap = set(parameter_names).intersection(function_names)
        if overlap:
            raise ModelValidationError(
                "Un nombre no puede ser parámetro y función: " + ", ".join(sorted(overlap))
            )

        self._validate_expression(self.lagrangian, "lagrangiano", allow_derivatives=False)
        self._validate_expression(self.normalization, "normalización", allow_derivatives=False)
        if infer_free_indices(self.lagrangian):
            raise ModelValidationError("El lagrangiano debe ser un escalar completamente contraído.")
        if infer_free_indices(self.normalization):
            raise ModelValidationError("La normalización global debe ser escalar.")
        allowed_normalization_scalars = set(parameter_names)
        if self.dimension.is_symbolic:
            allowed_normalization_scalars.add(str(self.dimension.value))
        for node in walk(self.normalization):
            if isinstance(node, (Tensor, Function, CovariantDerivative)):
                raise ModelValidationError(
                    "La normalización global solo puede depender de parámetros y de la dimensión."
                )
            if isinstance(node, Scalar) and node.name not in allowed_normalization_scalars:
                raise ModelValidationError(
                    f"El escalar {node.name!r} no es válido en la normalización global."
                )
        return self

    @property
    def tensor_declarations(self) -> tuple[TensorDeclaration, ...]:
        return (
            TensorDeclaration(
                self.symbols.metric,
                (Variance.UP, Variance.UP),
                TensorSymmetry.SYMMETRIC,
            ),
            TensorDeclaration(
                self.symbols.curvature,
                (Variance.DOWN,) * 4,
                TensorSymmetry.RIEMANN,
            ),
            TensorDeclaration(
                self.symbols.scalar_gradient,
                (Variance.DOWN,),
                TensorSymmetry.NONE,
            ),
        )

    def _validate_expression(self, expr: Expr, label: str, allow_derivatives: bool) -> None:
        allowed_scalars = {self.symbols.scalar, *[item.name for item in self.parameters]}
        if self.dimension.is_symbolic:
            allowed_scalars.add(str(self.dimension.value))
        function_arities = {item.name: item.arity for item in self.functions}

        declarations = {item.name: item for item in self.tensor_declarations}
        for node in walk(expr):
            if isinstance(node, Scalar) and node.name not in allowed_scalars:
                raise ModelValidationError(
                    f"Escalar no declarado {node.name!r} en {label}."
                )
            if isinstance(node, Tensor):
                declaration = declarations.get(node.name)
                if declaration is None:
                    raise ModelValidationError(f"Tensor no declarado {node.name!r} en {label}.")
                expected = declaration.slots
                actual = tuple(index.variance for index in node.indices)
                if actual != expected:
                    raise ModelValidationError(
                        f"{node.name} espera varianzas {[item.value for item in expected]}, "
                        f"pero recibió {[item.value for item in actual]}."
                    )
                if any(index.space != self.symbols.index_space for index in node.indices):
                    raise ModelValidationError(
                        f"Todos los índices iniciales deben pertenecer a {self.symbols.index_space}."
                    )
            if isinstance(node, Function):
                if node.name not in function_arities:
                    raise ModelValidationError(f"Función no declarada {node.name!r} en {label}.")
                if len(node.arguments) != function_arities[node.name]:
                    raise ModelValidationError(
                        f"{node.name} espera {function_arities[node.name]} argumentos."
                    )
            if isinstance(node, FunctionDerivative):
                raise ModelValidationError(
                    "Las derivadas formales de funciones son objetos calculados, no input del modelo."
                )
            if isinstance(node, (Variation, VolumeElement)):
                raise ModelValidationError(
                    "Las variaciones y la densidad de volumen son objetos calculados, no input de L."
                )
            if isinstance(node, CovariantDerivative) and not allow_derivatives:
                raise ModelValidationError(
                    "El lagrangiano inicial usa u_a como variable independiente; "
                    "no admite nodos explícitos de derivada covariante."
                )

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "dimension": self.dimension.to_data(),
            "normalization": self.normalization.to_data(),
            "lagrangian": self.lagrangian.to_data(),
            "parameters": [item.to_data() for item in self.parameters],
            "functions": [item.to_data() for item in self.functions],
            "assumptions": list(self.assumptions),
            "metadata": {key: value for key, value in self.metadata},
            "symbols": self.symbols.to_data(),
            "conventions": self.conventions.to_data(),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ModelSpec":
        dimension_data = data.get("dimension", {"value": "D"})
        conventions_data = data.get("conventions", {})
        return cls(
            name=str(data["name"]),
            lagrangian=expr_from_data(data["lagrangian"]),
            dimension=DimensionSpec(dimension_data.get("value", "D")),
            normalization=expr_from_data(data.get("normalization", Number(1).to_data())),
            parameters=tuple(ParameterSpec.from_data(item) for item in data.get("parameters", ())),
            functions=tuple(FunctionSpec.from_data(item) for item in data.get("functions", ())),
            assumptions=tuple(str(item) for item in data.get("assumptions", ())),
            metadata=tuple((str(key), str(value)) for key, value in data.get("metadata", {}).items()),
            symbols=GeometrySymbols.from_data(data.get("symbols", GeometrySymbols().to_data())),
            conventions=ConventionSpec(**{key: str(value) for key, value in conventions_data.items()}),
            schema_version=str(data.get("schema_version", MODEL_SCHEMA_VERSION)),
        )
