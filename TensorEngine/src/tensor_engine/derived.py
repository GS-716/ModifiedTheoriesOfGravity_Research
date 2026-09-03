"""Cantidades geométricas intermedias reutilizables y su trazabilidad."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .backends.base import TensorBackend
from .builders import ModelBuilder
from .components import (
    AnsatzSpecialization,
    ComponentEvaluation,
    ComponentFieldEquations,
    GeometryAnsatz,
    SympyComponentBackend,
)
from .contracts import VerificationStatus
from .errors import TensorEngineError
from .euler import EulerLagrangeResult
from .ir import Expr, Index, Number, Tensor, Variance, expr_from_data
from .model import GeometrySymbols, ModelSpec
from .variational import LagrangianMomenta
from .verification import VerificationReport


DERIVED_QUANTITY_KEYS = (
    "ricci_scalar",
    "ricci_squared",
    "riemann_tensor",
    "riemann_squared",
    "nabla_P",
    "nabla_nabla_P",
    "curvature_derivative_metric_term",
)

REPORT_QUANTITY_KEYS = (
    "lagrangian",
    "metric_momentum",
    "curvature_momentum",
    "scalar_gradient_momentum",
    "scalar_derivative",
    "metric_euler",
    "scalar_euler",
    "ricci_scalar",
    "ricci_squared",
    "riemann_tensor",
    "riemann_squared",
    "nabla_P",
    "nabla_nabla_P",
)


class SymbolicEvaluationStatus(str, Enum):
    CALCULATED = "calculated_symbolically"
    GEOMETRIC_INPUT = "geometric_input"


class ComponentProjectionStatus(str, Enum):
    PROJECTED = "projected_with_ansatz"
    NOT_REQUESTED = "not_requested"
    BACKEND_LIMITATION = "backend_limitation"


class XActValidationStatus(str, Enum):
    VALIDATED = "validated_with_xact"
    NOT_VALIDATED = "not_validated"
    NOT_REQUESTED = "not_requested"


class ProjectionStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    SYMBOLIC = "symbolic"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class AbstractQuantityRecord:
    key: str
    free_indices: tuple[Index, ...]
    description: str
    source_keys: tuple[str, ...]
    xact_status: XActValidationStatus
    xact_checks: tuple[str, ...] = ()
    validation_note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "free_indices", tuple(self.free_indices))
        object.__setattr__(self, "source_keys", tuple(self.source_keys))
        object.__setattr__(self, "xact_checks", tuple(self.xact_checks))
        if self.key not in REPORT_QUANTITY_KEYS:
            raise ValueError(f"Cantidad abstracta desconocida: {self.key!r}.")

    def to_data(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "free_indices": [item.to_data() for item in self.free_indices],
            "description": self.description,
            "source_keys": list(self.source_keys),
            "xact_status": self.xact_status.value,
            "xact_checks": list(self.xact_checks),
            "validation_note": self.validation_note,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "AbstractQuantityRecord":
        return cls(
            key=str(data["key"]),
            free_indices=tuple(Index.from_data(item) for item in data.get("free_indices", ())),
            description=str(data.get("description", "")),
            source_keys=tuple(str(item) for item in data.get("source_keys", ())),
            xact_status=XActValidationStatus(data["xact_status"]),
            xact_checks=tuple(str(item) for item in data.get("xact_checks", ())),
            validation_note=str(data.get("validation_note", "")),
        )


@dataclass(frozen=True, slots=True)
class AbstractTensorResults:
    """Vista covariante que referencia resultados ya calculados por el pipeline."""

    lagrangian: Expr
    metric_momentum: Expr
    curvature_momentum: Expr
    scalar_gradient_momentum: Expr
    scalar_derivative: Expr
    metric_euler: Expr
    scalar_euler: Expr
    ricci_scalar: Expr
    ricci_squared: Expr
    riemann_tensor: Expr
    riemann_squared: Expr
    nabla_P: Expr
    nabla_nabla_P: Expr
    records: tuple[AbstractQuantityRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        keys = tuple(item.key for item in self.records)
        if len(keys) != len(set(keys)) or set(keys) != set(REPORT_QUANTITY_KEYS):
            raise ValueError(
                f"La vista abstracta debe describir las {len(REPORT_QUANTITY_KEYS)} cantidades."
            )

    @property
    def L(self) -> Expr:
        return self.lagrangian

    def expression_items(self) -> tuple[tuple[str, Expr], ...]:
        return tuple((key, getattr(self, key)) for key in REPORT_QUANTITY_KEYS)

    def record(self, key: str) -> AbstractQuantityRecord:
        return next(item for item in self.records if item.key == key)

    def to_data(self) -> dict[str, Any]:
        return {
            "expressions": {
                key: expression.to_data() for key, expression in self.expression_items()
            },
            "records": [item.to_data() for item in self.records],
        }

    @classmethod
    def from_data(
        cls,
        data: Mapping[str, Any],
        *,
        symbols: GeometrySymbols | None = None,
    ) -> "AbstractTensorResults":
        expressions = dict(data["expressions"])
        records = [AbstractQuantityRecord.from_data(item) for item in data["records"]]
        missing = {"ricci_squared", "riemann_squared"} - set(expressions)
        if missing:
            if symbols is None:
                raise ValueError(
                    "La vista abstracta antigua requiere GeometrySymbols para reconstruir "
                    "los invariantes cuadráticos."
                )
            builder = ModelBuilder(symbols)
            generated = {
                "ricci_squared": builder.ricci_squared(),
                "riemann_squared": builder.riemann_squared(),
            }
            xact_requested = any(
                item.xact_status is not XActValidationStatus.NOT_REQUESTED
                for item in records
            )
            for key in REPORT_QUANTITY_KEYS:
                if key not in missing:
                    continue
                expressions[key] = generated[key].to_data()
                records.append(
                    AbstractQuantityRecord(
                        key=key,
                        free_indices=(),
                        description=_ABSTRACT_DESCRIPTIONS[key],
                        source_keys=_ABSTRACT_SOURCES[key],
                        xact_status=(
                            XActValidationStatus.NOT_VALIDATED
                            if xact_requested
                            else XActValidationStatus.NOT_REQUESTED
                        ),
                        validation_note=(
                            "Reconstruida desde un bundle anterior; no existe una "
                            "validación xAct independiente almacenada."
                        ),
                    )
                )
        return cls(
            **{
                key: expr_from_data(expressions[key])
                for key in REPORT_QUANTITY_KEYS
            },
            records=tuple(records),
        )


@dataclass(frozen=True, slots=True)
class ProjectedQuantityResult:
    key: str
    status: ProjectionStatus
    ansatz_name: str | None
    components: ComponentEvaluation | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.key not in REPORT_QUANTITY_KEYS:
            raise ValueError(f"Cantidad proyectada desconocida: {self.key!r}.")
        if self.status in {ProjectionStatus.COMPLETED, ProjectionStatus.PARTIAL}:
            if self.components is None:
                raise ValueError("Una proyección completada o parcial requiere componentes.")
        elif self.components is not None:
            raise ValueError("Una proyección simbólica o no disponible no contiene componentes.")

    @property
    def scalar(self) -> Expr:
        if self.components is None:
            raise ValueError("La cantidad no tiene una proyección escalar disponible.")
        return self.components.scalar

    def to_data(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "status": self.status.value,
            "ansatz": self.ansatz_name,
            "components": None if self.components is None else self.components.to_data(),
            "reason": self.reason,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ProjectedQuantityResult":
        component_data = data.get("components")
        return cls(
            key=str(data["key"]),
            status=ProjectionStatus(data["status"]),
            ansatz_name=None if data.get("ansatz") is None else str(data["ansatz"]),
            components=(
                None
                if component_data is None
                else ComponentEvaluation.from_data(component_data)
            ),
            reason=str(data.get("reason", "")),
        )


@dataclass(frozen=True, slots=True)
class ProjectedTensorResults:
    """Vista coordenada tolerante, ligada al ansatz exacto de la corrida."""

    ansatz_name: str | None
    quantities: tuple[ProjectedQuantityResult, ...]
    ansatz_geometry: GeometryAnsatz | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantities", tuple(self.quantities))
        keys = tuple(item.key for item in self.quantities)
        if len(keys) != len(set(keys)) or set(keys) != set(REPORT_QUANTITY_KEYS):
            raise ValueError(
                f"La vista proyectada debe contener las {len(REPORT_QUANTITY_KEYS)} cantidades."
            )
        if any(item.ansatz_name != self.ansatz_name for item in self.quantities):
            raise ValueError("Todas las proyecciones deben pertenecer al mismo ansatz.")
        if (
            self.ansatz_geometry is not None
            and self.ansatz_geometry.name != self.ansatz_name
        ):
            raise ValueError(
                "La geometría serializada debe coincidir con el nombre del ansatz proyectado."
            )

    def quantity(self, key: str) -> ProjectedQuantityResult:
        return next(item for item in self.quantities if item.key == key)

    @property
    def lagrangian(self) -> ProjectedQuantityResult:
        return self.quantity("lagrangian")

    @property
    def metric_momentum(self) -> ProjectedQuantityResult:
        return self.quantity("metric_momentum")

    @property
    def curvature_momentum(self) -> ProjectedQuantityResult:
        return self.quantity("curvature_momentum")

    @property
    def scalar_gradient_momentum(self) -> ProjectedQuantityResult:
        return self.quantity("scalar_gradient_momentum")

    @property
    def scalar_derivative(self) -> ProjectedQuantityResult:
        return self.quantity("scalar_derivative")

    @property
    def metric_euler(self) -> ProjectedQuantityResult:
        return self.quantity("metric_euler")

    @property
    def scalar_euler(self) -> ProjectedQuantityResult:
        return self.quantity("scalar_euler")

    @property
    def ricci_scalar(self) -> ProjectedQuantityResult:
        return self.quantity("ricci_scalar")

    @property
    def ricci_squared(self) -> ProjectedQuantityResult:
        return self.quantity("ricci_squared")

    @property
    def riemann_tensor(self) -> ProjectedQuantityResult:
        return self.quantity("riemann_tensor")

    @property
    def riemann_squared(self) -> ProjectedQuantityResult:
        return self.quantity("riemann_squared")

    @property
    def nabla_P(self) -> ProjectedQuantityResult:
        return self.quantity("nabla_P")

    @property
    def nabla_nabla_P(self) -> ProjectedQuantityResult:
        return self.quantity("nabla_nabla_P")

    def to_data(self) -> dict[str, Any]:
        return {
            "ansatz": self.ansatz_name,
            "ansatz_geometry": (
                None
                if self.ansatz_geometry is None
                else self.ansatz_geometry.to_data()
            ),
            "quantities": [item.to_data() for item in self.quantities],
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ProjectedTensorResults":
        ansatz_name = None if data.get("ansatz") is None else str(data["ansatz"])
        ansatz_data = data.get("ansatz_geometry")
        quantities = [
            ProjectedQuantityResult.from_data(item) for item in data["quantities"]
        ]
        present = {item.key for item in quantities}
        for key in ("ricci_squared", "riemann_squared"):
            if key not in present:
                quantities.append(
                    ProjectedQuantityResult(
                        key=key,
                        status=ProjectionStatus.SYMBOLIC,
                        ansatz_name=ansatz_name,
                        reason=(
                            "El bundle fue creado antes de incorporar esta proyección; "
                            "se conserva la expresión abstracta reconstruida."
                        ),
                    )
                )
        return cls(
            ansatz_name=ansatz_name,
            quantities=tuple(quantities),
            ansatz_geometry=(
                None
                if ansatz_data is None
                else GeometryAnsatz.from_data(ansatz_data)
            ),
        )


@dataclass(frozen=True, slots=True)
class SpecializedTensorResults:
    """Tercera vista obtenida al evaluar la IR sobre una especialización posterior."""

    base_ansatz_name: str
    specialization: AnsatzSpecialization
    ansatz_geometry: GeometryAnsatz
    quantities: tuple[ProjectedQuantityResult, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantities", tuple(self.quantities))
        keys = tuple(item.key for item in self.quantities)
        if len(keys) != len(set(keys)) or set(keys) != set(REPORT_QUANTITY_KEYS):
            raise ValueError(
                f"La vista especializada debe contener las {len(REPORT_QUANTITY_KEYS)} cantidades."
            )
        if any(item.ansatz_name != self.ansatz_geometry.name for item in self.quantities):
            raise ValueError("Las cantidades especializadas pertenecen a otro ansatz.")

    @property
    def ansatz_name(self) -> str:
        return self.ansatz_geometry.name

    def quantity(self, key: str) -> ProjectedQuantityResult:
        return next(item for item in self.quantities if item.key == key)

    @property
    def lagrangian(self) -> ProjectedQuantityResult:
        return self.quantity("lagrangian")

    @property
    def metric_momentum(self) -> ProjectedQuantityResult:
        return self.quantity("metric_momentum")

    @property
    def curvature_momentum(self) -> ProjectedQuantityResult:
        return self.quantity("curvature_momentum")

    @property
    def scalar_gradient_momentum(self) -> ProjectedQuantityResult:
        return self.quantity("scalar_gradient_momentum")

    @property
    def scalar_derivative(self) -> ProjectedQuantityResult:
        return self.quantity("scalar_derivative")

    @property
    def metric_euler(self) -> ProjectedQuantityResult:
        return self.quantity("metric_euler")

    @property
    def scalar_euler(self) -> ProjectedQuantityResult:
        return self.quantity("scalar_euler")

    @property
    def ricci_scalar(self) -> ProjectedQuantityResult:
        return self.quantity("ricci_scalar")

    @property
    def ricci_squared(self) -> ProjectedQuantityResult:
        return self.quantity("ricci_squared")

    @property
    def riemann_tensor(self) -> ProjectedQuantityResult:
        return self.quantity("riemann_tensor")

    @property
    def riemann_squared(self) -> ProjectedQuantityResult:
        return self.quantity("riemann_squared")

    @property
    def nabla_P(self) -> ProjectedQuantityResult:
        return self.quantity("nabla_P")

    @property
    def nabla_nabla_P(self) -> ProjectedQuantityResult:
        return self.quantity("nabla_nabla_P")

    def to_data(self) -> dict[str, Any]:
        return {
            "base_ansatz": self.base_ansatz_name,
            "specialization": self.specialization.to_data(),
            "ansatz_geometry": self.ansatz_geometry.to_data(),
            "quantities": [item.to_data() for item in self.quantities],
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "SpecializedTensorResults":
        return cls(
            base_ansatz_name=str(data["base_ansatz"]),
            specialization=AnsatzSpecialization.from_data(data["specialization"]),
            ansatz_geometry=GeometryAnsatz.from_data(data["ansatz_geometry"]),
            quantities=tuple(
                ProjectedQuantityResult.from_data(item)
                for item in data["quantities"]
            ),
        )


@dataclass(frozen=True, slots=True)
class DerivedQuantityRecord:
    """Estado de cálculo, proyección y validación de una cantidad derivada."""

    key: str
    free_indices: tuple[Index, ...]
    symbolic_status: SymbolicEvaluationStatus
    component_status: ComponentProjectionStatus
    xact_status: XActValidationStatus
    components: ComponentEvaluation | None = None
    source_keys: tuple[str, ...] = ()
    contributes_to: tuple[str, ...] = ()
    note: str = ""
    xact_check: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "free_indices", tuple(self.free_indices))
        object.__setattr__(self, "source_keys", tuple(self.source_keys))
        object.__setattr__(self, "contributes_to", tuple(self.contributes_to))
        if self.key not in DERIVED_QUANTITY_KEYS:
            raise ValueError(f"Cantidad derivada desconocida: {self.key!r}.")
        if self.component_status is ComponentProjectionStatus.PROJECTED:
            if self.components is None:
                raise ValueError("Una proyección marcada como calculada debe conservar componentes.")
        elif self.components is not None:
            raise ValueError("Solo una proyección calculada puede contener componentes.")
        if self.xact_status is XActValidationStatus.VALIDATED and not self.xact_check:
            raise ValueError("Una validación xAct debe identificar la comprobación usada.")

    def to_data(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "free_indices": [item.to_data() for item in self.free_indices],
            "symbolic_status": self.symbolic_status.value,
            "component_status": self.component_status.value,
            "xact_status": self.xact_status.value,
            "components": None if self.components is None else self.components.to_data(),
            "source_keys": list(self.source_keys),
            "contributes_to": list(self.contributes_to),
            "note": self.note,
            "xact_check": self.xact_check,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "DerivedQuantityRecord":
        component_data = data.get("components")
        return cls(
            key=str(data["key"]),
            free_indices=tuple(Index.from_data(item) for item in data.get("free_indices", ())),
            symbolic_status=SymbolicEvaluationStatus(data["symbolic_status"]),
            component_status=ComponentProjectionStatus(data["component_status"]),
            xact_status=XActValidationStatus(data["xact_status"]),
            components=(
                None
                if component_data is None
                else ComponentEvaluation.from_data(component_data)
            ),
            source_keys=tuple(str(item) for item in data.get("source_keys", ())),
            contributes_to=tuple(str(item) for item in data.get("contributes_to", ())),
            note=str(data.get("note", "")),
            xact_check=(
                None if data.get("xact_check") is None else str(data["xact_check"])
            ),
        )


@dataclass(frozen=True, slots=True)
class DerivedQuantities:
    """Expresiones de primera clase accesibles como ``run.derived.<nombre>``."""

    ricci_scalar: Expr
    ricci_squared: Expr
    riemann_tensor: Expr
    riemann_squared: Expr
    nabla_P: Expr
    nabla_nabla_P: Expr
    curvature_derivative_metric_term: Expr
    records: tuple[DerivedQuantityRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        keys = tuple(item.key for item in self.records)
        if len(keys) != len(set(keys)) or set(keys) != set(DERIVED_QUANTITY_KEYS):
            raise ValueError(
                "DerivedQuantities debe contener exactamente un estado por cantidad."
            )

    def expression_items(self) -> tuple[tuple[str, Expr], ...]:
        return tuple((key, getattr(self, key)) for key in DERIVED_QUANTITY_KEYS)

    def record(self, key: str) -> DerivedQuantityRecord:
        return next(item for item in self.records if item.key == key)

    def to_data(self) -> dict[str, Any]:
        return {
            "expressions": {
                key: expression.to_data()
                for key, expression in self.expression_items()
            },
            "records": [item.to_data() for item in self.records],
        }

    @classmethod
    def from_data(
        cls,
        data: Mapping[str, Any],
        *,
        symbols: GeometrySymbols | None = None,
    ) -> "DerivedQuantities":
        expressions = dict(data["expressions"])
        records = [DerivedQuantityRecord.from_data(item) for item in data["records"]]
        missing = {"ricci_squared", "riemann_squared"} - set(expressions)
        if missing:
            if symbols is None:
                raise ValueError(
                    "Las cantidades derivadas antiguas requieren GeometrySymbols para "
                    "reconstruir los invariantes cuadráticos."
                )
            builder = ModelBuilder(symbols)
            generated = {
                "ricci_squared": builder.ricci_squared(),
                "riemann_squared": builder.riemann_squared(),
            }
            projected_before = any(
                item.component_status is not ComponentProjectionStatus.NOT_REQUESTED
                for item in records
            )
            xact_requested = any(
                item.xact_status is not XActValidationStatus.NOT_REQUESTED
                for item in records
            )
            for key in DERIVED_QUANTITY_KEYS:
                if key not in missing:
                    continue
                expressions[key] = generated[key].to_data()
                records.append(
                    DerivedQuantityRecord(
                        key=key,
                        free_indices=(),
                        symbolic_status=SymbolicEvaluationStatus.CALCULATED,
                        component_status=(
                            ComponentProjectionStatus.BACKEND_LIMITATION
                            if projected_before
                            else ComponentProjectionStatus.NOT_REQUESTED
                        ),
                        xact_status=(
                            XActValidationStatus.NOT_VALIDATED
                            if xact_requested
                            else XActValidationStatus.NOT_REQUESTED
                        ),
                        source_keys=("validated_model", "riemann_tensor"),
                        note=(
                            "Reconstruida desde un bundle anterior. La proyección y la "
                            "validación de esta cantidad no estaban almacenadas."
                        ),
                    )
                )
        return cls(
            ricci_scalar=expr_from_data(expressions["ricci_scalar"]),
            ricci_squared=expr_from_data(expressions["ricci_squared"]),
            riemann_tensor=expr_from_data(expressions["riemann_tensor"]),
            riemann_squared=expr_from_data(expressions["riemann_squared"]),
            nabla_P=expr_from_data(expressions["nabla_P"]),
            nabla_nabla_P=expr_from_data(expressions["nabla_nabla_P"]),
            curvature_derivative_metric_term=expr_from_data(
                expressions["curvature_derivative_metric_term"]
            ),
            records=tuple(records),
        )


def _project(
    expression: Expr,
    free_indices: tuple[Index, ...],
    component_backend: SympyComponentBackend | None,
    *,
    component_budget: int,
    unavailable_reason: str | None = None,
) -> tuple[ComponentProjectionStatus, ComponentEvaluation | None, str]:
    if component_backend is None:
        if unavailable_reason is not None:
            return (
                ComponentProjectionStatus.BACKEND_LIMITATION,
                None,
                unavailable_reason,
            )
        return (
            ComponentProjectionStatus.NOT_REQUESTED,
            None,
            "No se proporcionó un ansatz para esta corrida.",
        )
    dimension = component_backend.geometry.dimension
    limit_reason = component_backend.projection_limit_reason(expression)
    if limit_reason is not None:
        return (
            ComponentProjectionStatus.BACKEND_LIMITATION,
            None,
            limit_reason,
        )
    potential_components = dimension ** len(free_indices)
    if isinstance(expression, Number) and expression.value == 0:
        return (
            ComponentProjectionStatus.PROJECTED,
            ComponentEvaluation(free_indices, dimension, ()),
            "La expresión abstracta es nula; todas sus componentes son cero.",
        )
    if potential_components > component_budget:
        return (
            ComponentProjectionStatus.BACKEND_LIMITATION,
            None,
            (
                f"La proyección densa requeriría {potential_components} componentes, "
                f"por encima del límite seguro {component_budget}."
            ),
        )
    try:
        projected = component_backend.evaluate(expression)
    except (TensorEngineError, OverflowError, RecursionError) as error:
        return (
            ComponentProjectionStatus.BACKEND_LIMITATION,
            None,
            f"El backend de componentes no pudo evaluar la cantidad: {error}",
        )
    return (
        ComponentProjectionStatus.PROJECTED,
        projected,
        f"Proyectada respetando el ansatz {component_backend.geometry.ansatz.name!r}.",
    )


def _xact_status(
    verification: VerificationReport,
    check_key: str | None,
) -> tuple[XActValidationStatus, str | None, str]:
    xact_requested = any(
        operation == "verify_model" for operation, _, _ in verification.external_bindings
    )
    if not xact_requested:
        return (
            XActValidationStatus.NOT_REQUESTED,
            None,
            "La corrida no solicitó validación Wolfram/xAct.",
        )
    if check_key is None:
        return (
            XActValidationStatus.NOT_VALIDATED,
            None,
            "xAct se ejecutó, pero esta cantidad no tiene una identidad independiente.",
        )
    full_key = f"external.model.{check_key}"
    check = next((item for item in verification.checks if item.key == full_key), None)
    if check is not None and check.status is VerificationStatus.PASSED:
        return (
            XActValidationStatus.VALIDATED,
            full_key,
            "xAct redujo a cero la identidad que contiene esta cantidad.",
        )
    return (
        XActValidationStatus.NOT_VALIDATED,
        None,
        f"La comprobación xAct {full_key!r} no fue aprobada o no está disponible.",
    )


def derive_intermediate_quantities(
    model: ModelSpec,
    momenta: LagrangianMomenta,
    euler: EulerLagrangeResult,
    verification: VerificationReport,
    backend: TensorBackend,
    component_backend: SympyComponentBackend | None = None,
    *,
    component_budget: int = 2048,
    projection_unavailable_reason: str | None = None,
) -> DerivedQuantities:
    """Recupera resultados ya producidos y proyecta sin sustituir el ansatz activo."""

    space = model.symbols.index_space
    a, b, c, d = (Index(name, Variance.UP, space) for name in ("a", "b", "c", "d"))
    e = Index("e", Variance.DOWN, space)
    f = Index("f", Variance.DOWN, space)
    riemann_indices = tuple(
        Index(name, Variance.DOWN, space) for name in ("a", "b", "c", "d")
    )
    derivative_metric_indices = (
        Index("a", Variance.DOWN, space),
        Index("b", Variance.DOWN, space),
    )

    builder = ModelBuilder(model.symbols)
    ricci_scalar = builder.ricci_scalar()
    ricci_squared = builder.ricci_squared()
    riemann_tensor = Tensor(model.symbols.curvature, riemann_indices)
    riemann_squared = builder.riemann_squared()
    nabla_p = backend.covariant_derivative(momenta.curvature, e)
    nabla_nabla_p = backend.covariant_derivative(nabla_p, f)
    derivative_metric_term = euler.curvature_derivative_metric_term

    specifications = (
        (
            "ricci_scalar",
            ricci_scalar,
            (),
            SymbolicEvaluationStatus.CALCULATED,
            ("validated_model",),
            (),
            None,
        ),
        (
            "ricci_squared",
            ricci_squared,
            (),
            SymbolicEvaluationStatus.CALCULATED,
            ("validated_model", "riemann_tensor"),
            (),
            None,
        ),
        (
            "riemann_tensor",
            riemann_tensor,
            riemann_indices,
            SymbolicEvaluationStatus.GEOMETRIC_INPUT,
            ("validated_model",),
            (),
            None,
        ),
        (
            "riemann_squared",
            riemann_squared,
            (),
            SymbolicEvaluationStatus.CALCULATED,
            ("validated_model", "riemann_tensor"),
            (),
            None,
        ),
        (
            "nabla_P",
            nabla_p,
            (a, b, c, d, e),
            SymbolicEvaluationStatus.CALCULATED,
            ("curvature_momentum",),
            (),
            None,
        ),
        (
            "nabla_nabla_P",
            nabla_nabla_p,
            (a, b, c, d, e, f),
            SymbolicEvaluationStatus.CALCULATED,
            ("nabla_P",),
            ("metric_euler",),
            None,
        ),
        (
            "curvature_derivative_metric_term",
            derivative_metric_term,
            derivative_metric_indices,
            SymbolicEvaluationStatus.CALCULATED,
            ("curvature_momentum", "nabla_nabla_P"),
            ("metric_euler",),
            "metric_euler_curvature_derivative_term",
        ),
    )

    records: list[DerivedQuantityRecord] = []
    for key, expression, indices, symbolic_status, sources, targets, check_key in specifications:
        component_status, components, projection_note = _project(
            expression,
            indices,
            component_backend,
            component_budget=component_budget,
            unavailable_reason=projection_unavailable_reason,
        )
        xact_status, xact_check, xact_note = _xact_status(verification, check_key)
        records.append(
            DerivedQuantityRecord(
                key=key,
                free_indices=indices,
                symbolic_status=symbolic_status,
                component_status=component_status,
                xact_status=xact_status,
                components=components,
                source_keys=sources,
                contributes_to=targets,
                note=f"{projection_note} {xact_note}",
                xact_check=xact_check,
            )
        )

    return DerivedQuantities(
        ricci_scalar=ricci_scalar,
        ricci_squared=ricci_squared,
        riemann_tensor=riemann_tensor,
        riemann_squared=riemann_squared,
        nabla_P=nabla_p,
        nabla_nabla_P=nabla_nabla_p,
        curvature_derivative_metric_term=derivative_metric_term,
        records=tuple(records),
    )


_ABSTRACT_DESCRIPTIONS = {
    "lagrangian": "Densidad lagrangiana escalar normalizada de la teoría.",
    "metric_momentum": "Derivada de L respecto de la métrica inversa, M_ab.",
    "curvature_momentum": "Momento de curvatura P^{abcd}=partial L/partial R_abcd.",
    "scalar_gradient_momentum": "Momento J^a conjugado a nabla_a phi.",
    "scalar_derivative": "Derivada explícita F_phi=partial L/partial phi.",
    "metric_euler": "Ecuación de Euler-Lagrange métrica E_ab.",
    "scalar_euler": "Ecuación de Euler-Lagrange escalar E_phi.",
    "ricci_scalar": "Escalar de Ricci construido con la convención geométrica activa.",
    "ricci_squared": "Invariante cuadrático de Ricci R_ab R^ab.",
    "riemann_tensor": "Tensor de Riemann covariante tratado como entrada geométrica.",
    "riemann_squared": "Invariante de Kretschmann R_abcd R^abcd.",
    "nabla_P": "Primera derivada covariante del momento de curvatura.",
    "nabla_nabla_P": "Segunda derivada covariante del momento de curvatura.",
}

_ABSTRACT_SOURCES = {
    "lagrangian": ("validated_model",),
    "metric_momentum": ("lagrangian",),
    "curvature_momentum": ("lagrangian",),
    "scalar_gradient_momentum": ("lagrangian",),
    "scalar_derivative": ("lagrangian",),
    "metric_euler": ("delta_lagrangian", "curvature_derivative_metric_term"),
    "scalar_euler": ("scalar_derivative", "scalar_gradient_momentum"),
    "ricci_scalar": ("validated_model",),
    "ricci_squared": ("validated_model", "riemann_tensor"),
    "riemann_tensor": ("validated_model",),
    "riemann_squared": ("validated_model", "riemann_tensor"),
    "nabla_P": ("curvature_momentum",),
    "nabla_nabla_P": ("nabla_P",),
}

_XACT_CHECKS = {
    "metric_momentum": ("metric_momentum_symmetry",),
    "curvature_momentum": (
        "curvature_momentum_first_pair",
        "curvature_momentum_second_pair",
        "curvature_momentum_pair_exchange",
        "curvature_momentum_first_bianchi",
    ),
    "metric_euler": (
        "metric_euler_symmetry",
        "metric_euler_curvature_derivative_term",
    ),
}


def _xact_status_for_checks(
    verification: VerificationReport,
    check_keys: tuple[str, ...],
) -> tuple[XActValidationStatus, tuple[str, ...], str]:
    xact_requested = any(
        operation == "verify_model" for operation, _, _ in verification.external_bindings
    )
    if not xact_requested:
        return (
            XActValidationStatus.NOT_REQUESTED,
            (),
            "La corrida no solicitó validación Wolfram/xAct.",
        )
    if not check_keys:
        return (
            XActValidationStatus.NOT_VALIDATED,
            (),
            "xAct se ejecutó, pero no existe una identidad independiente para esta cantidad.",
        )
    full_keys = tuple(f"external.model.{key}" for key in check_keys)
    by_key = {item.key: item for item in verification.checks}
    if all(
        key in by_key and by_key[key].status is VerificationStatus.PASSED
        for key in full_keys
    ):
        return (
            XActValidationStatus.VALIDATED,
            full_keys,
            "xAct aprobó las identidades estructurales asociadas a esta cantidad.",
        )
    return (
        XActValidationStatus.NOT_VALIDATED,
        full_keys,
        "Una o más identidades xAct asociadas no están disponibles o no fueron aprobadas.",
    )


def _projection_result(
    key: str,
    expression: Expr,
    free_indices: tuple[Index, ...],
    ansatz_name: str | None,
    component_backend: SympyComponentBackend | None,
    *,
    component_budget: int,
    unavailable_reason: str | None,
) -> ProjectedQuantityResult:
    if component_backend is None and ansatz_name is not None and unavailable_reason is None:
        return ProjectedQuantityResult(
            key,
            ProjectionStatus.SYMBOLIC,
            ansatz_name,
            reason="La proyección fue desactivada mediante EngineOptions.",
        )
    status, components, reason = _project(
        expression,
        free_indices,
        component_backend,
        component_budget=component_budget,
        unavailable_reason=unavailable_reason,
    )
    mapped = {
        ComponentProjectionStatus.PROJECTED: ProjectionStatus.COMPLETED,
        ComponentProjectionStatus.NOT_REQUESTED: ProjectionStatus.SYMBOLIC,
        ComponentProjectionStatus.BACKEND_LIMITATION: ProjectionStatus.UNAVAILABLE,
    }[status]
    return ProjectedQuantityResult(key, mapped, ansatz_name, components, reason)


def _result_expressions_and_signatures(
    model: ModelSpec,
    lagrangian: Expr,
    momenta: LagrangianMomenta,
    euler: EulerLagrangeResult,
    derived: DerivedQuantities,
) -> tuple[dict[str, Expr], dict[str, tuple[Index, ...]]]:
    space = model.symbols.index_space
    down = lambda name: Index(name, Variance.DOWN, space)
    up = lambda name: Index(name, Variance.UP, space)
    expressions = {
        "lagrangian": lagrangian,
        "metric_momentum": momenta.metric,
        "curvature_momentum": momenta.curvature,
        "scalar_gradient_momentum": momenta.scalar_gradient,
        "scalar_derivative": momenta.scalar,
        "metric_euler": euler.metric_euler,
        "scalar_euler": euler.scalar_euler,
        "ricci_scalar": derived.ricci_scalar,
        "ricci_squared": derived.ricci_squared,
        "riemann_tensor": derived.riemann_tensor,
        "riemann_squared": derived.riemann_squared,
        "nabla_P": derived.nabla_P,
        "nabla_nabla_P": derived.nabla_nabla_P,
    }
    signatures = {
        "lagrangian": (),
        "metric_momentum": (down("a"), down("b")),
        "curvature_momentum": tuple(up(name) for name in ("a", "b", "c", "d")),
        "scalar_gradient_momentum": (up("a"),),
        "scalar_derivative": (),
        "metric_euler": (down("a"), down("b")),
        "scalar_euler": (),
        "ricci_scalar": (),
        "ricci_squared": (),
        "riemann_tensor": tuple(down(name) for name in ("a", "b", "c", "d")),
        "riemann_squared": (),
        "nabla_P": (*tuple(up(name) for name in ("a", "b", "c", "d")), down("e")),
        "nabla_nabla_P": (
            *tuple(up(name) for name in ("a", "b", "c", "d")),
            down("e"),
            down("f"),
        ),
    }
    return expressions, signatures


def build_result_views(
    model: ModelSpec,
    lagrangian: Expr,
    momenta: LagrangianMomenta,
    euler: EulerLagrangeResult,
    derived: DerivedQuantities,
    verification: VerificationReport,
    *,
    ansatz_name: str | None = None,
    ansatz: GeometryAnsatz | None = None,
    component_backend: SympyComponentBackend | None = None,
    field_components: ComponentFieldEquations | None = None,
    projection_unavailable_reason: str | None = None,
    field_equation_failure_reason: str | None = None,
    component_budget: int = 2048,
) -> tuple[AbstractTensorResults, ProjectedTensorResults]:
    """Organiza dos vistas sin volver a derivar ninguna expresión covariante."""

    if ansatz is not None:
        if ansatz_name is not None and ansatz.name != ansatz_name:
            raise ValueError("ansatz y ansatz_name identifican geometrías distintas.")
        ansatz_name = ansatz.name

    expressions, signatures = _result_expressions_and_signatures(
        model, lagrangian, momenta, euler, derived
    )

    abstract_records: list[AbstractQuantityRecord] = []
    for key in REPORT_QUANTITY_KEYS:
        xact_status, xact_checks, note = _xact_status_for_checks(
            verification,
            _XACT_CHECKS.get(key, ()),
        )
        abstract_records.append(
            AbstractQuantityRecord(
                key=key,
                free_indices=signatures[key],
                description=_ABSTRACT_DESCRIPTIONS[key],
                source_keys=_ABSTRACT_SOURCES[key],
                xact_status=xact_status,
                xact_checks=xact_checks,
                validation_note=note,
            )
        )
    abstract = AbstractTensorResults(
        **expressions,
        records=tuple(abstract_records),
    )

    derived_projection_keys = {
        "ricci_scalar",
        "ricci_squared",
        "riemann_tensor",
        "riemann_squared",
        "nabla_P",
        "nabla_nabla_P",
    }
    projected: list[ProjectedQuantityResult] = []
    for key in REPORT_QUANTITY_KEYS:
        if key == "metric_euler" and field_components is not None:
            projected.append(
                ProjectedQuantityResult(
                    key,
                    ProjectionStatus.COMPLETED,
                    ansatz_name,
                    field_components.metric,
                    "Reutilizada desde la etapa de componentes de E_ab.",
                )
            )
            continue
        if key == "scalar_euler" and field_components is not None:
            projected.append(
                ProjectedQuantityResult(
                    key,
                    ProjectionStatus.COMPLETED,
                    ansatz_name,
                    field_components.scalar,
                    "Reutilizada desde la etapa de componentes de E_phi.",
                )
            )
            continue
        if (
            key in {"metric_euler", "scalar_euler"}
            and field_equation_failure_reason
            and component_backend is None
        ):
            projected.append(
                ProjectedQuantityResult(
                    key,
                    ProjectionStatus.UNAVAILABLE,
                    ansatz_name,
                    reason=field_equation_failure_reason,
                )
            )
            continue
        if key in derived_projection_keys:
            existing = derived.record(key)
            if (
                component_backend is None
                and ansatz_name is not None
                and projection_unavailable_reason is None
            ):
                projected.append(
                    ProjectedQuantityResult(
                        key,
                        ProjectionStatus.SYMBOLIC,
                        ansatz_name,
                        reason="La proyección fue desactivada mediante EngineOptions.",
                    )
                )
                continue
            mapped = {
                ComponentProjectionStatus.PROJECTED: ProjectionStatus.COMPLETED,
                ComponentProjectionStatus.NOT_REQUESTED: ProjectionStatus.SYMBOLIC,
                ComponentProjectionStatus.BACKEND_LIMITATION: ProjectionStatus.UNAVAILABLE,
            }[existing.component_status]
            projected.append(
                ProjectedQuantityResult(
                    key,
                    mapped,
                    ansatz_name,
                    existing.components,
                    existing.note,
                )
            )
            continue
        projected.append(
            _projection_result(
                key,
                expressions[key],
                signatures[key],
                ansatz_name,
                component_backend,
                component_budget=component_budget,
                unavailable_reason=projection_unavailable_reason,
            )
        )

    return abstract, ProjectedTensorResults(
        ansatz_name,
        tuple(projected),
        ansatz_geometry=ansatz,
    )


def build_specialized_results(
    model: ModelSpec,
    lagrangian: Expr,
    momenta: LagrangianMomenta,
    euler: EulerLagrangeResult,
    derived: DerivedQuantities,
    *,
    base_ansatz_name: str,
    specialization: AnsatzSpecialization,
    specialized_ansatz: GeometryAnsatz,
    component_backend: SympyComponentBackend | None,
    field_components: ComponentFieldEquations | None = None,
    unavailable_reason: str | None = None,
    component_budget: int = 2048,
) -> SpecializedTensorResults:
    """Proyecta resultados ya derivados sobre una especialización posterior."""

    expressions, signatures = _result_expressions_and_signatures(
        model, lagrangian, momenta, euler, derived
    )
    quantities: list[ProjectedQuantityResult] = []
    for key in REPORT_QUANTITY_KEYS:
        if key == "metric_euler" and field_components is not None:
            quantities.append(
                ProjectedQuantityResult(
                    key,
                    ProjectionStatus.COMPLETED,
                    specialized_ansatz.name,
                    field_components.metric,
                    "Reutilizada desde la evaluación especializada de E_ab.",
                )
            )
            continue
        if key == "scalar_euler" and field_components is not None:
            quantities.append(
                ProjectedQuantityResult(
                    key,
                    ProjectionStatus.COMPLETED,
                    specialized_ansatz.name,
                    field_components.scalar,
                    "Reutilizada desde la evaluación especializada de E_phi.",
                )
            )
            continue
        quantities.append(
            _projection_result(
                key,
                expressions[key],
                signatures[key],
                specialized_ansatz.name,
                component_backend,
                component_budget=component_budget,
                unavailable_reason=unavailable_reason,
            )
        )
    return SpecializedTensorResults(
        base_ansatz_name=base_ansatz_name,
        specialization=specialization,
        ansatz_geometry=specialized_ansatz,
        quantities=tuple(quantities),
    )
