"""Exportación reproducible de corridas y vistas de presentación.

El JSON conserva la IR como fuente canónica. LaTeX es deliberadamente una
vista derivada: nunca se vuelve a interpretar para continuar un cálculo.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping

from .components import (
    ComponentFieldEquations,
    GeometryAnsatz,
    ScalarFieldMode,
    SympyComponentBackend,
)
from .delta import DeltaContractionAudit, delta_count
from .contracts import (
    ArtifactRecord,
    Diagnostic,
    ExpressionForm,
    ExpressionRecord,
    Severity,
    StageResult,
    StageStatus,
)
from .derived import (
    REPORT_QUANTITY_KEYS,
    AbstractTensorResults,
    DerivedQuantities,
    ProjectedTensorResults,
    ProjectionStatus,
    SpecializedTensorResults,
    XActValidationStatus,
)
from .euler import EulerLagrangeResult
from .ir import (
    Add,
    CovariantDerivative,
    Expr,
    Function,
    FunctionDerivative,
    Mul,
    Number,
    Power,
    Scalar,
    Tensor,
    Variance,
    Variation,
    VolumeElement,
    expr_from_data,
    walk,
)
from .model import ModelSpec
from .presentation import (
    CompactBlock,
    CompactDecomposition,
    CompactProjection,
    DisplayExpression,
    DisplayPolicy,
    ReportPresentation,
    build_presentation,
    independent_curvature_components,
)
from .noether import NoetherWaldResult
from .variational import LagrangianMomenta
from .verification import VerificationReport


EXPORT_SCHEMA_VERSION = "1.0"
_SAFE_SLUG = re.compile(r"[^a-z0-9]+")


def _canonical_json(data: Any, *, pretty: bool = False) -> str:
    separators = None if pretty else (",", ":")
    return json.dumps(
        data,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=separators,
        sort_keys=True,
    ) + ("\n" if pretty else "")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_data(data: Any) -> str:
    return _sha256_bytes(_canonical_json(data).encode("utf-8"))


def _slug(value: str) -> str:
    candidate = _SAFE_SLUG.sub("-", value.lower()).strip("-")
    return candidate[:64] or "model"


@dataclass(frozen=True, slots=True)
class RunPackage:
    """Objetos matemáticos y evidencia que forman una corrida exportable."""

    model: ModelSpec
    momenta: LagrangianMomenta
    raw_variation: Expr
    euler: EulerLagrangeResult
    verification: VerificationReport
    normalized_lagrangian: Expr | None = None
    noether: NoetherWaldResult | None = None
    components: ComponentFieldEquations | None = None
    derived: DerivedQuantities | None = None
    abstract: AbstractTensorResults | None = None
    projected: ProjectedTensorResults | None = None
    specialized: SpecializedTensorResults | None = None
    duration_seconds: float = 0.0
    stage_durations: tuple[tuple[str, float], ...] = ()
    delta_contractions: tuple[DeltaContractionAudit, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_durations", tuple(tuple(item) for item in self.stage_durations))
        object.__setattr__(self, "delta_contractions", tuple(self.delta_contractions))
        if self.verification.model_name != self.model.name:
            raise ValueError("El informe de verificación pertenece a otro modelo.")
        if self.duration_seconds < 0 or any(value < 0 for _, value in self.stage_durations):
            raise ValueError("Las duraciones de corrida no pueden ser negativas.")
        keys = [key for key, _ in self.stage_durations]
        if len(keys) != len(set(keys)):
            raise ValueError("Las duraciones por etapa contienen claves repetidas.")
        if (self.abstract is None) != (self.projected is None):
            raise ValueError("Las vistas abstracta y proyectada deben conservarse juntas.")
        if self.specialized is not None and (
            self.abstract is None or self.projected is None
        ):
            raise ValueError(
                "Una vista especializada requiere las vistas abstracta y proyectada."
            )

    @property
    def lagrangian(self) -> Expr:
        return self.normalized_lagrangian or self.model.lagrangian

    def expression_records(self) -> tuple[ExpressionRecord, ...]:
        """Inventario normativo de expresiones con trazabilidad entre etapas."""

        records = [
            ExpressionRecord("original_lagrangian", self.model.lagrangian),
            ExpressionRecord(
                "lagrangian",
                self.lagrangian,
                ExpressionForm.CANONICAL,
                ("validated_model",),
            ),
            ExpressionRecord("metric_momentum", self.momenta.metric, ExpressionForm.CANONICAL, ("lagrangian",)),
            ExpressionRecord("curvature_momentum", self.momenta.curvature, ExpressionForm.CANONICAL, ("lagrangian",)),
            ExpressionRecord("scalar_gradient_momentum", self.momenta.scalar_gradient, ExpressionForm.CANONICAL, ("lagrangian",)),
            ExpressionRecord("scalar_derivative", self.momenta.scalar, ExpressionForm.CANONICAL, ("lagrangian",)),
            ExpressionRecord(
                "delta_lagrangian",
                self.raw_variation,
                source_keys=(
                    "lagrangian",
                    "metric_momentum",
                    "curvature_momentum",
                    "scalar_gradient_momentum",
                    "scalar_derivative",
                ),
            ),
            ExpressionRecord("metric_euler", self.euler.metric_euler, ExpressionForm.CANONICAL, ("delta_lagrangian",)),
            ExpressionRecord("scalar_euler", self.euler.scalar_euler, ExpressionForm.CANONICAL, ("delta_lagrangian",)),
            ExpressionRecord("boundary_potential_metric", self.euler.boundary_metric, ExpressionForm.CANONICAL, ("delta_lagrangian",)),
            ExpressionRecord("boundary_potential_scalar", self.euler.boundary_scalar, ExpressionForm.CANONICAL, ("delta_lagrangian",)),
            ExpressionRecord(
                "boundary_potential_total",
                self.euler.boundary_total,
                ExpressionForm.CANONICAL,
                ("boundary_potential_metric", "boundary_potential_scalar"),
            ),
            ExpressionRecord(
                "full_variation",
                self.euler.full_variation,
                ExpressionForm.CANONICAL,
                ("metric_euler", "scalar_euler", "boundary_potential_total"),
            ),
            ExpressionRecord("density_variation", self.euler.density_variation, ExpressionForm.CANONICAL, ("full_variation",)),
        ]
        if self.noether is not None:
            records.extend(
                (
                    ExpressionRecord("noether_current", self.noether.noether_current, ExpressionForm.CANONICAL, ("metric_euler", "scalar_euler", "boundary_potential_total")),
                    ExpressionRecord("constraint_current", self.noether.constraint_current, ExpressionForm.CANONICAL, ("metric_euler", "scalar_euler")),
                    ExpressionRecord("charge_potential", self.noether.charge_potential, ExpressionForm.CANONICAL, ("curvature_momentum",)),
                    ExpressionRecord("noether_identity", self.noether.noether_identity, ExpressionForm.CANONICAL, ("noether_current", "metric_euler", "scalar_euler")),
                )
            )
        if self.derived is not None:
            records.extend(
                ExpressionRecord(
                    key,
                    expression,
                    ExpressionForm.CANONICAL,
                    self.derived.record(key).source_keys,
                )
                for key, expression in self.derived.expression_items()
            )
        return tuple(records)

    def semantic_data(self) -> dict[str, Any]:
        """Contenido que identifica la corrida; excluye reloj, rutas y tiempos."""

        data = {
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "model": self.model.to_data(),
            "normalized_lagrangian": self.lagrangian.to_data(),
            "momenta": self.momenta.to_data(),
            "raw_variation": self.raw_variation.to_data(),
            "euler_lagrange": self.euler.to_data(),
            "noether_wald": None if self.noether is None else self.noether.to_data(),
            "components": None if self.components is None else self.components.to_data(),
            "derived_quantities": (
                None if self.derived is None else self.derived.to_data()
            ),
            "abstract_results": (
                None if self.abstract is None else self.abstract.to_data()
            ),
            "projected_results": (
                None if self.projected is None else self.projected.to_data()
            ),
            "verification": self.verification.to_data(),
        }
        if self.specialized is not None:
            data["specialized_results"] = self.specialized.to_data()
        return data

    @property
    def run_id(self) -> str:
        return f"run_{_sha256_data(self.semantic_data())[:20]}"

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "run_id": self.run_id,
            **self.semantic_data(),
            **({"delta_contractions": [a.to_data() for a in self.delta_contractions]}
               if self.delta_contractions else {}),
            "normalization_explicit": self.normalized_lagrangian is not None,
            "timing": {
                "duration_seconds": self.duration_seconds,
                "stages": {key: value for key, value in self.stage_durations},
            },
            "expressions": [record.to_data() for record in self.expression_records()],
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "RunPackage":
        semantic_keys = (
            "export_schema_version",
            "model",
            "normalized_lagrangian",
            "momenta",
            "raw_variation",
            "euler_lagrange",
            "noether_wald",
            "components",
            "derived_quantities",
            "abstract_results",
            "projected_results",
            "specialized_results",
            "verification",
        )
        stored_semantic = {
            key: data[key] for key in semantic_keys if key in data
        }
        stored_run_id = f"run_{_sha256_data(stored_semantic)[:20]}"
        timing = data.get("timing", {})
        model = ModelSpec.from_data(data["model"])
        component_data = data.get("components")
        derived_data = data.get("derived_quantities")
        abstract_data = data.get("abstract_results")
        projected_data = data.get("projected_results")
        specialized_data = data.get("specialized_results")
        noether_data = data.get("noether_wald")
        package = cls(
            model=model,
            momenta=LagrangianMomenta.from_data(data["momenta"]),
            raw_variation=expr_from_data(data["raw_variation"]),
            euler=EulerLagrangeResult.from_data(data["euler_lagrange"]),
            verification=VerificationReport.from_data(data["verification"]),
            normalized_lagrangian=(
                expr_from_data(data["normalized_lagrangian"])
                if data.get("normalization_explicit", True)
                else None
            ),
            noether=None if noether_data is None else NoetherWaldResult.from_data(noether_data),
            components=None if component_data is None else ComponentFieldEquations.from_data(component_data),
            derived=(
                None
                if derived_data is None
                else DerivedQuantities.from_data(derived_data, symbols=model.symbols)
            ),
            abstract=(
                None
                if abstract_data is None
                else AbstractTensorResults.from_data(abstract_data, symbols=model.symbols)
            ),
            projected=(
                None
                if projected_data is None
                else ProjectedTensorResults.from_data(projected_data)
            ),
            specialized=(
                None
                if specialized_data is None
                else SpecializedTensorResults.from_data(specialized_data)
            ),
            duration_seconds=float(timing.get("duration_seconds", 0.0)),
            stage_durations=tuple(
                (str(key), float(value)) for key, value in timing.get("stages", {}).items()
            ),
            delta_contractions=tuple(DeltaContractionAudit.from_data(a)
                                     for a in data.get("delta_contractions", ())),
        )
        supplied = data.get("run_id")
        if supplied is not None and supplied != package.run_id:
            semantic = package.semantic_data()
            legacy_candidates: list[dict[str, Any]] = []
            without_views = dict(semantic)
            without_views.pop("abstract_results", None)
            without_views.pop("projected_results", None)
            legacy_candidates.append(without_views)
            before_derived = dict(without_views)
            before_derived.pop("derived_quantities", None)
            legacy_euler = dict(before_derived["euler_lagrange"])
            legacy_euler.pop("curvature_derivative_metric_term", None)
            before_derived["euler_lagrange"] = legacy_euler
            legacy_candidates.append(before_derived)
            legacy_ids = {
                f"run_{_sha256_data(candidate)[:20]}" for candidate in legacy_candidates
            }
            if supplied != stored_run_id and supplied not in legacy_ids:
                raise ValueError("El run_id no coincide con el contenido reconstruido.")
        return package


@dataclass(frozen=True, slots=True)
class ExportedFile:
    key: str
    relative_path: str
    media_type: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        relative = Path(self.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("La ruta de un artefacto debe ser relativa y no escapar del bundle.")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("La huella de un artefacto debe ser un SHA-256 hexadecimal.")
        if self.size_bytes < 0:
            raise ValueError("El tamaño de un artefacto no puede ser negativo.")

    def to_data(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "relative_path": self.relative_path,
            "media_type": self.media_type,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ExportedFile":
        return cls(
            str(data["key"]),
            str(data["relative_path"]),
            str(data["media_type"]),
            str(data["sha256"]),
            int(data["size_bytes"]),
        )


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id: str
    created_at_utc: str
    status: StageStatus
    model_name: str
    model_schema_version: str
    convention_id: str
    dimension: str
    backend_name: str
    backend_version: str
    external_sources: tuple[tuple[str, str, str], ...]
    duration_seconds: float
    stage_durations: tuple[tuple[str, float], ...]
    expressions: tuple[tuple[str, str, str], ...]
    files: tuple[ExportedFile, ...]
    abstract_quantities: tuple[str, ...] = ()
    projected_quantities: tuple[tuple[str, str, str], ...] = ()
    specialized_quantities: tuple[tuple[str, str, str], ...] = ()
    external_bindings: tuple[tuple[str, str, str], ...] = ()
    adjudications: tuple[tuple[str, str, str], ...] = ()
    schema_version: str = EXPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "external_sources", tuple(tuple(item) for item in self.external_sources))
        object.__setattr__(self, "stage_durations", tuple(tuple(item) for item in self.stage_durations))
        object.__setattr__(self, "expressions", tuple(tuple(item) for item in self.expressions))
        object.__setattr__(self, "files", tuple(self.files))
        object.__setattr__(self, "abstract_quantities", tuple(self.abstract_quantities))
        object.__setattr__(self, "projected_quantities", tuple(tuple(item) for item in self.projected_quantities))
        object.__setattr__(self, "specialized_quantities", tuple(tuple(item) for item in self.specialized_quantities))
        object.__setattr__(self, "external_bindings", tuple(tuple(item) for item in self.external_bindings))
        object.__setattr__(self, "adjudications", tuple(tuple(item) for item in self.adjudications))
        if self.schema_version != EXPORT_SCHEMA_VERSION:
            raise ValueError(f"Versión de manifiesto no soportada: {self.schema_version!r}.")
        if not re.fullmatch(r"run_[0-9a-f]{20}", self.run_id):
            raise ValueError("El manifiesto contiene un run_id inválido.")
        if self.duration_seconds < 0 or any(value < 0 for _, value in self.stage_durations):
            raise ValueError("Las duraciones del manifiesto no pueden ser negativas.")
        if len({key for key, _, _ in self.expressions}) != len(self.expressions):
            raise ValueError("El manifiesto repite claves de expresión.")
        if len({item.relative_path for item in self.files}) != len(self.files):
            raise ValueError("El manifiesto repite rutas de artefactos.")
        if len(self.abstract_quantities) != len(set(self.abstract_quantities)):
            raise ValueError("El manifiesto repite cantidades abstractas.")
        if len({item[0] for item in self.projected_quantities}) != len(self.projected_quantities):
            raise ValueError("El manifiesto repite cantidades proyectadas.")
        if any(len(item) != 3 for item in self.projected_quantities):
            raise ValueError("Las proyecciones del manifiesto tienen un contrato inválido.")
        if len({item[0] for item in self.specialized_quantities}) != len(self.specialized_quantities):
            raise ValueError("El manifiesto repite cantidades especializadas.")
        if any(len(item) != 3 for item in self.specialized_quantities):
            raise ValueError("Las especializaciones del manifiesto tienen un contrato inválido.")
        if any(len(item) != 3 for item in self.external_bindings):
            raise ValueError("Los vínculos externos del manifiesto tienen un contrato inválido.")
        if any(len(item) != 3 for item in self.adjudications):
            raise ValueError("Las adjudicaciones del manifiesto tienen un contrato inválido.")

    def to_data(self) -> dict[str, Any]:
        result_views = {
            "abstract": list(self.abstract_quantities),
            "projected": [
                {"key": key, "status": status, "sha256": digest}
                for key, status, digest in self.projected_quantities
            ],
        }
        if self.specialized_quantities:
            result_views["specialized"] = [
                {"key": key, "status": status, "sha256": digest}
                for key, status, digest in self.specialized_quantities
            ]
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "status": self.status.value,
            "model": {
                "name": self.model_name,
                "schema_version": self.model_schema_version,
                "convention_id": self.convention_id,
                "dimension": self.dimension,
            },
            "backend": {"name": self.backend_name, "version": self.backend_version},
            "external_sources": [
                {"name": name, "version": version, "status": status}
                for name, version, status in self.external_sources
            ],
            "external_bindings": [
                {
                    "operation": operation,
                    "model_fingerprint": model_hash,
                    "calculation_fingerprint": calculation_hash,
                }
                for operation, model_hash, calculation_hash in self.external_bindings
            ],
            "adjudications": [
                {
                    "internal_check": internal_check,
                    "operation": operation,
                    "external_check": external_check,
                }
                for internal_check, operation, external_check in self.adjudications
            ],
            "timing": {
                "duration_seconds": self.duration_seconds,
                "stages": {key: value for key, value in self.stage_durations},
            },
            "expressions": [
                {"key": key, "form": form, "sha256": digest}
                for key, form, digest in self.expressions
            ],
            "result_views": result_views,
            "files": [item.to_data() for item in self.files],
        }

    def verify_files(self, base_directory: str | Path) -> tuple[str, ...]:
        """Devuelve incidencias de ruta, ausencia, tamaño o hash del bundle."""

        root = Path(base_directory).resolve()
        incidents: list[str] = []
        for item in self.files:
            path = (root / item.relative_path).resolve()
            if root != path and root not in path.parents:
                incidents.append(f"{item.relative_path}: ruta fuera del bundle")
                continue
            if not path.is_file():
                incidents.append(f"{item.relative_path}: archivo ausente")
                continue
            content = path.read_bytes()
            if len(content) != item.size_bytes:
                incidents.append(f"{item.relative_path}: tamaño no coincide")
            if _sha256_bytes(content) != item.sha256:
                incidents.append(f"{item.relative_path}: SHA-256 no coincide")
        return tuple(incidents)

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "RunManifest":
        model = data["model"]
        backend = data["backend"]
        timing = data.get("timing", {})
        result_views = data.get("result_views", {})
        return cls(
            run_id=str(data["run_id"]),
            created_at_utc=str(data["created_at_utc"]),
            status=StageStatus(data["status"]),
            model_name=str(model["name"]),
            model_schema_version=str(model["schema_version"]),
            convention_id=str(model["convention_id"]),
            dimension=str(model["dimension"]),
            backend_name=str(backend["name"]),
            backend_version=str(backend["version"]),
            external_sources=tuple(
                (str(item["name"]), str(item["version"]), str(item["status"]))
                for item in data.get("external_sources", ())
            ),
            duration_seconds=float(timing.get("duration_seconds", 0.0)),
            stage_durations=tuple(
                (str(key), float(value)) for key, value in timing.get("stages", {}).items()
            ),
            expressions=tuple(
                (str(item["key"]), str(item["form"]), str(item["sha256"]))
                for item in data.get("expressions", ())
            ),
            files=tuple(ExportedFile.from_data(item) for item in data.get("files", ())),
            abstract_quantities=tuple(
                str(item) for item in result_views.get("abstract", ())
            ),
            projected_quantities=tuple(
                (
                    str(item["key"]),
                    str(item["status"]),
                    str(item.get("sha256", "")),
                )
                for item in result_views.get("projected", ())
            ),
            specialized_quantities=tuple(
                (
                    str(item["key"]),
                    str(item["status"]),
                    str(item.get("sha256", "")),
                )
                for item in result_views.get("specialized", ())
            ),
            external_bindings=tuple(
                (
                    str(item["operation"]),
                    str(item["model_fingerprint"]),
                    str(item["calculation_fingerprint"]),
                )
                for item in data.get("external_bindings", ())
            ),
            adjudications=tuple(
                (
                    str(item["internal_check"]),
                    str(item["operation"]),
                    str(item["external_check"]),
                )
                for item in data.get("adjudications", ())
            ),
            schema_version=str(data.get("schema_version", EXPORT_SCHEMA_VERSION)),
        )


_LATEX_NAMES = {
    "Phi": r"\Phi",
    "phi": r"\phi",
    "tau": r"\tau",
    "varphi": r"\varphi",
    "xi": r"\xi",
    "kappa": r"\kappa",
    "lambda": r"\lambda",
    "mu": r"\mu",
    "nu": r"\nu",
    "rho": r"\rho",
    "sigma": r"\sigma",
    "theta": r"\theta",
    "Gamma": r"\Gamma",
    "Riemann": "R",
    "delta_Gamma": r"\delta\Gamma",
}


def _latex_name(name: str) -> str:
    if name in _LATEX_NAMES:
        return _LATEX_NAMES[name]
    if "_" in name:
        head, *tail = name.split("_")
        return rf"{_latex_name(head)}_{{\mathrm{{{'_'.join(tail)}}}}}"
    return name if len(name) == 1 else rf"\mathrm{{{name}}}"


def _latex_index(name: str) -> str:
    return _LATEX_NAMES.get(name, name)


def _latex_text(value: str) -> str:
    return value.replace("_", r"\_")


def _precedence(expr: Expr) -> int:
    if isinstance(expr, Add):
        return 10
    if isinstance(expr, Mul):
        return 20
    if isinstance(expr, Power):
        return 30
    return 40


def _latex(expr: Expr, parent_precedence: int = 0) -> str:
    precedence = _precedence(expr)
    if isinstance(expr, Number):
        if expr.denominator == 1:
            result = str(expr.numerator)
        else:
            sign = "-" if expr.numerator < 0 else ""
            result = rf"{sign}\frac{{{abs(expr.numerator)}}}{{{expr.denominator}}}"
    elif isinstance(expr, Scalar):
        result = _latex_name(expr.name)
    elif isinstance(expr, Tensor):
        result = _latex_name(expr.name)
        for index in expr.indices:
            marker = "^" if index.variance is Variance.UP else "_"
            result += "{}" + marker + "{" + _latex_index(index.name) + "}"
    elif isinstance(expr, Add):
        parts: list[str] = []
        for position, term in enumerate(expr.terms):
            rendered = _latex(term, precedence)
            if position and rendered.startswith("-"):
                parts.append("- " + rendered[1:].lstrip())
            else:
                parts.append(rendered if position == 0 else "+ " + rendered)
        result = " ".join(parts)
    elif isinstance(expr, Mul):
        result = r"\,".join(_latex(item, precedence) for item in expr.factors)
    elif isinstance(expr, Power):
        result = rf"{{{_latex(expr.base, precedence)}}}^{{{_latex(expr.exponent)}}}"
    elif isinstance(expr, Function):
        arguments = ", ".join(_latex(item) for item in expr.arguments)
        result = rf"{_latex_name(expr.name)}\!\left({arguments}\right)"
    elif isinstance(expr, FunctionDerivative):
        arguments = ", ".join(_latex(item) for item in expr.arguments)
        total_order = sum(expr.derivative_orders)
        if len(expr.arguments) == 1:
            result = rf"{_latex_name(expr.name)}^{{({total_order})}}\!\left({arguments}\right)"
        else:
            denominator = "".join(
                rf"\partial {_latex(argument)}^{{{order}}}"
                for argument, order in zip(expr.arguments, expr.derivative_orders, strict=True)
                if order
            )
            result = rf"\frac{{\partial^{{{total_order}}} {_latex_name(expr.name)}}}{{{denominator}}}\!\left({arguments}\right)"
    elif isinstance(expr, CovariantDerivative):
        result = rf"\nabla_{{{_latex_index(expr.index.name)}}}\!\left({_latex(expr.operand)}\right)"
    elif isinstance(expr, Variation):
        result = rf"\delta\!\left({_latex(expr.operand)}\right)"
    elif isinstance(expr, VolumeElement):
        result = rf"\sqrt{{-{_latex_name(expr.metric_name)}}}"
    else:  # pragma: no cover - la jerarquía IR es cerrada en esta versión
        raise TypeError(f"Nodo IR sin impresor LaTeX: {type(expr).__name__}.")
    if precedence < parent_precedence:
        return rf"\left({result}\right)"
    return result


def expr_to_latex(expr: Expr) -> str:
    """Convierte toda la IR soportada a una vista LaTeX determinista."""

    return _latex(expr)


def _display_name(name: str) -> str:
    names = {"ell": r"\ell", "alpha": r"\alpha", "beta": r"\beta", "delta": r"\delta"}
    match = re.fullmatch(r"([A-Za-z_]+)([0-9]+)", name)
    if match:
        return rf"{_display_name(match[1])}_{{{match[2]}}}"
    return names.get(name, _latex_name(name))


def display_expr_to_latex(expr: Expr, parent_precedence: int = 0) -> str:
    """Readable printer only. Does not cancel, reorder slots or mutate the IR."""
    precedence = _precedence(expr)
    render = display_expr_to_latex
    if isinstance(expr, Scalar):
        result = _display_name(expr.name)
    elif isinstance(expr, Tensor):
        result = _display_name(expr.name)
        # Group only consecutive equal variances, preserving the tensor slots.
        groups: list[tuple[Variance, list[str]]] = []
        for index in expr.indices:
            if not groups or groups[-1][0] is not index.variance:
                groups.append((index.variance, []))
            groups[-1][1].append(_display_name(index.name))
        for variance, names in groups:
            marker = "^" if variance is Variance.UP else "_"
            result += "{}" + marker + "{" + r"\,".join(names) + "}"
    elif isinstance(expr, Add):
        parts = []
        for term in expr.terms:
            text = render(term, precedence)
            parts.append(text if not parts else ("- " + text[1:] if text.startswith("-") else "+ " + text))
        result = " ".join(parts)
    elif isinstance(expr, Mul):
        numerator, denominator = [], []
        sign = 1
        for factor in expr.factors:
            if isinstance(factor, Number):
                sign *= -1 if factor.numerator < 0 else 1
                if abs(factor.numerator) != 1:
                    numerator.append(str(abs(factor.numerator)))
                if factor.denominator != 1:
                    denominator.append(str(factor.denominator))
            elif (isinstance(factor, Power) and isinstance(factor.exponent, Number)
                  and factor.exponent.denominator == 1 and factor.exponent.numerator < 0):
                positive = -factor.exponent.numerator
                denominator.append(render(factor.base, 20) if positive == 1 else render(Power(factor.base, Number(positive))))
            else:
                numerator.append(render(factor, precedence))
        top = r"\,".join(numerator) or "1"
        bottom = r"\,".join(denominator)
        # Large fraction numerators cannot break over pages in TeX. Keep a
        # small reciprocal coefficient outside the breakable polynomial.
        if bottom and len(top) > 160:
            result = rf"\frac{{1}}{{{bottom}}}\,{top}"
        else:
            result = rf"\frac{{{top}}}{{{bottom}}}" if bottom else top
        if sign < 0:
            result = "-" + result
    elif isinstance(expr, Power):
        if isinstance(expr.exponent, Number) and expr.exponent.denominator == 1 and expr.exponent.numerator < 0:
            n = -expr.exponent.numerator
            bottom = render(expr.base) if n == 1 else render(Power(expr.base, Number(n)))
            result = rf"\frac{{1}}{{{bottom}}}"
        else:
            result = rf"{{{render(expr.base, precedence)}}}^{{{render(expr.exponent)}}}"
    elif isinstance(expr, Function):
        result = rf"{_display_name(expr.name)}\!\left({', '.join(render(a) for a in expr.arguments)}\right)"
    elif isinstance(expr, FunctionDerivative) and len(expr.arguments) == 1:
        n = expr.derivative_orders[0]
        marker = "'" * n if n in (1, 2) else rf"^{{({n})}}"
        result = rf"{_display_name(expr.name)}{marker}\!\left({render(expr.arguments[0])}\right)"
    elif isinstance(expr, CovariantDerivative):
        result = rf"\nabla_{{{_display_name(expr.index.name)}}}\!\left({render(expr.operand)}\right)"
    elif isinstance(expr, Variation):
        result = rf"\delta\!\left({render(expr.operand)}\right)"
    else:
        result = expr_to_latex(expr)
    return rf"\left({result}\right)" if precedence < parent_precedence else result


def _presentation_latex(view: ReportPresentation, key: str) -> str:
    record = view.record(key)
    return _display_record_latex(view, record)


def _display_record_latex(
    view: ReportPresentation,
    record: DisplayExpression,
) -> str:
    printer = display_expr_to_latex if view.policy.enabled else expr_to_latex
    return printer(record.presentation)


def _compact_audit(key: str, record: DisplayExpression) -> str:
    data = {
        "key": key,
        "status": record.status,
        "operations": record.operations,
        "assumptions_used": record.assumptions_used,
        "notes": record.notes,
    }
    return "% compact-display: " + json.dumps(
        data,
        ensure_ascii=True,
        sort_keys=True,
    )


def _presentation_audit(view: ReportPresentation, key: str) -> str:
    record = view.record(key)
    data = {"key": key, "status": record.status, "operations": record.operations,
            "assumptions_used": record.assumptions_used, "notes": record.notes}
    return "% display: " + json.dumps(data, ensure_ascii=True, sort_keys=True)


_REPORT_LABELS = {
    "lagrangian": r"L",
    "metric_momentum": r"M_{ab}",
    "curvature_momentum": r"P^{abcd}",
    "scalar_gradient_momentum": r"J^a",
    "scalar_derivative": r"F_{\phi}",
    "metric_euler": r"E_{ab}",
    "scalar_euler": r"E_{\phi}",
    "ricci_scalar": r"\mathcal{R}",
    "ricci_squared": r"R_{ab}R^{ab}",
    "riemann_tensor": r"R_{abcd}",
    "riemann_squared": r"R_{abcd}R^{abcd}",
    "nabla_P": r"\nabla_e P^{abcd}",
    "nabla_nabla_P": r"\nabla_f\nabla_e P^{abcd}",
}

_XACT_STATUS_TEXT = {
    XActValidationStatus.VALIDATED: "validación estructural aprobada con Wolfram Engine/xAct",
    XActValidationStatus.NOT_VALIDATED: "sin validación independiente con xAct",
    XActValidationStatus.NOT_REQUESTED: "validación xAct no solicitada",
}

_PROJECTION_STATUS_TEXT = {
    ProjectionStatus.COMPLETED: "completada",
    ProjectionStatus.PARTIAL: "parcial",
    ProjectionStatus.SYMBOLIC: "simbólica",
    ProjectionStatus.UNAVAILABLE: "no disponible por limitación del backend",
}


def _coordinate_latex(coordinate: Scalar) -> str:
    return expr_to_latex(coordinate)


def _signed_line_element_term(expression: Expr, differential: str) -> tuple[int, str]:
    rendered = display_expr_to_latex(expression)
    sign = -1 if rendered.startswith("-") else 1
    magnitude = rendered[1:] if sign < 0 else rendered
    if magnitude.startswith(r"1\,"):
        magnitude = magnitude[3:]
    return sign, rf"{magnitude}\,d{differential}^2"


def _ansatz_summary_latex(ansatz: GeometryAnsatz) -> tuple[str, ...]:
    coordinates = tuple(_coordinate_latex(item) for item in ansatz.chart.coordinates)
    terms: list[tuple[int, str]] = []
    diagonal = all(
        entry == Number(0)
        for row, entries in enumerate(ansatz.metric_covariant)
        for column, entry in enumerate(entries)
        if row != column
    )
    if diagonal:
        for position, coordinate in enumerate(coordinates):
            entry = ansatz.metric_covariant[position][position]
            if entry != Number(0):
                terms.append(_signed_line_element_term(entry, coordinate))
        line_element = ""
        for position, (sign, term) in enumerate(terms):
            if position == 0:
                line_element = ("-" if sign < 0 else "") + term
            else:
                line_element += (" - " if sign < 0 else " + ") + term
        metric_line = rf"ds^2 = {line_element}"
    else:
        rows = r" \\ ".join(
            " & ".join(expr_to_latex(entry) for entry in row)
            for row in ansatz.metric_covariant
        )
        metric_line = rf"(g_{{\mu\nu}})=\begin{{pmatrix}}{rows}\end{{pmatrix}}"

    mode_text = {
        ScalarFieldMode.ABSENT: "no fijado",
        ScalarFieldMode.GENERIC: "genérico, sin especialización",
        ScalarFieldMode.SPECIALIZED: "perfil especializado explícitamente",
    }[ansatz.scalar_field_mode]
    scalar_line = (
        rf"\phi\ \text{{{mode_text}}}"
        if ansatz.scalar_field is None
        else rf"\phi = {expr_to_latex(ansatz.scalar_field)}"
        rf"\qquad\text{{({mode_text})}}"
    )
    chart_line = rf"(x^\mu)=\left({', '.join(coordinates)}\right)"
    return (
        r"\begin{dmath*}[breakdepth={3}]",
        rf"{chart_line},\qquad {metric_line}",
        r"\end{dmath*}",
        r"\begin{dmath*}[breakdepth={3}]",
        scalar_line,
        r"\end{dmath*}",
    )


def _indices_to_latex(indices: tuple[Any, ...]) -> str:
    if not indices:
        return r"\varnothing\;\text{(escalar)}"
    return r",\;".join(
        (
            rf"{_latex_index(index.name)}^{{\uparrow}}"
            if index.variance is Variance.UP
            else rf"{_latex_index(index.name)}_{{\downarrow}}"
        )
        for index in indices
    )


def _component_label(
    label: str, indices: tuple[Any, ...], position: tuple[int, ...],
    *, component_indices: tuple[Any, ...] | None = None,
) -> str:
    # A canonical expression can enumerate its free axes in a different order
    # from the mathematical label. Reorder coordinates, not values or JSON keys.
    if component_indices is not None:
        by_index = dict(zip(component_indices, position, strict=True))
        position = tuple(by_index[index] for index in indices)
    upper = "".join(
        str(value)
        for index, value in zip(indices, position, strict=True)
        if index.variance is Variance.UP
    )
    lower = "".join(
        str(value)
        for index, value in zip(indices, position, strict=True)
        if index.variance is Variance.DOWN
    )
    result = rf"\left[{label}\right]"
    if lower:
        result += rf"_{{{lower}}}"
    if upper:
        result += rf"^{{{upper}}}"
    return result


_COMPACT_PROJECTION_STATUS = {
    "completed": "completada",
    "symbolic": "simbólica",
    "unavailable": "no disponible",
}


def _append_compact_expression(
    lines: list[str],
    view: ReportPresentation,
    *,
    key: str,
    label: str,
    record: DisplayExpression,
) -> None:
    lines.extend(
        (
            _compact_audit(key, record),
            r"\begin{dmath*}[breakdepth={5}]",
            rf"\text{{forma compacta:}}\quad {label} = {_display_record_latex(view, record)}",
            r"\end{dmath*}",
        )
    )


def _append_compact_abstract(
    lines: list[str],
    view: ReportPresentation,
) -> None:
    if not view.compact_decompositions:
        return
    lines.extend(
        (
            r"\Needspace{12\baselineskip}",
            r"\par\bigskip\noindent{\large\bfseries Descomposición compacta adicional}\par",
            r"Esta vista se agrega después de los resultados anteriores y reutiliza exactamente "
            r"la IR canónica ya calculada. No interviene en la validación ni en los fingerprints.\par",
        )
    )
    for decomposition in view.compact_decompositions:
        lines.extend(
            (
                r"\Needspace{10\baselineskip}",
                rf"\par\medskip\noindent\textbf{{Descomposición de $ {decomposition.label_latex} $.}}\par",
                r"\begin{dmath*}[breakdepth={5}]",
                decomposition.formula_latex,
                r"\end{dmath*}",
                (
                    rf"\textbf{{Reconstrucción IR:}} {_latex_text(decomposition.reconstruction_status)}. "
                    rf"{_latex_text(decomposition.reconstruction_reason)}\par"
                ),
            )
        )
        _append_compact_expression(
            lines,
            view,
            key=f"abstract.compact.{decomposition.key}",
            label=decomposition.label_latex,
            record=decomposition.expression,
        )
        if decomposition.key == "curvature_momentum":
            continue
        for block in decomposition.blocks:
            lines.append(
                rf"\par\smallskip\noindent\textbf{{Bloque $ {block.label_latex} $}} "
                rf"(fuentes: \texttt{{{_latex_text(', '.join(block.source_keys))}}}).\par"
            )
            _append_compact_expression(
                lines,
                view,
                key=f"abstract.compact.{decomposition.key}.{block.key}",
                label=block.label_latex,
                record=block.expression,
            )


def _append_compact_projection(
    lines: list[str],
    view: ReportPresentation,
    *,
    key: str,
    label: str,
    projection: CompactProjection,
) -> None:
    lines.append(
        rf"\textbf{{Estado de proyección:}} "
        rf"{_COMPACT_PROJECTION_STATUS[projection.status]}. "
        rf"{_latex_text(projection.reason)}\par"
    )
    if projection.status != "completed":
        return
    if not projection.free_indices:
        position, record = projection.components[0]
        _append_compact_expression(
            lines,
            view,
            key=key,
            label=rf"{label}\big|_{{\mathrm{{ansatz}}}}",
            record=record,
        )
        return
    if not projection.components:
        lines.append(r"Todas las componentes de este bloque son nulas.\par")
        return
    displayed = projection.components[:12]
    for position, record in displayed:
        component_label = _component_label(
            label,
            projection.free_indices,
            position,
            component_indices=projection.free_indices,
        )
        _append_compact_expression(
            lines,
            view,
            key=f"{key}[{','.join(map(str, position))}]",
            label=component_label,
            record=record,
        )
    if len(projection.components) > len(displayed):
        lines.append(
            rf"Se muestran {len(displayed)} de {len(projection.components)} "
            r"componentes no nulas; el conjunto completo está en "
            r"\texttt{presentation.json}.\par"
        )


def _append_compact_projected(
    lines: list[str],
    view: ReportPresentation,
    ansatz_name: str,
) -> None:
    if not view.compact_decompositions:
        return
    lines.extend(
        (
            r"\Needspace{12\baselineskip}",
            r"\par\bigskip\noindent{\large\bfseries Descomposición compacta adicional}\par",
            rf"\textbf{{Ansatz de estos bloques:}} \texttt{{{_latex_text(ansatz_name)}}}. "
            r"Las formas compactas siguientes son únicamente vistas de las "
            r"componentes canónicas. Las formas expandidas permanecen disponibles en "
            r"\texttt{presentation.json}.\par",
        )
    )
    for decomposition in view.compact_decompositions:
        lines.extend(
            (
                r"\Needspace{10\baselineskip}",
                rf"\par\medskip\noindent\textbf{{Descomposición proyectada de $ {decomposition.label_latex} $.}}\par",
                r"\begin{dmath*}[breakdepth={5}]",
                decomposition.formula_latex,
                r"\end{dmath*}",
                (
                    rf"\textbf{{Reconstrucción por componentes:}} "
                    rf"{_latex_text(decomposition.projection_reconstruction_status)}. "
                    rf"{_latex_text(decomposition.projection_reconstruction_reason)}\par"
                ),
            )
        )
        for block in decomposition.blocks:
            lines.append(
                rf"\par\smallskip\noindent\textbf{{Bloque $ {block.label_latex} $.}}\par"
            )
            _append_compact_projection(
                lines,
                view,
                key=f"projected.compact.{decomposition.key}.{block.key}",
                label=block.label_latex,
                projection=block.projection,
            )


def _append_abstract_results(lines: list[str], package: RunPackage, view: ReportPresentation) -> None:
    lines.append(r"\section*{Expresiones tensoriales abstractas}")
    lines.append(
        r"Las expresiones de esta sección son covariantes y no incorporan sustituciones "
        r"del ansatz."
    )
    if package.abstract is None:
        lines.append(
            r"Este bundle heredado no contiene la vista abstracta estructurada; "
            r"los objetos canónicos permanecen disponibles en \texttt{results.json}."
        )
        return
    for key, expression in package.abstract.expression_items():
        record = package.abstract.record(key)
        label = _REPORT_LABELS[key]
        lines.extend(
            (
                r"\Needspace{9\baselineskip}",
                rf"\subsection*{{$ {label} $}}",
                _presentation_audit(view, f"abstract.{key}"),
                r"\begin{dmath*}[breakdepth={5}]",
                rf"{label} = {_presentation_latex(view, f'abstract.{key}')}",
                r"\end{dmath*}",
                r"\par\vspace{1.2\baselineskip}\noindent",
                r"\begin{minipage}{\linewidth}",
                rf"\textbf{{Significado:}} {_latex_text(record.description)}\par",
                rf"\textbf{{Índices libres y varianzas:}} $ {_indices_to_latex(record.free_indices)} $\par",
                rf"\textbf{{Fuentes del pipeline:}} \texttt{{{_latex_text(', '.join(record.source_keys))}}}\par",
                rf"\textbf{{Wolfram/xAct:}} {_XACT_STATUS_TEXT[record.xact_status]}. {_latex_text(record.validation_note)}\par",
                r"\end{minipage}",
            )
        )
        if key == "metric_euler" and package.derived is not None:
            lines.extend(
                (
                    _presentation_audit(view, "abstract.curvature_derivative_metric_term"),
                    r"\begin{dmath*}[breakdepth={5}]",
                    (
                        r"\left.E_{ab}\right|_{\nabla\nabla P} "
                        rf"= {_presentation_latex(view, 'abstract.curvature_derivative_metric_term')}"
                    ),
                    r"\end{dmath*}",
                    r"\par\vspace{0.8\baselineskip}\noindent",
                    r"\begin{minipage}{\linewidth}",
                    r"La expresión anterior identifica la contribución "
                    r"$-2\nabla^c\nabla^dP_{acdb}$ dentro de $E_{ab}$.\par",
                    r"\end{minipage}",
                )
            )
    _append_compact_abstract(lines, view)


def _append_projected_results(lines: list[str], package: RunPackage, view: ReportPresentation) -> None:
    lines.append(r"\section*{Expresiones proyectadas mediante el ansatz}")
    if package.projected is None or package.abstract is None:
        lines.append(
            r"Este bundle heredado no contiene una vista proyectada estructurada."
        )
        return
    ansatz_text = (
        "ninguno"
        if package.projected.ansatz_name is None
        else package.projected.ansatz_name
    )
    lines.append(rf"\textbf{{Ansatz utilizado:}} \texttt{{{_latex_text(ansatz_text)}}}.")
    if package.projected.ansatz_geometry is not None:
        lines.extend(_ansatz_summary_latex(package.projected.ansatz_geometry))
    lines.append(
        r"Las componentes completas se conservan de forma dispersa en "
        r"\texttt{results.json}; el documento muestra como máximo doce componentes "
        r"no nulas por tensor."
    )
    for key in REPORT_QUANTITY_KEYS:
        abstract_expression = getattr(package.abstract, key)
        abstract_record = package.abstract.record(key)
        projected = package.projected.quantity(key)
        label = _REPORT_LABELS[key]
        lines.extend(
            (
                r"\Needspace{7\baselineskip}",
                rf"\subsection*{{$ {label} $}}",
                (
                    rf"\textbf{{Ansatz:}} \texttt{{{_latex_text(ansatz_text)}}}; "
                    rf"\textbf{{estado:}} {_PROJECTION_STATUS_TEXT[projected.status]}.\par"
                ),
            )
        )
        if projected.components is None:
            lines.extend(
                (
                    r"La forma abstracta se conserva sin sustitución:",
                    _presentation_audit(view, f"projected.{key}.abstract_fallback"),
                    r"\begin{dmath*}[breakdepth={5}]",
                    rf"{label} = {_presentation_latex(view, f'projected.{key}.abstract_fallback')}",
                    r"\end{dmath*}",
                    r"\par\smallskip\noindent",
                    rf"\textbf{{Motivo:}} {_latex_text(projected.reason)}\par",
                )
            )
            continue
        components = projected.components
        total = components.dimension ** len(abstract_record.free_indices)
        nonzero = len(components.values)
        if not abstract_record.free_indices:
            lines.extend(
                (
                    _presentation_audit(view, f"projected.{key}.scalar"),
                    r"\begin{dmath*}[breakdepth={5}]",
                    rf"{label}\big|_{{\mathrm{{ansatz}}}} = {_presentation_latex(view, f'projected.{key}.scalar')}",
                    r"\end{dmath*}",
                    r"\par\smallskip",
                )
            )
        elif nonzero == 0:
            lines.append(_presentation_audit(view, f"projected.{key}.zero"))
            lines.append(r"Todas las componentes de esta cantidad son nulas.")
        else:
            displayed = components.values[:12]
            for position, expression in displayed:
                display_key = f"projected.{key}[{','.join(map(str, position))}]"
                lines.extend(
                    (
                        _presentation_audit(view, display_key),
                        r"\begin{dmath*}[breakdepth={5}]",
                        rf"{_component_label(label, abstract_record.free_indices, position, component_indices=components.free_indices)} = {_presentation_latex(view, display_key)}",
                        r"\end{dmath*}",
                        r"\par\smallskip",
                    )
                )
            if nonzero > len(displayed):
                lines.append(
                    rf"Representación compacta: se muestran {len(displayed)} de "
                    rf"{nonzero} componentes no nulas."
                )
        lines.append(
            rf"Proyección completa almacenada: {nonzero} componentes no nulas de {total}. "
            rf"{_latex_text(projected.reason)}\par\medskip"
        )


def _append_scalar_profiles(lines: list[str], view: ReportPresentation) -> None:
    if not view.scalar_profiles:
        return
    lines.extend((
        r"\par\medskip Extras: perfiles escalares de una variable, con la misma métrica del ansatz genérico. "
        r"Se sustituyen únicamente las componentes de $L$ y $P^{abcd}$ ya calculadas; "
        r"sin validación xAct independiente de estos perfiles. "
        r"Las componentes completas de estos extras constan en \texttt{presentation.json}.",
    ))
    for profile in view.scalar_profiles:
        lines.extend((
            r"\Needspace{8\baselineskip}",
            rf"\subsection*{{Extra: $\phi={display_expr_to_latex(profile.scalar_field)}$}}",
        ))
        for key, projection in profile.quantities:
            label = _REPORT_LABELS[key]
            if projection.status != "completed":
                lines.append(rf"${label}$: {_COMPACT_PROJECTION_STATUS[projection.status]}. "
                             rf"{_latex_text(projection.reason)}\par")
                continue
            displayed, independent = (
                independent_curvature_components(projection)
                if key == "curvature_momentum" else (projection.components, False)
            )
            if not displayed:
                lines.append(rf"\[{label}=0\]")
                continue
            if independent:
                lines.append(r"Componentes independientes no nulas; "
                             r"$P^{abcd}=-P^{bacd}=-P^{abdc}=P^{cdab}$.")
            for position, record in displayed[:6]:
                component_label = _component_label(label, projection.free_indices, position)
                if not projection.free_indices:
                    component_label = label
                lines.extend((
                    r"\begin{dmath*}[breakdepth={5}]",
                    rf"{component_label}={_display_record_latex(view, record)}",
                    r"\end{dmath*}",
                ))
            if len(displayed) > 6:
                lines.append(rf"Se muestran 6 de {len(displayed)} componentes; "
                             r"resto en \texttt{presentation.json}.")


def _append_specialized_results(
    lines: list[str],
    package: RunPackage,
    view: ReportPresentation,
) -> None:
    if package.specialized is None or package.abstract is None or package.projected is None:
        return
    specialized = package.specialized
    lines.append(r"\section*{Resultados especializados mediante el ansatz}")
    lines.append(
        rf"\textbf{{Ansatz base:}} \texttt{{{_latex_text(specialized.base_ansatz_name)}}}; "
        rf"\textbf{{ansatz especializado:}} \texttt{{{_latex_text(specialized.ansatz_name)}}}.\par"
    )
    base_geometry = package.projected.ansatz_geometry

    def metric_function_label(name: str) -> str:
        if base_geometry is not None:
            for row in base_geometry.metric_covariant:
                for entry in row:
                    for node in walk(entry):
                        if isinstance(node, Function) and node.name == name:
                            arguments = ", ".join(expr_to_latex(arg) for arg in node.arguments)
                            return rf"{_latex_name(name)}\!\left({arguments}\right)"
        return _latex_name(name)

    metric_inputs = r",\quad ".join(
        rf"{metric_function_label(name)} = {display_expr_to_latex(expression)}"
        for name, expression in specialized.specialization.metric_functions.items()
    ) or r"\text{sin sustituciones métricas}"
    scalar_input = (
        r"\text{sin sustitución escalar}"
        if specialized.specialization.scalar_field is None
        else display_expr_to_latex(specialized.specialization.scalar_field)
    )
    lines.extend(
        (
            r"\begin{dmath*}[breakdepth={4}]",
            rf"{metric_inputs},\qquad \phi = {scalar_input}",
            r"\end{dmath*}",
            r"La derivación abstracta y la proyección genérica preceden a esta etapa. "
            r"Las componentes especializadas completas se conservan en \texttt{results.json}; "
            r"el documento muestra como máximo doce componentes no nulas por tensor.",
        )
    )
    for key in REPORT_QUANTITY_KEYS:
        label = _REPORT_LABELS[key]
        abstract_record = package.abstract.record(key)
        projected = package.projected.quantity(key)
        result = specialized.quantity(key)
        lines.extend(
            (
                r"\Needspace{10\baselineskip}",
                rf"\subsection*{{$ {label} $}}",
                rf"\textbf{{Funciones usadas:}} $ {metric_inputs},\;\phi={scalar_input} $.\par",
                r"\textbf{Resultado abstracto:}",
                _presentation_audit(view, f"abstract.{key}"),
                r"\begin{dmath*}[breakdepth={5}]",
                rf"{label} = {_presentation_latex(view, f'abstract.{key}')}",
                r"\end{dmath*}",
                (
                    rf"\textbf{{Resultado proyectado:}} "
                    rf"{_PROJECTION_STATUS_TEXT[projected.status]}. "
                    rf"{_latex_text(projected.reason)}\par"
                ),
                (
                    rf"\textbf{{Resultado especializado:}} "
                    rf"{_PROJECTION_STATUS_TEXT[result.status]}.\par"
                ),
            )
        )
        if result.components is None:
            lines.extend(
                (
                    _presentation_audit(view, f"specialized.{key}.abstract_fallback"),
                    r"\begin{dmath*}[breakdepth={5}]",
                    rf"{label} = {_presentation_latex(view, f'specialized.{key}.abstract_fallback')}",
                    r"\end{dmath*}",
                    rf"\textbf{{Motivo:}} {_latex_text(result.reason)}\par",
                )
            )
        else:
            components = result.components
            total = components.dimension ** len(abstract_record.free_indices)
            nonzero = len(components.values)
            if not abstract_record.free_indices:
                display_key = f"specialized.{key}.scalar"
                lines.extend(
                    (
                        _presentation_audit(view, display_key),
                        r"\begin{dmath*}[breakdepth={5}]",
                        rf"{label}\big|_{{\mathrm{{especializado}}}} = {_presentation_latex(view, display_key)}",
                        r"\end{dmath*}",
                    )
                )
            elif nonzero == 0:
                lines.append(_presentation_audit(view, f"specialized.{key}.zero"))
                lines.append(r"Todas las componentes especializadas son nulas.\par")
            else:
                for position, _ in components.values[:12]:
                    display_key = f"specialized.{key}[{','.join(map(str, position))}]"
                    lines.extend(
                        (
                            _presentation_audit(view, display_key),
                            r"\begin{dmath*}[breakdepth={5}]",
                            rf"{_component_label(label, abstract_record.free_indices, position, component_indices=components.free_indices)} = {_presentation_latex(view, display_key)}",
                            r"\end{dmath*}",
                        )
                    )
                if nonzero > 12:
                    lines.append(
                        rf"Se muestran 12 de {nonzero} componentes no nulas; "
                        r"el conjunto completo está en \texttt{results.json}.\par"
                    )
            lines.append(
                rf"\textbf{{Almacenamiento:}} {nonzero} componentes no nulas de {total}. "
                rf"{_latex_text(result.reason)}\par"
            )
        lines.append(
            rf"\textbf{{Wolfram/xAct:}} {_XACT_STATUS_TEXT[abstract_record.xact_status]}. "
            rf"{_latex_text(abstract_record.validation_note)} "
            r"La especialización coordenada no se presenta como una validación xAct independiente.\par\medskip"
        )


def latex_report(package: RunPackage, display_policy: DisplayPolicy | None = None, *,
                 projected_assumptions: tuple[str, ...] = (),
                 presentation: ReportPresentation | None = None) -> str:
    """Presenta las vistas abstracta, proyectada y, si existe, especializada."""

    view = presentation or build_presentation(package, display_policy, projected_assumptions=projected_assumptions)
    if view.run_id != package.run_id:
        raise ValueError("La presentación pertenece a otra corrida.")
    summary = package.verification.summary
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=2.2cm]{geometry}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{breqn}",
        r"\usepackage{needspace}",
        r"\usepackage[T1]{fontenc}",
        r"\begin{document}",
        r"\begin{center}",
        r"{\Large\bfseries TensorEngine: informe de corrida}",
        r"\end{center}",
        rf"\textbf{{Modelo:}} \texttt{{{_latex_text(package.model.name)}}}\\",
        rf"\textbf{{Run ID:}} \texttt{{{_latex_text(package.run_id)}}}\\",
        rf"\textbf{{Estado:}} \texttt{{{_latex_text(package.verification.status.value)}}}\\",
        rf"\textbf{{Backend:}} \texttt{{{_latex_text(package.verification.backend_name)} {_latex_text(package.verification.backend_version)}}}\\",
        rf"\textbf{{Verificaciones:}} {summary['passed']} aprobadas, {summary['failed']} fallidas, {summary['undetermined']} indeterminadas.",
    ]
    mode = "simplificación protegida" if view.policy.enabled else "simplificación desactivada"
    lines.extend((r"\par\smallskip", rf"\textbf{{Presentación:}} {mode}, sin modificar los objetos canónicos. "
                  r"Las operaciones e hipótesis usadas por expresión constan en \texttt{presentation.json} "
                  r"y en los comentarios del archivo LaTeX. La validación xAct corresponde a la IR canónica.\par"))
    if package.delta_contractions:
        substitutions = sum(e.action == "substitute" for a in package.delta_contractions for e in a.events)
        traces = sum(e.action == "trace" for a in package.delta_contractions for e in a.events)
        remaining = sum(delta_count(expr) for _, expr in package.abstract.expression_items()) if package.abstract else 0
        lines.append(
            rf"\textbf{{Deltas canónicos:}} {substitutions} sustituciones y {traces} trazas registradas "
            rf"en las pasadas del álgebra (incluidas verificaciones); {remaining} deltas explícitos "
            rf"en las {len(REPORT_QUANTITY_KEYS)} cantidades finales. Detalle e índices sustituidos en \texttt{{delta\_contractions.json}}.\par"
        )
    transport_failures = tuple(
        check for check in package.verification.checks if check.diagnostic is not None
    )
    if transport_failures:
        lines.extend(
            (
                r"\par\smallskip\textbf{Diagnósticos estructurados IR--xAct.} "
                r"El fragmento JSON completo se conserva en \texttt{verification.json} "
                r"y \texttt{results.json}.",
                r"\begin{itemize}",
            )
        )
        for check in transport_failures:
            diagnostic = check.diagnostic
            assert diagnostic is not None
            path = ".".join(diagnostic.path) or "residual"
            identity = diagnostic.node_type or diagnostic.category
            if diagnostic.symbol is not None:
                identity += f" {diagnostic.symbol}"
            lines.append(
                rf"\item \texttt{{{_latex_text(check.key)}}}: "
                rf"\texttt{{{_latex_text(diagnostic.code)}}} en "
                rf"\texttt{{{_latex_text(path)}}}; "
                rf"{_latex_text(identity)}. {_latex_text(diagnostic.reason)}"
            )
        lines.extend((r"\end{itemize}", r"\par\smallskip"))
    _append_abstract_results(lines, package, view)
    _append_projected_results(lines, package, view)
    _append_scalar_profiles(lines, view)
    _append_specialized_results(lines, package, view)
    lines.extend(
        (
            r"\bigskip\noindent\textbf{Política de lectura.} "
            r"Este PDF/LaTeX es una vista. El archivo \texttt{results.json} conserva la representación intermedia canónica y la trazabilidad completa.",
            r"\end{document}",
            "",
        )
    )
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ExportBundle:
    output_directory: Path
    manifest_path: Path
    manifest: RunManifest
    pdf_diagnostic: str | None = None
    presentation: ReportPresentation | None = None

    def to_stage_result(self, duration_seconds: float = 0.0) -> StageResult:
        status = self.manifest.status
        diagnostics: tuple[Diagnostic, ...] = ()
        if status is StageStatus.FAILED:
            diagnostics = (
                Diagnostic("E_EXPORTED_FAILED_RUN", "La corrida se exportó para auditoría, pero contiene verificaciones fallidas.", Severity.ERROR),
            )
        elif status is StageStatus.PARTIAL:
            diagnostics = (
                Diagnostic("W_EXPORTED_PARTIAL_RUN", "La corrida exportada conserva verificaciones indeterminadas.", Severity.WARNING),
            )
        if self.pdf_diagnostic is not None:
            diagnostics = diagnostics + (
                Diagnostic(
                    "W_PDF_NOT_GENERATED",
                    self.pdf_diagnostic,
                    Severity.WARNING,
                ),
            )
        manifest_payload = self.manifest.to_data()
        artifacts_payload = {
            "output_directory": str(self.output_directory.resolve()),
            "manifest_path": str(self.manifest_path.resolve()),
            "files": [item.to_data() for item in self.manifest.files],
        }
        return StageResult(
            stage_key="export",
            status=status,
            backend="python",
            operation="export_run",
            inputs=(
                "validated_model",
                "verification_report",
                "derived_quantities",
                "abstract_results",
                "projected_results",
            ),
            artifacts=(
                ArtifactRecord(
                    "run_manifest",
                    "run_manifest",
                    tuple(manifest_payload.items()),
                    (
                        "validated_model",
                        "verification_report",
                        "derived_quantities",
                        "abstract_results",
                        "projected_results",
                    ),
                ),
                ArtifactRecord("exported_artifacts", "file_bundle", tuple(artifacts_payload.items()), ("run_manifest",)),
            ),
            diagnostics=diagnostics,
            duration_seconds=duration_seconds,
        )


class RunExporter:
    """Materializa archivos atómicamente dentro de una raíz controlada."""

    def __init__(
        self,
        output_root: str | Path,
        *,
        compile_pdf: bool = True,
        pdf_timeout_seconds: float = 90.0,
        display_policy: DisplayPolicy | None = None,
        projected_assumptions: tuple[str, ...] = (),
        component_backend: SympyComponentBackend | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.compile_pdf = compile_pdf
        self.pdf_timeout_seconds = pdf_timeout_seconds
        self.display_policy = display_policy or DisplayPolicy()
        self.projected_assumptions = tuple(projected_assumptions)
        # This optional backend is used only to project presentation-only
        # decomposition blocks that are not persisted independently in a run.
        self.component_backend = component_backend

    def _compile_pdf(self, directory: Path) -> tuple[bytes | None, str | None]:
        compiler = shutil.which("pdflatex") or shutil.which("xelatex")
        if compiler is None:
            return None, "No se encontró pdflatex ni xelatex; report.tex sigue disponible."
        pdf_path = directory / "report.pdf"
        pdf_path.unlink(missing_ok=True)
        try:
            environment = os.environ.copy()
            environment["SOURCE_DATE_EPOCH"] = "946684800"
            environment["FORCE_SOURCE_DATE"] = "1"
            completed = subprocess.run(
                [
                    compiler,
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "report.tex",
                ],
                cwd=directory,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.pdf_timeout_seconds,
                shell=False,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return None, f"No se pudo compilar report.tex a PDF: {error}"
        finally:
            for suffix in ("aux", "log", "out"):
                (directory / f"report.{suffix}").unlink(missing_ok=True)
        if completed.returncode != 0 or not pdf_path.is_file():
            diagnostic = "\n".join(
                item.strip()
                for item in (completed.stderr, completed.stdout[-2000:])
                if item and item.strip()
            )
            pdf_path.unlink(missing_ok=True)
            return (
                None,
                "La compilación LaTeX no produjo report.pdf. " + diagnostic,
            )
        return pdf_path.read_bytes(), None

    @staticmethod
    def _write_atomic(path: Path, content: str) -> bytes:
        encoded = content.encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".tensor-engine-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return encoded

    def export(
        self,
        package: RunPackage,
        *,
        created_at_utc: str | None = None,
    ) -> ExportBundle:
        root = self.output_root.resolve()
        directory = (root / f"{_slug(package.model.name)}-{package.run_id[4:16]}").resolve()
        if root != directory and root not in directory.parents:
            raise ValueError("La carpeta calculada de exportación escapa de la raíz autorizada.")
        directory.mkdir(parents=True, exist_ok=True)

        presentation = build_presentation(
            package,
            self.display_policy,
            projected_assumptions=self.projected_assumptions,
            component_backend=self.component_backend,
        )
        presentation_data = presentation.to_data()
        for key, record in presentation.expressions:
            presentation_data["expressions"][key]["latex"] = _presentation_latex(presentation, key)
            presentation_data["expressions"][key]["formatting_operations"] = (
                ["latex_signs_fractions_index_groups"] if self.display_policy.enabled else []
            )
        for decomposition, decomposition_data in zip(
            presentation.compact_decompositions,
            presentation_data["compact_decompositions"],
            strict=True,
        ):
            decomposition_data["compact"]["latex"] = _display_record_latex(
                presentation,
                decomposition.expression,
            )
            decomposition_data["expanded_latex"] = expr_to_latex(
                decomposition.expression.canonical
            )
            for (_, component), component_data in zip(
                decomposition.projection.components,
                decomposition_data["projection"]["components"],
                strict=True,
            ):
                component_data["latex"] = _display_record_latex(
                    presentation,
                    component,
                )
                component_data["expanded_latex"] = expr_to_latex(
                    component.canonical
                )
            for block, block_data in zip(
                decomposition.blocks,
                decomposition_data["blocks"],
                strict=True,
            ):
                block_data["compact"]["latex"] = _display_record_latex(
                    presentation,
                    block.expression,
                )
                block_data["expanded_latex"] = expr_to_latex(
                    block.expression.canonical
                )
                for (_, component), component_data in zip(
                    block.projection.components,
                    block_data["projection"]["components"],
                    strict=True,
                ):
                    component_data["latex"] = _display_record_latex(
                        presentation,
                        component,
                    )
                    component_data["expanded_latex"] = expr_to_latex(
                        component.canonical
                    )
        contents = {
            "results": ("results.json", "application/json", _canonical_json(package.to_data(), pretty=True)),
            "verification": ("verification.json", "application/json", _canonical_json(package.verification.to_data(), pretty=True)),
            "latex_report": ("report.tex", "application/x-tex", latex_report(package, presentation=presentation)),
            "presentation": ("presentation.json", "application/json", _canonical_json(presentation_data, pretty=True)),
        }
        if package.delta_contractions:
            contents["delta_contractions"] = (
                "delta_contractions.json", "application/json", _canonical_json({
                    "schema_version": "1.0", "run_id": package.run_id,
                    "passes": [a.to_data() for a in package.delta_contractions],
                    "final_abstract_counts": {} if package.abstract is None else {
                        key: delta_count(expr) for key, expr in package.abstract.expression_items()
                    },
                }, pretty=True),
            )
        files: list[ExportedFile] = []
        for key, (name, media_type, content) in contents.items():
            encoded = self._write_atomic(directory / name, content)
            files.append(ExportedFile(key, name, media_type, _sha256_bytes(encoded), len(encoded)))

        pdf_diagnostic: str | None = None
        if self.compile_pdf:
            pdf_content, pdf_diagnostic = self._compile_pdf(directory)
            if pdf_content is not None:
                files.append(
                    ExportedFile(
                        "pdf_report",
                        "report.pdf",
                        "application/pdf",
                        _sha256_bytes(pdf_content),
                        len(pdf_content),
                    )
                )

        if created_at_utc is None:
            created_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        records = package.expression_records()
        manifest = RunManifest(
            run_id=package.run_id,
            created_at_utc=created_at_utc,
            status=package.verification.status,
            model_name=package.model.name,
            model_schema_version=package.model.schema_version,
            convention_id=package.model.conventions.convention_id,
            dimension=str(package.model.dimension.value),
            backend_name=package.verification.backend_name,
            backend_version=package.verification.backend_version,
            external_sources=package.verification.external_sources,
            duration_seconds=package.duration_seconds,
            stage_durations=package.stage_durations,
            expressions=tuple(
                (record.key, record.form.value, _sha256_data(record.expression.to_data()))
                for record in records
            ),
            files=tuple(files),
            abstract_quantities=(
                ()
                if package.abstract is None
                else tuple(key for key, _ in package.abstract.expression_items())
            ),
            projected_quantities=(
                ()
                if package.projected is None
                else tuple(
                    (
                        item.key,
                        item.status.value,
                        _sha256_data(
                            item.components.to_data()
                            if item.components is not None
                            else {"reason": item.reason, "ansatz": item.ansatz_name}
                        ),
                    )
                    for item in package.projected.quantities
                )
            ),
            specialized_quantities=(
                ()
                if package.specialized is None
                else tuple(
                    (
                        item.key,
                        item.status.value,
                        _sha256_data(
                            item.components.to_data()
                            if item.components is not None
                            else {
                                "reason": item.reason,
                                "ansatz": item.ansatz_name,
                            }
                        ),
                    )
                    for item in package.specialized.quantities
                )
            ),
            external_bindings=package.verification.external_bindings,
            adjudications=package.verification.adjudications,
        )
        manifest_path = directory / "manifest.json"
        self._write_atomic(manifest_path, _canonical_json(manifest.to_data(), pretty=True))
        return ExportBundle(directory, manifest_path, manifest, pdf_diagnostic, presentation)
