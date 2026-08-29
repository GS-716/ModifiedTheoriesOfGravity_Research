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
import tempfile
from typing import Any, Mapping

from .components import ComponentFieldEquations
from .contracts import (
    ArtifactRecord,
    Diagnostic,
    ExpressionForm,
    ExpressionRecord,
    Severity,
    StageResult,
    StageStatus,
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
)
from .model import ModelSpec
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
    duration_seconds: float = 0.0
    stage_durations: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_durations", tuple(tuple(item) for item in self.stage_durations))
        if self.verification.model_name != self.model.name:
            raise ValueError("El informe de verificación pertenece a otro modelo.")
        if self.duration_seconds < 0 or any(value < 0 for _, value in self.stage_durations):
            raise ValueError("Las duraciones de corrida no pueden ser negativas.")
        keys = [key for key, _ in self.stage_durations]
        if len(keys) != len(set(keys)):
            raise ValueError("Las duraciones por etapa contienen claves repetidas.")

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
        return tuple(records)

    def semantic_data(self) -> dict[str, Any]:
        """Contenido que identifica la corrida; excluye reloj, rutas y tiempos."""

        return {
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "model": self.model.to_data(),
            "normalized_lagrangian": self.lagrangian.to_data(),
            "momenta": self.momenta.to_data(),
            "raw_variation": self.raw_variation.to_data(),
            "euler_lagrange": self.euler.to_data(),
            "noether_wald": None if self.noether is None else self.noether.to_data(),
            "components": None if self.components is None else self.components.to_data(),
            "verification": self.verification.to_data(),
        }

    @property
    def run_id(self) -> str:
        return f"run_{_sha256_data(self.semantic_data())[:20]}"

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "run_id": self.run_id,
            **self.semantic_data(),
            "normalization_explicit": self.normalized_lagrangian is not None,
            "timing": {
                "duration_seconds": self.duration_seconds,
                "stages": {key: value for key, value in self.stage_durations},
            },
            "expressions": [record.to_data() for record in self.expression_records()],
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "RunPackage":
        timing = data.get("timing", {})
        component_data = data.get("components")
        noether_data = data.get("noether_wald")
        package = cls(
            model=ModelSpec.from_data(data["model"]),
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
            duration_seconds=float(timing.get("duration_seconds", 0.0)),
            stage_durations=tuple(
                (str(key), float(value)) for key, value in timing.get("stages", {}).items()
            ),
        )
        supplied = data.get("run_id")
        if supplied is not None and supplied != package.run_id:
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
    external_bindings: tuple[tuple[str, str, str], ...] = ()
    adjudications: tuple[tuple[str, str, str], ...] = ()
    schema_version: str = EXPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "external_sources", tuple(tuple(item) for item in self.external_sources))
        object.__setattr__(self, "stage_durations", tuple(tuple(item) for item in self.stage_durations))
        object.__setattr__(self, "expressions", tuple(tuple(item) for item in self.expressions))
        object.__setattr__(self, "files", tuple(self.files))
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
        if any(len(item) != 3 for item in self.external_bindings):
            raise ValueError("Los vínculos externos del manifiesto tienen un contrato inválido.")
        if any(len(item) != 3 for item in self.adjudications):
            raise ValueError("Las adjudicaciones del manifiesto tienen un contrato inválido.")

    def to_data(self) -> dict[str, Any]:
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
    "phi": r"\phi",
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


_DISPLAY_LABELS = {
    "original_lagrangian": r"L_{\mathrm{original}}",
    "lagrangian": r"L",
    "metric_momentum": r"M_{ab}",
    "curvature_momentum": r"P^{abcd}",
    "scalar_gradient_momentum": r"J^a",
    "scalar_derivative": r"F_{\phi}",
    "delta_lagrangian": r"\delta L",
    "metric_euler": r"E_{ab}",
    "scalar_euler": r"E_{\phi}",
    "boundary_potential_metric": r"\Theta^a_g",
    "boundary_potential_scalar": r"\Theta^a_\phi",
    "boundary_potential_total": r"\Theta^a",
    "full_variation": r"\delta L_{\mathrm{IBP}}",
    "density_variation": r"\delta(\sqrt{-g}L)",
    "noether_current": r"J^a_\xi",
    "constraint_current": r"C^a_\xi",
    "charge_potential": r"Q^{ab}_\xi",
    "noether_identity": r"\mathcal{N}_\xi",
}


def latex_report(package: RunPackage) -> str:
    """Crea un documento autocontenido de presentación para la corrida."""

    summary = package.verification.summary
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=2.2cm]{geometry}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{breqn}",
        r"\usepackage[T1]{fontenc}",
        r"\begin{document}",
        r"\section*{TensorEngine: informe de corrida}",
        rf"\textbf{{Modelo:}} \texttt{{{_latex_text(package.model.name)}}}\\",
        rf"\textbf{{Run ID:}} \texttt{{{_latex_text(package.run_id)}}}\\",
        rf"\textbf{{Estado:}} \texttt{{{_latex_text(package.verification.status.value)}}}\\",
        rf"\textbf{{Backend:}} \texttt{{{_latex_text(package.verification.backend_name)} {_latex_text(package.verification.backend_version)}}}\\",
        rf"\textbf{{Verificaciones:}} {summary['passed']} aprobadas, {summary['failed']} fallidas, {summary['undetermined']} indeterminadas.",
        r"\section*{Resultados covariantes}",
    ]
    for record in package.expression_records():
        label = _DISPLAY_LABELS.get(record.key, rf"\mathrm{{{_latex_text(record.key)}}}")
        lines.extend((r"\begin{dmath*}[breakdepth={5}]", rf"{label} = {expr_to_latex(record.expression)}", r"\end{dmath*}"))
    if package.components is not None:
        lines.append(r"\section*{Componentes independientes}")
        lines.append(rf"Ansatz: \texttt{{{_latex_text(package.components.ansatz_name)}}}.")
        for position, expression in package.components.independent_metric:
            lines.extend((r"\begin{dmath*}[breakdepth={5}]", rf"E_{{{position[0]}{position[1]}}} = {expr_to_latex(expression)}", r"\end{dmath*}"))
    lines.extend(
        (
            r"\section*{Política de lectura}",
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
            inputs=("validated_model", "verification_report"),
            artifacts=(
                ArtifactRecord("run_manifest", "run_manifest", tuple(manifest_payload.items()), ("validated_model", "verification_report")),
                ArtifactRecord("exported_artifacts", "file_bundle", tuple(artifacts_payload.items()), ("run_manifest",)),
            ),
            diagnostics=diagnostics,
            duration_seconds=duration_seconds,
        )


class RunExporter:
    """Materializa archivos atómicamente dentro de una raíz controlada."""

    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root)

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

        contents = {
            "results": ("results.json", "application/json", _canonical_json(package.to_data(), pretty=True)),
            "verification": ("verification.json", "application/json", _canonical_json(package.verification.to_data(), pretty=True)),
            "latex_report": ("report.tex", "application/x-tex", latex_report(package)),
        }
        files: list[ExportedFile] = []
        for key, (name, media_type, content) in contents.items():
            encoded = self._write_atomic(directory / name, content)
            files.append(ExportedFile(key, name, media_type, _sha256_bytes(encoded), len(encoded)))

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
            external_bindings=package.verification.external_bindings,
            adjudications=package.verification.adjudications,
        )
        manifest_path = directory / "manifest.json"
        self._write_atomic(manifest_path, _canonical_json(manifest.to_data(), pretty=True))
        return ExportBundle(directory, manifest_path, manifest)
