"""Orquestación integral y observable del pipeline de TensorEngine."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, TypeVar

from .backends.base import TensorBackend
from .backends.structural import StructuralTensorBackend
from .components import (
    ComponentFieldEquations,
    GeometryAnsatz,
    SympyComponentBackend,
    evaluate_field_equations,
)
from .contracts import (
    ArtifactRecord,
    Diagnostic,
    ExpressionForm,
    ExpressionRecord,
    StageResult,
    StageStatus,
    Severity,
    validate_stage_result,
)
from .derived import (
    AbstractTensorResults,
    DerivedQuantities,
    ProjectedTensorResults,
    build_result_views,
    derive_intermediate_quantities,
)
from .errors import PipelineExecutionError, TensorEngineError
from .euler import EulerLagrangeResult
from .exporting import ExportBundle, RunExporter, RunPackage
from .presentation import DisplayPolicy
from .ir import Expr, Number, mul
from .model import ModelSpec
from .noether import NoetherWaldResult
from .stages import DEFAULT_PIPELINE
from .variational import LagrangianMomenta
from .verification import VerificationReport, RunVerifier, adjudicate_external_evidence
from .wolfram_bridge import (
    WolframValidationReport,
    WolframXActBridge,
    calculation_fingerprint,
    model_fingerprint,
)


ENGINE_SCHEMA_VERSION = "1.0"
T = TypeVar("T")
EventHandler = Callable[["RunEvent"], None]
BackendFactory = Callable[[ModelSpec], TensorBackend]


@dataclass(frozen=True, slots=True)
class EngineOptions:
    """Ramas opcionales; la exportación requiere además una raíz de salida."""

    include_noether: bool = True
    include_components: bool = True
    include_export: bool = True

    def to_data(self) -> dict[str, bool]:
        return {
            "include_noether": self.include_noether,
            "include_components": self.include_components,
            "include_export": self.include_export,
        }


@dataclass(frozen=True, slots=True)
class RunEvent:
    """Evento pequeño para notebooks, logs o una futura interfaz gráfica."""

    stage_key: str
    state: str
    duration_seconds: float = 0.0
    message: str = ""

    def __post_init__(self) -> None:
        if self.state not in {"started", "completed", "failed"}:
            raise ValueError(f"Estado de evento no soportado: {self.state!r}.")
        if self.duration_seconds < 0:
            raise ValueError("La duración de un evento no puede ser negativa.")

    def to_data(self) -> dict[str, Any]:
        return {
            "stage_key": self.stage_key,
            "state": self.state,
            "duration_seconds": self.duration_seconds,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class EngineRun:
    """Resultado de alto nivel, incluida la bitácora contractual por etapas."""

    package: RunPackage
    stages: tuple[StageResult, ...]
    skipped_stages: tuple[str, ...] = ()
    export_bundle: ExportBundle | None = None
    schema_version: str = ENGINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(self, "skipped_stages", tuple(self.skipped_stages))
        if self.schema_version != ENGINE_SCHEMA_VERSION:
            raise ValueError(f"Versión de EngineRun no soportada: {self.schema_version!r}.")
        keys = [stage.stage_key for stage in self.stages]
        if len(keys) != len(set(keys)):
            raise ValueError("Una corrida integral no puede repetir etapas.")

    @property
    def status(self) -> StageStatus:
        return self.package.verification.status

    @property
    def derived(self) -> DerivedQuantities | None:
        """Cantidades geométricas intermedias de la corrida."""

        return self.package.derived

    @property
    def abstract(self) -> AbstractTensorResults | None:
        return self.package.abstract

    @property
    def projected(self) -> ProjectedTensorResults | None:
        return self.package.projected

    @property
    def delta_contractions(self):
        return self.package.delta_contractions

    @property
    def duration_seconds(self) -> float:
        return sum(stage.duration_seconds for stage in self.stages)

    def acceptable(self, *, strict: bool = False) -> bool:
        if strict:
            return self.status is StageStatus.SUCCESS
        return self.status is not StageStatus.FAILED

    def summary_data(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.package.run_id,
            "model_name": self.package.model.name,
            "status": self.status.value,
            "verification": self.package.verification.summary,
            "duration_seconds": self.duration_seconds,
            "completed_stages": [stage.stage_key for stage in self.stages],
            "skipped_stages": list(self.skipped_stages),
            "output_directory": (
                None
                if self.export_bundle is None
                else str(self.export_bundle.output_directory.resolve())
            ),
        }

    def to_data(self) -> dict[str, Any]:
        return {
            **self.summary_data(),
            "package": self.package.to_data(),
            "stages": [stage.to_data() for stage in self.stages],
        }


def _default_backend(model: ModelSpec) -> TensorBackend:
    return StructuralTensorBackend.from_model(model)


class TensorEngine:
    """Ejecuta un `ModelSpec` de extremo a extremo con una sola llamada."""

    def __init__(
        self,
        *,
        backend_factory: BackendFactory = _default_backend,
        options: EngineOptions | None = None,
        event_handler: EventHandler | None = None,
    ) -> None:
        self.backend_factory = backend_factory
        self.options = options or EngineOptions()
        self.event_handler = event_handler
        self._specifications = {stage.key: stage for stage in DEFAULT_PIPELINE}

    def _emit(self, event: RunEvent) -> None:
        if self.event_handler is not None:
            self.event_handler(event)

    def _execute(
        self,
        stage_key: str,
        operation: Callable[[], T],
        result_builder: Callable[[T, float], StageResult],
        completed: list[StageResult],
    ) -> T:
        self._emit(RunEvent(stage_key, "started"))
        started = perf_counter()
        try:
            value = operation()
            duration = perf_counter() - started
            result = result_builder(value, duration)
            validate_stage_result(self._specifications[stage_key], result)
        except Exception as error:
            duration = perf_counter() - started
            self._emit(RunEvent(stage_key, "failed", duration, str(error)))
            if isinstance(error, PipelineExecutionError):
                raise
            raise PipelineExecutionError(stage_key, str(error)) from error
        completed.append(result)
        self._emit(RunEvent(stage_key, "completed", duration))
        return value

    def run(
        self,
        model: ModelSpec,
        *,
        ansatz: GeometryAnsatz | None = None,
        output_root: str | Path | None = None,
        external_reports: tuple[WolframValidationReport, ...] = (),
        wolfram_bridge: WolframXActBridge | None = None,
        display_policy: DisplayPolicy | None = None,
    ) -> EngineRun:
        """Calcula, verifica y opcionalmente exporta una teoría declarada."""

        completed: list[StageResult] = []
        skipped: list[str] = []

        validated_model = self._execute(
            "validate_model",
            model.validate,
            lambda value, duration: StageResult(
                "validate_model",
                StageStatus.SUCCESS,
                "python",
                "validate_model",
                inputs=("model_spec",),
                artifacts=(
                    ArtifactRecord(
                        "validated_model",
                        "model_spec",
                        tuple(value.to_data().items()),
                        ("model_spec",),
                    ),
                ),
                duration_seconds=duration,
            ),
            completed,
        )
        backend = self.backend_factory(validated_model)

        normalized_lagrangian = self._execute(
            "normalize_lagrangian",
            lambda: backend.canonicalize(
                mul(validated_model.normalization, validated_model.lagrangian)
            ),
            lambda value, duration: StageResult(
                "normalize_lagrangian",
                StageStatus.SUCCESS,
                backend.info.name,
                "normalize_lagrangian",
                inputs=("validated_model",),
                outputs=(
                    ExpressionRecord(
                        "lagrangian",
                        value,
                        ExpressionForm.CANONICAL,
                        ("validated_model",),
                        validated_model.assumptions,
                    ),
                ),
                duration_seconds=duration,
            ),
            completed,
        )
        calculation_model = replace(
            validated_model,
            lagrangian=normalized_lagrangian,
            normalization=Number(1),
        )

        momenta = self._execute(
            "derive_momenta",
            lambda: backend.derive_momenta(normalized_lagrangian),
            lambda value, duration: StageResult(
                "derive_momenta",
                StageStatus.SUCCESS,
                backend.info.name,
                "derive_momenta",
                inputs=("lagrangian",),
                outputs=(
                    ExpressionRecord("metric_momentum", value.metric, ExpressionForm.CANONICAL, ("lagrangian",), validated_model.assumptions),
                    ExpressionRecord("curvature_momentum", value.curvature, ExpressionForm.CANONICAL, ("lagrangian",), validated_model.assumptions),
                    ExpressionRecord("scalar_gradient_momentum", value.scalar_gradient, ExpressionForm.CANONICAL, ("lagrangian",), validated_model.assumptions),
                    ExpressionRecord("scalar_derivative", value.scalar, ExpressionForm.CANONICAL, ("lagrangian",), validated_model.assumptions),
                ),
                duration_seconds=duration,
            ),
            completed,
        )

        raw_variation = self._execute(
            "raw_variation",
            lambda: backend.raw_lagrangian_variation(momenta),
            lambda value, duration: StageResult(
                "raw_variation",
                StageStatus.SUCCESS,
                backend.info.name,
                "raw_lagrangian_variation",
                inputs=("lagrangian", "metric_momentum", "curvature_momentum", "scalar_gradient_momentum", "scalar_derivative"),
                outputs=(
                    ExpressionRecord(
                        "delta_lagrangian",
                        value,
                        ExpressionForm.RAW,
                        ("lagrangian", "metric_momentum", "curvature_momentum", "scalar_gradient_momentum", "scalar_derivative"),
                        validated_model.assumptions,
                    ),
                ),
                duration_seconds=duration,
            ),
            completed,
        )

        euler = self._execute(
            "integrate_by_parts",
            lambda: backend.derive_euler_lagrange(normalized_lagrangian, momenta),
            lambda value, duration: self._euler_stage(value, backend, validated_model, duration),
            completed,
        )

        noether: NoetherWaldResult | None = None
        if self.options.include_noether:
            noether = self._execute(
                "noether",
                lambda: backend.derive_noether_wald(normalized_lagrangian, momenta, euler),
                lambda value, duration: StageResult(
                    "noether",
                    StageStatus.SUCCESS,
                    backend.info.name,
                    "derive_noether_wald",
                    inputs=("metric_euler", "scalar_euler", "boundary_potential_total"),
                    outputs=(
                        ExpressionRecord("noether_current", value.noether_current, ExpressionForm.CANONICAL, ("metric_euler", "scalar_euler", "boundary_potential_total"), validated_model.assumptions),
                        ExpressionRecord("charge_potential", value.charge_potential, ExpressionForm.CANONICAL, ("curvature_momentum",), validated_model.assumptions),
                    ),
                    duration_seconds=duration,
                ),
                completed,
            )
        else:
            skipped.append("noether")

        components: ComponentFieldEquations | None = None
        component_backend: SympyComponentBackend | None = None
        component_failure_reason: str | None = None
        if ansatz is not None and self.options.include_components:
            def component_operation() -> tuple[
                ComponentFieldEquations | None,
                SympyComponentBackend | None,
                str | None,
            ]:
                active_backend: SympyComponentBackend | None = None
                try:
                    active_backend = SympyComponentBackend.from_model(
                        calculation_model,
                        ansatz,
                    )
                    result = evaluate_field_equations(
                        euler.metric_euler,
                        euler.scalar_euler,
                        active_backend,
                    )
                    return result, active_backend, None
                except (TensorEngineError, OverflowError, RecursionError) as error:
                    return None, active_backend, str(error)

            components, component_backend, component_failure_reason = self._execute(
                "components",
                component_operation,
                lambda value, duration: StageResult(
                    "components",
                    StageStatus.SUCCESS if value[0] is not None else StageStatus.PARTIAL,
                    value[1].name if value[1] is not None else "sympy-components",
                    "evaluate_field_equations",
                    inputs=("validated_model", "geometry_ansatz", "metric_euler", "scalar_euler"),
                    artifacts=(
                        ArtifactRecord(
                            "component_results",
                            "component_field_equations",
                            tuple(
                                (
                                    value[0].to_data()
                                    if value[0] is not None
                                    else {
                                        "ansatz": ansatz.name,
                                        "available": False,
                                        "reason": value[2],
                                    }
                                ).items()
                            ),
                            ("validated_model", "geometry_ansatz", "metric_euler", "scalar_euler"),
                        ),
                    ),
                    diagnostics=(
                        ()
                        if value[0] is not None
                        else (
                            Diagnostic(
                                "W_COMPONENT_BACKEND_LIMITATION",
                                value[2] or "No fue posible proyectar las ecuaciones de campo.",
                                Severity.WARNING,
                            ),
                        )
                    ),
                    duration_seconds=duration,
                ),
                completed,
            )
        else:
            skipped.append("components")

        active_external_reports = list(external_reports)
        expected_model_fingerprint = model_fingerprint(validated_model)
        expected_calculation_fingerprint = calculation_fingerprint(
            validated_model,
            normalized_lagrangian,
            momenta,
            euler,
            noether,
        )
        for report in active_external_reports:
            if report.operation == "verify_model" and (
                report.model_name != validated_model.name
                or report.model_fingerprint != expected_model_fingerprint
                or report.calculation_fingerprint != expected_calculation_fingerprint
            ):
                raise PipelineExecutionError(
                    "verify",
                    "Un reporte Wolfram genérico pertenece a otro modelo o cálculo.",
                )
        if wolfram_bridge is not None:
            active_external_reports.append(
                self._execute(
                    "wolfram_model_validation",
                    lambda: wolfram_bridge.validate_model(
                        validated_model,
                        momenta,
                        euler,
                        normalized_lagrangian=normalized_lagrangian,
                        noether=noether,
                    ),
                    lambda value, duration: self._wolfram_stage(value, duration),
                    completed,
                )
            )
        else:
            skipped.append("wolfram_model_validation")

        verifier = RunVerifier(calculation_model, backend)

        def verification_operation() -> VerificationReport:
            report = verifier.verify(
                momenta,
                euler,
                raw_variation=raw_variation,
                noether=noether,
                components=components,
                component_backend=(
                    component_backend if components is not None else None
                ),
                external_reports=tuple(active_external_reports),
            )
            return adjudicate_external_evidence(
                report,
                tuple(active_external_reports),
            )

        verification = self._execute(
            "verify",
            verification_operation,
            lambda value, duration: value.to_stage_result(duration),
            completed,
        )

        derived = self._execute(
            "derive_intermediate_quantities",
            lambda: derive_intermediate_quantities(
                calculation_model,
                momenta,
                euler,
                verification,
                backend,
                component_backend,
                projection_unavailable_reason=component_failure_reason,
            ),
            lambda value, duration: StageResult(
                "derive_intermediate_quantities",
                StageStatus.SUCCESS,
                backend.info.name,
                "derive_intermediate_quantities",
                inputs=(
                    "validated_model",
                    "curvature_momentum",
                    "metric_euler",
                    "verification_report",
                ),
                artifacts=(
                    ArtifactRecord(
                        "derived_quantities",
                        "derived_quantities",
                        tuple(value.to_data().items()),
                        (
                            "validated_model",
                            "curvature_momentum",
                            "metric_euler",
                            "verification_report",
                        ),
                    ),
                ),
                duration_seconds=duration,
            ),
            completed,
        )

        abstract_results, projected_results = self._execute(
            "organize_result_views",
            lambda: build_result_views(
                calculation_model,
                normalized_lagrangian,
                momenta,
                euler,
                derived,
                verification,
                ansatz_name=None if ansatz is None else ansatz.name,
                component_backend=component_backend,
                field_components=components,
                projection_unavailable_reason=component_failure_reason,
                field_equation_failure_reason=(
                    component_failure_reason
                    if ansatz is not None and components is None
                    else None
                ),
            ),
            lambda value, duration: StageResult(
                "organize_result_views",
                StageStatus.SUCCESS,
                backend.info.name,
                "build_result_views",
                inputs=(
                    "lagrangian",
                    "metric_momentum",
                    "curvature_momentum",
                    "metric_euler",
                    "scalar_euler",
                    "derived_quantities",
                ),
                artifacts=(
                    ArtifactRecord(
                        "abstract_results",
                        "abstract_tensor_results",
                        tuple(value[0].to_data().items()),
                        ("lagrangian", "metric_momentum", "metric_euler", "derived_quantities"),
                    ),
                    ArtifactRecord(
                        "projected_results",
                        "projected_tensor_results",
                        tuple(value[1].to_data().items()),
                        ("geometry_ansatz", "abstract_results"),
                    ),
                ),
                duration_seconds=duration,
            ),
            completed,
        )

        pre_export_duration = sum(stage.duration_seconds for stage in completed)
        package = RunPackage(
            validated_model,
            momenta,
            raw_variation,
            euler,
            verification,
            normalized_lagrangian=normalized_lagrangian,
            noether=noether,
            components=components,
            derived=derived,
            abstract=abstract_results,
            projected=projected_results,
            duration_seconds=pre_export_duration,
            stage_durations=tuple((stage.stage_key, stage.duration_seconds) for stage in completed),
            delta_contractions=tuple(getattr(backend, "delta_contractions", ())),
        )

        export_bundle: ExportBundle | None = None
        if self.options.include_export and output_root is not None:
            export_bundle = self._execute(
                "export",
                lambda: RunExporter(
                    output_root, display_policy=display_policy,
                    projected_assumptions=() if ansatz is None else ansatz.assumptions,
                ).export(package),
                lambda value, duration: value.to_stage_result(duration),
                completed,
            )
        else:
            skipped.append("export")

        return EngineRun(package, tuple(completed), tuple(skipped), export_bundle)

    @staticmethod
    def _euler_stage(
        value: EulerLagrangeResult,
        backend: TensorBackend,
        model: ModelSpec,
        duration: float,
    ) -> StageResult:
        source = ("delta_lagrangian",)
        return StageResult(
            "integrate_by_parts",
            StageStatus.SUCCESS,
            backend.info.name,
            "derive_euler_lagrange",
            inputs=source,
            outputs=(
                ExpressionRecord("metric_euler", value.metric_euler, ExpressionForm.CANONICAL, source, model.assumptions),
                ExpressionRecord("scalar_euler", value.scalar_euler, ExpressionForm.CANONICAL, source, model.assumptions),
                ExpressionRecord("boundary_potential_metric", value.boundary_metric, ExpressionForm.CANONICAL, source, model.assumptions),
                ExpressionRecord("boundary_potential_scalar", value.boundary_scalar, ExpressionForm.CANONICAL, source, model.assumptions),
                ExpressionRecord("boundary_potential_total", value.boundary_total, ExpressionForm.CANONICAL, ("boundary_potential_metric", "boundary_potential_scalar"), model.assumptions),
                ExpressionRecord("full_variation", value.full_variation, ExpressionForm.CANONICAL, ("metric_euler", "scalar_euler", "boundary_potential_total"), model.assumptions),
            ),
            duration_seconds=duration,
        )

    @staticmethod
    def _wolfram_stage(value: WolframValidationReport, duration: float) -> StageResult:
        status = StageStatus(value.status)
        diagnostics: tuple[Diagnostic, ...] = ()
        if status is StageStatus.FAILED:
            diagnostics = (
                Diagnostic(
                    "E_WOLFRAM_MODEL_VALIDATION",
                    "xAct reportó uno o más residuales fallidos para el cálculo concreto.",
                    Severity.ERROR,
                ),
            )
        inputs = ("validated_model", "metric_momentum", "curvature_momentum", "metric_euler", "scalar_euler")
        return StageResult(
            "wolfram_model_validation",
            status,
            "wolfram-xact",
            "verify_model",
            inputs=inputs,
            artifacts=(
                ArtifactRecord(
                    "wolfram_model_report",
                    "wolfram_validation_report",
                    tuple(value.to_data().items()),
                    inputs,
                ),
            ),
            diagnostics=diagnostics,
            duration_seconds=duration,
        )


def run_model(
    model: ModelSpec,
    **options: Any,
) -> EngineRun:
    """Atajo para la configuración predeterminada del orquestador."""

    return TensorEngine().run(model, **options)
