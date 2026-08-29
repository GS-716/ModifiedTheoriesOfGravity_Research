"""Verificación integral y reproducible de una corrida de TensorEngine."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from .backends.base import TensorBackend
from .backends.structural import StructuralTensorBackend
from .components import ComponentFieldEquations, SympyComponentBackend
from .contracts import (
    ArtifactRecord,
    Diagnostic,
    Severity,
    StageResult,
    StageStatus,
    VerificationRecord,
    VerificationStatus,
)
from .differential import DifferentialContext
from .euler import EulerLagrangeResult
from .indices import index_key, rename_free_indices
from .ir import (
    Expr,
    Function,
    FunctionDerivative,
    Number,
    Scalar,
    Tensor,
    Variance,
    VolumeElement,
    add,
    infer_free_indices,
    mul,
    walk,
)
from .model import ModelSpec
from .noether import NoetherWaldResult
from .variational import (
    LagrangianMomenta,
    VariationalContext,
    raw_lagrangian_variation,
)
from .wolfram_bridge import WolframValidationReport


VERIFICATION_SCHEMA_VERSION = "1.0"


def _is_zero(expr: Expr) -> bool:
    return isinstance(expr, Number) and expr.value == 0


def _signature(expr: Expr) -> tuple[tuple[str, str], ...]:
    return tuple((item.space, item.variance.value) for item in infer_free_indices(expr))


def _expected_signature(space: str, *variances: Variance) -> tuple[tuple[str, str], ...]:
    return tuple((space, variance.value) for variance in variances)


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Resultado agregado; un `undetermined` nunca se presenta como éxito total."""

    model_name: str
    backend_name: str
    backend_version: str
    checks: tuple[VerificationRecord, ...]
    external_sources: tuple[tuple[str, str, str], ...] = ()
    external_bindings: tuple[tuple[str, str, str], ...] = ()
    adjudications: tuple[tuple[str, str, str], ...] = ()
    schema_version: str = VERIFICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(self.checks))
        object.__setattr__(self, "external_sources", tuple(tuple(item) for item in self.external_sources))
        object.__setattr__(self, "external_bindings", tuple(tuple(item) for item in self.external_bindings))
        object.__setattr__(self, "adjudications", tuple(tuple(item) for item in self.adjudications))
        if self.schema_version != VERIFICATION_SCHEMA_VERSION:
            raise ValueError(f"Versión de VerificationReport no soportada: {self.schema_version!r}.")
        keys = [item.key for item in self.checks]
        if len(keys) != len(set(keys)):
            raise ValueError("VerificationReport contiene claves de comprobación repetidas.")
        if any(len(item) != 3 for item in self.external_sources):
            raise ValueError("Cada fuente externa debe registrar nombre, versión y estado.")
        if any(len(item) != 3 for item in self.external_bindings):
            raise ValueError("Cada vínculo externo debe registrar operación y dos fingerprints.")
        if any(len(item) != 3 for item in self.adjudications):
            raise ValueError(
                "Cada adjudicación debe registrar comprobación interna, operación y prueba externa."
            )

    @property
    def status(self) -> StageStatus:
        if any(item.status is VerificationStatus.FAILED for item in self.checks):
            return StageStatus.FAILED
        if any(item.status is VerificationStatus.UNDETERMINED for item in self.checks):
            return StageStatus.PARTIAL
        return StageStatus.SUCCESS

    @property
    def passed(self) -> bool:
        return bool(self.checks) and self.status is StageStatus.SUCCESS

    @property
    def summary(self) -> dict[str, int]:
        return {
            status.value: sum(item.status is status for item in self.checks)
            for status in VerificationStatus
        }

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "backend": {"name": self.backend_name, "version": self.backend_version},
            "status": self.status.value,
            "checks": [item.to_data() for item in self.checks],
            "summary": self.summary,
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
        }

    def to_stage_result(self, duration_seconds: float = 0.0) -> StageResult:
        """Adapta el informe al contrato declarativo de la etapa `verify`."""

        inputs = (
            "validated_model",
            "metric_momentum",
            "curvature_momentum",
            "metric_euler",
            "scalar_euler",
            "full_variation",
        )
        diagnostics: tuple[Diagnostic, ...] = ()
        if self.status is StageStatus.FAILED:
            diagnostics = (
                Diagnostic(
                    "E_VERIFICATION_FAILED",
                    "Una o más verificaciones matemáticas fallaron.",
                    Severity.ERROR,
                ),
            )
        elif self.status is StageStatus.PARTIAL:
            diagnostics = (
                Diagnostic(
                    "W_VERIFICATION_UNDETERMINED",
                    "El informe conserva verificaciones que el backend no pudo decidir.",
                    Severity.WARNING,
                ),
            )
        payload = self.to_data()
        return StageResult(
            stage_key="verify",
            status=self.status,
            backend=self.backend_name,
            operation="verify_run",
            inputs=inputs,
            artifacts=(
                ArtifactRecord(
                    "verification_report",
                    "verification_report",
                    tuple(payload.items()),
                    inputs,
                ),
            ),
            verifications=self.checks,
            diagnostics=diagnostics,
            duration_seconds=duration_seconds,
        )

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "VerificationReport":
        backend = data["backend"]
        return cls(
            model_name=str(data["model_name"]),
            backend_name=str(backend["name"]),
            backend_version=str(backend["version"]),
            checks=tuple(VerificationRecord.from_data(item) for item in data["checks"]),
            external_sources=tuple(
                (str(item["name"]), str(item["version"]), str(item["status"]))
                for item in data.get("external_sources", ())
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
            schema_version=str(data.get("schema_version", VERIFICATION_SCHEMA_VERSION)),
        )


def adjudicate_external_evidence(
    report: VerificationReport,
    external_reports: Sequence[WolframValidationReport],
) -> VerificationReport:
    """Resuelve solo indeterminados respaldados unánimemente por evidencia ligada.

    El llamador debe haber comprobado antes los fingerprints del modelo y del cálculo.
    Reportes fijos de fases anteriores y pruebas sin vínculo explícito nunca adjudican.
    """

    observations: dict[str, list[tuple[str, str, VerificationStatus]]] = {}
    for external in external_reports:
        if (
            external.operation != "verify_model"
            or external.model_fingerprint is None
            or external.calculation_fingerprint is None
        ):
            continue
        for check in external.checks:
            for target in check.adjudicates:
                observations.setdefault(target, []).append(
                    (external.operation, check.key, check.status)
                )

    checks: list[VerificationRecord] = []
    adjudications = list(report.adjudications)
    for record in report.checks:
        evidence = observations.get(record.key, [])
        if (
            record.status is VerificationStatus.UNDETERMINED
            and evidence
            and all(status is VerificationStatus.PASSED for _, _, status in evidence)
        ):
            sources = ", ".join(f"{operation}:{key}" for operation, key, _ in evidence)
            checks.append(
                VerificationRecord(
                    record.key,
                    VerificationStatus.PASSED,
                    message=f"{record.message} Adjudicada por evidencia xAct ligada: {sources}.",
                )
            )
            adjudications.extend(
                (record.key, operation, key) for operation, key, _ in evidence
            )
        else:
            checks.append(record)

    return replace(
        report,
        checks=tuple(checks),
        adjudications=tuple(dict.fromkeys(adjudications)),
    )


class RunVerifier:
    """Aplica la política común de verificación a objetos ya calculados."""

    def __init__(self, model: ModelSpec, backend: TensorBackend | None = None) -> None:
        self.model = model
        self.backend = backend or StructuralTensorBackend.from_model(model)
        self.space = model.symbols.index_space
        self.variational_context = VariationalContext.from_model(model)
        self.differential_context = DifferentialContext.from_model(model)

    def _record_from_residual(
        self,
        key: str,
        residual: Expr,
        message: str,
        *,
        undecidable: bool = False,
    ) -> VerificationRecord:
        reduced = self.backend.simplify(residual)
        if _is_zero(reduced):
            return VerificationRecord(key, VerificationStatus.PASSED, message=message)
        status = VerificationStatus.UNDETERMINED if undecidable else VerificationStatus.FAILED
        return VerificationRecord(key, status, reduced, message)

    def _signature_record(
        self,
        key: str,
        expr: Expr,
        variances: tuple[Variance, ...],
    ) -> VerificationRecord:
        if _is_zero(expr) or _signature(expr) == _expected_signature(self.space, *variances):
            return VerificationRecord(
                key,
                VerificationStatus.PASSED,
                message="La firma de índices coincide con el contrato.",
            )
        return VerificationRecord(
            key,
            VerificationStatus.FAILED,
            expr,
            "La expresión conserva índices libres o varianzas inesperadas.",
        )

    def _permutation_residual(self, expr: Expr, order: tuple[int, ...], sign: int) -> Expr:
        if _is_zero(expr):
            return Number(0)
        free = tuple(sorted(infer_free_indices(expr), key=lambda item: (item.space, item.name)))
        if len(free) != len(order):
            return expr
        mapping = {
            index_key(index): free[target].name
            for index, target in zip(free, order, strict=True)
        }
        permuted = rename_free_indices(expr, mapping)
        return add(expr, mul(-sign, permuted))

    def _equivalence_records(
        self,
        prefix: str,
        actual: Sequence[tuple[str, Expr]],
        expected: Sequence[tuple[str, Expr]],
    ) -> list[VerificationRecord]:
        expected_map = dict(expected)
        records: list[VerificationRecord] = []
        for name, expression in actual:
            records.append(
                self._record_from_residual(
                    f"{prefix}.{name}",
                    add(expression, mul(-1, expected_map[name])),
                    f"{name} coincide con su reconstrucción independiente dentro del pipeline.",
                )
            )
        return records

    def _model_and_symbol_checks(self, expressions: Iterable[Expr]) -> list[VerificationRecord]:
        self.model.validate()
        records = [
            VerificationRecord(
                "model.valid",
                VerificationStatus.PASSED,
                message="ModelSpec satisface el esquema y las convenciones activas.",
            )
        ]
        allowed_scalars = {
            self.model.symbols.scalar,
            *(item.name for item in self.model.parameters),
        }
        if self.model.dimension.is_symbolic:
            allowed_scalars.add(str(self.model.dimension.value))
        allowed_functions = {item.name for item in self.model.functions}.union({"Log"})
        allowed_tensors = {
            self.model.symbols.metric,
            self.model.symbols.curvature,
            self.model.symbols.scalar_gradient,
            "delta",
            "delta_Gamma",
            "xi",
        }
        offending: Expr | None = None
        for expression in expressions:
            for node in walk(expression):
                if isinstance(node, Scalar) and node.name not in allowed_scalars:
                    offending = expression
                    break
                if isinstance(node, (Function, FunctionDerivative)) and node.name not in allowed_functions:
                    offending = expression
                    break
                if isinstance(node, Tensor) and node.name not in allowed_tensors:
                    offending = expression
                    break
                if isinstance(node, VolumeElement) and node.metric_name != self.model.symbols.metric:
                    offending = expression
                    break
            if offending is not None:
                break
        if offending is None:
            records.append(
                VerificationRecord(
                    "symbols.declared",
                    VerificationStatus.PASSED,
                    message="Todos los símbolos calculados pertenecen al modelo o al vocabulario interno.",
                )
            )
        else:
            records.append(
                VerificationRecord(
                    "symbols.declared",
                    VerificationStatus.FAILED,
                    offending,
                    "Se encontró al menos un símbolo no declarado.",
                )
            )
        return records

    def _canonicalization_check(self, expressions: Iterable[Expr]) -> VerificationRecord:
        for expression in expressions:
            once = self.backend.canonicalize(expression)
            twice = self.backend.canonicalize(once)
            if once != twice:
                return self._record_from_residual(
                    "canonicalization.idempotent",
                    add(once, mul(-1, twice)),
                    "La canonización debe ser idempotente en todos los resultados.",
                )
        return VerificationRecord(
            "canonicalization.idempotent",
            VerificationStatus.PASSED,
            message="La segunda canonización no modifica ningún resultado.",
        )

    def _momenta_checks(self, momenta: LagrangianMomenta) -> list[VerificationRecord]:
        records = [
            self._signature_record("signature.metric_momentum", momenta.metric, (Variance.DOWN, Variance.DOWN)),
            self._signature_record("signature.curvature_momentum", momenta.curvature, (Variance.UP,) * 4),
            self._signature_record("signature.scalar_gradient_momentum", momenta.scalar_gradient, (Variance.UP,)),
            self._signature_record("signature.scalar_derivative", momenta.scalar, ()),
            self._record_from_residual(
                "symmetry.metric_momentum",
                self._permutation_residual(momenta.metric, (1, 0), 1),
                "M_ab es simétrico.",
                undecidable=True,
            ),
            self._record_from_residual(
                "symmetry.curvature_momentum.first_pair",
                self._permutation_residual(momenta.curvature, (1, 0, 2, 3), -1),
                "P^{abcd} es antisimétrico en el primer par.",
            ),
            self._record_from_residual(
                "symmetry.curvature_momentum.second_pair",
                self._permutation_residual(momenta.curvature, (0, 1, 3, 2), -1),
                "P^{abcd} es antisimétrico en el segundo par.",
            ),
            self._record_from_residual(
                "symmetry.curvature_momentum.pair_exchange",
                self._permutation_residual(momenta.curvature, (2, 3, 0, 1), 1),
                "P^{abcd}=P^{cdab}.",
            ),
        ]
        if _is_zero(momenta.curvature):
            bianchi = Number(0)
        else:
            free = {
                item.name: item
                for item in infer_free_indices(momenta.curvature)
                if item.space == self.space
            }
            first = momenta.curvature
            second = rename_free_indices(
                momenta.curvature,
                {
                    index_key(free["b"]): "c",
                    index_key(free["c"]): "d",
                    index_key(free["d"]): "b",
                },
            )
            third = rename_free_indices(
                momenta.curvature,
                {
                    index_key(free["b"]): "d",
                    index_key(free["c"]): "b",
                    index_key(free["d"]): "c",
                },
            )
            bianchi = add(first, second, third)
        records.append(
            self._record_from_residual(
                "symmetry.curvature_momentum.first_bianchi",
                bianchi,
                "P^{a[bcd]}=0.",
            )
        )
        expected = self.backend.derive_momenta(self.model.lagrangian)
        records.extend(
            self._equivalence_records(
                "recompute_momenta",
                (
                    ("metric", momenta.metric),
                    ("curvature", momenta.curvature),
                    ("scalar_gradient", momenta.scalar_gradient),
                    ("scalar", momenta.scalar),
                ),
                (
                    ("metric", expected.metric),
                    ("curvature", expected.curvature),
                    ("scalar_gradient", expected.scalar_gradient),
                    ("scalar", expected.scalar),
                ),
            )
        )
        return records

    def _euler_checks(
        self,
        momenta: LagrangianMomenta,
        euler: EulerLagrangeResult,
    ) -> list[VerificationRecord]:
        records = [
            self._signature_record("signature.metric_euler", euler.metric_euler, (Variance.DOWN, Variance.DOWN)),
            self._signature_record("signature.scalar_euler", euler.scalar_euler, ()),
            self._signature_record("signature.boundary_metric", euler.boundary_metric, (Variance.UP,)),
            self._signature_record("signature.boundary_scalar", euler.boundary_scalar, (Variance.UP,)),
            self._signature_record("signature.boundary_total", euler.boundary_total, (Variance.UP,)),
            self._signature_record("signature.full_variation", euler.full_variation, ()),
            self._signature_record("signature.density_variation", euler.density_variation, ()),
            self._record_from_residual(
                "symmetry.metric_euler",
                self._permutation_residual(euler.metric_euler, (1, 0), 1),
                "E_ab es simétrico.",
                undecidable=True,
            ),
            self._record_from_residual(
                "boundary.total",
                add(euler.boundary_total, mul(-1, add(euler.boundary_metric, euler.boundary_scalar))),
                "Theta^a coincide con la suma de sus sectores métrico y escalar.",
            ),
            self._record_from_residual(
                "density.factorization",
                add(
                    euler.density_variation,
                    mul(-1, VolumeElement(self.model.symbols.metric), euler.full_variation),
                ),
                "La variación de densidad es sqrt(-g) por la variación completa.",
            ),
            self.backend.check_scalar_integration_by_parts(momenta),
        ]
        expected = self.backend.derive_euler_lagrange(self.model.lagrangian, momenta)
        records.extend(
            self._equivalence_records(
                "recompute_euler",
                (
                    ("metric", euler.metric_euler),
                    ("scalar", euler.scalar_euler),
                    ("boundary_metric", euler.boundary_metric),
                    ("boundary_scalar", euler.boundary_scalar),
                    ("boundary_total", euler.boundary_total),
                    ("full_variation", euler.full_variation),
                    ("density_variation", euler.density_variation),
                ),
                (
                    ("metric", expected.metric_euler),
                    ("scalar", expected.scalar_euler),
                    ("boundary_metric", expected.boundary_metric),
                    ("boundary_scalar", expected.boundary_scalar),
                    ("boundary_total", expected.boundary_total),
                    ("full_variation", expected.full_variation),
                    ("density_variation", expected.density_variation),
                ),
            )
        )
        return records

    def _noether_checks(
        self,
        momenta: LagrangianMomenta,
        euler: EulerLagrangeResult,
        noether: NoetherWaldResult,
    ) -> list[VerificationRecord]:
        records = [
            self._signature_record("signature.noether_current", noether.noether_current, (Variance.UP,)),
            self._signature_record("signature.charge_potential", noether.charge_potential, (Variance.UP, Variance.UP)),
            self._record_from_residual(
                "symmetry.charge_potential",
                self._permutation_residual(noether.charge_potential, (1, 0), -1),
                "Q_xi^{ab} es antisimétrico.",
            ),
            self.backend.check_noether_decomposition(noether),
            self.backend.check_noether_identity(noether),
        ]
        expected = self.backend.derive_noether_wald(self.model.lagrangian, momenta, euler)
        records.extend(
            self._equivalence_records(
                "recompute_noether",
                (
                    ("current", noether.noether_current),
                    ("charge", noether.charge_potential),
                    ("identity", noether.noether_identity),
                ),
                (
                    ("current", expected.noether_current),
                    ("charge", expected.charge_potential),
                    ("identity", expected.noether_identity),
                ),
            )
        )
        return records

    def _component_checks(
        self,
        euler: EulerLagrangeResult,
        components: ComponentFieldEquations,
        component_backend: SympyComponentBackend,
    ) -> list[VerificationRecord]:
        expected_metric = component_backend.evaluate(euler.metric_euler)
        expected_scalar = component_backend.evaluate(euler.scalar_euler)
        metric_equal = components.metric == expected_metric
        scalar_equal = components.scalar == expected_scalar
        symmetry_residual: Expr = Number(0)
        for a in range(components.metric.dimension):
            for b in range(a + 1, components.metric.dimension):
                residual = self.backend.simplify(
                    add(components.metric.component(a, b), mul(-1, components.metric.component(b, a)))
                )
                if not _is_zero(residual):
                    symmetry_residual = residual
                    break
            if not _is_zero(symmetry_residual):
                break
        return [
            VerificationRecord(
                "components.metric_projection",
                VerificationStatus.PASSED if metric_equal else VerificationStatus.FAILED,
                None if metric_equal else Scalar("component_metric_projection_mismatch"),
                "Las componentes métricas coinciden con la proyección directa de E_ab.",
            ),
            VerificationRecord(
                "components.scalar_projection",
                VerificationStatus.PASSED if scalar_equal else VerificationStatus.FAILED,
                None if scalar_equal else Scalar("component_scalar_projection_mismatch"),
                "La componente escalar coincide con la proyección directa de E_phi.",
            ),
            self._record_from_residual(
                "components.metric_symmetry",
                symmetry_residual,
                "La matriz coordenada de E_ab es simétrica.",
            ),
        ]

    def verify(
        self,
        momenta: LagrangianMomenta,
        euler: EulerLagrangeResult,
        *,
        raw_variation: Expr | None = None,
        noether: NoetherWaldResult | None = None,
        components: ComponentFieldEquations | None = None,
        component_backend: SympyComponentBackend | None = None,
        external_reports: Sequence[WolframValidationReport] = (),
    ) -> VerificationReport:
        """Ejecuta verificaciones obligatorias y extensiones disponibles."""

        expressions: list[Expr] = [
            self.model.lagrangian,
            momenta.metric,
            momenta.curvature,
            momenta.scalar_gradient,
            momenta.scalar,
            euler.metric_euler,
            euler.scalar_euler,
            euler.boundary_metric,
            euler.boundary_scalar,
            euler.boundary_total,
            euler.full_variation,
            euler.density_variation,
        ]
        if raw_variation is not None:
            expressions.append(raw_variation)
        if noether is not None:
            expressions.extend(
                (
                    noether.noether_current,
                    noether.charge_potential,
                    noether.decomposition_residual,
                    noether.noether_identity,
                )
            )

        checks = self._model_and_symbol_checks(expressions)
        checks.extend(self._momenta_checks(momenta))
        checks.extend(self._euler_checks(momenta, euler))
        checks.append(self._canonicalization_check(expressions))

        if raw_variation is not None:
            expected_raw = raw_lagrangian_variation(momenta, self.variational_context)
            checks.append(
                self._record_from_residual(
                    "variation.raw_reconstruction",
                    add(raw_variation, mul(-1, expected_raw)),
                    "La variación cruda coincide con la reconstrucción desde los cuatro momentos.",
                )
            )
        if noether is not None:
            checks.extend(self._noether_checks(momenta, euler, noether))
        if (components is None) != (component_backend is None):
            raise ValueError("components y component_backend deben proporcionarse juntos.")
        if components is not None and component_backend is not None:
            checks.extend(self._component_checks(euler, components, component_backend))

        external_sources: list[tuple[str, str, str]] = []
        external_bindings: list[tuple[str, str, str]] = []
        for report in external_reports:
            prefix = report.operation.removeprefix("verify_")
            external_sources.append((report.operation, report.wolfram_version, report.status))
            if report.model_fingerprint is not None and report.calculation_fingerprint is not None:
                external_bindings.append(
                    (report.operation, report.model_fingerprint, report.calculation_fingerprint)
                )
            for record in report.verification_records:
                checks.append(
                    VerificationRecord(
                        f"external.{prefix}.{record.key}",
                        record.status,
                        record.residual,
                        record.message,
                    )
                )

        return VerificationReport(
            model_name=self.model.name,
            backend_name=self.backend.info.name,
            backend_version=self.backend.info.version,
            checks=tuple(checks),
            external_sources=tuple(external_sources),
            external_bindings=tuple(external_bindings),
        )


def verify_run(
    model: ModelSpec,
    momenta: LagrangianMomenta,
    euler: EulerLagrangeResult,
    **options: Any,
) -> VerificationReport:
    """Atajo público para la verificación estructural predeterminada."""

    return RunVerifier(model).verify(momenta, euler, **options)
