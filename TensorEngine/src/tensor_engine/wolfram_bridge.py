"""Transporte local y explícito entre Python y Wolfram Engine/xAct."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping

from .contracts import VerificationRecord, VerificationStatus
from .errors import BackendExecutionError, BackendUnavailableError
from .euler import EulerLagrangeResult
from .indices import all_indices, index_key, rename_free_indices
from .ir import Expr, Number, Scalar, VolumeElement, add, expr_from_data, infer_free_indices, mul
from .model import ModelSpec, TensorDeclaration
from .noether import NoetherWaldResult
from .variational import LagrangianMomenta


BRIDGE_SCHEMA_VERSION = "1.3"
_OPERATION_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")


def _fingerprint(data: Any) -> str:
    encoded = json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_fingerprint(model: ModelSpec) -> str:
    """Identidad SHA-256 del ModelSpec, independiente de rutas y tiempos."""

    return _fingerprint(model.to_data())


def calculation_fingerprint(
    model: ModelSpec,
    normalized_lagrangian: Expr,
    momenta: LagrangianMomenta,
    euler: EulerLagrangeResult,
    noether: NoetherWaldResult | None = None,
) -> str:
    """Identidad de los objetos sometidos a validación externa."""

    return _fingerprint(
        {
            "model_fingerprint": model_fingerprint(model),
            "normalized_lagrangian": normalized_lagrangian.to_data(),
            "momenta": momenta.to_data(),
            "euler_lagrange": euler.to_data(),
            "noether_wald": None if noether is None else noether.to_data(),
        }
    )


@dataclass(frozen=True, slots=True)
class WolframRuntime:
    available: bool
    executable: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class WolframComponentInfo:
    """Disponibilidad y versión de un componente local de xAct."""

    available: bool
    version: str | None = None
    release_date: tuple[int, ...] | None = None

    def to_data(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "version": self.version,
            "release_date": None if self.release_date is None else list(self.release_date),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "WolframComponentInfo":
        release_date = data.get("release_date")
        if release_date is not None and not isinstance(release_date, list):
            raise BackendExecutionError("La fecha de versión de xAct debe ser una lista JSON.")
        return cls(
            available=bool(data.get("available", False)),
            version=None if data.get("version") is None else str(data["version"]),
            release_date=(
                None
                if release_date is None
                else tuple(int(item) for item in release_date)
            ),
        )


@dataclass(frozen=True, slots=True)
class WolframPhase5Check:
    """Resultado textual de una identidad evaluada y canonizada por xAct."""

    key: str
    status: VerificationStatus
    message: str
    residual: str | None = None
    strategy: str | None = None
    adjudicates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "adjudicates", tuple(self.adjudicates))
        if self.strategy not in {None, "algebraic", "riemann_bianchi", "differential"}:
            raise BackendExecutionError(f"Estrategia xAct no soportada: {self.strategy!r}.")
        if self.status is VerificationStatus.PASSED and self.residual is not None:
            raise BackendExecutionError(
                "Una comprobación xAct aprobada no puede conservar residual."
            )
        if self.status is not VerificationStatus.PASSED and self.residual is None:
            raise BackendExecutionError(
                "Una comprobación xAct no aprobada debe conservar su residual."
            )

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "WolframPhase5Check":
        try:
            status = VerificationStatus(str(data["status"]))
            key = str(data["key"])
        except (KeyError, ValueError) as error:
            raise BackendExecutionError("La comprobación xAct tiene un contrato inválido.") from error
        residual = data.get("residual")
        return cls(
            key=key,
            status=status,
            message=str(data.get("message", "")),
            residual=None if residual is None else str(residual),
            strategy=None if data.get("strategy") is None else str(data["strategy"]),
            adjudicates=tuple(str(item) for item in data.get("adjudicates", ())),
        )

    def to_data(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "status": self.status.value,
            "message": self.message,
            "residual": self.residual,
            "strategy": self.strategy,
            "adjudicates": list(self.adjudicates),
        }

    def to_verification_record(self) -> VerificationRecord:
        """Adapta el resultado al contrato común sin fingir traducir el residual xAct."""

        residual_marker = None
        message = self.message
        if self.residual is not None:
            safe_key = re.sub(r"[^a-zA-Z0-9_]", "_", self.key)
            residual_marker = Scalar(f"wolfram_residual_{safe_key}")
            message = f"{message} Residual xAct: {self.residual}"
        return VerificationRecord(self.key, self.status, residual_marker, message)


@dataclass(frozen=True, slots=True)
class WolframValidationReport:
    """Informe tipado de una validación de fase ejecutada en Wolfram/xAct."""

    status: str
    wolfram_version: str
    wolfram_version_number: float
    wolfram_release_number: int
    system_id: str
    xact_xtensor: WolframComponentInfo
    xact_xpert: WolframComponentInfo
    xact_xtras: WolframComponentInfo
    conventions: tuple[tuple[str, Any], ...]
    checks: tuple[WolframPhase5Check, ...]
    operation: str = "verify_phase5"
    xact_xcoba: WolframComponentInfo = field(
        default_factory=lambda: WolframComponentInfo(False)
    )
    model_name: str | None = None
    model_fingerprint: str | None = None
    calculation_fingerprint: str | None = None

    def to_data(self) -> dict[str, Any]:
        data = {
            "schema_version": BRIDGE_SCHEMA_VERSION,
            "status": self.status,
            "operation": self.operation,
            "runtime": {
                "wolfram_version": self.wolfram_version,
                "wolfram_version_number": self.wolfram_version_number,
                "wolfram_release_number": self.wolfram_release_number,
                "system_id": self.system_id,
            },
            "components": {
                "xact_xtensor": self.xact_xtensor.to_data(),
                "xact_xpert": self.xact_xpert.to_data(),
                "xact_xtras": self.xact_xtras.to_data(),
                "xact_xcoba": self.xact_xcoba.to_data(),
            },
            "conventions": dict(self.conventions),
            "checks": [item.to_data() for item in self.checks],
            "summary": self.summary,
        }
        if self.model_name is not None:
            data["subject"] = {
                "model_name": self.model_name,
                "model_fingerprint": self.model_fingerprint,
                "calculation_fingerprint": self.calculation_fingerprint,
            }
        return data

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "WolframValidationReport":
        operation = str(data.get("operation", ""))
        if operation not in {"verify_phase5", "verify_phase6", "verify_phase7", "verify_model"}:
            raise BackendExecutionError("La respuesta no corresponde a una validación conocida.")
        runtime = data.get("runtime")
        components = data.get("components")
        checks_data = data.get("checks")
        conventions = data.get("conventions", {})
        if not isinstance(runtime, Mapping) or not isinstance(components, Mapping):
            raise BackendExecutionError("La respuesta xAct no contiene runtime/components válidos.")
        if not isinstance(checks_data, list) or not isinstance(conventions, Mapping):
            raise BackendExecutionError("La respuesta xAct no contiene checks/conventions válidos.")
        xTensor_data = components.get("xact_xtensor")
        xPert_data = components.get("xact_xpert")
        xTras_data = components.get("xact_xtras")
        xCoba_data = components.get("xact_xcoba", {"available": False})
        if (
            not isinstance(xTensor_data, Mapping)
            or not isinstance(xPert_data, Mapping)
            or not isinstance(xTras_data, Mapping)
            or not isinstance(xCoba_data, Mapping)
        ):
            raise BackendExecutionError("La respuesta xAct omite la versión de sus componentes.")
        subject = data.get("subject")
        if operation == "verify_model" and not isinstance(subject, Mapping):
            raise BackendExecutionError("La validación genérica no identifica el modelo calculado.")
        try:
            report = cls(
                status=str(data["status"]),
                wolfram_version=str(runtime["wolfram_version"]),
                wolfram_version_number=float(runtime["wolfram_version_number"]),
                wolfram_release_number=int(runtime["wolfram_release_number"]),
                system_id=str(runtime["system_id"]),
                xact_xtensor=WolframComponentInfo.from_data(xTensor_data),
                xact_xpert=WolframComponentInfo.from_data(xPert_data),
                xact_xtras=WolframComponentInfo.from_data(xTras_data),
                conventions=tuple((str(key), value) for key, value in conventions.items()),
                checks=tuple(WolframPhase5Check.from_data(item) for item in checks_data),
                operation=operation,
                xact_xcoba=WolframComponentInfo.from_data(xCoba_data),
                model_name=(None if not isinstance(subject, Mapping) else str(subject["model_name"])),
                model_fingerprint=(None if not isinstance(subject, Mapping) else str(subject["model_fingerprint"])),
                calculation_fingerprint=(None if not isinstance(subject, Mapping) else str(subject["calculation_fingerprint"])),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise BackendExecutionError(
                f"La respuesta {operation} está incompleta."
            ) from error
        if operation == "verify_model" and (
            not _FINGERPRINT_RE.fullmatch(report.model_fingerprint or "")
            or not _FINGERPRINT_RE.fullmatch(report.calculation_fingerprint or "")
        ):
            raise BackendExecutionError("La validación genérica contiene fingerprints inválidos.")
        return report

    @property
    def passed(self) -> bool:
        return self.status == "success" and bool(self.checks) and all(
            item.status is VerificationStatus.PASSED for item in self.checks
        )

    @property
    def summary(self) -> dict[str, int]:
        return {
            status.value: sum(item.status is status for item in self.checks)
            for status in VerificationStatus
        }

    @property
    def verification_records(self) -> tuple[VerificationRecord, ...]:
        return tuple(item.to_verification_record() for item in self.checks)


WolframPhase5Report = WolframValidationReport
WolframPhase6Report = WolframValidationReport
WolframPhase7Report = WolframValidationReport


def detect_wolfram_runtime(executable: str | None = None) -> WolframRuntime:
    """Detecta wolframscript sin iniciar procesos ni modificar el sistema."""

    candidate = executable or shutil.which("wolframscript")
    if not candidate:
        return WolframRuntime(
            False,
            None,
            "No se encontró wolframscript en PATH ni se proporcionó una ruta explícita.",
        )
    resolved = shutil.which(candidate) or candidate
    path = Path(resolved)
    if not path.is_file():
        return WolframRuntime(False, str(path), "La ruta de wolframscript no existe.")
    return WolframRuntime(True, str(path.resolve()), "WolframScript disponible.")


def _declaration_data(declaration: TensorDeclaration) -> dict[str, Any]:
    return {
        "name": declaration.name,
        "slots": [item.value for item in declaration.slots],
        "symmetry": declaration.symmetry.value,
    }


def _permutation_residual(expr: Expr, order: tuple[int, ...], sign: int) -> Expr:
    if isinstance(expr, Number) and expr.value == 0:
        return expr
    free = tuple(sorted(infer_free_indices(expr), key=lambda item: (item.space, item.name)))
    if len(free) != len(order):
        return expr
    mapping = {
        index_key(index): free[target].name
        for index, target in zip(free, order, strict=True)
    }
    return add(expr, mul(-sign, rename_free_indices(expr, mapping)))


def _permuted(expr: Expr, order: tuple[int, ...]) -> Expr:
    if isinstance(expr, Number) and expr.value == 0:
        return expr
    free = tuple(sorted(infer_free_indices(expr), key=lambda item: (item.space, item.name)))
    if len(free) != len(order):
        return expr
    return rename_free_indices(
        expr,
        {
            index_key(index): free[target].name
            for index, target in zip(free, order, strict=True)
        },
    )


def _generic_checks(
    model: ModelSpec,
    momenta: LagrangianMomenta,
    euler: EulerLagrangeResult,
    noether: NoetherWaldResult | None,
) -> tuple[dict[str, Any], ...]:
    checks: list[tuple[str, str, Expr, str, str, tuple[str, ...]]] = [
        ("metric_momentum_symmetry", "M_ab es simétrico para este modelo.", _permutation_residual(momenta.metric, (1, 0), 1), "failed", "algebraic", ()),
        ("curvature_momentum_first_pair", "P^{abcd} es antisimétrico en el primer par.", _permutation_residual(momenta.curvature, (1, 0, 2, 3), -1), "failed", "algebraic", ()),
        ("curvature_momentum_second_pair", "P^{abcd} es antisimétrico en el segundo par.", _permutation_residual(momenta.curvature, (0, 1, 3, 2), -1), "failed", "algebraic", ()),
        ("curvature_momentum_pair_exchange", "P^{abcd}=P^{cdab}.", _permutation_residual(momenta.curvature, (2, 3, 0, 1), 1), "failed", "algebraic", ()),
        ("curvature_momentum_first_bianchi", "P^{a[bcd]}=0.", add(momenta.curvature, _permuted(momenta.curvature, (0, 2, 3, 1)), _permuted(momenta.curvature, (0, 3, 1, 2))), "failed", "riemann_bianchi", ()),
        ("metric_euler_symmetry", "E_ab es simétrico para este modelo.", _permutation_residual(euler.metric_euler, (1, 0), 1), "failed", "algebraic", ()),
        ("boundary_sum", "La frontera total coincide con sus sectores.", add(euler.boundary_total, mul(-1, add(euler.boundary_metric, euler.boundary_scalar))), "failed", "algebraic", ()),
        ("density_factorization", "La variación de densidad factoriza sqrt(-g).", add(euler.density_variation, mul(-1, VolumeElement(model.symbols.metric), euler.full_variation)), "failed", "algebraic", ()),
    ]
    if noether is not None:
        checks.extend(
            (
                ("wald_charge_antisymmetry", "Q_xi^{ab} es antisimétrico.", _permutation_residual(noether.charge_potential, (1, 0), -1), "failed", "algebraic", ()),
                ("noether_current_decomposition", "La descomposición de la corriente se anula tras ordenar derivadas.", noether.decomposition_residual, "undetermined", "differential", ("noether_current_decomposition",)),
                ("diffeomorphism_noether_identity", "La identidad diferencial de Noether se anula.", noether.noether_identity, "undetermined", "differential", ("diffeomorphism_noether_identity",)),
            )
        )
    return tuple(
        {
            "key": key,
            "message": message,
            "residual": residual.to_data(),
            "on_nonzero": on_nonzero,
            "strategy": strategy,
            "adjudicates": list(adjudicates),
        }
        for key, message, residual, on_nonzero, strategy, adjudicates in checks
    )


class WolframXActBridge:
    """Ejecuta solicitudes JSON locales; nunca requiere una API web."""

    def __init__(
        self,
        executable: str | None = None,
        script_path: str | Path | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.runtime = detect_wolfram_runtime(executable)
        default_script = Path(__file__).resolve().parents[2] / "wolfram" / "TensorEngineBridge.wl"
        self.script_path = Path(script_path) if script_path is not None else default_script
        self.timeout_seconds = timeout_seconds

    @property
    def available(self) -> bool:
        return self.runtime.available and self.script_path.is_file()

    def build_request(
        self,
        operation: str,
        expression: Expr | None = None,
        declarations: tuple[TensorDeclaration, ...] = (),
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not _OPERATION_RE.fullmatch(operation):
            raise ValueError(f"Operación Wolfram inválida: {operation!r}")
        return {
            "schema_version": BRIDGE_SCHEMA_VERSION,
            "operation": operation,
            "expression": None if expression is None else expression.to_data(),
            "declarations": [_declaration_data(item) for item in declarations],
            "options": dict(options or {}),
        }

    def execute(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if not self.runtime.available:
            raise BackendUnavailableError(self.runtime.reason)
        if not self.script_path.is_file():
            raise BackendUnavailableError(
                f"No se encontró el script puente: {self.script_path}"
            )
        assert self.runtime.executable is not None
        with tempfile.TemporaryDirectory(prefix="tensor_engine_wolfram_") as temporary:
            directory = Path(temporary)
            request_path = directory / "request.json"
            response_path = directory / "response.json"
            request_path.write_text(
                json.dumps(dict(request), ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            try:
                completed = subprocess.run(
                    [
                        self.runtime.executable,
                        "-file",
                        str(self.script_path),
                        str(request_path),
                        str(response_path),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    shell=False,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise BackendExecutionError(f"No se pudo ejecutar WolframScript: {error}") from error
            if completed.returncode != 0:
                diagnostic = "\n".join(
                    item.strip()
                    for item in (completed.stderr, completed.stdout)
                    if item and item.strip()
                )
                raise BackendExecutionError(
                    f"WolframScript terminó con código {completed.returncode}: {diagnostic}"
                )
            if not response_path.is_file():
                diagnostic = "\n".join(
                    item.strip()
                    for item in (completed.stderr, completed.stdout)
                    if item and item.strip()
                )
                suffix = "" if not diagnostic else f" Salida del kernel: {diagnostic}"
                raise BackendExecutionError(
                    "WolframScript no produjo response.json." + suffix
                )
            try:
                response = json.loads(response_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise BackendExecutionError("La respuesta Wolfram no es JSON válido.") from error
        if not isinstance(response, dict):
            raise BackendExecutionError("La respuesta Wolfram debe ser un objeto JSON.")
        return response

    def ping(self) -> dict[str, Any]:
        return self.execute(self.build_request("ping"))

    def validate_phase5(self) -> WolframPhase5Report:
        """Ejecuta las identidades de referencia de fase 5 en xTensor/xPert."""

        response = self.execute(self.build_request("verify_phase5"))
        return WolframPhase5Report.from_data(response)

    def validate_phase6(self) -> WolframPhase6Report:
        """Ejecuta las identidades de Noether-Wald de fase 6 en xTensor/xTras."""

        response = self.execute(self.build_request("verify_phase6"))
        return WolframPhase6Report.from_data(response)

    def validate_phase7(self) -> WolframPhase7Report:
        """Ejecuta la geometría FLRW de referencia mediante xCoba."""

        response = self.execute(self.build_request("verify_phase7"))
        return WolframPhase7Report.from_data(response)

    def build_model_validation_request(
        self,
        model: ModelSpec,
        momenta: LagrangianMomenta,
        euler: EulerLagrangeResult,
        *,
        normalized_lagrangian: Expr | None = None,
        noether: NoetherWaldResult | None = None,
    ) -> dict[str, Any]:
        """Construye una solicitud genérica ligada al modelo y sus resultados."""

        normalized = normalized_lagrangian or model.lagrangian
        checks = _generic_checks(model, momenta, euler, noether)
        residuals = tuple(expr_from_data(check["residual"]) for check in checks)
        indices = sorted(
            {
                index.name
                for residual in residuals
                for index in all_indices(residual)
            }
        )
        return self.build_request(
            "verify_model",
            normalized,
            model.tensor_declarations,
            {
                "subject": {
                    "model_name": model.name,
                    "model_fingerprint": model_fingerprint(model),
                    "calculation_fingerprint": calculation_fingerprint(
                        model,
                        normalized,
                        momenta,
                        euler,
                        noether,
                    ),
                },
                "model": model.to_data(),
                "indices": indices,
                "checks": list(checks),
            },
        )

    def validate_model(
        self,
        model: ModelSpec,
        momenta: LagrangianMomenta,
        euler: EulerLagrangeResult,
        *,
        normalized_lagrangian: Expr | None = None,
        noether: NoetherWaldResult | None = None,
    ) -> WolframValidationReport:
        """Valida residuales del modelo concreto y comprueba el eco criptográfico."""

        request = self.build_model_validation_request(
            model,
            momenta,
            euler,
            normalized_lagrangian=normalized_lagrangian,
            noether=noether,
        )
        response = self.execute(request)
        report = WolframValidationReport.from_data(response)
        expected = request["options"]["subject"]
        if (
            report.model_name != expected["model_name"]
            or report.model_fingerprint != expected["model_fingerprint"]
            or report.calculation_fingerprint != expected["calculation_fingerprint"]
        ):
            raise BackendExecutionError(
                "Wolfram devolvió evidencia ligada a un modelo o cálculo diferente."
            )
        return report
