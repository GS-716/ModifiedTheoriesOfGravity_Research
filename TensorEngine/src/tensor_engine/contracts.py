"""Contratos inmutables para etapas, resultados y verificaciones."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Any, Mapping

from .errors import ContractValidationError
from .ir import Expr, expr_from_data


def _validate_json_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, (list, tuple)):
        for position, item in enumerate(value):
            _validate_json_value(item, f"{path}[{position}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractValidationError(f"{path} contiene una clave no textual.")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise ContractValidationError(
        f"{path} contiene {type(value).__name__}, que no es serializable como JSON."
    )


class StageStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


class VerificationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    UNDETERMINED = "undetermined"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ExpressionForm(str, Enum):
    RAW = "raw"
    CANONICAL = "canonical"
    MODEL_REDUCED = "model_reduced"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    code: str
    message: str
    severity: Severity = Severity.ERROR

    def to_data(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "severity": self.severity.value}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "Diagnostic":
        return cls(
            str(data["code"]),
            str(data["message"]),
            Severity(data.get("severity", Severity.ERROR.value)),
        )


@dataclass(frozen=True, slots=True)
class VerificationDiagnostic:
    """Contexto estructurado y JSON-seguro de una verificación no concluida."""

    code: str
    reason: str
    category: str = "expression"
    path: tuple[str, ...] = ()
    node_type: str | None = None
    symbol: str | None = None
    fragment_json: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", tuple(str(item) for item in self.path))
        if not self.code or not self.reason or not self.category:
            raise ContractValidationError(
                "Un diagnóstico de verificación requiere código, categoría y motivo."
            )
        if self.fragment_json is not None:
            try:
                fragment = json.loads(self.fragment_json)
            except (TypeError, json.JSONDecodeError) as error:
                raise ContractValidationError(
                    "El fragmento diagnóstico no contiene JSON válido."
                ) from error
            _validate_json_value(fragment, "diagnostic.fragment")

    @property
    def fragment(self) -> Any:
        return None if self.fragment_json is None else json.loads(self.fragment_json)

    def to_data(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "reason": self.reason,
            "category": self.category,
            "path": list(self.path),
            "node_type": self.node_type,
            "symbol": self.symbol,
            "fragment": self.fragment,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "VerificationDiagnostic":
        fragment = data.get("fragment")
        _validate_json_value(fragment, "diagnostic.fragment")
        return cls(
            code=str(data["code"]),
            reason=str(data["reason"]),
            category=str(data.get("category", "expression")),
            path=tuple(str(item) for item in data.get("path", ())),
            node_type=None if data.get("node_type") is None else str(data["node_type"]),
            symbol=None if data.get("symbol") is None else str(data["symbol"]),
            fragment_json=(
                None
                if fragment is None
                else json.dumps(
                    fragment,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class VerificationRecord:
    key: str
    status: VerificationStatus
    residual: Expr | None = None
    message: str = ""
    diagnostic: VerificationDiagnostic | None = None

    def __post_init__(self) -> None:
        if self.status is VerificationStatus.PASSED and self.residual is not None:
            raise ContractValidationError(
                "Una verificación aprobada no debe conservar un residual no nulo."
            )
        if self.status is not VerificationStatus.PASSED and self.residual is None:
            raise ContractValidationError(
                "Una verificación fallida o indeterminada debe conservar su residual."
            )

    def to_data(self) -> dict[str, Any]:
        data = {
            "key": self.key,
            "status": self.status.value,
            "residual": None if self.residual is None else self.residual.to_data(),
            "message": self.message,
        }
        if self.diagnostic is not None:
            data["diagnostic"] = self.diagnostic.to_data()
        return data

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "VerificationRecord":
        residual_data = data.get("residual")
        diagnostic_data = data.get("diagnostic")
        if diagnostic_data is not None and not isinstance(diagnostic_data, Mapping):
            raise ContractValidationError("El diagnóstico de verificación debe ser un objeto JSON.")
        return cls(
            str(data["key"]),
            VerificationStatus(data["status"]),
            None if residual_data is None else expr_from_data(residual_data),
            str(data.get("message", "")),
            (
                None
                if diagnostic_data is None
                else VerificationDiagnostic.from_data(diagnostic_data)
            ),
        )


@dataclass(frozen=True, slots=True)
class ExpressionRecord:
    key: str
    expression: Expr
    form: ExpressionForm = ExpressionForm.RAW
    source_keys: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_keys", tuple(self.source_keys))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))

    def to_data(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "form": self.form.value,
            "expression": self.expression.to_data(),
            "source_keys": list(self.source_keys),
            "assumptions": list(self.assumptions),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ExpressionRecord":
        return cls(
            str(data["key"]),
            expr_from_data(data["expression"]),
            ExpressionForm(data.get("form", ExpressionForm.RAW.value)),
            tuple(str(item) for item in data.get("source_keys", ())),
            tuple(str(item) for item in data.get("assumptions", ())),
        )


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Salida estructurada no tensorial, por ejemplo un manifiesto o reporte."""

    key: str
    artifact_type: str
    payload: tuple[tuple[str, Any], ...]
    source_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", tuple(tuple(item) for item in self.payload))
        object.__setattr__(self, "source_keys", tuple(self.source_keys))
        keys = [item[0] for item in self.payload]
        if len(keys) != len(set(keys)):
            raise ContractValidationError(f"El artefacto {self.key} repite claves en su payload.")
        for key, value in self.payload:
            if not isinstance(key, str):
                raise ContractValidationError("Las claves de un artefacto deben ser texto.")
            _validate_json_value(value, f"{self.key}.{key}")

    def to_data(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "artifact_type": self.artifact_type,
            "payload": {key: value for key, value in self.payload},
            "source_keys": list(self.source_keys),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ArtifactRecord":
        return cls(
            str(data["key"]),
            str(data["artifact_type"]),
            tuple((str(key), value) for key, value in data.get("payload", {}).items()),
            tuple(str(item) for item in data.get("source_keys", ())),
        )


@dataclass(frozen=True, slots=True)
class StageSpec:
    key: str
    requires: tuple[str, ...]
    produces: tuple[str, ...]
    optional: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "requires", tuple(self.requires))
        object.__setattr__(self, "produces", tuple(self.produces))
        if not self.key or not self.produces:
            raise ContractValidationError("Una etapa necesita clave y al menos una salida.")
        if len(set(self.produces)) != len(self.produces):
            raise ContractValidationError(f"La etapa {self.key} repite claves de salida.")

    def to_data(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "requires": list(self.requires),
            "produces": list(self.produces),
            "optional": self.optional,
            "description": self.description,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "StageSpec":
        return cls(
            str(data["key"]),
            tuple(str(item) for item in data.get("requires", ())),
            tuple(str(item) for item in data["produces"]),
            bool(data.get("optional", False)),
            str(data.get("description", "")),
        )


@dataclass(frozen=True, slots=True)
class StageResult:
    stage_key: str
    status: StageStatus
    backend: str
    operation: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[ExpressionRecord, ...] = ()
    artifacts: tuple[ArtifactRecord, ...] = ()
    verifications: tuple[VerificationRecord, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    duration_seconds: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "verifications", tuple(self.verifications))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        if self.duration_seconds < 0:
            raise ContractValidationError("La duración de una etapa no puede ser negativa.")
        output_keys = [(item.key, item.form) for item in self.outputs]
        if len(output_keys) != len(set(output_keys)):
            raise ContractValidationError("Una etapa repite una salida con la misma forma.")
        artifact_keys = [item.key for item in self.artifacts]
        if len(artifact_keys) != len(set(artifact_keys)):
            raise ContractValidationError("Una etapa repite una salida estructurada.")
        if self.status is StageStatus.SUCCESS and any(
            item.severity is Severity.ERROR for item in self.diagnostics
        ):
            raise ContractValidationError("Una etapa exitosa no puede contener diagnósticos de error.")
        if self.status is StageStatus.FAILED and not self.diagnostics:
            raise ContractValidationError("Una etapa fallida debe incluir un diagnóstico.")

    def to_data(self) -> dict[str, Any]:
        return {
            "stage_key": self.stage_key,
            "status": self.status.value,
            "backend": self.backend,
            "operation": self.operation,
            "inputs": list(self.inputs),
            "outputs": [item.to_data() for item in self.outputs],
            "artifacts": [item.to_data() for item in self.artifacts],
            "verifications": [item.to_data() for item in self.verifications],
            "diagnostics": [item.to_data() for item in self.diagnostics],
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "StageResult":
        return cls(
            stage_key=str(data["stage_key"]),
            status=StageStatus(data["status"]),
            backend=str(data["backend"]),
            operation=str(data["operation"]),
            inputs=tuple(str(item) for item in data.get("inputs", ())),
            outputs=tuple(ExpressionRecord.from_data(item) for item in data.get("outputs", ())),
            artifacts=tuple(ArtifactRecord.from_data(item) for item in data.get("artifacts", ())),
            verifications=tuple(
                VerificationRecord.from_data(item) for item in data.get("verifications", ())
            ),
            diagnostics=tuple(Diagnostic.from_data(item) for item in data.get("diagnostics", ())),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
        )


def validate_stage_result(spec: StageSpec, result: StageResult) -> None:
    """Comprueba un resultado contra las entradas y salidas declaradas."""

    if result.stage_key != spec.key:
        raise ContractValidationError(
            f"El resultado de {result.stage_key!r} no corresponde a la etapa {spec.key!r}."
        )
    missing_inputs = set(spec.requires).difference(result.inputs)
    if missing_inputs:
        raise ContractValidationError(
            f"El resultado de {spec.key} no registra entradas requeridas: {sorted(missing_inputs)}"
        )
    produced = {item.key for item in result.outputs}.union(
        item.key for item in result.artifacts
    )
    unexpected = produced.difference(spec.produces)
    if unexpected:
        raise ContractValidationError(
            f"La etapa {spec.key} produjo claves no declaradas: {sorted(unexpected)}"
        )
    if result.status is StageStatus.SUCCESS:
        missing_outputs = set(spec.produces).difference(produced)
        if missing_outputs:
            raise ContractValidationError(
                f"La etapa exitosa {spec.key} no produjo: {sorted(missing_outputs)}"
            )
