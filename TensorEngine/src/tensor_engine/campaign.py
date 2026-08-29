"""Ejecución uniforme y comparable de colecciones de lagrangianos."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Callable, Mapping

from .contracts import StageStatus
from .engine import EngineOptions, RunEvent, TensorEngine
from .model import ModelSpec
from .wolfram_bridge import WolframXActBridge


CAMPAIGN_SCHEMA_VERSION = "1.0"
_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class CampaignEntry:
    key: str
    model: ModelSpec

    def __post_init__(self) -> None:
        if not _KEY_RE.fullmatch(self.key):
            raise ValueError(f"Clave de campaña inválida: {self.key!r}.")

    def to_data(self) -> dict[str, Any]:
        return {"key": self.key, "model": self.model.to_data()}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "CampaignEntry":
        return cls(str(data["key"]), ModelSpec.from_data(data["model"]))


@dataclass(frozen=True, slots=True)
class CampaignSpec:
    name: str
    entries: tuple[CampaignEntry, ...]
    metadata: tuple[tuple[str, str], ...] = ()
    schema_version: str = CAMPAIGN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))
        object.__setattr__(self, "metadata", tuple(tuple(item) for item in self.metadata))
        if self.schema_version != CAMPAIGN_SCHEMA_VERSION:
            raise ValueError(f"Versión de campaña no soportada: {self.schema_version!r}.")
        if not _KEY_RE.fullmatch(self.name):
            raise ValueError(f"Nombre de campaña inválido: {self.name!r}.")
        if not self.entries:
            raise ValueError("Una campaña debe contener al menos un modelo.")
        keys = [item.key for item in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("Las claves de una campaña deben ser únicas.")
        model_names = [item.model.name for item in self.entries]
        if len(model_names) != len(set(model_names)):
            raise ValueError("Los nombres de modelo de una campaña deben ser únicos.")
        if any(len(item) != 2 for item in self.metadata):
            raise ValueError("La metadata de campaña debe contener pares clave/valor.")

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "entries": [item.to_data() for item in self.entries],
            "metadata": {key: value for key, value in self.metadata},
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "CampaignSpec":
        return cls(
            str(data["name"]),
            tuple(CampaignEntry.from_data(item) for item in data["entries"]),
            tuple((str(key), str(value)) for key, value in data.get("metadata", {}).items()),
            str(data.get("schema_version", CAMPAIGN_SCHEMA_VERSION)),
        )


@dataclass(frozen=True, slots=True)
class CampaignRecord:
    key: str
    model_name: str
    status: StageStatus
    verification: tuple[tuple[str, int], ...]
    duration_seconds: float
    run_id: str | None = None
    output_directory: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "verification",
            tuple(sorted((str(key), int(value)) for key, value in self.verification)),
        )
        if self.duration_seconds < 0:
            raise ValueError("La duración de una entrada de campaña no puede ser negativa.")
        if self.status is StageStatus.FAILED and self.run_id is None and self.error is None:
            raise ValueError("Una entrada fallida sin corrida debe conservar el error.")

    @property
    def summary(self) -> dict[str, int]:
        return dict(self.verification)

    def to_data(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "model_name": self.model_name,
            "status": self.status.value,
            "verification": self.summary,
            "duration_seconds": self.duration_seconds,
            "run_id": self.run_id,
            "output_directory": self.output_directory,
            "error": self.error,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "CampaignRecord":
        return cls(
            str(data["key"]),
            str(data["model_name"]),
            StageStatus(str(data["status"])),
            tuple((str(key), int(value)) for key, value in data.get("verification", {}).items()),
            float(data.get("duration_seconds", 0.0)),
            None if data.get("run_id") is None else str(data["run_id"]),
            None if data.get("output_directory") is None else str(data["output_directory"]),
            None if data.get("error") is None else str(data["error"]),
        )


@dataclass(frozen=True, slots=True)
class CampaignReport:
    campaign_name: str
    records: tuple[CampaignRecord, ...]
    created_at_utc: str
    duration_seconds: float
    engine_options: tuple[tuple[str, bool], ...]
    wolfram_enabled: bool
    schema_version: str = CAMPAIGN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(
            self,
            "engine_options",
            tuple(sorted((str(key), bool(value)) for key, value in self.engine_options)),
        )
        if self.schema_version != CAMPAIGN_SCHEMA_VERSION:
            raise ValueError(f"Versión de reporte de campaña no soportada: {self.schema_version!r}.")
        if self.duration_seconds < 0:
            raise ValueError("La duración total de campaña no puede ser negativa.")

    @property
    def status(self) -> StageStatus:
        if any(item.status is StageStatus.FAILED for item in self.records):
            return StageStatus.FAILED
        if any(item.status is StageStatus.PARTIAL for item in self.records):
            return StageStatus.PARTIAL
        return StageStatus.SUCCESS

    @property
    def summary(self) -> dict[str, int]:
        return {
            status.value: sum(item.status is status for item in self.records)
            for status in StageStatus
            if status in {StageStatus.SUCCESS, StageStatus.PARTIAL, StageStatus.FAILED}
        }

    def acceptable(self, *, strict: bool = False) -> bool:
        return self.status is StageStatus.SUCCESS if strict else self.status is not StageStatus.FAILED

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_name": self.campaign_name,
            "status": self.status.value,
            "summary": self.summary,
            "created_at_utc": self.created_at_utc,
            "duration_seconds": self.duration_seconds,
            "engine_options": dict(self.engine_options),
            "wolfram_enabled": self.wolfram_enabled,
            "records": [item.to_data() for item in self.records],
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "CampaignReport":
        return cls(
            str(data["campaign_name"]),
            tuple(CampaignRecord.from_data(item) for item in data["records"]),
            str(data["created_at_utc"]),
            float(data.get("duration_seconds", 0.0)),
            tuple(
                (str(key), bool(value))
                for key, value in data.get("engine_options", {}).items()
            ),
            bool(data.get("wolfram_enabled", False)),
            str(data.get("schema_version", CAMPAIGN_SCHEMA_VERSION)),
        )


CampaignEventHandler = Callable[[str, RunEvent], None]


class CampaignRunner:
    """Ejecuta cada modelo aisladamente y conserva todos los resultados."""

    def __init__(
        self,
        *,
        options: EngineOptions | None = None,
        event_handler: CampaignEventHandler | None = None,
    ) -> None:
        self.options = options or EngineOptions()
        self.event_handler = event_handler

    def run(
        self,
        campaign: CampaignSpec,
        *,
        output_root: str | Path | None = None,
        wolfram_bridge: WolframXActBridge | None = None,
    ) -> CampaignReport:
        started = perf_counter()
        root = None if output_root is None else Path(output_root)
        records: list[CampaignRecord] = []
        for entry in campaign.entries:
            entry_started = perf_counter()

            def emit(event: RunEvent, key: str = entry.key) -> None:
                if self.event_handler is not None:
                    self.event_handler(key, event)

            try:
                result = TensorEngine(options=self.options, event_handler=emit).run(
                    entry.model,
                    output_root=None if root is None else root / entry.key,
                    wolfram_bridge=wolfram_bridge,
                )
                records.append(
                    CampaignRecord(
                        entry.key,
                        entry.model.name,
                        result.status,
                        tuple(result.package.verification.summary.items()),
                        perf_counter() - entry_started,
                        result.package.run_id,
                        (
                            None
                            if result.export_bundle is None
                            else str(result.export_bundle.output_directory.resolve())
                        ),
                    )
                )
            except Exception as error:
                records.append(
                    CampaignRecord(
                        entry.key,
                        entry.model.name,
                        StageStatus.FAILED,
                        (),
                        perf_counter() - entry_started,
                        error=str(error),
                    )
                )
        return CampaignReport(
            campaign.name,
            tuple(records),
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            perf_counter() - started,
            tuple(self.options.to_data().items()),
            wolfram_bridge is not None,
        )
