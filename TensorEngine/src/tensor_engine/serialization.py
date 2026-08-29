"""Entrada y salida JSON explícita para modelos y ansatz."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .components import GeometryAnsatz
from .campaign import CampaignReport, CampaignSpec
from .model import ModelSpec
from .source import LagrangianSourceSpec


def _read_mapping(path: str | Path) -> Mapping[str, Any]:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{source} debe contener un objeto JSON en la raíz.")
    return data


def _write_mapping(path: str | Path, data: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".tensor-engine-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return target


def load_model(path: str | Path) -> ModelSpec:
    return ModelSpec.from_data(_read_mapping(path))


def save_model(model: ModelSpec, path: str | Path) -> Path:
    return _write_mapping(path, model.to_data())


def load_ansatz(path: str | Path) -> GeometryAnsatz:
    return GeometryAnsatz.from_data(_read_mapping(path))


def save_ansatz(ansatz: GeometryAnsatz, path: str | Path) -> Path:
    return _write_mapping(path, ansatz.to_data())


def load_campaign(path: str | Path) -> CampaignSpec:
    return CampaignSpec.from_data(_read_mapping(path))


def save_campaign(campaign: CampaignSpec, path: str | Path) -> Path:
    return _write_mapping(path, campaign.to_data())


def load_campaign_report(path: str | Path) -> CampaignReport:
    return CampaignReport.from_data(_read_mapping(path))


def save_campaign_report(report: CampaignReport, path: str | Path) -> Path:
    return _write_mapping(path, report.to_data())


def load_lagrangian_source(path: str | Path) -> LagrangianSourceSpec:
    return LagrangianSourceSpec.from_data(_read_mapping(path))


def save_lagrangian_source(source: LagrangianSourceSpec, path: str | Path) -> Path:
    return _write_mapping(path, source.to_data())
