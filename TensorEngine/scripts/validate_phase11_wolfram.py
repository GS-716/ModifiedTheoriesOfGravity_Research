"""Valida con xAct un ModelSpec JSON concreto y conserva sus fingerprints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tensor_engine import (  # noqa: E402
    StructuralTensorBackend,
    TensorEngineError,
    WolframXActBridge,
    load_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida algebraicamente un modelo mediante xAct.")
    parser.add_argument("model", type=Path, help="ModelSpec JSON que identifica la teoría.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "phase11_wolfram_validation.json",
    )
    parser.add_argument("--wolframscript")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    try:
        model = load_model(args.model)
        backend = StructuralTensorBackend.from_model(model)
        normalized = backend.canonicalize(model.normalization * model.lagrangian)
        momenta = backend.derive_momenta(normalized)
        euler = backend.derive_euler_lagrange(normalized, momenta)
        noether = backend.derive_noether_wald(normalized, momenta, euler)
        report = WolframXActBridge(
            executable=args.wolframscript,
            timeout_seconds=args.timeout,
        ).validate_model(
            model,
            momenta,
            euler,
            normalized_lagrangian=normalized,
            noether=noether,
        )
    except TensorEngineError as error:
        print(f"Validación Wolfram/xAct no ejecutada: {error}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.to_data(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Fase 11: {report.status}; passed={report.summary['passed']}, "
        f"failed={report.summary['failed']}, undetermined={report.summary['undetermined']}"
    )
    print(f"Modelo: {report.model_name}")
    print(f"Fingerprint cálculo: {report.calculation_fingerprint}")
    print(f"Informe: {args.output.resolve()}")
    return 0 if report.summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
