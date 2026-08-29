"""Ejecuta la referencia integral con validación diferencial y adjudicación xAct."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tensor_engine import RunEvent, TensorEngine, WolframXActBridge, save_model  # noqa: E402
from verify_phase8_reference import reference_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Corrida integral de fase 12 con evidencia diferencial xAct."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "phase12_reference",
    )
    parser.add_argument("--wolframscript")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    model = replace(
        reference_model(),
        name="phase12_scalar_tensor_reference",
        metadata=(("purpose", "phase12_differential_adjudication"),),
    )
    model_path = save_model(model, args.output_root / "model.json")

    def show(event: RunEvent) -> None:
        if event.state == "completed":
            print(f"  [ok] {event.stage_key}: {event.duration_seconds:.3f} s")

    result = TensorEngine(event_handler=show).run(
        model,
        output_root=args.output_root / "runs",
        wolfram_bridge=WolframXActBridge(
            executable=args.wolframscript,
            timeout_seconds=args.timeout,
        ),
    )
    verification = result.package.verification
    summary = verification.summary
    print(
        f"Fase 12: {result.status.value}; run_id={result.package.run_id}; "
        f"passed={summary['passed']}, failed={summary['failed']}, "
        f"undetermined={summary['undetermined']}"
    )
    print(f"Adjudicaciones: {len(verification.adjudications)}")
    for internal, operation, external in verification.adjudications:
        print(f"  {internal} <- {operation}:{external}")
    print(f"Modelo JSON: {model_path.resolve()}")
    if result.export_bundle is not None:
        print(f"Manifiesto: {result.export_bundle.manifest_path.resolve()}")
    return 0 if result.status.value == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
