"""Ejecuta el pipeline completo mediante la API pública de fase 10."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tensor_engine import RunEvent, TensorEngine, save_model  # noqa: E402
from verify_phase8_reference import reference_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Corrida integral escalar-tensor de fase 10.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "phase10_reference",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    base = reference_model()
    model = replace(
        base,
        name="phase10_scalar_tensor_reference",
        metadata=(("purpose", "phase10_end_to_end_orchestration"),),
    )
    model_path = save_model(model, args.output_root / "model.json")

    def show(event: RunEvent) -> None:
        if event.state == "completed":
            print(f"  [ok] {event.stage_key}: {event.duration_seconds:.3f} s")

    result = TensorEngine(event_handler=show).run(
        model,
        output_root=args.output_root / "runs",
    )
    summary = result.package.verification.summary
    print(
        f"Fase 10: {result.status.value}; run_id={result.package.run_id}; "
        f"passed={summary['passed']}, failed={summary['failed']}, "
        f"undetermined={summary['undetermined']}"
    )
    print(f"Modelo JSON: {model_path.resolve()}")
    if result.export_bundle is not None:
        print(f"Manifiesto: {result.export_bundle.manifest_path.resolve()}")
    if result.status.value == "failed":
        return 1
    if args.strict and result.status.value == "partial":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
