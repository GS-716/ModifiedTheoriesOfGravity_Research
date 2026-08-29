"""Ejecuta y exporta la corrida escalar-tensor de referencia de fase 9."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tensor_engine import RunExporter, RunPackage, StructuralTensorBackend, verify_run  # noqa: E402
from verify_phase8_reference import reference_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Exporta JSON, manifiesto e informe LaTeX.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "phase9_reference",
    )
    args = parser.parse_args()

    model = reference_model()
    backend = StructuralTensorBackend.from_model(model)
    momenta = backend.derive_momenta(model.lagrangian)
    raw = backend.raw_lagrangian_variation(momenta)
    euler = backend.derive_euler_lagrange(model.lagrangian, momenta)
    noether = backend.derive_noether_wald(model.lagrangian, momenta, euler)
    verification = verify_run(
        model,
        momenta,
        euler,
        raw_variation=raw,
        noether=noether,
    )
    package = RunPackage(model, momenta, raw, euler, verification, noether=noether)
    bundle = RunExporter(args.output_root).export(package)

    summary = verification.summary
    print(
        f"Fase 9: {verification.status.value}; run_id={package.run_id}; "
        f"passed={summary['passed']}, failed={summary['failed']}, "
        f"undetermined={summary['undetermined']}"
    )
    print(f"Manifiesto: {bundle.manifest_path.resolve()}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
