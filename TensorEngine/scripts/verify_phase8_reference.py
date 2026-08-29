"""Genera el informe integral del modelo escalar-tensor de referencia."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tensor_engine import (  # noqa: E402
    DimensionSpec,
    FunctionSpec,
    ModelBuilder,
    ModelSpec,
    Number,
    StructuralTensorBackend,
    function,
    verify_run,
)


def reference_model() -> ModelSpec:
    builder = ModelBuilder()
    ricci_scalar = (
        builder.metric("a", "c")
        * builder.metric("b", "d")
        * builder.riemann("a", "b", "c", "d")
    )
    kinetic = (
        builder.metric("a", "b")
        * builder.scalar_gradient("a")
        * builder.scalar_gradient("b")
    )
    lagrangian = (
        function("F", builder.phi) * ricci_scalar
        - Number(1, 2) * function("Z", builder.phi) * kinetic
        - function("V", builder.phi)
    )
    return ModelSpec(
        name="phase8_scalar_tensor_reference",
        lagrangian=lagrangian,
        dimension=DimensionSpec(4),
        functions=(FunctionSpec("F"), FunctionSpec("Z"), FunctionSpec("V")),
        metadata=(("purpose", "phase8_integral_verification"),),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verifica momentos, Euler-Lagrange y Noether del modelo de fase 8."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "phase8_reference_verification.json",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Devuelve error también cuando existen comprobaciones indeterminadas.",
    )
    args = parser.parse_args()

    model = reference_model()
    backend = StructuralTensorBackend.from_model(model)
    momenta = backend.derive_momenta(model.lagrangian)
    raw = backend.raw_lagrangian_variation(momenta)
    euler = backend.derive_euler_lagrange(model.lagrangian, momenta)
    noether = backend.derive_noether_wald(model.lagrangian, momenta, euler)
    report = verify_run(
        model,
        momenta,
        euler,
        raw_variation=raw,
        noether=noether,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.to_data(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Fase 8: {report.status.value}; "
        f"passed={report.summary['passed']}, "
        f"failed={report.summary['failed']}, "
        f"undetermined={report.summary['undetermined']}"
    )
    print(f"Informe: {args.output.resolve()}")
    if report.summary["failed"]:
        return 1
    if args.strict and not report.passed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
