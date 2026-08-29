"""Compila y ejecuta una fuente textual compuesta de fase 14."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tensor_engine import (  # noqa: E402
    FunctionSpec,
    LagrangianSourceSpec,
    ParameterSpec,
    RunEvent,
    TensorEngine,
    WolframXActBridge,
    save_lagrangian_source,
    save_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Referencia declarativa de fase 14.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "phase14_reference",
    )
    parser.add_argument("--wolframscript")
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args()

    source = LagrangianSourceSpec(
        "phase14_declarative_reference",
        "F(phi)*R + alpha*R**2 + K(phi, X) - V(phi)",
        normalization="1/kappa",
        parameters=(
            ParameterSpec("alpha", description="Acoplamiento cuadrático"),
            ParameterSpec("kappa", assumptions=("nonzero",), description="Normalización global"),
        ),
        functions=(
            FunctionSpec("F", 1, "Acoplamiento no mínimo"),
            FunctionSpec("K", 2, "Sector K(phi,X)"),
            FunctionSpec("V", 1, "Potencial"),
        ),
        metadata=(("purpose", "phase14_safe_source_compilation"),),
    )
    source_path = save_lagrangian_source(source, args.output_root / "source.json")
    model = source.compile()
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
    summary = result.package.verification.summary
    print(
        f"Fase 14: {result.status.value}; run_id={result.package.run_id}; "
        f"passed={summary['passed']}, failed={summary['failed']}, "
        f"undetermined={summary['undetermined']}"
    )
    print(f"Fingerprint fuente: {source.fingerprint}")
    print(f"Fuente: {source_path.resolve()}")
    print(f"Modelo: {model_path.resolve()}")
    if result.export_bundle is not None:
        print(f"Manifiesto: {result.export_bundle.manifest_path.resolve()}")
    return 0 if result.status.value == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
