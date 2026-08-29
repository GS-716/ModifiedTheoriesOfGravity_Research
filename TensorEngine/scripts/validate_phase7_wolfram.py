"""Ejecuta y conserva la validación de geometría coordenada de fase 7."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tensor_engine import TensorEngineError, WolframXActBridge  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valida el ansatz FLRW y sus componentes mediante Wolfram Engine/xCoba."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "phase7_wolfram_validation.json",
    )
    parser.add_argument("--wolframscript", help="Ruta explícita a wolframscript.")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    try:
        report = WolframXActBridge(
            executable=args.wolframscript,
            timeout_seconds=args.timeout,
        ).validate_phase7()
    except TensorEngineError as error:
        print(f"Validación Wolfram/xCoba no ejecutada: {error}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.to_data(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wolfram/xCoba fase 7: {report.status}; "
        f"passed={report.summary['passed']}, "
        f"failed={report.summary['failed']}, "
        f"undetermined={report.summary['undetermined']}"
    )
    print(f"Informe: {args.output.resolve()}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
