"""Interfaz de línea de comandos para ejecutar modelos serializados."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .campaign import CampaignRunner
from .catalog import catalog_entries, catalog_model
from .engine import EngineOptions, TensorEngine
from .errors import TensorEngineError
from .serialization import (
    load_ansatz,
    load_campaign,
    load_lagrangian_source,
    load_model,
    save_campaign_report,
    save_model,
)
from .wolfram_bridge import WolframXActBridge


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tensor-engine",
        description="Motor variacional para L(g,R,phi,nabla phi).",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser("validate", help="Valida un ModelSpec JSON.")
    validate.add_argument("model", type=Path)
    validate.add_argument("--json", action="store_true", dest="as_json")

    compile_source = subcommands.add_parser(
        "compile", help="Compila una fuente lagrangiana declarativa a ModelSpec."
    )
    compile_source.add_argument("source", type=Path)
    compile_source.add_argument("output", type=Path)
    compile_source.add_argument("--json", action="store_true", dest="as_json")

    catalog = subcommands.add_parser("catalog", help="Lista o exporta modelos de referencia.")
    catalog_commands = catalog.add_subparsers(dest="catalog_command", required=True)
    catalog_list = catalog_commands.add_parser("list", help="Lista el catálogo incorporado.")
    catalog_list.add_argument("--json", action="store_true", dest="as_json")
    catalog_export = catalog_commands.add_parser("export", help="Exporta un ModelSpec del catálogo.")
    catalog_export.add_argument("key")
    catalog_export.add_argument("output", type=Path)
    catalog_export.add_argument("--name")
    catalog_export.add_argument("--json", action="store_true", dest="as_json")

    run = subcommands.add_parser("run", help="Ejecuta el pipeline integral.")
    run.add_argument("model", type=Path)
    run.add_argument("--ansatz", type=Path)
    run.add_argument("--output-root", type=Path, default=Path("outputs") / "runs")
    run.add_argument("--no-export", action="store_true")
    run.add_argument("--no-noether", action="store_true")
    run.add_argument("--no-components", action="store_true")
    run.add_argument("--strict", action="store_true", help="Un estado partial devuelve código 2.")
    run.add_argument("--wolfram", action="store_true", help="Añade validación ligada al modelo mediante xAct.")
    run.add_argument("--wolframscript", help="Ruta explícita al ejecutable wolframscript.")
    run.add_argument("--wolfram-timeout", type=float, default=180.0)
    run.add_argument("--json", action="store_true", dest="as_json")

    run_source = subcommands.add_parser(
        "run-source", help="Compila y ejecuta una fuente lagrangiana en un paso."
    )
    run_source.add_argument("source", type=Path)
    run_source.add_argument("--ansatz", type=Path)
    run_source.add_argument("--output-root", type=Path, default=Path("outputs") / "runs")
    run_source.add_argument("--no-export", action="store_true")
    run_source.add_argument("--no-noether", action="store_true")
    run_source.add_argument("--no-components", action="store_true")
    run_source.add_argument("--strict", action="store_true")
    run_source.add_argument("--wolfram", action="store_true")
    run_source.add_argument("--wolframscript")
    run_source.add_argument("--wolfram-timeout", type=float, default=180.0)
    run_source.add_argument("--json", action="store_true", dest="as_json")

    campaign = subcommands.add_parser("campaign", help="Ejecuta varios modelos bajo el mismo pipeline.")
    campaign.add_argument("spec", type=Path)
    campaign.add_argument("--output-root", type=Path, default=Path("outputs") / "campaigns")
    campaign.add_argument("--no-export", action="store_true")
    campaign.add_argument("--no-noether", action="store_true")
    campaign.add_argument("--strict", action="store_true")
    campaign.add_argument("--wolfram", action="store_true")
    campaign.add_argument("--wolframscript")
    campaign.add_argument("--wolfram-timeout", type=float, default=180.0)
    campaign.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _print_data(data: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(data, ensure_ascii=False, sort_keys=True))
        return
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        print(f"{key}: {value}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "compile":
            source = load_lagrangian_source(args.source)
            model = source.compile()
            path = save_model(model, args.output)
            _print_data(
                {
                    "status": "success",
                    "model_name": model.name,
                    "source_fingerprint": source.fingerprint,
                    "path": str(path.resolve()),
                },
                args.as_json,
            )
            return 0

        if args.command == "catalog":
            if args.catalog_command == "list":
                _print_data(
                    {"status": "success", "models": [item.to_data() for item in catalog_entries()]},
                    args.as_json,
                )
                return 0
            model = catalog_model(args.key, name=args.name)
            path = save_model(model, args.output)
            _print_data(
                {"status": "success", "model_name": model.name, "path": str(path.resolve())},
                args.as_json,
            )
            return 0

        if args.command == "campaign":
            campaign = load_campaign(args.spec)
            options = EngineOptions(
                include_noether=not args.no_noether,
                include_components=False,
                include_export=not args.no_export,
            )
            bridge = None
            if args.wolfram:
                bridge = WolframXActBridge(
                    executable=args.wolframscript,
                    timeout_seconds=args.wolfram_timeout,
                )
            report = CampaignRunner(options=options).run(
                campaign,
                output_root=args.output_root,
                wolfram_bridge=bridge,
            )
            report_path = save_campaign_report(
                report,
                args.output_root / f"{campaign.name}-campaign-report.json",
            )
            data = report.to_data()
            data["report_path"] = str(report_path.resolve())
            _print_data(data, args.as_json)
            if report.status.value == "failed":
                return 1
            if args.strict and report.status.value == "partial":
                return 2
            return 0

        if args.command == "run-source":
            source = load_lagrangian_source(args.source)
            model = source.compile()
        else:
            model = load_model(args.model)
        if args.command == "validate":
            _print_data(
                {
                    "status": "success",
                    "model_name": model.name,
                    "schema_version": model.schema_version,
                    "dimension": model.dimension.value,
                },
                args.as_json,
            )
            return 0

        ansatz = None if args.ansatz is None else load_ansatz(args.ansatz)
        options = EngineOptions(
            include_noether=not args.no_noether,
            include_components=not args.no_components,
            include_export=not args.no_export,
        )
        engine = TensorEngine(options=options)
        bridge = None
        if args.wolfram:
            bridge = WolframXActBridge(
                executable=args.wolframscript,
                timeout_seconds=args.wolfram_timeout,
            )
        result = engine.run(
            model,
            ansatz=ansatz,
            output_root=None if args.no_export else args.output_root,
            wolfram_bridge=bridge,
        )
        _print_data(result.summary_data(), args.as_json)
        if result.status.value == "failed":
            return 1
        if args.strict and result.status.value == "partial":
            return 2
        return 0
    except (TensorEngineError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"tensor-engine: error: {error}", file=sys.stderr)
        return 1
