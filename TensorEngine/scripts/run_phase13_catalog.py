"""Ejecuta el catálogo completo bajo una campaña común de fase 13."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tensor_engine import (  # noqa: E402
    CampaignEntry,
    CampaignRunner,
    CampaignSpec,
    EngineOptions,
    WolframXActBridge,
    catalog_entries,
    save_campaign,
    save_campaign_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Campaña comparativa del catálogo de fase 13.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "phase13_reference",
    )
    parser.add_argument("--no-wolfram", action="store_true")
    parser.add_argument("--wolframscript")
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args()

    campaign = CampaignSpec(
        "phase13_catalog_reference",
        tuple(
            CampaignEntry(
                entry.key,
                entry.create(name=f"phase13_{entry.key}"),
            )
            for entry in catalog_entries()
        ),
        (("purpose", "cross_model_uniform_pipeline"),),
    )
    spec_path = save_campaign(campaign, args.output_root / "campaign.json")
    bridge = None
    if not args.no_wolfram:
        bridge = WolframXActBridge(
            executable=args.wolframscript,
            timeout_seconds=args.timeout,
        )

    def show(key, event) -> None:
        if event.state == "completed" and event.stage_key in {"wolfram_model_validation", "verify", "export"}:
            print(f"  [{key}] {event.stage_key}: {event.duration_seconds:.3f} s")

    report = CampaignRunner(
        options=EngineOptions(
            include_noether=True,
            include_components=False,
            include_export=True,
        ),
        event_handler=show,
    ).run(
        campaign,
        output_root=args.output_root / "runs",
        wolfram_bridge=bridge,
    )
    report_path = save_campaign_report(report, args.output_root / "campaign-report.json")
    print(f"Fase 13: {report.status.value}; {report.summary}")
    for record in report.records:
        print(
            f"  {record.key}: {record.status.value}; "
            f"passed={record.summary.get('passed', 0)}, "
            f"failed={record.summary.get('failed', 0)}, "
            f"undetermined={record.summary.get('undetermined', 0)}"
        )
        if record.error:
            print(f"    error: {record.error}")
    print(f"Especificación: {spec_path.resolve()}")
    print(f"Reporte: {report_path.resolve()}")
    return 0 if report.status.value == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
