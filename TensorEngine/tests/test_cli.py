from __future__ import annotations

import json

from tensor_engine import (
    CampaignEntry,
    CampaignSpec,
    DimensionSpec,
    ModelSpec,
    Number,
    LagrangianSourceSpec,
    FunctionSpec,
    catalog_model,
    load_model,
    save_campaign,
    save_model,
    save_lagrangian_source,
)
from tensor_engine.cli import main


def model_file(tmp_path):
    model = ModelSpec("cli_constant", Number(1), dimension=DimensionSpec(4))
    return save_model(model, tmp_path / "model.json")


def test_cli_validates_model_json(tmp_path, capsys) -> None:
    path = model_file(tmp_path)
    exit_code = main(("validate", str(path), "--json"))
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["status"] == "success"
    assert output["model_name"] == "cli_constant"


def test_cli_runs_without_export_when_requested(tmp_path, capsys) -> None:
    path = model_file(tmp_path)
    exit_code = main(("run", str(path), "--no-export", "--no-noether", "--json"))
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["status"] == "success"
    assert output["output_directory"] is None
    assert output["skipped_stages"] == [
        "noether",
        "components",
        "wolfram_model_validation",
        "export",
    ]


def test_cli_exports_to_selected_root(tmp_path, capsys) -> None:
    path = model_file(tmp_path)
    output_root = tmp_path / "runs"
    exit_code = main(("run", str(path), "--output-root", str(output_root), "--json"))
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["output_directory"] is not None
    assert (output_root).is_dir()


def test_cli_reports_invalid_json_without_traceback(tmp_path, capsys) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("not json", encoding="utf-8")
    exit_code = main(("validate", str(path)))
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "tensor-engine: error:" in captured.err


def test_cli_lists_and_exports_catalog_models(tmp_path, capsys) -> None:
    assert main(("catalog", "list", "--json")) == 0
    listed = json.loads(capsys.readouterr().out)
    assert len(listed["models"]) == 5
    target = tmp_path / "kessence.json"
    assert main(("catalog", "export", "k_essence", str(target), "--json")) == 0
    exported = json.loads(capsys.readouterr().out)
    assert exported["model_name"] == "k_essence"
    assert load_model(target).name == "k_essence"


def test_cli_runs_campaign_and_writes_comparative_report(tmp_path, capsys) -> None:
    campaign = CampaignSpec(
        "cli_campaign",
        (CampaignEntry("eh", catalog_model("einstein_hilbert", name="cli_eh")),),
    )
    spec = save_campaign(campaign, tmp_path / "campaign.json")
    output = tmp_path / "campaign-output"
    exit_code = main((
        "campaign",
        str(spec),
        "--output-root",
        str(output),
        "--no-export",
        "--no-noether",
        "--json",
    ))
    data = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert data["status"] == "success"
    assert (output / "cli_campaign-campaign-report.json").is_file()


def test_cli_compiles_and_runs_declarative_source(tmp_path, capsys) -> None:
    source = LagrangianSourceSpec(
        "cli_source",
        "R - X/2 - V(phi)",
        functions=(FunctionSpec("V", 1),),
    )
    source_path = save_lagrangian_source(source, tmp_path / "source.json")
    model_path = tmp_path / "compiled-model.json"
    assert main(("compile", str(source_path), str(model_path), "--json")) == 0
    compiled = json.loads(capsys.readouterr().out)
    assert compiled["source_fingerprint"] == source.fingerprint
    assert load_model(model_path).name == "cli_source"

    assert main((
        "run-source",
        str(source_path),
        "--no-export",
        "--no-noether",
        "--json",
    )) == 0
    executed = json.loads(capsys.readouterr().out)
    assert executed["status"] == "success"
    assert executed["model_name"] == "cli_source"
