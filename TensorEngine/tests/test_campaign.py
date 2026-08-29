from __future__ import annotations

import json

from tensor_engine import (
    CampaignEntry,
    CampaignReport,
    CampaignRunner,
    CampaignSpec,
    EngineOptions,
    StageStatus,
    catalog_model,
    load_campaign,
    load_campaign_report,
    save_campaign,
    save_campaign_report,
    TensorEngine,
)


def reference_campaign() -> CampaignSpec:
    return CampaignSpec(
        "phase13_smoke",
        (
            CampaignEntry("eh", catalog_model("einstein_hilbert", name="campaign_eh")),
            CampaignEntry(
                "canonical",
                catalog_model("canonical_scalar", name="campaign_canonical"),
            ),
        ),
        (("purpose", "uniform_pipeline_test"),),
    )


def test_campaign_spec_roundtrip_and_file_io(tmp_path) -> None:
    campaign = reference_campaign()
    encoded = json.loads(json.dumps(campaign.to_data()))
    assert CampaignSpec.from_data(encoded) == campaign
    path = save_campaign(campaign, tmp_path / "campaign.json")
    assert load_campaign(path) == campaign


def test_campaign_executes_models_under_same_options(tmp_path) -> None:
    report = CampaignRunner(
        options=EngineOptions(
            include_noether=False,
            include_components=False,
            include_export=False,
        )
    ).run(reference_campaign())
    assert report.status is StageStatus.SUCCESS
    assert report.summary == {"success": 2, "failed": 0, "partial": 0}
    assert all(record.run_id is not None for record in report.records)
    assert all(record.output_directory is None for record in report.records)
    path = save_campaign_report(report, tmp_path / "report.json")
    loaded = load_campaign_report(path)
    assert isinstance(loaded, CampaignReport)
    assert loaded == report


def test_campaign_rejects_duplicate_keys() -> None:
    model = catalog_model("einstein_hilbert")
    try:
        CampaignSpec(
            "duplicate",
            (CampaignEntry("same", model), CampaignEntry("same", model)),
        )
    except ValueError as error:
        assert "claves" in str(error)
    else:
        raise AssertionError("La campaña aceptó claves duplicadas.")


def test_campaign_isolates_a_model_failure(monkeypatch) -> None:
    original = TensorEngine.run

    def flaky_run(self, model, **options):
        if model.name == "campaign_canonical":
            raise RuntimeError("synthetic model failure")
        return original(self, model, **options)

    monkeypatch.setattr(TensorEngine, "run", flaky_run)
    report = CampaignRunner(
        options=EngineOptions(
            include_noether=False,
            include_components=False,
            include_export=False,
        )
    ).run(reference_campaign())
    assert report.status is StageStatus.FAILED
    assert report.records[0].status is StageStatus.SUCCESS
    assert report.records[1].status is StageStatus.FAILED
    assert report.records[1].error == "synthetic model failure"
