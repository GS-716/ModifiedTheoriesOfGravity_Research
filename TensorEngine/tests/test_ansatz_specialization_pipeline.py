from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import sympy as sp

from tensor_engine import (
    AnsatzSpecialization,
    DimensionSpec,
    EngineOptions,
    Function,
    LagrangianSourceSpec,
    ModelValidationError,
    Number,
    ParameterSpec,
    RunExporter,
    RunPackage,
    Scalar,
    TensorEngine,
    calculation_fingerprint,
    draft4_circular_ansatz,
    ir_scalar_to_sympy,
    latex_report,
    model_fingerprint,
    spatially_flat_flrw_ansatz,
)


def _case0():
    ansatz = draft4_circular_ansatz()
    _, r, _ = ansatz.chart.coordinates
    ell, mass = Scalar("ell"), Scalar("lambda")
    source = LagrangianSourceSpec(
        name="draft4_case_0_test",
        expression="R + 2/ell**2",
        dimension=DimensionSpec(3),
        parameters=(ParameterSpec("ell"), ParameterSpec("lambda")),
    )
    specialization = AnsatzSpecialization(
        metric_functions={"f": r**2 / ell**2 - mass},
        scalar_field=Number(0),
        name="draft4_case_0_test_specialized",
    )
    return source.compile(), ansatz, specialization


def _calculation_hash(run):
    package = run.package
    return calculation_fingerprint(
        package.model,
        package.lagrangian,
        package.momenta,
        package.euler,
        package.noether,
    )


def test_specialization_replaces_user_metric_and_scalar_without_mutating_base() -> None:
    base = draft4_circular_ansatz()
    _, r, varphi = base.chart.coordinates
    custom_f = Number(1) + r**2
    custom_phi = Scalar("q") * varphi
    spec = AnsatzSpecialization(
        metric_functions={"f": custom_f},
        scalar_field=custom_phi,
    )
    specialized = spec.apply(base)

    assert specialized.metric_covariant[0][0] == -custom_f
    assert specialized.scalar_field == custom_phi
    assert base.scalar_field == Function("Phi", base.chart.coordinates[1:])
    assert AnsatzSpecialization.from_data(spec.to_data()) == spec


def test_specialization_cannot_reintroduce_draft4_time_dependence() -> None:
    base = draft4_circular_ansatz()
    tau, r, varphi = base.chart.coordinates
    with pytest.raises(ModelValidationError, match="tau"):
        AnsatzSpecialization(
            scalar_field=Function("Psi", (tau, r, varphi)),
        ).apply(base)


def test_stationary_scalar_propagates_through_projected_equations() -> None:
    source = LagrangianSourceSpec(
        name="draft4_stationary_scalar_pipeline",
        expression="R - alpha*X",
        dimension=DimensionSpec(3),
        parameters=(ParameterSpec("alpha"),),
    )
    run = TensorEngine(
        options=EngineOptions(include_noether=False, include_export=False)
    ).run(source.compile(), ansatz=draft4_circular_ansatz())
    tau = sp.Symbol("tau")
    for quantity in run.projected.quantities:
        if quantity.components is None:
            continue
        for _, expression in quantity.components.values:
            assert not ir_scalar_to_sympy(expression).has(tau), quantity.key


def test_specialization_is_after_derivation_and_keeps_generic_projection() -> None:
    model, ansatz, specialization = _case0()
    options = EngineOptions(include_noether=False, include_export=False)
    generic = TensorEngine(options=options).run(model, ansatz=ansatz)
    run = TensorEngine(options=options).run(
        model,
        ansatz=ansatz,
        specialization=specialization,
    )

    assert run.abstract == generic.abstract
    assert run.projected == generic.projected
    assert run.projected.ansatz_geometry == ansatz
    assert run.specialized is not None
    assert run.specialized.ansatz_geometry != ansatz
    assert model_fingerprint(run.package.model) == model_fingerprint(generic.package.model)
    assert _calculation_hash(run) == _calculation_hash(generic)
    assert [stage.stage_key for stage in run.stages].index("specialize_ansatz") > [
        stage.stage_key for stage in run.stages
    ].index("organize_result_views")


def test_case0_specialized_results_roundtrip_and_report(tmp_path: Path) -> None:
    model, ansatz, specialization = _case0()
    run = TensorEngine(
        options=EngineOptions(include_noether=False, include_export=False)
    ).run(model, ansatz=ansatz, specialization=specialization)
    assert run.specialized is not None
    assert len(run.specialized.quantities) == 13
    assert all(item.status.value == "completed" for item in run.specialized.quantities)
    assert run.specialized.metric_euler.components is not None
    assert run.specialized.metric_euler.components.values == ()
    rebuilt = RunPackage.from_data(json.loads(json.dumps(run.package.to_data())))
    assert rebuilt == run.package
    assert rebuilt.run_id == run.package.run_id

    bundle = RunExporter(tmp_path, compile_pdf=False).export(run.package)
    tex = (bundle.output_directory / "report.tex").read_text(encoding="utf-8")
    presentation = json.loads(
        (bundle.output_directory / "presentation.json").read_text(encoding="utf-8")
    )
    results = json.loads(
        (bundle.output_directory / "results.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    assert "Resultados especializados mediante el ansatz" in tex
    assert tex.index("Expresiones tensoriales abstractas") < tex.index(
        "Expresiones proyectadas mediante el ansatz"
    ) < tex.index("Resultados especializados mediante el ansatz")
    assert results["specialized_results"]["specialization"]["metric_functions"]["f"]
    assert len(manifest["result_views"]["specialized"]) == 13
    assert any(key.startswith("specialized.") for key in presentation["expressions"])
    assert bundle.manifest.verify_files(bundle.output_directory) == ()


def test_backend_limitation_in_specialization_is_nonfatal() -> None:
    model, ansatz, _ = _case0()
    degenerate = AnsatzSpecialization(
        metric_functions={"f": Number(0)},
        scalar_field=Number(0),
    )
    run = TensorEngine(
        options=EngineOptions(include_noether=False, include_export=False)
    ).run(model, ansatz=ansatz, specialization=degenerate)
    assert run.specialized is not None
    assert all(item.status.value == "unavailable" for item in run.specialized.quantities)
    assert all(item.reason for item in run.specialized.quantities)


def test_notebook_contains_exactly_one_independent_cell_per_draft4_case() -> None:
    notebook_path = Path(__file__).parents[2] / "ResearchWorkflow" / "01_modified_gravity_workflow.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    by_id = {cell.get("id"): cell for cell in notebook["cells"]}
    assert {"draft4-case-0", "draft4-case-1", "draft4-case-2"}.issubset(by_id)
    assert sum(cell.get("id", "").startswith("draft4-case-") for cell in notebook["cells"]) == 3
    for number in range(3):
        source = "".join(by_id[f"draft4-case-{number}"]["source"])
        assert "f_input =" in source
        assert "phi_input =" in source
        assert "AnsatzSpecialization(" in source
        assert "specialization=specialization" in source
        assert f'name="draft4_case_{number}"' in source


def test_latex_without_specialization_keeps_the_two_existing_sections() -> None:
    model, ansatz, _ = _case0()
    run = TensorEngine(
        options=EngineOptions(include_noether=False, include_export=False)
    ).run(model, ansatz=ansatz)
    tex = latex_report(run.package)
    assert tex.count(r"\section*{") == 2
    assert "Resultados especializados mediante el ansatz" not in tex


def test_specialization_remains_generic_for_flrw_and_custom_ansatz() -> None:
    ansatz = spatially_flat_flrw_ansatz()
    specialization = AnsatzSpecialization(
        metric_functions={"a": Number(1)},
        scalar_field=Number(0),
        name="flat_flrw_specialized_test",
    )
    model = LagrangianSourceSpec(
        name="flrw_specialization_test",
        expression="R",
        dimension=DimensionSpec(4),
    ).compile()
    run = TensorEngine(
        options=EngineOptions(include_noether=False, include_export=False)
    ).run(model, ansatz=ansatz, specialization=specialization)
    assert run.projected.ansatz_geometry == ansatz
    assert run.specialized is not None
    assert run.specialized.ricci_scalar.scalar == Number(0)


@pytest.mark.skipif(
    os.environ.get("TENSOR_ENGINE_RUN_DRAFT4_SLOW") != "1",
    reason="La proyección completa del Caso 2 especializado tarda varios minutos.",
)
@pytest.mark.parametrize("case_number", (0, 1, 2))
def test_draft4_cases_specialize_to_the_documented_solutions(case_number: int) -> None:
    ansatz = draft4_circular_ansatz()
    _, r, varphi = ansatz.chart.coordinates
    ell, p, mass = Scalar("ell"), Scalar("p"), Scalar("lambda")
    if case_number == 0:
        expression = "R + 2/ell**2"
        parameters = ("ell", "lambda")
        f_input, phi_input = r**2 / ell**2 - mass, Number(0)
    elif case_number == 1:
        alpha1, r0 = Scalar("alpha1"), Scalar("r0")
        expression = "R + 2/ell**2 - alpha1*X"
        parameters = ("ell", "alpha1", "p", "r0", "lambda")
        f_input = r**2 / ell**2 - mass - alpha1*p**2*Function("log", (r/r0,))
        phi_input = p*varphi
    else:
        beta0 = Scalar("beta0")
        expression = "R + 2/ell**2 + ell**2*beta0*(3*RicciUU - X*R)"
        parameters = ("ell", "beta0", "p", "lambda")
        f_input = (r**2 / ell**2 - mass) / (1 + beta0*p**2*ell**2/r**2)
        phi_input = p*varphi
    model = LagrangianSourceSpec(
        name=f"draft4_case_{case_number}_integration",
        expression=expression,
        dimension=DimensionSpec(3),
        parameters=tuple(ParameterSpec(name) for name in parameters),
    ).compile()
    run = TensorEngine(
        options=EngineOptions(include_noether=False, include_export=False)
    ).run(
        model,
        ansatz=ansatz,
        specialization=AnsatzSpecialization(
            metric_functions={"f": f_input},
            scalar_field=phi_input,
            name=f"draft4_case_{case_number}_integration_specialized",
        ),
    )
    assert run.specialized is not None
    assert all(item.status.value == "completed" for item in run.specialized.quantities)
    assert run.specialized.metric_euler.components.values == ()
    assert run.specialized.scalar_euler.components.values == ()
