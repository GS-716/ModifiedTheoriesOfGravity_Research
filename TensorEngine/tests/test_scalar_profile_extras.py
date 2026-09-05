from dataclasses import replace
import json

import pytest
import sympy as sp

from tensor_engine import (
    DimensionSpec, EngineOptions, Function, FunctionSpec, LagrangianSourceSpec, ParameterSpec,
    RunExporter, RunPackage, Scalar, StructuralTensorBackend, TensorEngine,
    build_presentation, draft4_circular_ansatz, latex_report,
    spatially_flat_flrw_ansatz,
)
from tensor_engine.components import (
    ComponentEvaluation, SympyComponentBackend, ir_scalar_to_sympy,
    specialize_scalar_components, sympy_scalar_to_ir,
)
from tensor_engine.derived import ProjectionStatus
from tensor_engine.presentation import independent_curvature_components


@pytest.mark.parametrize("coordinate", ["r", "varphi"])
def test_profile_substitution_includes_mixed_and_higher_derivatives(coordinate):
    r, angle = sp.symbols("r varphi")
    phi = sp.Function("Phi")(r, angle)
    kinetic = sp.diff(phi, r)**2 + sp.diff(phi, angle)**2
    expression = (phi + sp.diff(phi, r, 2) + sp.diff(phi, angle, 2)
                  + sp.diff(phi, r, angle) + sp.Function("K")(phi, kinetic))
    components = ComponentEvaluation((), 3, (((), sympy_scalar_to_ir(expression)),))
    reduced = specialize_scalar_components(
        components, Function("Phi", (Scalar("r"), Scalar("varphi"))),
        Function("Phi", (Scalar(coordinate),)),
    )
    x = sp.Symbol(coordinate)
    restricted = sp.Function("Phi")(x)
    expected = (restricted + sp.diff(restricted, x, 2)
                + sp.Function("K")(restricted, sp.diff(restricted, x)**2))
    assert sp.simplify(ir_scalar_to_sympy(reduced.scalar) - expected) == 0


@pytest.mark.parametrize("expression", [
    "R + 2/ell**2 + ell**2*beta0*(3*RicciUU-X*R)",
    "F(phi)*R + V(phi)",
])
def test_profile_L_and_P_match_direct_projection(expression):
    model = LagrangianSourceSpec(
        "profile_equivalence", expression, dimension=DimensionSpec(3),
        parameters=(ParameterSpec("ell"), ParameterSpec("beta0")),
        functions=(FunctionSpec("F", 1), FunctionSpec("V", 1)) if "F(phi)" in expression else (),
    ).compile()
    ansatz = draft4_circular_ansatz()
    backend = StructuralTensorBackend.from_model(model)
    expressions = (model.lagrangian, backend.derive_momenta(model.lagrangian).curvature)
    generic = SympyComponentBackend.from_model(model, ansatz)
    originals = tuple(generic.evaluate_sympy(expr).to_ir() for expr in expressions)
    for coordinate in ansatz.scalar_field.arguments:
        field = Function("Phi", (coordinate,))
        direct = SympyComponentBackend.from_model(model, ansatz.specialize_scalar(field))
        for expr, original in zip(expressions, originals):
            reduced = specialize_scalar_components(original, ansatz.scalar_field, field)
            expected = direct.evaluate_sympy(expr).to_ir()
            assert reduced.free_indices == expected.free_indices
            positions = dict(reduced.values).keys() | dict(expected.values).keys()
            for position in positions:
                assert sp.simplify(ir_scalar_to_sympy(reduced.component(*position))
                                   - ir_scalar_to_sympy(expected.component(*position))) == 0


@pytest.fixture(scope="module")
def package():
    model = LagrangianSourceSpec(
        "scalar_profile_report", "R-alpha*X", dimension=DimensionSpec(3),
        parameters=(ParameterSpec("alpha"),),
    ).compile()
    return TensorEngine(options=EngineOptions(include_noether=False, include_export=False)).run(
        model, ansatz=draft4_circular_ansatz(),
    ).package


def test_extras_preserve_run_and_existing_report(package, tmp_path):
    before = json.dumps(package.to_data(), sort_keys=True)
    view = build_presentation(package)
    assert len(view.scalar_profiles) == 2
    assert all(tuple(key for key, _ in profile.quantities)
               == ("lagrangian", "curvature_momentum") for profile in view.scalar_profiles)
    for profile in view.scalar_profiles:
        components, verified = independent_curvature_components(dict(profile.quantities)["curvature_momentum"])
        assert verified and len(components) == 3
    tex = latex_report(package, presentation=view)
    old = latex_report(package, presentation=replace(view, scalar_profiles=()))
    start = tex.index(r"\par\medskip Extras:")
    end = tex.index(r"\bigskip\noindent\textbf{Política de lectura.}", start)
    assert tex[:start] + tex[end:] == old
    assert tex.count(r"\subsection*{Extra:") == 2
    assert tex.count(r"\section*{") == old.count(r"\section*{")
    assert before == json.dumps(package.to_data(), sort_keys=True)
    assert RunPackage.from_data(json.loads(json.dumps(package.to_data()))) == package
    bundle = RunExporter(tmp_path, compile_pdf=False).export(package)
    presentation = json.loads((bundle.output_directory / "presentation.json").read_text(encoding="utf-8"))
    assert len(presentation["scalar_profiles"]) == 2
    assert json.loads((bundle.output_directory / "results.json").read_text(encoding="utf-8")) == json.loads(before)
    assert bundle.manifest.verify_files(bundle.output_directory) == ()


def test_missing_base_projection_does_not_fail_report(package):
    missing = replace(package.projected.lagrangian, status=ProjectionStatus.SYMBOLIC,
                      components=None, reason="Límite del backend de prueba.")
    projected = replace(package.projected, quantities=tuple(
        missing if item.key == "lagrangian" else item for item in package.projected.quantities
    ))
    changed = replace(package, projected=projected)
    view = build_presentation(changed)
    assert all(dict(profile.quantities)["lagrangian"].status == "symbolic"
               for profile in view.scalar_profiles)
    assert "Límite del backend de prueba." in latex_report(changed, presentation=view)


def test_unrelated_ansatz_has_no_profile_extras():
    model = LagrangianSourceSpec("flrw_no_extras", "R", dimension=DimensionSpec(4)).compile()
    run = TensorEngine(options=EngineOptions(include_noether=False, include_export=False)).run(
        model, ansatz=spatially_flat_flrw_ansatz(),
    )
    assert build_presentation(run.package).scalar_profiles == ()
