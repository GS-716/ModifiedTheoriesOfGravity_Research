from dataclasses import replace
import json
import os

import pytest
import sympy as sp

from tensor_engine import (
    AnsatzSpecialization, ComponentEvaluation, CoordinateChart, DimensionSpec,
    EngineOptions, Function, GeometryAnsatz, Index, LagrangianSourceSpec,
    Number, ParameterSpec, RunPackage, Scalar, TensorEngine, Variance,
    WolframXActBridge, draft4_circular_ansatz, ir_scalar_to_sympy,
    spatially_flat_flrw_ansatz, sympy_scalar_to_ir,
)
from field_equations_solver import FieldEquationWolframBridge
from field_equations_solver.solving import (
    FieldEquationSolution, ReducedEquation, analyze_redundancy, classify_system,
    raise_metric_equation, solveFieldEquations, SolverSearchPolicy,
)
from field_equations_solver.reporting import solution_latex


def make_run(expression="R + 2/ell**2", *, ansatz=None, name="solver_test"):
    ansatz = ansatz or draft4_circular_ansatz()
    model = LagrangianSourceSpec(name=name, expression=expression,
                                dimension=DimensionSpec(ansatz.dimension),
                                parameters=tuple(ParameterSpec(n) for n in ("ell", "alpha", "beta0"))).compile()
    return TensorEngine(options=EngineOptions(include_noether=False, include_export=False)).run(model, ansatz=ansatz)


@pytest.fixture(scope="module")
def eh():
    return make_run()


@pytest.fixture(scope="module")
def kinetic():
    return make_run("R - alpha*X")


@pytest.fixture(scope="module")
def case2():
    return make_run("R + 2/ell**2 + ell**2*beta0*(3*RicciUU - X*R)", name="case2_solver")


def test_raise_index_uses_full_inverse_metric():
    chart = CoordinateChart("oblique", (Scalar("x"), Scalar("y")))
    ansatz = GeometryAnsatz("oblique", chart, ((Number(-2), Number(1)), (Number(1), Number(3))))
    indices = (Index("a", Variance.DOWN), Index("b", Variance.DOWN))
    comp = ComponentEvaluation(indices, 2, (((0, 0), Number(7)), ((0, 1), Number(5)),
                                             ((1, 0), Number(5)), ((1, 1), Number(11))))
    expected = sp.Matrix([[-2, 1], [1, 3]]).inv() * sp.Matrix([[7, 5], [5, 11]])
    assert raise_metric_equation(comp, ansatz) == expected
    with pytest.raises(ValueError, match="covariantes"):
        raise_metric_equation(replace(comp, free_indices=(indices[0].flipped(), indices[1])), ansatz)


def test_draft4_combinations_keep_absolute_and_all_originals(eh):
    result = solveFieldEquations(eh, solve=False)
    r, ell = sp.symbols("r ell")
    f = sp.Function("f")(r)
    e = {item.key: ir_scalar_to_sympy(item.reduced) for item in result.equations}
    assert sp.simplify(e["difference_0_1"]) == 0
    assert sp.simplify(e["difference_0_2"] - (sp.diff(f, r)/r - sp.diff(f, r, 2))/2) == 0
    assert sp.simplify(e["difference_0_2"] - e["difference_1_2"]) == 0
    assert sp.simplify(e["absolute_0"] - (sp.diff(f, r)/(2*r) - 1/ell**2)) == 0
    assert len(result.original_equations) == 10  # all 3x3 entries plus scalar, including zero entries
    assert sum(item.role == "off_diagonal" for item in result.equations) == 6


def test_rational_redundancy_certificates_and_parameter_branches():
    x, y, a = sp.symbols("x y a")
    def eq(k, e):
        ir = sympy_scalar_to_ir(e)
        return ReducedEquation(k, k, ir, ir, "test")
    equations = analyze_redundancy(tuple(eq(str(i), e) for i, e in enumerate([x-y, y, x, 0, a*x])))
    assert [e.status for e in equations] == ["linearly_independent_over_Q", "linearly_independent_over_Q",
                                           "redundant", "zero", "linearly_independent_over_Q"]
    for e in equations:
        if e.dependencies:
            reference = {v.key: ir_scalar_to_sympy(v.reduced) for v in equations}
            assert sp.expand(ir_scalar_to_sympy(e.reduced) - sum(ir_scalar_to_sympy(c)*reference[k] for k, c in e.dependencies)) == 0


def test_ode_pde_dae_mixed_classification():
    r, v, a = sp.symbols("r v a")
    f, phi = sp.Function("f")(r), sp.Function("Phi")(r, v)
    assert classify_system([sp.diff(f, r, 2)], [f], [r])["kind"] == "ODE"
    assert classify_system([sp.diff(phi, r, 2)+sp.diff(phi, v, 2)], [phi], [r, v])["kind"] == "PDE"
    assert classify_system([sp.diff(f, r), f**2-a], [f, a], [r])["kind"] == "DAE"
    assert classify_system([sp.diff(phi, r)+sp.diff(f, r)], [phi, f], [r, v])["kind"] == "mixed"
    assert classify_system([a**2-1], [a], [r])["kind"] == "algebraic"


def test_phi_radial_and_exact_q_varphi_reuse_projection(kinetic):
    original = json.dumps(kinetic.to_data(), sort_keys=True)
    r, angle, q = Scalar("r"), Scalar("varphi"), Scalar("q")
    radial = solveFieldEquations(kinetic, specialization=AnsatzSpecialization(scalar_field=Function("Phi", (r,))), solve=False)
    angular = solveFieldEquations(kinetic, specialization=AnsatzSpecialization(scalar_field=q*angle), solve=False)
    assert radial.classification["kind"] in ("ODE", "DAE")
    assert not radial.classification["contains_pde"]
    assert angular.ansatz.scalar_field == q*angle
    assert "phi0" not in json.dumps(angular.to_data())
    assert not any("Phi(" in str(ir_scalar_to_sympy(e.reduced)) for e in angular.equations)
    assert json.dumps(kinetic.to_data(), sort_keys=True) == original
    assert solveFieldEquations(kinetic, solve=False).classification["contains_pde"]
    with pytest.raises(Exception, match="tau"):
        solveFieldEquations(kinetic, specialization=AnsatzSpecialization(scalar_field=Function("Phi", (Scalar("tau"),))), solve=False)


def test_solution_verification_rejects_trace_only_and_degenerate_solutions(eh):
    r, ell, c = Scalar("r"), Scalar("ell"), Scalar("constant")
    f = Function("f", (r,))
    result = solveFieldEquations(eh, solve=False)
    good = result.verify({f: r**2/ell**2+c})
    assert good.status == "verified_on_domain"
    assert len(good.residuals) == 10 and all(v == Number(0) for _, v in good.residuals)
    assert result.verify({f: Number(1)}).status != "verified_on_domain"  # differences alone vanish!
    assert result.verify({f: Number(0)}).status == "rejected"
    assert result.verify({f: r**2/ell**2+c, ell: Number(0)}).status == "rejected"
    changed = replace(result.original_equations[1], original=Number(1))
    adversarial = replace(result, original_equations=(result.original_equations[0], changed, *result.original_equations[2:]))
    assert adversarial.verify({f: r**2/ell**2+c}).status == "rejected"
    assigned = result.verify({f: r**2/ell**2+c, ell: Number(2)})
    assert assigned.status == "verified_on_domain"
    with pytest.raises(ValueError, match="autorreferentes"):
        result.verify({f: f+1})


def test_metric_reparameterization_is_applied_exactly_once(eh):
    r = Scalar("r")
    f = Function("f", (r,))
    result = solveFieldEquations(eh, specialization=AnsatzSpecialization(metric_functions={"f": f+1}), solve=False)
    assert result.ansatz.metric_covariant[0][0] == -(f+1)
    # Inspect a covariant equation, whose f prefactor detects a double shift.
    from field_equations_solver.solving import _substitute
    fr, rr = sp.Function("f")(sp.Symbol("r")), sp.Symbol("r")
    assert _substitute(fr**2, {fr: fr+1}) == (fr+1)**2


def test_case2_known_family_verified_in_all_original_components(case2):
    r, ell, beta, q, c = (Scalar(n) for n in ("r", "ell", "beta0", "q", "constant"))
    result = solveFieldEquations(case2, specialization=AnsatzSpecialization(scalar_field=q*Scalar("varphi")), solve=False)
    family = (r**2/ell**2+c)/(1+beta*q**2*ell**2/r**2)
    verified = result.verify({Function("f", (r,)): family})
    assert verified.status == "verified_on_domain", verified.unresolved
    assert all(value == Number(0) for _, value in verified.residuals)
    assert result.classification["kind"] == "ODE"


def test_solver_json_and_presentation_do_not_change_source(eh, tmp_path):
    before = json.dumps(eh.to_data(), sort_keys=True)
    result = solveFieldEquations(eh, solve=False)
    restored = FieldEquationSolution.from_data(json.loads(json.dumps(result.to_data())))
    assert restored == result
    assert RunPackage.from_data(restored.source_results) == eh.package
    directory = result.export(tmp_path, compile_pdf=False)
    assert FieldEquationSolution.from_data(json.loads((directory/"results.json").read_text(encoding="utf-8"))) == result
    tex, _ = solution_latex(result)
    assert "Ecuaciones combinadas" in tex
    assert "Cantidades geométricas" not in tex
    assert json.dumps(eh.to_data(), sort_keys=True) == before
    assert (directory/"manifest.json").is_file()
    from tensor_engine import DisplayPolicy
    canonical = (directory/"results.json").read_bytes()
    digest = json.loads((directory/"manifest.json").read_text())["result_sha256"]
    result.export(tmp_path, compile_pdf=False, display_policy=DisplayPolicy(enabled=False))
    assert (directory/"results.json").read_bytes() == canonical
    assert json.loads((directory/"manifest.json").read_text())["result_sha256"] == digest


def test_missing_components_is_nonfatal(eh):
    view = eh.projected
    from tensor_engine.derived import ProjectionStatus
    bad = replace(view.metric_euler, components=None, status=ProjectionStatus.UNAVAILABLE, reason="test backend limit")
    view = replace(view, quantities=tuple(bad if q.key==bad.key else q for q in view.quantities))
    package = replace(eh.package, projected=view)
    result = solveFieldEquations(package, solve=False)
    assert result.status == "unavailable" and "test backend limit" in result.diagnostics[0]
    assert any(e.key == "scalar" for e in result.equations)
    assert result.verify({}).status == "undetermined"


def test_domain_signs_and_unknown_assumptions_are_not_ignored(eh):
    r, ell = Scalar("r"), Scalar("ell")
    model = replace(eh.package.model, parameters=tuple(
        replace(p, assumptions=("positive",)) if p.name == "ell" else p for p in eh.package.model.parameters))
    run = replace(eh.package, model=model)
    result = solveFieldEquations(run, solve=False)
    assert result.verify({Function("f", (r,)): r**2/ell**2+1, ell: Number(-1)}).status == "rejected"
    domain = result.verify({Function("f", (r,)): r**2/ell**2+1})
    assert domain.status == "verified_on_domain"
    assert "r > 0" in domain.domain_conditions and "ell > 0" in domain.domain_conditions
    unclear = replace(run, model=replace(model, assumptions=("unsupported_assumption",)))
    assert solveFieldEquations(unclear, solve=False).verify({Function("f", (r,)): r**2/ell**2+1}).status == "undetermined"


def test_already_specialized_selection_is_explicit(eh):
    from tensor_engine.derived import SpecializedTensorResults
    spec = AnsatzSpecialization(scalar_field=Scalar("q")*Scalar("varphi"))
    ansatz = spec.apply(eh.projected.ansatz_geometry)
    view = SpecializedTensorResults(eh.projected.ansatz_name, spec, ansatz,
                                    tuple(replace(q, ansatz_name=ansatz.name) for q in eh.projected.quantities))
    package = replace(eh.package, specialized=view)
    assert solveFieldEquations(package, solve=False).ansatz == eh.projected.ansatz_geometry
    assert solveFieldEquations(package, use_specialized=True, solve=False).ansatz == ansatz
    with pytest.raises(ValueError, match="no ambos"):
        solveFieldEquations(package, specialization=spec, use_specialized=True)


def test_unavailable_wolfram_and_unverified_candidates_never_pass(eh):
    class Unavailable:
        available = False
    result = solveFieldEquations(eh, wolfram_bridge=Unavailable())
    assert result.backend["status"] == "unavailable" and result.equations
    assert all(solution.origin != "Wolfram" for solution in result.solutions
               if solution.status == "verified_on_domain")
    class Unverified:
        available = True
        timeout_seconds = 2
        def build_request(self, operation, options):
            assert operation == "solve_field_equations"
            assert len(options["original_equations"]) == 10
            return options
        def execute(self, request):
            r, ell = Scalar("r"), Scalar("ell")
            return {"candidates": [{"rules": [[Function("f", (r,)).to_data(), (r**2/ell**2+1).to_data()]],
                                    "verification": "undetermined"}]}
    result = solveFieldEquations(eh, wolfram_bridge=Unverified())
    wolfram = [solution for solution in result.solutions if solution.origin == "Wolfram"]
    assert wolfram and all(solution.status == "undetermined" for solution in wolfram)


def test_notebook_solver_cell_is_optional():
    from pathlib import Path
    notebook = json.loads((Path(__file__).parents[2]/"ResearchWorkflow/01_modified_gravity_workflow.ipynb").read_text(encoding="utf-8"))
    cell = next(c for c in notebook["cells"] if c.get("id") == "optional-field-equation-solver")
    namespace = {}
    exec("".join(cell["source"]), namespace)
    assert callable(namespace["solveFieldEq"])
    assert namespace["solveFieldEq"](False, object()) is None


@pytest.mark.skipif(os.environ.get("TENSOR_ENGINE_RUN_WOLFRAM_TESTS") != "1", reason="Wolfram opt-in")
def test_live_solve_reduce_eliminate_and_unsupported_node_diagnostic():
    a, b = Scalar("a"), Scalar("b")
    bridge = FieldEquationWolframBridge(timeout_seconds=60)
    equations = [(a+b-3).to_data(), (a-b-1).to_data()]
    response = bridge.execute(bridge.build_request("solve_field_equations", options={
        "equations": equations, "original_equations": equations,
        "unknowns": [a.to_data(), b.to_data()], "eliminate": [b.to_data()],
        "nonzero": [], "time_limit": 5,
    }))
    assert response["status"] == "evaluated"
    operations = {item["operation"]: item for item in response["operations"]}
    assert {"Solve", "Reduce", "Eliminate", "DSolve"}.issubset(operations)
    assert all(o["status"] == "evaluated" for o in response["operations"] if o["operation"] in ("Solve", "Reduce", "Eliminate"))
    assert all(candidate["origin"] in {"Wolfram Solve", "Wolfram Reduce"}
               for candidate in response["candidates"])
    assert response["candidates"][0]["verification"] == "verified"
    from tensor_engine import expr_from_data
    rules = {expr_from_data(k): expr_from_data(v) for k, v in response["candidates"][0]["rules"]}
    assert rules == {a: Number(2), b: Number(1)}
    malformed = {"type": "tensor", "name": "not_a_scalar"}
    failure = bridge.execute(bridge.build_request("solve_field_equations", options={"equations": [malformed]}))
    assert failure["status"] == "unavailable"
    assert failure["failed_node"]["node"] == malformed


def test_flrw_remains_an_arbitrary_ansatz():
    run = make_run("R", ansatz=spatially_flat_flrw_ansatz())
    result = solveFieldEquations(run, solve=False)
    assert len(result.original_equations) == 17
    assert len([e for e in result.equations if e.role == "diagonal_difference"]) == 6
    assert result.classification["kind"] == "ODE"


@pytest.mark.skipif(os.environ.get("TENSOR_ENGINE_RUN_WOLFRAM_TESTS") != "1", reason="Wolfram opt-in")
@pytest.mark.parametrize("fixture_name", ["eh", "case2"])
def test_live_wolfram_solver_and_original_xact_validation(request, fixture_name, tmp_path):
    run = request.getfixturevalue(fixture_name)
    solver_bridge = FieldEquationWolframBridge(timeout_seconds=180)
    xact_bridge = WolframXActBridge(timeout_seconds=180)
    result = solveFieldEquations(run, specialization=AnsatzSpecialization(scalar_field=Scalar("q")*Scalar("varphi")), wolfram_bridge=solver_bridge)
    (tmp_path/"solver-response.json").write_text(json.dumps(result.to_data(), ensure_ascii=False, indent=2), encoding="utf-8")
    assert result.backend["status"] == "evaluated", result.diagnostics
    assert any(s.status == "verified_on_domain" for s in result.solutions), ([(s.status, s.unresolved) for s in result.solutions], result.diagnostics, tmp_path)
    assert all(len(s.residuals) == 10 for s in result.solutions)
    package = run.package
    evidence = xact_bridge.execute(xact_bridge.build_model_validation_request(package.model, package.momenta, package.euler,
                              normalized_lagrangian=package.lagrangian))
    assert evidence["summary"]["failed"] == 0, evidence
    assert evidence["summary"]["passed"] > 0
    assert all(c.get("residual") is None for c in evidence["checks"] if c["status"] == "passed")
