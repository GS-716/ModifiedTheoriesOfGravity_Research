from dataclasses import replace
import json

import pytest
import sympy as sp

from tensor_engine import (
    DisplayPolicy, PresentationBuilder, ModelBuilder, ModelSpec, ParameterSpec,
    FunctionSpec, DimensionSpec, LagrangianSourceSpec, EngineOptions, TensorEngine,
    build_presentation, add, mul,
    Scalar, Number, Tensor, Index, Variance, Function, FunctionDerivative,
    CovariantDerivative, StructuralTensorBackend,
)
from tensor_engine.ir import Add, Mul, Power, walk
from tensor_engine.components import ir_scalar_to_sympy
from tensor_engine.exporting import display_expr_to_latex, expr_to_latex


def builder(assumptions=(), **policy):
    model = ModelSpec("display", ModelBuilder().ricci_scalar(),
                     parameters=(ParameterSpec("ell"), ParameterSpec("beta0")),
                     assumptions=assumptions)
    return PresentationBuilder(model, DisplayPolicy(**policy))


def test_default_policy_is_conservative_and_validated():
    assert DisplayPolicy().factor and not DisplayPolicy().aggressive
    for kw in ({"max_nodes": 0}, {"max_nodes": True}, {"factor": "yes"}):
        with pytest.raises(ValueError):
            DisplayPolicy(**kw)


@pytest.mark.parametrize("base", [Scalar("ell"), Scalar("r"), Function("a", (Scalar("t"),))])
@pytest.mark.parametrize("aggressive", [False, True])
def test_poles_never_cancel_without_explicit_assumptions(base, aggressive):
    for expr in (base * base**-1, base**-1 - base**-1):
        result = builder(aggressive=aggressive).expression(expr)
        assert result.canonical is expr
        assert result.presentation != Number(0) and result.presentation != Number(1)
        assert any(isinstance(n, Power) and n.exponent == Number(-1) for n in walk(result.presentation))
        assert result.assumptions_used == ()


@pytest.mark.parametrize("base,assumption", [
    (Scalar("ell"), "ell != 0"), (Scalar("r"), "r>0"),
    (Function("a", (Scalar("t"),)), "a(t)>0"),
])
def test_explicit_nonzero_hypotheses_allow_and_record_cancellation(base, assumption):
    result = builder((assumption,)).expression(base * base**-1)
    assert result.presentation == Number(1)
    assert assumption in result.assumptions_used
    assert result.status == "simplified"


def test_real_or_nonnegative_does_not_imply_nonzero():
    ell = Scalar("ell")
    for assumption in ("ell is real", "ell>=0", "0<=ell", "ell=1"):
        result = builder((assumption,)).expression(ell * ell**-1)
        assert result.presentation != Number(1)
        assert not result.assumptions_used


def test_parameter_assumptions_are_used_but_never_invented():
    model = replace(builder().model, parameters=(ParameterSpec("ell", ("positive",)),))
    ell = Scalar("ell")
    result = PresentationBuilder(model).expression(ell / ell)
    assert result.presentation == Number(1)
    assert result.assumptions_used == ("ell!=0",)


def test_radicals_and_symbolic_powers_are_not_rewritten():
    x = Scalar("x")
    radical = Power(x*x, Number(1, 2))
    expr = Power(radical, Number(2))
    result = builder(aggressive=True).expression(expr)
    assert any(n == radical for n in walk(result.presentation))


def test_scalar_polynomial_factoring_collection_and_exact_fractions():
    ell, beta, x, y = map(Scalar, ("ell", "beta0", "x", "y"))
    expr = ell*beta*x + ell*beta*y + Number(1, 3)*x + Number(2, 3)*x
    result = builder().expression(expr)
    assert result.status == "simplified"
    assert sp.expand(ir_scalar_to_sympy(expr) - ir_scalar_to_sympy(result.presentation)) == 0
    assert len(list(walk(result.presentation))) < len(list(walk(expr)))
    assert result.assumptions_used == ()
    assert "combine_like_terms" in result.operations


def test_fraction_together_with_explicit_domain():
    ell, r = Scalar("ell"), Scalar("r")
    expr = ell**-1 + r**-1
    result = builder(("ell!=0", "r!=0")).expression(expr)
    assert sp.cancel(ir_scalar_to_sympy(expr)-ir_scalar_to_sympy(result.presentation)) == 0
    # A longer common-denominator representation is intentionally not preferred.
    assert len(list(walk(result.presentation))) <= len(list(walk(expr)))


def test_tensor_factoring_preserves_free_indices_and_reuses_metric_rules():
    b = ModelBuilder()
    a, c = Index("a", Variance.DOWN), Index("c", Variance.DOWN)
    tensor = Tensor("T", (a, c))
    ell = Scalar("ell")
    expr = 2*ell*tensor + 3*ell*tensor
    presenter = builder()
    result = presenter.expression(expr)
    assert set(result.presentation.free_indices) == {a, c}
    assert StructuralTensorBackend.from_model(presenter.model).simplify(expr-result.presentation) == Number(0)
    contracted = b.metric("a", "b")*Tensor("T", (a, c))
    reduced = presenter.expression(contracted)
    assert set(reduced.presentation.free_indices) == set(contracted.free_indices)
    assert any(isinstance(n, Tensor) and n.name == "g" for n in walk(reduced.presentation))


def test_tensor_like_term_cancellation_must_not_remove_poles():
    t = Tensor("T", (Index("a", Variance.DOWN),))
    ell = Scalar("ell")
    expr = ell**-1*t - ell**-1*t
    assert builder().expression(expr).presentation != Number(0)
    permitted = builder(("ell!=0",)).expression(expr)
    assert permitted.presentation == Number(0)
    assert permitted.assumptions_used == ("ell!=0",)


def test_covariant_derivatives_are_not_commuted_or_expanded():
    t = Tensor("T", (Index("a", Variance.UP),))
    x, y = Index("e", Variance.DOWN), Index("f", Variance.DOWN)
    first = CovariantDerivative(x, CovariantDerivative(y, t))
    second = CovariantDerivative(y, CovariantDerivative(x, t))
    result = builder().expression(first-second)
    assert result.presentation != Number(0)
    assert first in tuple(walk(result.presentation)) and second in tuple(walk(result.presentation))


def test_deterministic_local_cache_and_json_audit():
    expr = 2*Scalar("ell")+3*Scalar("ell")
    presenter = builder()
    first = presenter.expression(expr)
    assert presenter.expression(expr) is first
    assert builder().expression(expr).to_data() == first.to_data()
    assert json.loads(json.dumps(first.to_data())) == first.to_data()
    assert {"status", "operations", "assumptions_used", "canonical_sha256"} <= first.to_data().keys()


def test_disabled_budget_and_backend_error_are_nonfatal(monkeypatch):
    expr = ModelBuilder().ricci_scalar()
    assert builder(enabled=False).expression(expr).status == "disabled"
    assert builder(max_nodes=1).expression(expr).presentation is expr
    def unavailable(*args):
        raise RuntimeError("test backend unavailable")
    monkeypatch.setattr("tensor_engine.presentation.canonicalize_dummy_indices", unavailable)
    result = builder().expression(expr)
    assert result.status == "fallback" and result.presentation is expr
    assert "test backend unavailable" in result.notes[0]


def test_readable_printer_keeps_poles_slots_signs_and_derivative_order():
    ell = Scalar("ell")
    assert display_expr_to_latex(-ell) == r"-\ell"
    assert display_expr_to_latex(ell*ell**-1) == r"\frac{\ell}{\ell}"
    assert display_expr_to_latex(Number(-1, 2)) == r"-\frac{1}{2}"
    mixed = Tensor("T", (Index("a", Variance.DOWN), Index("b", Variance.UP), Index("c", Variance.DOWN)))
    assert display_expr_to_latex(mixed) == r"T{}_{a}{}^{b}{}_{c}"
    derivative = FunctionDerivative("f", (2,), (Scalar("r"),))
    assert display_expr_to_latex(derivative) == r"f''\!\left(r\right)"
    assert expr_to_latex(-ell) == r"-1\,\mathrm{ell}"


def test_index_dummy_names_do_not_collide_with_free_names():
    b = ModelBuilder()
    t = Tensor("T", (Index("d0", Variance.DOWN),))
    expr = b.ricci_scalar()*t
    result = builder().expression(expr)
    assert result.status != "fallback"
    assert result.presentation.free_indices == t.free_indices


def test_large_fraction_numerator_stays_breakable_in_latex():
    terms = tuple(Scalar("coefficient_with_a_long_name_" + str(i)) for i in range(8))
    expr = Number(1, 4)*Add(terms)
    rendered = display_expr_to_latex(expr)
    assert rendered.startswith(r"\frac{1}{4}\,")
    assert r"\left(" in rendered


def test_engine_accepts_display_policy_with_a_user_defined_ansatz(tmp_path, monkeypatch):
    from tensor_engine import (
        CoordinateChart, GeometryAnsatz, DimensionSpec, LagrangianSourceSpec,
        EngineOptions, TensorEngine, RunExporter,
    )
    monkeypatch.setattr(RunExporter, "_compile_pdf", lambda *args: (None, "PDF disabled in test"))
    t, r = Scalar("t"), Scalar("r")
    ansatz = GeometryAnsatz("user_geometry", CoordinateChart("user_chart", (t, r)),
                           ((Number(-1), Number(0)), (Number(0), Number(1))),
                           scalar_field=t, assumptions=("r>0",))
    source = LagrangianSourceSpec("user_geometry_model", "R+X", dimension=DimensionSpec(2))
    run = TensorEngine(options=EngineOptions(include_noether=False)).run(
        source.compile(), ansatz=ansatz, output_root=tmp_path,
        display_policy=DisplayPolicy(enabled=False),
    )
    assert run.projected.ansatz_name == "user_geometry"
    assert all(record.status == "disabled" for _, record in run.export_bundle.presentation.expressions)
    assert {
        item.key for item in run.export_bundle.presentation.compact_decompositions
    } == {"metric_euler", "scalar_euler", "curvature_momentum"}
    assert all(
        block.projection.status == "completed"
        for item in run.export_bundle.presentation.compact_decompositions
        for block in item.blocks
    )


@pytest.mark.parametrize(
    "name,expression,parameters,functions,dimension",
    (
        ("compact_eh", "R", (), (), 4),
        (
            "compact_scalar",
            "R - alpha*X",
            (ParameterSpec("alpha"),),
            (),
            4,
        ),
        (
            "compact_case2",
            "R + 2/ell**2 + ell**2*beta0*(3*RicciUU - X*R)",
            tuple(ParameterSpec(item) for item in ("ell", "beta0", "p")),
            (),
            3,
        ),
        (
            "compact_functions",
            "F(phi)*R + K(phi, X)",
            (),
            (FunctionSpec("F", 1), FunctionSpec("K", 2)),
            4,
        ),
        (
            "compact_quadratic",
            "R + alpha*R**2",
            (ParameterSpec("alpha"),),
            (),
            4,
        ),
    ),
)
def test_compact_decompositions_are_generic_exact_and_presentation_only(
    name, expression, parameters, functions, dimension,
):
    model = LagrangianSourceSpec(
        name,
        expression,
        dimension=DimensionSpec(dimension),
        parameters=parameters,
        functions=functions,
    ).compile()
    run = TensorEngine(
        options=EngineOptions(
            include_noether=False,
            include_components=False,
            include_export=False,
        )
    ).run(model)
    before = json.dumps(run.package.to_data(), sort_keys=True)
    old_keys = tuple(key for key, _ in build_presentation(run.package).expressions)
    view = build_presentation(run.package)
    backend = StructuralTensorBackend.from_model(model)

    assert tuple(item.key for item in view.compact_decompositions) == (
        "metric_euler",
        "scalar_euler",
        "curvature_momentum",
    )
    assert tuple(key for key, _ in view.expressions) == old_keys
    expected_blocks = {
        "metric_euler": (
            "metric_momentum",
            "curvature_algebraic",
            "volume_term",
            "curvature_derivative",
        ),
        "scalar_euler": ("scalar_force", "scalar_current_divergence"),
        "curvature_momentum": ("curvature_momentum",),
    }
    for decomposition in view.compact_decompositions:
        assert decomposition.reconstruction_status == "verified"
        assert tuple(block.key for block in decomposition.blocks) == expected_blocks[
            decomposition.key
        ]
        residual = backend.simplify(
            add(
                *(block.expression.canonical for block in decomposition.blocks),
                mul(-1, decomposition.expression.canonical),
            )
        )
        assert residual == Number(0)
        for record in (
            decomposition.expression,
            *(block.expression for block in decomposition.blocks),
        ):
            assert backend.simplify(
                add(record.canonical, mul(-1, record.presentation))
            ) == Number(0)

    payload = view.to_data()
    assert payload["purpose"] == "presentation_only"
    assert len(payload["compact_decompositions"]) == 3
    assert json.dumps(run.package.to_data(), sort_keys=True) == before
    assert run.package.run_id == view.run_id
