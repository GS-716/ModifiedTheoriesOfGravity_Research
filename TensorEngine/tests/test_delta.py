import json

import pytest
import sympy as sp

from tensor_engine import (
    Add, CovariantDerivative, DeltaContractionAudit, GeometrySymbols, Index,
    ModelBuilder, Number, Scalar, StructuralTensorBackend, Tensor, Variance,
    contract_deltas, delta_count, canonicalize_dummy_indices, rename_free_indices,
    CoordinateChart, GeometryAnsatz, SympyComponentBackend, ComponentTensor,
    DisplayPolicy, ModelSpec, PresentationBuilder,
)
from tensor_engine.ir import Mul, Function, Power, walk


def up(name, space="M"):
    return Index(name, Variance.UP, space)


def down(name, space="M"):
    return Index(name, Variance.DOWN, space)


def delta(a, b, space="M"):
    return Tensor("delta", (up(a, space), down(b, space)))


def vector(name, index):
    return Tensor(name, (index,))


def D(index, expr):
    return CovariantDerivative(down(index), expr)


def canonical(expr):
    return StructuralTensorBackend(dimension=3).canonicalize(expr)


@pytest.mark.parametrize("reverse", [False, True])
def test_vector_and_rank_two_with_either_delta_slot_order(reverse):
    identity = delta("a", "b")
    if reverse:
        identity = Tensor("delta", tuple(reversed(identity.indices)))
    assert canonical(identity*vector("V", up("b"))) == vector("V", up("a"))
    assert canonical(identity*Tensor("T", (up("b"), down("c")))) == Tensor("T", (up("a"), down("c")))
    assert canonical(identity*vector("W", down("a"))) == vector("W", down("b"))


def test_delta_on_derivative_slot_does_not_raise_or_move_derivative():
    result = canonical(delta("a", "b")*D("a", vector("V", up("c"))))
    assert result == D("b", vector("V", up("c")))


def test_second_derivative_internal_divergence_and_order():
    expr = delta("a", "b")*D("c", D("a", vector("V", up("b"))))
    result = canonical(expr)
    expected = canonicalize_dummy_indices(D("c", D("b", vector("V", up("b")))))
    assert result == expected
    assert result.index == down("c")
    assert result.free_indices == (down("c"),)
    assert result != canonical(D("b", D("c", vector("V", up("b")))))


def test_derivative_operand_index_is_substituted_without_commuting():
    expr = delta("a", "b")*D("e", D("f", vector("V", up("b"))))
    assert canonical(expr) == D("e", D("f", vector("V", up("a"))))


def test_chain_trace_and_free_identity():
    assert canonical(delta("a", "b")*delta("b", "c")) == delta("a", "c")
    assert canonical(delta("a", "b")*delta("b", "c")*vector("V", up("c"))) == vector("V", up("a"))
    assert canonical(delta("a", "b")*delta("b", "a")) == Number(3)
    kept = contract_deltas(delta("a", "b"), dimension=3)
    assert kept.expression == delta("a", "b")
    assert kept.audit.status == "symbolic"
    assert kept.audit.events[0].action == "retained"
    assert "ámbito" in kept.audit.events[0].reason


def test_inverse_metrics_yield_identity_then_act_on_derivative_block():
    backend = StructuralTensorBackend(dimension=3)
    metric = Tensor("g", (up("a"), up("c")))*Tensor("g", (down("c"), down("b")))
    assert backend.simplify(metric) == delta("a", "b")
    result = backend.simplify(metric*D("e", vector("V", up("b"))))
    assert result == D("e", vector("V", up("a")))
    assert any(e.action == "substitute" for a in backend.delta_contractions for e in a.events)


def test_scope_capture_target_named_like_canonical_dummy():
    inner = vector("U", up("d0"))*vector("W", down("d0"))
    expr = delta("d0", "b")*(inner*vector("V", up("b")))
    result = contract_deltas(expr).expression
    assert result.free_indices == (up("d0"),)
    vectors = [n for n in walk(result) if isinstance(n, Tensor)]
    assert next(n for n in vectors if n.name == "V").indices == (up("d0"),)
    assert next(n for n in vectors if n.name == "U").indices[0].name != "d0"


def test_rename_free_indices_reserves_targets_in_nested_scopes():
    traced = Tensor("T", (up("x"), down("x")))
    expr = vector("V", up("a"))*Function("F", (traced,))
    result = rename_free_indices(expr, {("M", "a"): "d0"})
    assert result.free_indices == (up("d0"),)
    tensor = next(n for n in walk(result) if isinstance(n, Tensor) and n.name == "T")
    assert tensor.indices[0].name != "d0"


def test_sum_factor_and_nested_scalar_products_keep_independent_dummies():
    trace = Tensor("T", (up("b"), down("b")))
    block = Add((trace*vector("V", up("b")), vector("W", up("b"))))
    expr = delta("a", "b")*block
    result = canonical(expr)
    assert result.free_indices == (up("a"),)
    assert delta_count(result) == 0
    assert sum(n.name == "T" for n in walk(result) if isinstance(n, Tensor)) == 1


def test_block_with_both_contracted_slots_can_form_a_trace():
    expr = delta("a", "b")*Tensor("T", (up("b"), down("a")))
    result = canonical(expr)
    assert result.is_scalar
    assert isinstance(result, Tensor) and result.indices[0].name == result.indices[1].name


def test_custom_space_and_space_specific_dimensions():
    expr = delta("a", "b", "N")*vector("V", up("b", "N"))
    assert contract_deltas(expr).expression == vector("V", up("a", "N"))
    trace = delta("a", "a", "N")
    unknown = contract_deltas(trace, dimension=3)
    assert delta_count(unknown.expression) == 1
    assert "dimensión" in unknown.audit.events[-1].reason
    assert contract_deltas(trace, dimensions={"N": 7}).expression == Number(7)
    assert contract_deltas(trace, index_space="N", dimension="n").expression == Scalar("n")


def test_mixed_riemann_symmetries_move_indices_with_their_variance():
    from tensor_engine import TensorDeclaration, TensorSymmetry
    backend = StructuralTensorBackend((
        TensorDeclaration("Riemann", (Variance.DOWN,)*4, TensorSymmetry.RIEMANN),
    ))
    a, b, c, d = up("a"), down("b"), down("c"), down("d")
    left = Tensor("Riemann", (a, b, c, d))
    right = Tensor("Riemann", (b, a, c, d))
    assert backend.canonicalize(left + right) == Number(0)
    assert set(backend.canonicalize(left).free_indices) == {a, b, c, d}
    # Ricci symmetry is a consequence of the same pair symmetries plus dummy
    # renaming, not a special rule for a model or the metric momentum.
    ricci = Tensor("Riemann", (down("a"), up("i"), down("b"), down("i")))
    swapped = rename_free_indices(ricci, {("M", "a"): "b", ("M", "b"): "a"})
    assert backend.canonicalize(ricci - swapped) == Number(0)


def test_incompatible_spaces_and_unmixed_variances_remain_explicit():
    cases = [Tensor("delta", (up("a", "N"), down("b", "M"))),
             Tensor("delta", (up("a"), up("b")))]
    for identity in cases:
        result = contract_deltas(identity*vector("V", up("c")))
        assert delta_count(result.expression) == 1
        assert result.audit.status == "symbolic"
        assert result.audit.events[-1].reason
    expr = delta("a", "b")*vector("V", up("b", "N"))
    assert contract_deltas(expr).expression == expr


def test_partial_result_reports_remaining_identity():
    expr = delta("a", "b")*vector("V", up("b"))*delta("c", "d")
    result = contract_deltas(expr)
    assert result.audit.status == "partial"
    assert result.audit.deltas_before == 2 and result.audit.deltas_after == 1
    assert {e.action for e in result.audit.events} == {"substitute", "retained"}


def test_malformed_ir_is_preserved_not_guessed():
    ambiguous = Mul((delta("a", "b"), vector("V", up("b")), vector("W", up("b"))))
    result = contract_deltas(ambiguous)
    assert result.expression is ambiguous
    assert result.audit.status == "symbolic"
    assert "indeterminada" in result.audit.events[0].reason


def test_zero_and_scalar_function_arguments():
    assert contract_deltas(Number(0)).expression == Number(0)
    expr = Function("F", (delta("a", "a"),))
    assert contract_deltas(expr, dimension=3).expression == Function("F", (Number(3),))
    assert contract_deltas(Number(0)*delta("a", "b")).expression == Number(0)


def test_audit_has_exact_substitution_hashes_and_roundtrip():
    expr = delta("a", "b")*vector("V", up("b"))
    result = contract_deltas(expr)
    event = result.audit.events[0]
    assert event.source.variance is event.replacement.variance is Variance.UP
    assert event.replacement == up("a")
    assert result.audit.input_sha256 != result.audit.output_sha256
    rebuilt = DeltaContractionAudit.from_data(json.loads(json.dumps(result.audit.to_data())))
    assert rebuilt == result.audit
    assert contract_deltas(result.expression).expression == result.expression


def component_backend(space="M"):
    t, x = Scalar("t"), Scalar("x")
    ansatz = GeometryAnsatz("user_delta", CoordinateChart("tx", (t, x)),
                           ((Number(-1), Number(0)), (Number(0), Number(1))), scalar_field=t*x)
    return SympyComponentBackend(ansatz, GeometrySymbols(index_space=space))


def test_components_free_identity_trace_and_covariant_derivative():
    backend = component_backend()
    identity = backend.evaluate(delta("a", "b"))
    assert identity.component(0, 0) == Number(1) and identity.component(0, 1) == Number(0)
    assert backend.evaluate(delta("a", "a")).scalar == Number(2)
    assert backend.evaluate(D("c", delta("a", "b"))).values == ()
    expr = delta("a", "b")*D("c", D("a", Tensor("u", (up("b"),))))
    assert backend.evaluate(expr).to_data() == backend.evaluate(canonical(expr)).to_data()


def test_components_intrinsic_trace_reuses_product_summation():
    backend = component_backend()
    backend.tensors["T"] = ComponentTensor.from_mapping((up("x"), down("y")), 2,
                                                        {(0, 0): sp.Integer(2), (1, 1): sp.Integer(5)})
    assert backend.evaluate(Tensor("T", (up("a"), down("a")))).scalar == Number(7)
    expr = delta("a", "b")*Tensor("T", (up("b"), down("a")))
    assert backend.evaluate(expr).scalar == backend.evaluate(canonical(expr)).scalar


def test_curved_user_ansatz_nonzero_derivatives_agree_before_and_after_contraction():
    from tensor_engine.components import ir_scalar_to_sympy
    t, x = Scalar("t"), Scalar("x")
    ansatz = GeometryAnsatz(
        "user_curved_delta", CoordinateChart("tx", (t, x)),
        ((Number(-1), Number(0)), (Number(0), Number(1) + t**2)),
        scalar_field=t**3*x**2,
    )
    backend = SympyComponentBackend(ansatz)
    structural = StructuralTensorBackend.from_model(ModelSpec("curved_check", ModelBuilder().ricci_scalar()))
    expressions = (
        delta("a", "b")*D("a", Tensor("u", (up("c"),))),
        delta("a", "b")*D("c", D("a", Tensor("u", (up("b"),)))),
        Tensor("Riemann", (down("a"), up("i"), down("b"), down("i"))),
        Tensor("Riemann", (up("i"), up("j"), down("i"), down("j"))),
    )
    for expr in expressions:
        normalized = structural.canonicalize(expr)
        assert structural.canonicalize(normalized) == normalized
        before, after = backend.evaluate(expr), backend.evaluate(normalized)
        assert set(before.free_indices) == set(after.free_indices)
        assert before.values and after.values  # not a vacuous zero comparison
        assert len(before.values) == len(after.values)
        for indices, value in before.values:
            reordered = tuple(indices[before.free_indices.index(i)] for i in after.free_indices)
            assert sp.simplify(ir_scalar_to_sympy(value) -
                               ir_scalar_to_sympy(after.component(*reordered))) == 0


def test_component_delta_rejects_foreign_spaces_and_accepts_user_space():
    from tensor_engine import BackendExecutionError
    with pytest.raises(BackendExecutionError, match="espacio"):
        component_backend().evaluate(delta("a", "b", "N"))
    assert component_backend("N").evaluate(delta("a", "a", "N")).scalar == Number(2)


def test_display_never_contracts_even_with_aggressive_policy():
    model = ModelSpec("delta_display", ModelBuilder().ricci_scalar())
    expr = delta("a", "b")*vector("V", up("b"))
    for aggressive in (False, True):
        result = PresentationBuilder(model, DisplayPolicy(aggressive=aggressive)).expression(expr)
        assert delta_count(result.presentation) == 1
        assert delta_count(result.canonical) == 1
        canonical_result = canonical(expr)
        assert PresentationBuilder(model).expression(canonical_result).presentation == canonical_result
