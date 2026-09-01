from __future__ import annotations

import json

import pytest

from tensor_engine import (
    DEFAULT_INVARIANTS, DimensionSpec, FunctionSpec, GeometrySymbols,
    InvariantRegistry, InvariantSpec, LagrangianSourceSpec, ModelBuilder,
    ModelSpec, Number, ParameterSpec, SourceCompilationError,
    StructuralTensorBackend, TensorAlgebraError, model_fingerprint, mul,
)


CASE2 = "R + 2/ell**2 + ell**2*beta0*(3*RicciUU - X*R)"
PARAMETERS = tuple(ParameterSpec(name) for name in ("ell", "beta0", "p"))
GENERIC = (
    'contract(Riemann("a","b","c","d"), metric("a","c"), '
    'metric("b","e"), metric("d","f"), gradient("e"), gradient("f"))'
)


def manual_ricci_uu(b):
    return mul(
        b.metric("a", "c"), b.riemann("a", "b", "c", "d"),
        b.metric("b", "e"), b.metric("d", "f"),
        b.scalar_gradient("e"), b.scalar_gradient("f"),
    )


@pytest.mark.parametrize("alias", ("R", "X", "RicciUU", "phi"))
def test_aliases_are_existing_ir_not_new_tensor_heads(alias):
    b = ModelBuilder()
    expected = {
        "R": b.ricci_scalar(), "X": b.kinetic_scalar(),
        "RicciUU": manual_ricci_uu(b), "phi": b.phi,
    }[alias]
    model = LagrangianSourceSpec("alias", alias).compile()
    assert model.lagrangian == expected
    assert model.lagrangian.is_scalar
    provenance = json.loads(dict(model.metadata)["source_invariants"])
    assert list(provenance) == [alias]
    assert len(provenance[alias]["ir_sha256"]) == 64


def test_case2_exact_ir_matches_low_level_construction():
    b = ModelBuilder()
    ell, beta0 = b.scalar("ell"), b.scalar("beta0")
    expected = (
        b.ricci_scalar() + Number(2) / ell**2
        + ell**2 * beta0 * (
            Number(3) * manual_ricci_uu(b) - b.kinetic_scalar() * b.ricci_scalar()
        )
    )
    model = LagrangianSourceSpec("case2", CASE2, parameters=PARAMETERS).compile()
    assert model.lagrangian.to_data() == expected.to_data()
    backend = StructuralTensorBackend.from_model(model)
    assert backend.simplify(model.lagrangian - expected) == Number(0)
    # The variational path receives the expanded IR without special dispatch.
    assert backend.derive_momenta(model.lagrangian) == backend.derive_momenta(expected)


@pytest.mark.parametrize(
    "text,expected,functions,parameters",
    (
        (
            "F(phi)*R + K(phi, X)",
            lambda b: b.function("F", b.phi) * b.ricci_scalar()
            + b.function("K", b.phi, b.kinetic_scalar()),
            (FunctionSpec("F"), FunctionSpec("K", 2)), (),
        ),
        ("V(phi)", lambda b: b.function("V", b.phi), (FunctionSpec("V"),), ()),
        (
            "R + alpha*R**2",
            lambda b: b.ricci_scalar() + b.scalar("alpha") * b.ricci_scalar()**2,
            (), (ParameterSpec("alpha"),),
        ),
    ),
)
def test_functions_parameters_and_powers_are_unchanged(text, expected, functions, parameters):
    model = LagrangianSourceSpec(
        "compatible", text, functions=functions, parameters=parameters,
    ).compile()
    assert model.lagrangian == expected(ModelBuilder())


@pytest.mark.parametrize("expression", (GENERIC, GENERIC.replace('"a"', '"z"')))
def test_generic_contraction_equals_alias_after_index_canonicalization(expression):
    model = LagrangianSourceSpec("generic", expression).compile()
    backend = StructuralTensorBackend.from_model(model)
    alias = LagrangianSourceSpec("alias", "RicciUU").compile()
    assert backend.canonicalize(model.lagrangian) == backend.canonicalize(alias.lagrangian)


def test_repeated_scalar_aliases_preserve_dummy_index_scopes():
    model = LagrangianSourceSpec("scopes", "RicciUU*RicciUU + X*R").compile()
    b = ModelBuilder()
    expected = manual_ricci_uu(b) * manual_ricci_uu(b) + b.kinetic_scalar() * b.ricci_scalar()
    backend = StructuralTensorBackend.from_model(model)
    assert model.lagrangian.is_scalar
    assert backend.simplify(model.lagrangian - expected) == Number(0)


def test_custom_geometry_symbols_are_respected():
    symbols = GeometrySymbols(
        index_space="N", metric="h", curvature="C", scalar="psi", scalar_gradient="v",
    )
    b = ModelBuilder(symbols)
    model = LagrangianSourceSpec("renamed", "RicciUU + phi", symbols=symbols).compile()
    assert model.lagrangian == manual_ricci_uu(b) + b.phi
    generic = LagrangianSourceSpec("renamed_generic", GENERIC, symbols=symbols).compile()
    backend = StructuralTensorBackend.from_model(generic)
    assert backend.simplify(generic.lagrangian - b.ricci_uu()) == Number(0)


def test_extension_is_scoped_cached_and_reproducible():
    calls = []

    def xx(builder):
        calls.append(True)
        return builder.kinetic_scalar()**2

    registry = DEFAULT_INVARIANTS.with_invariant(InvariantSpec("XX", xx, "X squared", "2"))
    source = LagrangianSourceSpec("extension", "XX + XX")
    model = source.compile(registry=registry)
    assert len(calls) == 1  # expand once, not once per occurrence
    assert "XX" not in DEFAULT_INVARIANTS.aliases
    assert json.loads(dict(model.metadata)["source_invariants"])["XX"]["version"] == "2"
    restored_source = LagrangianSourceSpec.from_data(json.loads(json.dumps(source.to_data())))
    assert restored_source.compile(registry=registry) == model
    assert ModelSpec.from_data(json.loads(json.dumps(model.to_data()))) == model
    with pytest.raises(SourceCompilationError, match="no declarado"):
        source.compile()


def test_registry_rejects_duplicates_and_non_scalar_expansions():
    with pytest.raises(SourceCompilationError, match="duplicados"):
        DEFAULT_INVARIANTS.with_invariant(DEFAULT_INVARIANTS.get("R"))
    registry = DEFAULT_INVARIANTS.with_invariant(
        InvariantSpec("Bad", lambda b: b.scalar_gradient("a"), "not scalar"),
    )
    with pytest.raises(SourceCompilationError, match="IR escalar"):
        LagrangianSourceSpec("bad", "Bad").compile(registry=registry)
    with pytest.raises(SourceCompilationError):
        InvariantRegistry((object(),))
    with pytest.raises(SourceCompilationError, match="inválido"):
        InvariantSpec("contract", ModelBuilder.ricci_scalar, "collision")


def test_expansion_hash_binds_custom_meaning_even_if_source_text_is_unchanged():
    source = LagrangianSourceSpec("meaning", "Custom")
    first = DEFAULT_INVARIANTS.with_invariant(
        InvariantSpec("Custom", ModelBuilder.kinetic_scalar, "X"),
    )
    second = DEFAULT_INVARIANTS.with_invariant(
        InvariantSpec("Custom", lambda b: b.kinetic_scalar()**2, "X squared"),
    )
    model1, model2 = source.compile(registry=first), source.compile(registry=second)
    assert dict(model1.metadata)["source_fingerprint"] == dict(model2.metadata)["source_fingerprint"]
    assert dict(model1.metadata)["source_invariants"] != dict(model2.metadata)["source_invariants"]
    assert model_fingerprint(model1) != model_fingerprint(model2)


@pytest.mark.parametrize("name", ("R", "X", "RicciUU", "phi", "contract", "metric", "gradient", "Riemann"))
def test_reserved_names_cannot_shadow_syntax(name):
    with pytest.raises(SourceCompilationError, match="reservados"):
        LagrangianSourceSpec("shadow", "R", parameters=(ParameterSpec(name),))
    with pytest.raises(SourceCompilationError, match="reservados"):
        LagrangianSourceSpec("shadow", "R", functions=(FunctionSpec(name),))


def test_custom_alias_and_dimension_collisions_are_rejected():
    registry = DEFAULT_INVARIANTS.with_invariant(InvariantSpec("XX", ModelBuilder.kinetic_scalar, "X"))
    with pytest.raises(SourceCompilationError, match="reservados"):
        LagrangianSourceSpec("shadow", "XX", parameters=(ParameterSpec("XX"),)).compile(registry=registry)
    with pytest.raises(SourceCompilationError, match="reservados"):
        LagrangianSourceSpec("dimension", "R", dimension=DimensionSpec("R")).compile()


@pytest.mark.parametrize("expression", (
    "contract()", 'contract(gradient("a"))',
    'contract(metric("a","b"), gradient("a"), gradient("a"))',
    'metric("a")', 'gradient(phi)', 'gradient("a-b")',
    'contract(*[R])', 'contract(R, bad=X)',
    'metric("a", __import__("os").getcwd())', 'gradient("a").indices',
    'contract([R][0])', '"R"', '(lambda: R)()',
))
def test_generic_syntax_is_safe_and_checks_indices(expression):
    with pytest.raises(SourceCompilationError):
        LagrangianSourceSpec("bad_syntax", expression).compile()


@pytest.mark.parametrize("normalization", ("R", "X", "RicciUU", "phi", GENERIC))
def test_geometry_cannot_enter_constant_normalization(normalization):
    with pytest.raises(SourceCompilationError):
        LagrangianSourceSpec("normalization", "R", normalization=normalization).compile()


def test_source_model_roundtrip_preserves_assumptions_normalization_and_metadata():
    source = LagrangianSourceSpec(
        "roundtrip", CASE2, dimension=DimensionSpec(3),
        parameters=PARAMETERS + (ParameterSpec("kappa"),),
        normalization="1/kappa", assumptions=("ell != 0", "beta0 != 0"),
        metadata=(("note", "test"),),
    )
    restored = LagrangianSourceSpec.from_data(json.loads(json.dumps(source.to_data())))
    assert restored == source
    assert restored.compile() == source.compile()
    assert ModelSpec.from_data(restored.compile().to_data()) == source.compile()


def test_builder_contract_reuses_ir_checks():
    b = ModelBuilder()
    assert b.contract(b.kinetic_scalar(), b.ricci_scalar()).is_scalar
    with pytest.raises(TensorAlgebraError, match="libres"):
        b.contract(b.scalar_gradient("a"))
