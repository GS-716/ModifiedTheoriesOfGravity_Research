from __future__ import annotations

import json

import pytest

from tensor_engine import (
    FunctionSpec,
    LagrangianSourceSpec,
    ParameterSpec,
    Number,
    SourceCompilationError,
    StructuralTensorBackend,
    catalog_model,
    load_lagrangian_source,
    save_lagrangian_source,
)


CASES = (
    ("einstein_hilbert", "R", (), ()),
    (
        "canonical_scalar",
        "R - X/2 - V(phi)",
        (FunctionSpec("V", 1),),
        (),
    ),
    (
        "nonminimal_scalar_tensor",
        "F(phi)*R - Z(phi)*X/2 - V(phi)",
        (FunctionSpec("F", 1), FunctionSpec("Z", 1), FunctionSpec("V", 1)),
        (),
    ),
    (
        "k_essence",
        "R + K(phi, X)",
        (FunctionSpec("K", 2),),
        (),
    ),
    (
        "quadratic_ricci_scalar",
        "R + alpha*R**2",
        (),
        (ParameterSpec("alpha"),),
    ),
)


@pytest.mark.parametrize("key,expression,functions,parameters", CASES)
def test_declarative_sources_match_catalog_ir(key, expression, functions, parameters) -> None:
    source = LagrangianSourceSpec(
        key,
        expression,
        functions=functions,
        parameters=parameters,
    )
    compiled = source.compile()
    expected = catalog_model(key)
    backend = StructuralTensorBackend.from_model(compiled)
    assert backend.simplify(compiled.lagrangian - expected.lagrangian) == Number(0)
    assert dict(compiled.metadata)["source_fingerprint"] == source.fingerprint
    assert len(source.fingerprint) == 64


def test_source_roundtrip_and_exact_normalization(tmp_path) -> None:
    source = LagrangianSourceSpec(
        "normalized_source",
        "R - X/2",
        normalization="1/kappa",
        parameters=(ParameterSpec("kappa"),),
    )
    encoded = json.loads(json.dumps(source.to_data()))
    assert LagrangianSourceSpec.from_data(encoded) == source
    path = save_lagrangian_source(source, tmp_path / "source.json")
    assert load_lagrangian_source(path) == source
    assert source.compile().normalization.is_scalar


@pytest.mark.parametrize(
    "expression,match",
    (
        ("__import__('os')", "Función no declarada"),
        ("phi.__class__", "Attribute"),
        ("F(phi, X)", "espera 1 argumentos"),
        ("unknown + R", "Símbolo no declarado"),
        ("R ^ 2", r"Use \*\*"),
        ("0.5*X", "No se admiten decimales"),
        ("[R][0]", "Subscript"),
    ),
)
def test_unsafe_or_ambiguous_syntax_is_rejected(expression, match) -> None:
    source = LagrangianSourceSpec(
        "unsafe_source",
        expression,
        functions=(FunctionSpec("F", 1),),
    )
    with pytest.raises(SourceCompilationError, match=match):
        source.compile()


def test_dsl_aliases_cannot_be_redeclared() -> None:
    with pytest.raises(SourceCompilationError, match="reservados"):
        LagrangianSourceSpec(
            "collision",
            "R",
            parameters=(ParameterSpec("R"),),
        )
