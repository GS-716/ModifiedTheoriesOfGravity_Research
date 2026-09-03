from __future__ import annotations

import json
import hashlib

import pytest
import sympy as sp

from tensor_engine import (
    DimensionSpec,
    LagrangianSourceSpec,
    ModelBuilder,
    Number,
    ParameterSpec,
    RunPackage,
    StructuralTensorBackend,
    SympyComponentBackend,
    TensorEngine,
    draft4_circular_ansatz,
    spatially_flat_flrw_ansatz,
)
from tensor_engine.components import ir_scalar_to_sympy


@pytest.mark.parametrize(
    "alias,builder_method",
    (("RicciSq", "ricci_squared"), ("RiemannSq", "riemann_squared")),
)
def test_quadratic_curvature_models_roundtrip_and_reuse_expanded_ir(
    alias, builder_method,
):
    source = LagrangianSourceSpec(
        name="integration_" + alias,
        expression=f"R + alpha*{alias}",
        parameters=(ParameterSpec("alpha"),),
    )
    model = source.compile()
    run = TensorEngine().run(model)
    rebuilt = RunPackage.from_data(json.loads(json.dumps(run.package.to_data())))
    builder = ModelBuilder(model.symbols)
    expected = builder.ricci_scalar() + builder.scalar("alpha") * getattr(
        builder, builder_method
    )()
    backend = StructuralTensorBackend.from_model(model)
    expected_momenta = backend.derive_momenta(expected)

    assert model.lagrangian == expected
    assert run.package.momenta == expected_momenta
    assert run.package.euler == backend.derive_euler_lagrange(expected, expected_momenta)
    assert rebuilt == run.package
    assert rebuilt.run_id == run.package.run_id
    assert run.package.verification.summary["failed"] == 0
    assert run.package.momenta.scalar_gradient == Number(0)
    assert run.package.momenta.scalar == Number(0)
    assert tuple(key for key, _ in run.abstract.expression_items())[-6:] == (
        "ricci_scalar",
        "ricci_squared",
        "riemann_tensor",
        "riemann_squared",
        "nabla_P",
        "nabla_nabla_P",
    )
    assert run.projected.ricci_squared.reason
    assert run.projected.riemann_squared.reason

    if alias == "RicciSq":
        legacy = json.loads(json.dumps(run.package.to_data()))
        for section in ("derived_quantities", "abstract_results"):
            for key in ("ricci_squared", "riemann_squared"):
                legacy[section]["expressions"].pop(key)
            legacy[section]["records"] = [
                item
                for item in legacy[section]["records"]
                if item["key"] not in {"ricci_squared", "riemann_squared"}
            ]
        legacy["projected_results"]["quantities"] = [
            item
            for item in legacy["projected_results"]["quantities"]
            if item["key"] not in {"ricci_squared", "riemann_squared"}
        ]
        legacy["expressions"] = [
            item
            for item in legacy["expressions"]
            if item["key"] not in {"ricci_squared", "riemann_squared"}
        ]
        semantic_keys = (
            "export_schema_version", "model", "normalized_lagrangian", "momenta",
            "raw_variation", "euler_lagrange", "noether_wald", "components",
            "derived_quantities", "abstract_results", "projected_results", "verification",
        )
        semantic = {key: legacy[key] for key in semantic_keys}
        digest = hashlib.sha256(
            json.dumps(
                semantic, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
        ).hexdigest()
        legacy["run_id"] = f"run_{digest[:20]}"
        migrated = RunPackage.from_data(legacy)
        assert len(migrated.abstract.records) == 13
        assert migrated.abstract.ricci_squared == builder.ricci_squared()
        assert migrated.projected.ricci_squared.status.value == "symbolic"
        assert "bundle" in migrated.projected.ricci_squared.reason


def test_no_dimension_three_curvature_identity_is_applied_to_abstract_ir():
    builder = ModelBuilder()
    model = LagrangianSourceSpec(
        "three_dimensional_ir",
        "RiemannSq - 4*RicciSq + R**2",
        dimension=DimensionSpec(3),
    ).compile()
    backend = StructuralTensorBackend.from_model(model)

    assert backend.simplify(model.lagrangian) != Number(0)
    assert model.lagrangian == (
        builder.riemann_squared()
        - Number(4) * builder.ricci_squared()
        + builder.ricci_scalar() ** Number(2)
    )


@pytest.mark.parametrize("kind", ("draft4", "flrw"))
def test_quadratic_invariants_project_to_independent_coordinate_formulas(kind):
    builder = ModelBuilder()
    if kind == "draft4":
        ansatz = draft4_circular_ansatz()
        radial = sp.Symbol("r")
        f = sp.Function("f")(radial)
        ricci_expected = (
            radial**2 * sp.diff(f, radial, 2) ** 2
            + 2 * radial * sp.diff(f, radial) * sp.diff(f, radial, 2)
            + 3 * sp.diff(f, radial) ** 2
        ) / (2 * radial**2)
        riemann_expected = (
            radial**2 * sp.diff(f, radial, 2) ** 2
            + 2 * sp.diff(f, radial) ** 2
        ) / radial**2
    else:
        ansatz = spatially_flat_flrw_ansatz()
        time = sp.Symbol("t")
        a = sp.Function("a")(time)
        adot, addot = sp.diff(a, time), sp.diff(a, time, 2)
        ricci_expected = 12 * (
            a**2 * addot**2 + a * adot**2 * addot + adot**4
        ) / a**4
        riemann_expected = 12 * (a**2 * addot**2 + adot**4) / a**4

    backend = SympyComponentBackend(ansatz)
    ricci_actual = backend.evaluate_sympy(builder.ricci_squared()).scalar
    riemann_actual = backend.evaluate_sympy(builder.riemann_squared()).scalar

    assert sp.simplify(ricci_actual - ricci_expected) == 0
    assert sp.simplify(riemann_actual - riemann_expected) == 0


def test_first_class_projection_uses_the_same_component_backend():
    source = LagrangianSourceSpec(
        "quadratic_projection",
        "R + alpha*RicciSq",
        dimension=DimensionSpec(3),
        parameters=(ParameterSpec("alpha"),),
    )
    run = TensorEngine().run(source.compile(), ansatz=draft4_circular_ansatz())
    component_backend = SympyComponentBackend.from_model(
        run.package.model, draft4_circular_ansatz()
    )

    assert run.projected.ricci_squared.status.value == "completed"
    assert run.projected.riemann_squared.status.value == "completed"
    assert sp.simplify(
        ir_scalar_to_sympy(run.projected.ricci_squared.scalar)
        - component_backend.evaluate_sympy(run.derived.ricci_squared).scalar
    ) == 0
    assert sp.simplify(
        ir_scalar_to_sympy(run.projected.riemann_squared.scalar)
        - component_backend.evaluate_sympy(run.derived.riemann_squared).scalar
    ) == 0


@pytest.mark.parametrize(
    "expression,parameters",
    (
        ("R", ()),
        ("R + alpha*R**2", (ParameterSpec("alpha"),)),
        ("RicciUU", ()),
        ("R*X", ()),
        (
            "R + 2/ell**2 + ell**2*beta0*(3*RicciUU - X*R)",
            (ParameterSpec("ell"), ParameterSpec("beta0"), ParameterSpec("p")),
        ),
    ),
)
def test_existing_frontend_models_regress_with_new_first_class_results(
    expression, parameters,
):
    model = LagrangianSourceSpec(
        "regression", expression, parameters=parameters
    ).compile()
    run = TensorEngine().run(model)

    assert run.package.verification.summary["failed"] == 0
    assert len(run.abstract.records) == 13
    assert len(run.projected.quantities) == 13
    assert run.abstract.ricci_squared == ModelBuilder(model.symbols).ricci_squared()
    assert run.abstract.riemann_squared == ModelBuilder(model.symbols).riemann_squared()
