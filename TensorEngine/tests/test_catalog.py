from __future__ import annotations

import pytest

from tensor_engine import (
    Function,
    ModelBuilder,
    ModelSpec,
    catalog_entries,
    catalog_model,
    infer_free_indices,
)


def test_high_level_invariants_are_fully_contracted() -> None:
    builder = ModelBuilder()
    assert infer_free_indices(builder.ricci_scalar()) == ()
    assert infer_free_indices(builder.kinetic_scalar()) == ()
    symbolic = builder.function("K", builder.phi, builder.kinetic_scalar())
    assert isinstance(symbolic, Function)
    assert len(symbolic.arguments) == 2


def test_catalog_models_are_unique_and_valid() -> None:
    entries = catalog_entries()
    assert len(entries) == 5
    assert len({entry.key for entry in entries}) == len(entries)
    for entry in entries:
        model = entry.create()
        assert isinstance(model, ModelSpec)
        assert model.validate() is model
        assert dict(model.metadata)["catalog_key"] == entry.key


def test_catalog_can_rename_without_mutating_template() -> None:
    renamed = catalog_model("k_essence", name="my_k_essence")
    original = catalog_model("k_essence")
    assert renamed.name == "my_k_essence"
    assert original.name == "k_essence"
    assert renamed.lagrangian == original.lagrangian


def test_unknown_catalog_key_is_explicit() -> None:
    with pytest.raises(KeyError, match="Disponibles"):
        catalog_model("unknown")
