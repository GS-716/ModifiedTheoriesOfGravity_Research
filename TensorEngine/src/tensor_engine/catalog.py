"""Catálogo pequeño de teorías representativas del dominio soportado."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from .builders import ModelBuilder
from .ir import Number, Scalar
from .model import DimensionSpec, FunctionSpec, ModelSpec, ParameterSpec


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    key: str
    title: str
    description: str
    factory: Callable[[], ModelSpec]

    def create(self, *, name: str | None = None) -> ModelSpec:
        model = self.factory()
        return model if name is None else replace(model, name=name)

    def to_data(self) -> dict[str, str]:
        return {"key": self.key, "title": self.title, "description": self.description}


def _model(
    key: str,
    lagrangian,
    *,
    functions: tuple[FunctionSpec, ...] = (),
    parameters: tuple[ParameterSpec, ...] = (),
) -> ModelSpec:
    return ModelSpec(
        key,
        lagrangian,
        dimension=DimensionSpec(4),
        functions=functions,
        parameters=parameters,
        metadata=(("catalog_key", key), ("domain", "L_g_R_phi_nabla_phi")),
    )


def _einstein_hilbert() -> ModelSpec:
    builder = ModelBuilder()
    return _model("einstein_hilbert", builder.ricci_scalar())


def _canonical_scalar() -> ModelSpec:
    builder = ModelBuilder()
    potential = builder.function("V", builder.phi)
    lagrangian = builder.ricci_scalar() - Number(1, 2) * builder.kinetic_scalar() - potential
    return _model(
        "canonical_scalar",
        lagrangian,
        functions=(FunctionSpec("V", 1, "Potencial escalar"),),
    )


def _nonminimal_scalar_tensor() -> ModelSpec:
    builder = ModelBuilder()
    coupling = builder.function("F", builder.phi)
    kinetic = builder.function("Z", builder.phi)
    potential = builder.function("V", builder.phi)
    lagrangian = (
        coupling * builder.ricci_scalar()
        - Number(1, 2) * kinetic * builder.kinetic_scalar()
        - potential
    )
    return _model(
        "nonminimal_scalar_tensor",
        lagrangian,
        functions=(
            FunctionSpec("F", 1, "Acoplamiento no mínimo"),
            FunctionSpec("Z", 1, "Acoplamiento cinético"),
            FunctionSpec("V", 1, "Potencial escalar"),
        ),
    )


def _k_essence() -> ModelSpec:
    builder = ModelBuilder()
    matter = builder.function("K", builder.phi, builder.kinetic_scalar())
    return _model(
        "k_essence",
        builder.ricci_scalar() + matter,
        functions=(FunctionSpec("K", 2, "K(phi,X)"),),
    )


def _quadratic_ricci_scalar() -> ModelSpec:
    builder = ModelBuilder()
    curvature = builder.ricci_scalar()
    return _model(
        "quadratic_ricci_scalar",
        curvature + Scalar("alpha") * curvature**2,
        parameters=(ParameterSpec("alpha", description="Acoplamiento de R^2"),),
    )


_CATALOG = (
    CatalogEntry(
        "einstein_hilbert",
        "Einstein–Hilbert",
        "L=R; referencia gravitatoria mínima.",
        _einstein_hilbert,
    ),
    CatalogEntry(
        "canonical_scalar",
        "Escalar canónico",
        "L=R-X/2-V(phi).",
        _canonical_scalar,
    ),
    CatalogEntry(
        "nonminimal_scalar_tensor",
        "Escalar–tensor no mínimo",
        "L=F(phi)R-Z(phi)X/2-V(phi).",
        _nonminimal_scalar_tensor,
    ),
    CatalogEntry(
        "k_essence",
        "K-essence",
        "L=R+K(phi,X), con X=(nabla phi)^2.",
        _k_essence,
    ),
    CatalogEntry(
        "quadratic_ricci_scalar",
        "Gravedad R+alpha R^2",
        "Caso de curvatura no lineal dentro de L(g,R,phi,nabla phi).",
        _quadratic_ricci_scalar,
    ),
)


def catalog_entries() -> tuple[CatalogEntry, ...]:
    return _CATALOG


def catalog_model(key: str, *, name: str | None = None) -> ModelSpec:
    for entry in _CATALOG:
        if entry.key == key:
            return entry.create(name=name)
    available = ", ".join(item.key for item in _CATALOG)
    raise KeyError(f"Modelo de catálogo desconocido {key!r}. Disponibles: {available}.")
