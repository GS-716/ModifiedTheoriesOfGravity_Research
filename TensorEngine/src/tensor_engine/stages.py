"""Topología declarativa del pipeline; no ejecuta todavía cálculos."""

from __future__ import annotations

from .contracts import StageSpec
from .errors import ContractValidationError


DEFAULT_PIPELINE: tuple[StageSpec, ...] = (
    StageSpec(
        "validate_model",
        ("model_spec",),
        ("validated_model",),
        description="Valida alcance, símbolos, índices y convenciones.",
    ),
    StageSpec(
        "normalize_lagrangian",
        ("validated_model",),
        ("lagrangian",),
        description="Construye la forma interna canónica sin aplicar identidades del modelo.",
    ),
    StageSpec(
        "derive_momenta",
        ("lagrangian",),
        ("metric_momentum", "curvature_momentum", "scalar_gradient_momentum", "scalar_derivative"),
        description="Calcula M_ab, P^abcd, J^a y F_phi.",
    ),
    StageSpec(
        "raw_variation",
        (
            "lagrangian",
            "metric_momentum",
            "curvature_momentum",
            "scalar_gradient_momentum",
            "scalar_derivative",
        ),
        ("delta_lagrangian",),
        description="Aplica la regla de cadena antes de integrar por partes.",
    ),
    StageSpec(
        "integrate_by_parts",
        ("delta_lagrangian",),
        (
            "metric_euler",
            "scalar_euler",
            "boundary_potential_metric",
            "boundary_potential_scalar",
            "boundary_potential_total",
            "full_variation",
        ),
        description="Separa bulk y frontera y obtiene las ecuaciones de Euler-Lagrange.",
    ),
    StageSpec(
        "noether",
        ("metric_euler", "scalar_euler", "boundary_potential_total"),
        ("noether_current", "charge_potential"),
        optional=True,
        description="Construye corriente de Noether, restricción e Iyer-Wald para difeomorfismos.",
    ),
    StageSpec(
        "components",
        ("validated_model", "geometry_ansatz", "metric_euler", "scalar_euler"),
        ("component_results",),
        optional=True,
        description=(
            "Evalúa una geometría concreta usando un GeometryAnsatz externo y reutilizable."
        ),
    ),
    StageSpec(
        "wolfram_model_validation",
        (
            "validated_model",
            "metric_momentum",
            "curvature_momentum",
            "metric_euler",
            "scalar_euler",
        ),
        ("wolfram_model_report",),
        optional=True,
        description="Valida residuales IR del cálculo concreto mediante xAct y fingerprints.",
    ),
    StageSpec(
        "verify",
        (
            "validated_model",
            "metric_momentum",
            "curvature_momentum",
            "metric_euler",
            "scalar_euler",
            "full_variation",
        ),
        ("verification_report",),
        description="Ejecuta verificaciones matemáticas y de integridad.",
    ),
    StageSpec(
        "export",
        ("validated_model", "verification_report"),
        ("run_manifest", "exported_artifacts"),
        optional=True,
        description="Genera manifestación y artefactos de presentación.",
    ),
)


def validate_pipeline(
    stages: tuple[StageSpec, ...] = DEFAULT_PIPELINE,
    initial_inputs: tuple[str, ...] = ("model_spec", "geometry_ansatz"),
) -> None:
    """Comprueba orden, claves únicas y disponibilidad de dependencias."""

    stage_keys: set[str] = set()
    available = set(initial_inputs)
    producers: dict[str, str] = {}
    for stage in stages:
        if stage.key in stage_keys:
            raise ContractValidationError(f"Clave de etapa repetida: {stage.key}")
        stage_keys.add(stage.key)
        missing = set(stage.requires).difference(available)
        if missing:
            raise ContractValidationError(
                f"La etapa {stage.key} requiere entradas aún no producidas: {sorted(missing)}"
            )
        for output in stage.produces:
            if output in producers:
                raise ContractValidationError(
                    f"{output} es producido por {producers[output]} y {stage.key}."
                )
            producers[output] = stage.key
            available.add(output)


validate_pipeline()
