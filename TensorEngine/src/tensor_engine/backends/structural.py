"""Backend puro Python para operaciones estructurales auditables."""

from __future__ import annotations

from collections.abc import Mapping

from .base import BackendInfo, Capability, TensorBackend
from ..canonical import (
    canonicalize_monoterm,
    first_bianchi_residual,
    lower_index,
    raise_index,
    simplify_metrics,
)
from ..contracts import VerificationRecord, VerificationStatus
from ..differential import (
    DifferentialContext,
    commutator_residual,
    covariant_derivative as apply_covariant_derivative,
    differential_bianchi_residual,
    divergence as apply_divergence,
    gradient as apply_gradient,
    hessian as apply_hessian,
    laplacian as apply_laplacian,
    lie_derivative as apply_lie_derivative,
)
from ..euler import (
    EulerLagrangeResult,
    derive_euler_lagrange as apply_derive_euler_lagrange,
    scalar_integration_by_parts_residual,
)
from ..indices import tensor_product
from ..ir import Expr, Index, Number
from ..delta import DeltaContractionAudit, contract_deltas, delta_count
from ..model import ModelSpec, TensorDeclaration
from ..noether import (
    DiffeomorphismVariation,
    NoetherWaldResult,
    derive_noether_wald as apply_derive_noether_wald,
)
from ..palatini import (
    all_down_curvature_variation as apply_all_down_curvature_variation,
    connection_variation as apply_connection_variation,
    mixed_curvature_variation as apply_mixed_curvature_variation,
)
from ..transform import antisymmetrize, expand, substitute, symmetrize
from ..variational import (
    LagrangianMomenta,
    VariationalContext,
    covariant_metric_variation as apply_covariant_metric_variation,
    derive_momenta as apply_derive_momenta,
    direct_variation as apply_direct_variation,
    raw_lagrangian_variation as apply_raw_lagrangian_variation,
    riemann_independent_variation as apply_riemann_independent_variation,
    scalar_gradient_geometric_variation as apply_scalar_gradient_variation,
    volume_element_variation as apply_volume_element_variation,
)


class StructuralTensorBackend(TensorBackend):
    """Referencia determinista; no pretende reemplazar xAct."""

    info = BackendInfo(
        name="structural-python",
        version="0.9.0",
        capabilities=frozenset(
            {
                Capability.INDEX_HYGIENE,
                Capability.STRUCTURAL_SUBSTITUTION,
                Capability.EXPANSION,
                Capability.MONOTERM_SYMMETRY,
                Capability.METRIC_CONTRACTION,
                Capability.COVARIANT_DERIVATIVES,
                Capability.FUNCTION_CHAIN_RULE,
                Capability.CURVATURE_COMMUTATOR,
                Capability.LIE_DERIVATIVE,
                Capability.ELEMENTARY_VARIATION,
                Capability.LAGRANGIAN_MOMENTA,
                Capability.RIEMANN_PROJECTION,
                Capability.PALATINI_VARIATION,
                Capability.INTEGRATION_BY_PARTS,
                Capability.EULER_LAGRANGE,
                Capability.BOUNDARY_POTENTIAL,
                Capability.DIFFEOMORPHISM_VARIATION,
                Capability.NOETHER_CURRENT,
                Capability.WALD_CHARGE,
                Capability.NOETHER_IDENTITY,
            }
        ),
    )

    def __init__(
        self,
        declarations: tuple[TensorDeclaration, ...] = (),
        metric_name: str = "g",
        curvature_name: str = "Riemann",
        delta_name: str = "delta",
        dimension: int | str | Expr = "D",
        differential_context: DifferentialContext | None = None,
        variational_context: VariationalContext | None = None,
    ) -> None:
        self.declarations = tuple(declarations)
        self.metric_name = metric_name
        self.curvature_name = curvature_name
        self.delta_name = delta_name
        self.dimension = dimension
        self._delta_audits: dict[tuple[str, str], DeltaContractionAudit] = {}
        self.differential_context = differential_context or DifferentialContext(
            metric_name=metric_name,
            delta_name=delta_name,
            curvature_name=curvature_name,
            dimension=dimension,
        )
        self.variational_context = variational_context or VariationalContext(
            metric_name=metric_name,
            curvature_name=curvature_name,
            delta_name=delta_name,
        )

    @classmethod
    def from_model(cls, model: ModelSpec) -> "StructuralTensorBackend":
        return cls(
            declarations=model.tensor_declarations,
            metric_name=model.symbols.metric,
            curvature_name=model.symbols.curvature,
            dimension=model.dimension.value,
            differential_context=DifferentialContext.from_model(model),
            variational_context=VariationalContext.from_model(model),
        )

    def canonicalize(self, expr: Expr) -> Expr:
        current = self._contract_deltas(expr)
        current = canonicalize_monoterm(current, self.declarations)
        contracted = self._contract_deltas(current)
        return (canonicalize_monoterm(contracted, self.declarations)
                if contracted != current else current)

    @property
    def delta_contractions(self) -> tuple[DeltaContractionAudit, ...]:
        return tuple(self._delta_audits.values())

    def _remember_delta_audit(self, audit: DeltaContractionAudit) -> None:
        if audit.events:
            self._delta_audits[(audit.input_sha256, audit.output_sha256)] = audit

    def _contract_deltas(self, expr: Expr) -> Expr:
        if not delta_count(expr, self.delta_name):
            return expr
        result = contract_deltas(expr, delta_name=self.delta_name, dimension=self.dimension,
                                 index_space=self.variational_context.index_space)
        self._remember_delta_audit(result.audit)
        return result.expression

    def simplify(self, expr: Expr) -> Expr:
        current = self.canonicalize(expr)
        for _ in range(8):
            audits: list[DeltaContractionAudit] = []
            simplified = simplify_metrics(
                current,
                metric_name=self.metric_name,
                delta_name=self.delta_name,
                dimension=self.dimension,
                index_space=self.variational_context.index_space,
                audit=audits,
            )
            for audit in audits:
                self._remember_delta_audit(audit)
            simplified = self.canonicalize(simplified)
            if simplified == current:
                break
            current = simplified
        return current

    def expand(self, expr: Expr) -> Expr:
        return expand(expr)

    def substitute(self, expr: Expr, replacements: Mapping[Expr, Expr]) -> Expr:
        return substitute(expr, replacements)

    def symmetrize(self, expr: Expr, indices: tuple[Index, ...]) -> Expr:
        return self.canonicalize(symmetrize(expr, indices))

    def antisymmetrize(self, expr: Expr, indices: tuple[Index, ...]) -> Expr:
        return self.canonicalize(antisymmetrize(expr, indices))

    def tensor_product(self, *factors: Expr) -> Expr:
        return tensor_product(*factors)

    def raise_index(self, expr: Expr, index: Index) -> Expr:
        return raise_index(expr, index, self.metric_name)

    def lower_index(self, expr: Expr, index: Index) -> Expr:
        return lower_index(expr, index, self.metric_name)

    def check_first_bianchi(
        self,
        tensor_name: str,
        indices: tuple[Index, Index, Index, Index],
    ) -> VerificationRecord:
        residual = self.canonicalize(first_bianchi_residual(tensor_name, *indices))
        if residual == Number(0):
            return VerificationRecord(
                "riemann_first_bianchi",
                VerificationStatus.PASSED,
                message="El residual se redujo estructuralmente a cero.",
            )
        return VerificationRecord(
            "riemann_first_bianchi",
            VerificationStatus.UNDETERMINED,
            residual=residual,
            message=(
                "El backend estructural no aplica identidades multitémino; "
                "el residual debe enviarse a un backend con esa capacidad."
            ),
        )

    def covariant_derivative(self, expr: Expr, index: Index) -> Expr:
        return self.canonicalize(
            apply_covariant_derivative(expr, index, self.differential_context)
        )

    def gradient(self, scalar: Expr, index: Index) -> Expr:
        return self.canonicalize(apply_gradient(scalar, index, self.differential_context))

    def hessian(self, scalar: Expr, first: Index, second: Index) -> Expr:
        return self.canonicalize(
            apply_hessian(scalar, first, second, self.differential_context)
        )

    def divergence(self, expr: Expr, index: Index) -> Expr:
        return self.simplify(apply_divergence(expr, index, self.differential_context))

    def laplacian(self, scalar: Expr, index_space: str = "M") -> Expr:
        return self.simplify(
            apply_laplacian(scalar, self.differential_context, index_space)
        )

    def lie_derivative(self, expr: Expr) -> Expr:
        return self.canonicalize(apply_lie_derivative(expr, self.differential_context))

    def check_commutator(
        self,
        expr: Expr,
        first: Index,
        second: Index,
    ) -> VerificationRecord:
        residual = self.simplify(
            commutator_residual(expr, first, second, self.differential_context)
        )
        if residual == Number(0):
            return VerificationRecord(
                "covariant_derivative_commutator",
                VerificationStatus.PASSED,
                message="El residual del conmutador se redujo a cero.",
            )
        return VerificationRecord(
            "covariant_derivative_commutator",
            VerificationStatus.UNDETERMINED,
            residual=residual,
            message=(
                "El backend construyó ambos lados, pero no dispone de una "
                "reducción multitémino suficiente para decidir el residual."
            ),
        )

    def check_differential_bianchi(
        self,
        tensor_name: str,
        indices: tuple[Index, Index, Index, Index, Index],
    ) -> VerificationRecord:
        residual = self.canonicalize(
            differential_bianchi_residual(
                tensor_name,
                *indices,
                context=self.differential_context,
            )
        )
        if residual == Number(0):
            return VerificationRecord(
                "riemann_differential_bianchi",
                VerificationStatus.PASSED,
                message="El residual diferencial se redujo a cero.",
            )
        return VerificationRecord(
            "riemann_differential_bianchi",
            VerificationStatus.UNDETERMINED,
            residual=residual,
            message=(
                "La identidad diferencial de Bianchi requiere una capacidad "
                "multitémino no disponible en este backend."
            ),
        )

    def derive_momenta(self, lagrangian: Expr) -> LagrangianMomenta:
        raw = apply_derive_momenta(lagrangian, self.variational_context)
        return LagrangianMomenta(
            metric=self.simplify(raw.metric),
            curvature=self.simplify(raw.curvature),
            scalar_gradient=self.simplify(raw.scalar_gradient),
            scalar=self.simplify(raw.scalar),
        )

    def direct_variation(self, expr: Expr) -> Expr:
        return self.canonicalize(apply_direct_variation(expr, self.variational_context))

    def raw_lagrangian_variation(self, momenta: LagrangianMomenta) -> Expr:
        return self.canonicalize(
            apply_raw_lagrangian_variation(momenta, self.variational_context)
        )

    def covariant_metric_variation(self, first: Index, second: Index) -> Expr:
        return self.canonicalize(
            apply_covariant_metric_variation(first, second, self.variational_context)
        )

    def volume_element_variation(self) -> Expr:
        return self.canonicalize(apply_volume_element_variation(self.variational_context))

    def scalar_gradient_geometric_variation(self, index: Index) -> Expr:
        return self.canonicalize(
            apply_scalar_gradient_variation(index, self.variational_context)
        )

    def riemann_independent_variation(
        self,
        indices: tuple[Index, Index, Index, Index],
    ) -> Expr:
        return self.canonicalize(
            apply_riemann_independent_variation(indices, self.variational_context)
        )

    def connection_variation(self, upper: Index, first: Index, second: Index) -> Expr:
        return self.canonicalize(
            apply_connection_variation(upper, first, second, self.variational_context)
        )

    def mixed_curvature_variation(
        self,
        upper: Index,
        lower: Index,
        first: Index,
        second: Index,
        *,
        expand_connection: bool = False,
    ) -> Expr:
        return self.canonicalize(
            apply_mixed_curvature_variation(
                upper,
                lower,
                first,
                second,
                self.variational_context,
                expand_connection=expand_connection,
            )
        )

    def all_down_curvature_variation(
        self,
        indices: tuple[Index, Index, Index, Index],
        *,
        expand_connection: bool = False,
    ) -> Expr:
        return self.canonicalize(
            apply_all_down_curvature_variation(
                indices,
                self.variational_context,
                expand_connection=expand_connection,
            )
        )

    def derive_euler_lagrange(
        self,
        lagrangian: Expr,
        momenta: LagrangianMomenta | None = None,
    ) -> EulerLagrangeResult:
        current_momenta = momenta or self.derive_momenta(lagrangian)
        raw = apply_derive_euler_lagrange(
            lagrangian,
            current_momenta,
            self.variational_context,
            self.differential_context,
        )
        return EulerLagrangeResult(
            metric_euler=self.simplify(raw.metric_euler),
            scalar_euler=self.simplify(raw.scalar_euler),
            boundary_metric=self.canonicalize(raw.boundary_metric),
            boundary_scalar=self.canonicalize(raw.boundary_scalar),
            boundary_total=self.canonicalize(raw.boundary_total),
            full_variation=self.canonicalize(raw.full_variation),
            density_variation=self.canonicalize(raw.density_variation),
            curvature_derivative_metric_term=self.simplify(
                raw.curvature_derivative_metric_term
            ),
        )

    def check_scalar_integration_by_parts(
        self,
        momenta: LagrangianMomenta,
    ) -> VerificationRecord:
        residual = self.canonicalize(
            scalar_integration_by_parts_residual(
                momenta,
                self.variational_context,
                self.differential_context,
            )
        )
        if residual == Number(0):
            return VerificationRecord(
                "scalar_integration_by_parts",
                VerificationStatus.PASSED,
                message="La regla de cadena escalar coincide con bulk más frontera.",
            )
        return VerificationRecord(
            "scalar_integration_by_parts",
            VerificationStatus.UNDETERMINED,
            residual=residual,
            message="El backend estructural no pudo reducir por completo el residual escalar.",
        )

    def derive_noether_wald(
        self,
        lagrangian: Expr,
        momenta: LagrangianMomenta | None = None,
        euler: EulerLagrangeResult | None = None,
    ) -> NoetherWaldResult:
        current_momenta = momenta or self.derive_momenta(lagrangian)
        current_euler = euler or self.derive_euler_lagrange(lagrangian, current_momenta)
        raw = apply_derive_noether_wald(
            lagrangian,
            current_momenta,
            current_euler,
            self.variational_context,
            self.differential_context,
        )
        return NoetherWaldResult(
            diffeomorphism=DiffeomorphismVariation(
                inverse_metric=self.simplify(raw.diffeomorphism.inverse_metric),
                scalar=self.simplify(raw.diffeomorphism.scalar),
            ),
            boundary_metric=self.simplify(raw.boundary_metric),
            boundary_scalar=self.simplify(raw.boundary_scalar),
            boundary_total=self.simplify(raw.boundary_total),
            noether_current=self.simplify(raw.noether_current),
            constraint_current=self.simplify(raw.constraint_current),
            charge_potential=self.simplify(raw.charge_potential),
            charge_divergence=self.simplify(raw.charge_divergence),
            decomposition_residual=self.simplify(raw.decomposition_residual),
            noether_identity=self.simplify(raw.noether_identity),
        )

    def check_noether_identity(
        self,
        result: NoetherWaldResult,
    ) -> VerificationRecord:
        residual = self.simplify(result.noether_identity)
        if residual == Number(0):
            return VerificationRecord(
                "diffeomorphism_noether_identity",
                VerificationStatus.PASSED,
                message="La identidad 2 nabla^a E_ab + E_phi nabla_b phi se redujo a cero.",
            )
        return VerificationRecord(
            "diffeomorphism_noether_identity",
            VerificationStatus.UNDETERMINED,
            residual=residual,
            message=(
                "La identidad requiere Bianchi diferencial o conmutación de derivadas "
                "que el backend estructural no fuerza."
            ),
        )

    def check_noether_decomposition(
        self,
        result: NoetherWaldResult,
    ) -> VerificationRecord:
        residual = self.simplify(result.decomposition_residual)
        if residual == Number(0):
            return VerificationRecord(
                "noether_current_decomposition",
                VerificationStatus.PASSED,
                message="J_xi^a=2E^a_b xi^b+nabla_b Q_xi^{ab}.",
            )
        return VerificationRecord(
            "noether_current_decomposition",
            VerificationStatus.UNDETERMINED,
            residual=residual,
            message=(
                "La descomposición fue construida, pero su reducción completa requiere "
                "identidades multitémino de un backend externo."
            ),
        )
