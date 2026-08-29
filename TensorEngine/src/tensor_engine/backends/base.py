"""Interfaz común para backends tensoriales intercambiables."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from collections.abc import Mapping

from ..contracts import VerificationRecord
from ..euler import EulerLagrangeResult
from ..errors import BackendCapabilityError
from ..ir import Expr, Index
from ..noether import NoetherWaldResult
from ..variational import LagrangianMomenta


class Capability(str, Enum):
    INDEX_HYGIENE = "index_hygiene"
    STRUCTURAL_SUBSTITUTION = "structural_substitution"
    EXPANSION = "expansion"
    MONOTERM_SYMMETRY = "monoterm_symmetry"
    METRIC_CONTRACTION = "metric_contraction"
    MULTITERM_BIANCHI = "multiterm_bianchi"
    DIMENSION_IDENTITIES = "dimension_identities"
    COVARIANT_DERIVATIVES = "covariant_derivatives"
    FUNCTION_CHAIN_RULE = "function_chain_rule"
    CURVATURE_COMMUTATOR = "curvature_commutator"
    LIE_DERIVATIVE = "lie_derivative"
    DIFFERENTIAL_BIANCHI = "differential_bianchi"
    ELEMENTARY_VARIATION = "elementary_variation"
    LAGRANGIAN_MOMENTA = "lagrangian_momenta"
    RIEMANN_PROJECTION = "riemann_projection"
    PALATINI_VARIATION = "palatini_variation"
    INTEGRATION_BY_PARTS = "integration_by_parts"
    EULER_LAGRANGE = "euler_lagrange"
    BOUNDARY_POTENTIAL = "boundary_potential"
    DIFFEOMORPHISM_VARIATION = "diffeomorphism_variation"
    NOETHER_CURRENT = "noether_current"
    WALD_CHARGE = "wald_charge"
    NOETHER_IDENTITY = "noether_identity"
    GEOMETRY_ANSATZ = "geometry_ansatz"
    COORDINATE_COMPONENTS = "coordinate_components"


@dataclass(frozen=True, slots=True)
class BackendInfo:
    name: str
    version: str
    capabilities: frozenset[Capability]

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities


class TensorBackend(ABC):
    """Contrato mínimo que deberán implementar Python, SymPy y xAct."""

    info: BackendInfo

    def require(self, capability: Capability) -> None:
        if not self.info.supports(capability):
            raise BackendCapabilityError(
                f"El backend {self.info.name} no soporta {capability.value}."
            )

    @abstractmethod
    def canonicalize(self, expr: Expr) -> Expr:
        raise NotImplementedError

    @abstractmethod
    def simplify(self, expr: Expr) -> Expr:
        raise NotImplementedError

    @abstractmethod
    def expand(self, expr: Expr) -> Expr:
        raise NotImplementedError

    @abstractmethod
    def substitute(self, expr: Expr, replacements: Mapping[Expr, Expr]) -> Expr:
        raise NotImplementedError

    @abstractmethod
    def symmetrize(self, expr: Expr, indices: tuple[Index, ...]) -> Expr:
        raise NotImplementedError

    @abstractmethod
    def antisymmetrize(self, expr: Expr, indices: tuple[Index, ...]) -> Expr:
        raise NotImplementedError

    @abstractmethod
    def tensor_product(self, *factors: Expr) -> Expr:
        raise NotImplementedError

    @abstractmethod
    def raise_index(self, expr: Expr, index: Index) -> Expr:
        raise NotImplementedError

    @abstractmethod
    def lower_index(self, expr: Expr, index: Index) -> Expr:
        raise NotImplementedError

    @abstractmethod
    def check_first_bianchi(
        self,
        tensor_name: str,
        indices: tuple[Index, Index, Index, Index],
    ) -> VerificationRecord:
        raise NotImplementedError

    @abstractmethod
    def covariant_derivative(self, expr: Expr, index: Index) -> Expr:
        raise NotImplementedError

    @abstractmethod
    def gradient(self, scalar: Expr, index: Index) -> Expr:
        raise NotImplementedError

    @abstractmethod
    def hessian(self, scalar: Expr, first: Index, second: Index) -> Expr:
        raise NotImplementedError

    @abstractmethod
    def divergence(self, expr: Expr, index: Index) -> Expr:
        raise NotImplementedError

    @abstractmethod
    def laplacian(self, scalar: Expr, index_space: str = "M") -> Expr:
        raise NotImplementedError

    @abstractmethod
    def lie_derivative(self, expr: Expr) -> Expr:
        raise NotImplementedError

    @abstractmethod
    def check_commutator(
        self,
        expr: Expr,
        first: Index,
        second: Index,
    ) -> VerificationRecord:
        raise NotImplementedError

    @abstractmethod
    def check_differential_bianchi(
        self,
        tensor_name: str,
        indices: tuple[Index, Index, Index, Index, Index],
    ) -> VerificationRecord:
        raise NotImplementedError

    @abstractmethod
    def derive_momenta(self, lagrangian: Expr) -> LagrangianMomenta:
        raise NotImplementedError

    @abstractmethod
    def direct_variation(self, expr: Expr) -> Expr:
        raise NotImplementedError

    @abstractmethod
    def raw_lagrangian_variation(self, momenta: LagrangianMomenta) -> Expr:
        raise NotImplementedError

    @abstractmethod
    def covariant_metric_variation(self, first: Index, second: Index) -> Expr:
        raise NotImplementedError

    @abstractmethod
    def volume_element_variation(self) -> Expr:
        raise NotImplementedError

    @abstractmethod
    def scalar_gradient_geometric_variation(self, index: Index) -> Expr:
        raise NotImplementedError

    @abstractmethod
    def riemann_independent_variation(
        self,
        indices: tuple[Index, Index, Index, Index],
    ) -> Expr:
        raise NotImplementedError

    @abstractmethod
    def connection_variation(self, upper: Index, first: Index, second: Index) -> Expr:
        raise NotImplementedError

    @abstractmethod
    def mixed_curvature_variation(
        self,
        upper: Index,
        lower: Index,
        first: Index,
        second: Index,
        *,
        expand_connection: bool = False,
    ) -> Expr:
        raise NotImplementedError

    @abstractmethod
    def all_down_curvature_variation(
        self,
        indices: tuple[Index, Index, Index, Index],
        *,
        expand_connection: bool = False,
    ) -> Expr:
        raise NotImplementedError

    @abstractmethod
    def derive_euler_lagrange(
        self,
        lagrangian: Expr,
        momenta: LagrangianMomenta | None = None,
    ) -> EulerLagrangeResult:
        raise NotImplementedError

    @abstractmethod
    def check_scalar_integration_by_parts(
        self,
        momenta: LagrangianMomenta,
    ) -> VerificationRecord:
        raise NotImplementedError

    @abstractmethod
    def derive_noether_wald(
        self,
        lagrangian: Expr,
        momenta: LagrangianMomenta | None = None,
        euler: EulerLagrangeResult | None = None,
    ) -> NoetherWaldResult:
        raise NotImplementedError

    @abstractmethod
    def check_noether_identity(
        self,
        result: NoetherWaldResult,
    ) -> VerificationRecord:
        raise NotImplementedError

    @abstractmethod
    def check_noether_decomposition(
        self,
        result: NoetherWaldResult,
    ) -> VerificationRecord:
        raise NotImplementedError
