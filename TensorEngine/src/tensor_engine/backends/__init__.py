"""Backends disponibles en TensorEngine."""

from .base import BackendInfo, Capability, TensorBackend
from .structural import StructuralTensorBackend

__all__ = ["BackendInfo", "Capability", "StructuralTensorBackend", "TensorBackend"]

