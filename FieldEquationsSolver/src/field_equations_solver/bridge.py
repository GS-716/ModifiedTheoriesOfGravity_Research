"""Local Wolfram transport dedicated to formal field-equation solving."""
from pathlib import Path

from tensor_engine.wolfram_bridge import WolframXActBridge


class FieldEquationWolframBridge(WolframXActBridge):
    """Use the solver script without loading TensorEngine's xAct dispatcher."""

    def __init__(self, executable=None, script_path=None, timeout_seconds=120.0):
        default = Path(__file__).resolve().parents[2] / "wolfram" / "FieldEquationSolver.wl"
        super().__init__(
            executable=executable,
            script_path=default if script_path is None else script_path,
            timeout_seconds=timeout_seconds,
        )

