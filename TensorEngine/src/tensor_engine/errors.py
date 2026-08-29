"""Excepciones públicas de TensorEngine."""


class TensorEngineError(Exception):
    """Base para errores controlados del motor."""


class IRValidationError(TensorEngineError, ValueError):
    """La expresión intermedia no es tensorialmente válida."""


class ModelValidationError(TensorEngineError, ValueError):
    """La especificación de un modelo viola el contrato matemático."""


class SourceCompilationError(TensorEngineError, ValueError):
    """Una fuente declarativa no pertenece a la gramática segura."""


class ContractValidationError(TensorEngineError, ValueError):
    """Un resultado o una etapa viola su contrato de datos."""


class TensorAlgebraError(TensorEngineError, ValueError):
    """Una operación tensorial solicitada es inconsistente."""


class BackendCapabilityError(TensorEngineError, NotImplementedError):
    """El backend seleccionado no implementa una capacidad declarada."""


class BackendUnavailableError(TensorEngineError, RuntimeError):
    """El runtime externo solicitado no está instalado o no es accesible."""


class BackendExecutionError(TensorEngineError, RuntimeError):
    """Un backend externo terminó con error o devolvió una respuesta inválida."""


class PipelineExecutionError(TensorEngineError, RuntimeError):
    """Una etapa del pipeline integral no pudo completarse."""

    def __init__(self, stage_key: str, message: str) -> None:
        self.stage_key = stage_key
        super().__init__(f"La etapa {stage_key!r} falló: {message}")
