# Orquestador integral

## Objetivo

La fase 10 convierte las operaciones ya verificadas en un flujo ejecutable con
una sola llamada. `TensorEngine.run(model)` recorre el contrato declarado en
`DEFAULT_PIPELINE` y devuelve un `EngineRun` con resultados, tiempos y estados.

El orden efectivo es:

```text
ModelSpec
  -> validación
  -> normalización de la acción
  -> momentos
  -> variación cruda
  -> Euler-Lagrange y frontera
  -> Noether-Wald [opcional]
  -> componentes [opcional, requiere ansatz]
  -> verificación integral
  -> exportación [opcional, requiere ruta]
```

Cada etapa genera un `StageResult` y se comprueba inmediatamente contra su
`StageSpec`. Una divergencia entre implementación y contrato detiene la corrida
con `PipelineExecutionError` e identifica la etapa responsable.

## Normalización global

La etapa `normalize_lagrangian` aplica explícitamente
`model.normalization * model.lagrangian`. Para las operaciones posteriores se
construye una vista interna del modelo con esa expresión y normalización unidad.
El paquete final conserva separadamente el lagrangiano original y el efectivo.

Esta etapa canoniza el orden algebraico, pero no aplica contracciones métricas
que cambien `R_abcd` por una forma con índices elevados. Esa reducción pertenece
a los resultados calculados; el `ModelSpec` efectivo debe seguir cumpliendo el
vocabulario de entrada fijado en la fase 1.
